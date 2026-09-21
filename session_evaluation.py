"""
Session-level leakage check for the drowsiness classifier.

STANDALONE and READ-ONLY with respect to the existing pipeline:
  - does NOT modify src/dataset.py, src/training.py, main.py or collect_data.py
  - does NOT modify or overwrite models/drowsiness_model.joblib
  - only ever WRITES new files under reports/session_evaluation/

Why this script exists
-----------------------
src/dataset.py's split_and_scale() does a random, ROW-level stratified split.
Every CSV in dataset/ is one recording session, and consecutive feature
windows within a session overlap heavily (each window is 10s of a sliding
1s-step signal). A row-level split can and typically will put near-duplicate
windows from the *same* session on both sides of the split, so the model is
partly being tested on data it has effectively already seen. That inflates
test accuracy - most dramatically for Random Forest, which can memorise
near-duplicate rows easily.

This script instead keeps every CSV file (= one recording session) entirely
on one side of the split. That is a strictly harder, more honest test: the
model has to generalise to a session it has never seen at all, not just to
unseen rows.

It reuses, unmodified, the exact building blocks the real pipeline uses:
  - src.dataset.load_dataset()     for CSV loading + validation
  - src.config.FEATURE_COLUMNS     for the feature set
  - src.training.build_models()    for the same 3 model definitions
  - src.training.evaluate()        for the same metric definitions

Usage
-----
    python session_evaluation.py
    python session_evaluation.py --test-size 0.3 --seed 7
    python session_evaluation.py --no-row-level     # skip the comparison block
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from src import config
from src import dataset as dataset_module
from src import training

REPORT_SUBDIR = "session_evaluation"
SEARCH_TRIALS = 4000   # candidate splits tried; cheap, so generous is fine


# ==========================================================================
# Session table
# ==========================================================================

def build_session_table(data):
    """
    Group the already-loaded DataFrame by its 'source_file' column (added by
    src.dataset.load_dataset) into one entry per recording session.
    """
    sessions = []
    for name, group in data.groupby("source_file", sort=True):
        counts = group[config.LABEL_COLUMN].value_counts().to_dict()
        class_counts = {c: int(counts.get(c, 0)) for c in sorted(config.CLASS_NAMES)}
        sessions.append({
            "name": name,
            "rows": int(len(group)),
            "class_counts": class_counts,
            "classes_present": sorted(c for c, n in class_counts.items() if n > 0),
        })
    return sessions


def achievable_classes(sessions):
    """
    A class can only appear on BOTH sides of a session-level split if it
    occurs in at least 2 distinct sessions (a session cannot be split in
    half). Returns (achievable_set, sessions_per_class_count).
    """
    session_count = {c: 0 for c in config.CLASS_NAMES}
    for s in sessions:
        for c in s["classes_present"]:
            session_count[c] += 1
    achievable = {c for c, n in session_count.items() if n >= 2}
    return achievable, session_count


# ==========================================================================
# Session-aware split search
# ==========================================================================

def _score_candidate(test_names, sessions_by_name, total_rows, total_counts,
                     achievable, target_frac):
    """
    Lower is better. Hard-constraint violations (an achievable class missing
    from one side) dominate the score; row-fraction distance from the target
    test size is a soft tie-breaker.
    """
    test_rows = sum(sessions_by_name[n]["rows"] for n in test_names)
    train_rows = total_rows - test_rows
    if test_rows == 0 or train_rows == 0:
        return float("inf"), None

    test_counts = {c: 0 for c in config.CLASS_NAMES}
    for n in test_names:
        for c, cnt in sessions_by_name[n]["class_counts"].items():
            test_counts[c] += cnt
    train_counts = {c: total_counts[c] - test_counts[c]
                    for c in config.CLASS_NAMES}

    penalty = 0.0
    for c in achievable:
        if test_counts[c] == 0:
            penalty += 1000.0
        if train_counts[c] == 0:
            penalty += 1000.0

    frac = test_rows / total_rows
    penalty += abs(frac - target_frac) * 10.0

    info = {
        "test_rows": test_rows, "train_rows": train_rows,
        "test_frac": frac, "test_counts": test_counts,
        "train_counts": train_counts,
    }
    return penalty, info


def find_session_split(sessions, target_frac, seed, n_trials=SEARCH_TRIALS):
    """
    Search for a session-level train/test assignment that:
      1. keeps every session fully on one side (never split a session), and
      2. puts every "achievable" class (one present in 2+ sessions) on both
         sides where that is mathematically possible, and
      3. lands close to the requested test-set row fraction.

    A class present in only a single session can never be on both sides -
    that is reported explicitly, not silently hidden.

    Deterministic for a fixed seed: candidates come from row-fraction
    prefixes of RNG-shuffled session orders, scored, and the best kept.
    """
    sessions_by_name = {s["name"]: s for s in sessions}
    names = list(sessions_by_name)
    total_rows = sum(s["rows"] for s in sessions)
    total_counts = {c: sum(s["class_counts"].get(c, 0) for s in sessions)
                    for c in config.CLASS_NAMES}
    achievable, session_count = achievable_classes(sessions)

    if len(names) < 2:
        raise RuntimeError(
            "Need at least 2 CSV sessions in dataset/ to build a session-"
            f"level split; found {len(names)}."
        )

    rng = np.random.default_rng(seed)
    best_names, best_score, best_info = None, None, None

    for _ in range(n_trials):
        order = rng.permutation(names)
        cum, test_names = 0, []
        for n in order:
            if cum / total_rows >= target_frac:
                break
            test_names.append(n)
            cum += sessions_by_name[n]["rows"]
        if not test_names or len(test_names) == len(names):
            continue  # a degenerate all-or-nothing split is never useful

        score, info = _score_candidate(test_names, sessions_by_name,
                                       total_rows, total_counts, achievable,
                                       target_frac)
        if best_score is None or score < best_score:
            best_score, best_names, best_info = score, list(test_names), info

    if best_names is None:
        raise RuntimeError(
            "Could not find any valid session split. This can happen with "
            "very few sessions or a very skewed row distribution - try "
            "--test-size closer to 0.5 or collect more sessions."
        )

    train_names = sorted(n for n in names if n not in best_names)
    test_names = sorted(best_names)
    return {
        "train_sessions": train_names,
        "test_sessions": test_names,
        "achievable_classes": sorted(achievable),
        "sessions_per_class": session_count,
        "split_penalty": best_score,
        **best_info,
    }


# ==========================================================================
# Reporting
# ==========================================================================

def _print_session_table(sessions):
    print(f"\n{len(sessions)} recording session(s) found in dataset/:")
    header = f"  {'session file':<34}{'rows':>6}   classes present"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for s in sessions:
        names = ", ".join(config.CLASS_NAMES[c] for c in s["classes_present"])
        mixed = " (mixed)" if len(s["classes_present"]) > 1 else ""
        print(f"  {s['name']:<34}{s['rows']:>6}   {names}{mixed}")


def _print_split(split):
    print(f"\nSession-level split (penalty={split['split_penalty']:.2f}, "
          f"0.0 = all hard constraints satisfied):")
    print(f"  Train sessions ({len(split['train_sessions'])}, "
          f"{split['train_rows']} rows):")
    for name in split["train_sessions"]:
        print(f"    - {name}")
    print(f"  Test sessions  ({len(split['test_sessions'])}, "
          f"{split['test_rows']} rows, "
          f"{split['test_frac'] * 100:.1f}% of all rows):")
    for name in split["test_sessions"]:
        print(f"    - {name}")

    print("\n  Row count per class:")
    print(f"    {'class':<10}{'train':>8}{'test':>8}")
    for c in sorted(config.CLASS_NAMES):
        print(f"    {config.CLASS_NAMES[c]:<10}"
              f"{split['train_counts'][c]:>8}{split['test_counts'][c]:>8}")

    unachievable = set(config.CLASS_NAMES) - set(split["achievable_classes"])
    if unachievable:
        print("\n  NOTE - structurally limited classes (present in only one "
              "session each, so they cannot appear on both sides of any "
              "session-level split):")
        for c in sorted(unachievable):
            n_sessions = split["sessions_per_class"][c]
            side = ("test" if split["test_counts"][c] > 0
                    else "train" if split["train_counts"][c] > 0 else "neither")
            print(f"    - {config.CLASS_NAMES[c]}: in {n_sessions} session(s), "
                  f"ends up in: {side}")


def _print_results_table(results, title):
    header = f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}{'Recall':>9}{'F1':>8}"
    print(f"\n{title}")
    print(header)
    print("-" * len(header))
    for name, entry in results.items():
        m = entry["metrics"]
        print(f"{name:<20}{m['accuracy']:>10.3f}{m['precision_macro']:>11.3f}"
              f"{m['recall_macro']:>9.3f}{m['f1_macro']:>8.3f}")
    print("-" * len(header))
    print("Precision / Recall / F1 are macro-averaged over the three classes.")


def _print_confusion(results):
    labels = sorted(config.CLASS_NAMES)
    names = [config.CLASS_NAMES[c] for c in labels]
    for model_name, entry in results.items():
        print(f"\nConfusion matrix - {model_name} "
              f"(rows = true, cols = predicted, order {names}):")
        for row in entry["metrics"]["confusion_matrix"]:
            print("  " + "".join(f"{v:>6}" for v in row))


def _save_confusion_plot(matrix, model_name, path):
    """PNG of a confusion matrix. Skipped silently if matplotlib is missing."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    labels = sorted(config.CLASS_NAMES)
    names = [config.CLASS_NAMES[c] for c in labels]
    matrix = np.array(matrix)

    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(names)), names)
    ax.set_yticks(range(len(names)), names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Session-level confusion matrix - {model_name}")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def _json_safe(obj):
    """Recursively convert numpy scalars/arrays so json.dump doesn't choke."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


# ==========================================================================
# Main
# ==========================================================================

def run(dataset_dir=None, output_dir=None, test_size=None, seed=None,
       trials=SEARCH_TRIALS, run_row_level=True):
    dataset_dir = config.DATASET_DIR if dataset_dir is None else dataset_dir
    test_size = config.TEST_SIZE if test_size is None else test_size
    seed = config.RANDOM_STATE if seed is None else seed
    output_dir = (os.path.join(config.REPORT_DIR, REPORT_SUBDIR)
                  if output_dir is None else output_dir)

    # ---- load (reuses the existing, unmodified loader/validator) --------
    try:
        data = dataset_module.load_dataset(dataset_dir)
    except dataset_module.DatasetError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    sessions = build_session_table(data)
    _print_session_table(sessions)

    n_files = data.attrs.get("files", [])
    print(f"\nTotal: {len(data)} samples across {len(sessions)} session "
          f"file(s) (source: {len(n_files)} CSV file(s) in '{dataset_dir}/').")

    # ---- session-aware split --------------------------------------------
    try:
        split = find_session_split(sessions, test_size, seed, trials)
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1
    _print_split(split)

    train_mask = data["source_file"].isin(split["train_sessions"])
    test_mask = data["source_file"].isin(split["test_sessions"])

    X_train = data.loc[train_mask, config.FEATURE_COLUMNS].to_numpy(np.float64)
    y_train = data.loc[train_mask, config.LABEL_COLUMN].to_numpy(int)
    X_test = data.loc[test_mask, config.FEATURE_COLUMNS].to_numpy(np.float64)
    y_test = data.loc[test_mask, config.LABEL_COLUMN].to_numpy(int)

    # Scaler fit on TRAIN SESSIONS ONLY - no test-session information leaks
    # into preprocessing, matching what split_and_scale() does at the row
    # level, just applied at the session level here.
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ---- train the same 3 models via the existing, unmodified helper ----
    print("\nTraining Logistic Regression, Decision Tree, Random Forest "
          "on the session-level split...")
    session_results = training.train_all(X_train_scaled, y_train,
                                         X_test_scaled, y_test)
    _print_results_table(session_results, "SESSION-LEVEL results "
                         "(sessions never split - the honest number)")
    _print_confusion(session_results)

    for name, entry in session_results.items():
        print(f"\nPer-class report ({name}):\n{entry['metrics']['report']}")

    # ---- optional side-by-side: today's row-level split, for contrast ---
    row_results = None
    if run_row_level:
        print("\n" + "=" * 74)
        print("For comparison, this is what the EXISTING row-level split "
              "(src.dataset.split_and_scale, the same one train_model.py "
              "uses) reports on the identical data:")
        X_tr_r, X_te_r, y_tr_r, y_te_r, _ = dataset_module.split_and_scale(
            data, test_size=test_size, random_state=seed)
        row_results = training.train_all(X_tr_r, y_tr_r, X_te_r, y_te_r)
        _print_results_table(row_results, "ROW-LEVEL results (rows from the "
                             "same session can land on both sides)")

        print("\nGap (row-level accuracy minus session-level accuracy) - a "
              "large positive gap is the signature of session leakage:")
        for name in session_results:
            gap = (row_results[name]["metrics"]["accuracy"]
                   - session_results[name]["metrics"]["accuracy"])
            print(f"  {name:<20}{gap:+.3f}")

    # ---- save report ------------------------------------------------------
    os.makedirs(output_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")

    summary = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_dir": dataset_dir,
        "n_samples": int(len(data)),
        "n_sessions": len(sessions),
        "test_size_target": test_size,
        "random_seed": seed,
        "sessions": sessions,
        "split": {k: v for k, v in split.items()},
        "session_level": {
            name: {"metrics": {k: v for k, v in entry["metrics"].items()
                               if k != "report"}}
            for name, entry in session_results.items()
        },
    }
    if row_results is not None:
        summary["row_level_comparison"] = {
            name: {"metrics": {k: v for k, v in entry["metrics"].items()
                               if k != "report"}}
            for name, entry in row_results.items()
        }
        summary["accuracy_gap_row_minus_session"] = {
            name: (row_results[name]["metrics"]["accuracy"]
                   - session_results[name]["metrics"]["accuracy"])
            for name in session_results
        }

    json_path = os.path.join(output_dir, f"session_evaluation_{stamp}.json")
    with open(json_path, "w") as handle:
        json.dump(_json_safe(summary), handle, indent=2)

    plot_paths = []
    for name, entry in session_results.items():
        p = os.path.join(output_dir, f"confusion_{name}_{stamp}.png")
        saved = _save_confusion_plot(entry["metrics"]["confusion_matrix"],
                                     name, p)
        if saved:
            plot_paths.append(saved)

    print(f"\nSaved JSON report to: {json_path}")
    if plot_paths:
        print("Saved confusion matrix plots to:")
        for p in plot_paths:
            print(f"  {p}")
    print("\nNothing under models/ was created, modified or overwritten by "
          "this script.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Session-aware leakage check for the drowsiness "
                    "classifier. Read-only with respect to the existing "
                    "training/prediction pipeline.")
    parser.add_argument("--dataset-dir", default=None,
                        help="defaults to config.DATASET_DIR ('dataset')")
    parser.add_argument("--output-dir", default=None,
                        help="defaults to reports/session_evaluation/")
    parser.add_argument("--test-size", type=float, default=None,
                        help="target fraction of rows in the test sessions "
                             "(defaults to config.TEST_SIZE)")
    parser.add_argument("--seed", type=int, default=None,
                        help="defaults to config.RANDOM_STATE")
    parser.add_argument("--trials", type=int, default=SEARCH_TRIALS,
                        help="candidate splits to search (default %(default)s)")
    parser.add_argument("--no-row-level", action="store_true",
                        help="skip the row-level comparison block")
    args = parser.parse_args()

    sys.exit(run(
        dataset_dir=args.dataset_dir,
        output_dir=args.output_dir,
        test_size=args.test_size,
        seed=args.seed,
        trials=args.trials,
        run_row_level=not args.no_row_level,
    ))
