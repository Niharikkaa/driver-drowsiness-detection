"""
Step 2 - train and compare classifiers.

    python train_model.py

Loads every CSV in dataset/, preprocesses it, trains Logistic Regression,
Decision Tree and Random Forest, prints a comparison, and saves the best model
plus its scaler to models/.

All reported numbers come from your collected data. Nothing here generates
samples or invents scores.
"""

import argparse
import os
import sys

from src import config
from src import dataset as dataset_module
from src import training


def _print_table(results):
    header = (f"{'Model':<20}{'Accuracy':>10}{'Precision':>11}"
              f"{'Recall':>9}{'F1':>8}{'CV F1':>16}")
    print("\n" + header)
    print("-" * len(header))
    for name, entry in results.items():
        m = entry["metrics"]
        if entry["cv_mean"] is None:
            cv = "n/a"
        else:
            cv = f"{entry['cv_mean']:.3f} +/- {entry['cv_std']:.3f}"
        print(f"{name:<20}{m['accuracy']:>10.3f}{m['precision_macro']:>11.3f}"
              f"{m['recall_macro']:>9.3f}{m['f1_macro']:>8.3f}{cv:>16}")
    print("-" * len(header))
    print("Precision / Recall / F1 are macro-averaged over the three classes.")


def _save_confusion_plot(matrix, model_name, path):
    """Confusion matrix PNG. Skipped silently if matplotlib is unavailable."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
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
    ax.set_title(f"Confusion matrix - {model_name}")

    threshold = matrix.max() / 2 if matrix.max() else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, int(matrix[i, j]), ha="center", va="center",
                    color="white" if matrix[i, j] > threshold else "black")

    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def run(dataset_dir=None, model_dir=None, no_plot=False):
    dataset_dir = config.DATASET_DIR if dataset_dir is None else dataset_dir
    model_dir = config.MODEL_DIR if model_dir is None else model_dir

    # ---- load -------------------------------------------------------
    try:
        data = dataset_module.load_dataset(dataset_dir)
        warnings = dataset_module.check_dataset(data)
    except dataset_module.DatasetError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    counts = dataset_module.class_counts(data)
    files = data.attrs.get("files", [])

    print(f"Loaded {len(data)} samples from {len(files)} file(s) "
          f"in '{dataset_dir}/'")
    for cls, n in counts.items():
        share = n / len(data) * 100 if len(data) else 0
        print(f"  {cls} {config.CLASS_NAMES[cls]:<8}: {n:>5}  ({share:5.1f}%)")

    for warning in warnings:
        print(f"[WARN] {warning}")

    # ---- preprocess -------------------------------------------------
    X_train, X_test, y_train, y_test, scaler = dataset_module.split_and_scale(
        data)
    print(f"\nTrain / test split: {len(y_train)} / {len(y_test)} "
          f"(test_size={config.TEST_SIZE}), features standardised.")

    # Cross-validation runs on the whole scaled set for a steadier estimate.
    X_all = scaler.transform(data[config.FEATURE_COLUMNS].to_numpy(float))
    y_all = data[config.LABEL_COLUMN].to_numpy(int)

    # ---- train ------------------------------------------------------
    print("Training Logistic Regression, Decision Tree, Random Forest...")
    results = training.train_all(X_train, y_train, X_test, y_test,
                                 X_all, y_all)
    _print_table(results)

    best_name = training.pick_best(results)
    best = results[best_name]
    print(f"\nBest model by macro F1: {best_name}")
    print(f"\nPer-class report ({best_name}):\n{best['metrics']['report']}")

    labels = sorted(config.CLASS_NAMES)
    print("Confusion matrix (rows = true, cols = predicted, order "
          f"{[config.CLASS_NAMES[c] for c in labels]}):")
    for row in best["metrics"]["confusion_matrix"]:
        print("  " + "".join(f"{v:>6}" for v in row))

    # ---- feature importance, when the model exposes it ---------------
    estimator = best["model"]
    if hasattr(estimator, "feature_importances_"):
        print("\nMost useful features:")
        ranked = sorted(zip(config.FEATURE_COLUMNS,
                            estimator.feature_importances_),
                        key=lambda pair: pair[1], reverse=True)
        for name, importance in ranked[:5]:
            print(f"  {name:<18}{importance:.3f}")

    # ---- save -------------------------------------------------------
    path = training.save_model(
        model=estimator,
        scaler=scaler,
        model_name=best_name,
        metrics=best["metrics"],
        dataset_info={
            "n_samples": int(len(data)),
            "class_counts": counts,
            "files": [os.path.basename(f) for f in files],
        },
        model_dir=model_dir,
    )
    print(f"\nSaved model + scaler to: {path}")

    if not no_plot:
        plot_path = _save_confusion_plot(
            best["metrics"]["confusion_matrix"], best_name,
            os.path.join(config.REPORT_DIR, "confusion_matrix.png"))
        if plot_path:
            print(f"Saved confusion matrix to: {plot_path}")

    print("\nTest it live with:  python main.py --model")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train drowsiness models.")
    parser.add_argument("--dataset-dir", default=None)
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    sys.exit(run(args.dataset_dir, args.model_dir, args.no_plot))
