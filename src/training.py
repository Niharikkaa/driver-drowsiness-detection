"""
Step 2 - model training, evaluation and persistence.

Trains three lightweight classifiers, scores them on a held-out test set, and
saves the best one together with its scaler so real-time prediction applies
exactly the same preprocessing.

No deep learning: these are all plain scikit-learn estimators that train in
under a second on a dataset of this size.
"""

import os
import time

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier

from src import config


def build_models(random_state=None):
    """The three candidate classifiers, in the order they are reported."""
    random_state = (config.RANDOM_STATE if random_state is None
                    else random_state)
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=6,                # shallow: keeps it fast and readable
            min_samples_leaf=3,         # guards against overfitting tiny sets
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=120,
            max_depth=10,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }


def evaluate(model, X_test, y_test):
    """Accuracy, macro/weighted precision, recall and F1, plus per-class."""
    y_pred = model.predict(X_test)
    kwargs = {"zero_division": 0}

    labels = sorted(config.CLASS_NAMES)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision_macro": precision_score(y_test, y_pred, average="macro",
                                           **kwargs),
        "recall_macro": recall_score(y_test, y_pred, average="macro",
                                     **kwargs),
        "f1_macro": f1_score(y_test, y_pred, average="macro", **kwargs),
        "precision_weighted": precision_score(y_test, y_pred,
                                              average="weighted", **kwargs),
        "recall_weighted": recall_score(y_test, y_pred, average="weighted",
                                        **kwargs),
        "f1_weighted": f1_score(y_test, y_pred, average="weighted", **kwargs),
        "per_class": {
            int(cls): {
                "precision": float(p), "recall": float(r), "f1": float(f),
                "support": int(s),
            }
            for cls, p, r, f, s in zip(
                labels,
                precision_score(y_test, y_pred, labels=labels, average=None,
                                **kwargs),
                recall_score(y_test, y_pred, labels=labels, average=None,
                             **kwargs),
                f1_score(y_test, y_pred, labels=labels, average=None,
                         **kwargs),
                np.bincount(y_test, minlength=len(labels))[labels],
            )
        },
        "confusion_matrix": confusion_matrix(y_test, y_pred,
                                             labels=labels).tolist(),
        "report": classification_report(
            y_test, y_pred, labels=labels,
            target_names=[config.CLASS_NAMES[c] for c in labels],
            zero_division=0,
        ),
    }


def cross_validate(model, X, y, folds=None):
    """
    Mean and std of cross-validated macro F1.

    A single train/test split on a small dataset is noisy; this gives a more
    honest picture. Returns (mean, std, folds_used) or (None, None, 0) when
    there are too few samples per class to fold.
    """
    folds = config.CV_FOLDS if folds is None else folds
    counts = np.bincount(y)
    smallest = counts[counts > 0].min()
    usable = min(folds, int(smallest))

    if usable < 2:
        return None, None, 0

    splitter = StratifiedKFold(n_splits=usable, shuffle=True,
                               random_state=config.RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=splitter, scoring="f1_macro")
    return float(scores.mean()), float(scores.std()), usable


def train_all(X_train, y_train, X_test, y_test, X_all=None, y_all=None):
    """
    Fit and score every candidate.

    Returns an ordered dict: name -> {model, metrics, cv_mean, cv_std,
    cv_folds, fit_seconds}.
    """
    results = {}
    for name, model in build_models().items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        fit_seconds = time.perf_counter() - started

        metrics = evaluate(model, X_test, y_test)

        cv_mean = cv_std = None
        cv_folds = 0
        if X_all is not None and y_all is not None:
            fresh = build_models()[name]
            cv_mean, cv_std, cv_folds = cross_validate(fresh, X_all, y_all)

        results[name] = {
            "model": model,
            "metrics": metrics,
            "cv_mean": cv_mean,
            "cv_std": cv_std,
            "cv_folds": cv_folds,
            "fit_seconds": fit_seconds,
        }
    return results


def pick_best(results):
    """
    Best by macro F1 on the test set; cross-validated F1 breaks ties.

    Macro F1 rather than accuracy, because DROWSY is the class that matters
    and it is usually the rarest - accuracy would let a model ignore it.
    """
    def score(item):
        entry = item[1]
        return (entry["metrics"]["f1_macro"], entry["cv_mean"] or 0.0)

    return max(results.items(), key=score)[0]


def save_model(model, scaler, model_name, metrics, dataset_info,
               model_dir=None, filename=None):
    """Persist the classifier, the scaler and enough metadata to audit it."""
    model_dir = config.MODEL_DIR if model_dir is None else model_dir
    filename = config.MODEL_FILE if filename is None else filename
    os.makedirs(model_dir, exist_ok=True)
    path = os.path.join(model_dir, filename)

    joblib.dump({
        "model": model,
        "scaler": scaler,
        "model_name": model_name,
        "feature_columns": list(config.FEATURE_COLUMNS),
        "class_names": dict(config.CLASS_NAMES),
        "metrics": {k: v for k, v in metrics.items() if k != "report"},
        "dataset_info": dataset_info,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "window_sec": config.FEATURE_WINDOW_SEC,
    }, path)

    return path


def load_model(model_dir=None, filename=None):
    """Load a saved bundle. Raises FileNotFoundError with actionable text."""
    model_dir = config.MODEL_DIR if model_dir is None else model_dir
    filename = config.MODEL_FILE if filename is None else filename
    path = os.path.join(model_dir, filename)

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No trained model at '{path}'.\n"
            "Collect data then train:\n"
            "  python collect_data.py\n"
            "  python train_model.py"
        )

    bundle = joblib.load(path)

    saved_columns = bundle.get("feature_columns", [])
    if list(saved_columns) != list(config.FEATURE_COLUMNS):
        raise ValueError(
            "The saved model expects different features than config.py "
            "defines. FEATURE_COLUMNS changed since training - retrain with "
            "python train_model.py"
        )
    return bundle
