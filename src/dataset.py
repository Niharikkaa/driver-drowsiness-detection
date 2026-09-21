"""
Step 2 - dataset loading and preprocessing.

Reads every CSV in dataset/, validates it, and produces a scaled, stratified
train/test split. No data is ever generated here: if dataset/ is empty this
module raises, it does not invent rows.
"""

import glob
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src import config


class DatasetError(RuntimeError):
    pass


def dataset_files(dataset_dir=None):
    dataset_dir = config.DATASET_DIR if dataset_dir is None else dataset_dir
    return sorted(glob.glob(os.path.join(dataset_dir, "*.csv")))


def load_dataset(dataset_dir=None):
    """
    Concatenate every CSV in dataset/ into one DataFrame.

    Raises DatasetError with actionable text if nothing usable is found.
    """
    dataset_dir = config.DATASET_DIR if dataset_dir is None else dataset_dir
    files = dataset_files(dataset_dir)

    if not files:
        raise DatasetError(
            f"No CSV files found in '{dataset_dir}/'.\n"
            "Collect data first:  python collect_data.py"
        )

    expected = set(config.FEATURE_COLUMNS) | {config.LABEL_COLUMN}
    frames = []
    for path in files:
        frame = pd.read_csv(path)
        missing = expected - set(frame.columns)
        if missing:
            raise DatasetError(
                f"'{path}' is missing column(s): {sorted(missing)}. "
                "It was probably written by an older version - delete or "
                "re-collect it."
            )
        frame["source_file"] = os.path.basename(path)
        frames.append(frame)

    data = pd.concat(frames, ignore_index=True)

    before = len(data)
    data = data.dropna(subset=config.FEATURE_COLUMNS + [config.LABEL_COLUMN])
    data = data[np.isfinite(data[config.FEATURE_COLUMNS]).all(axis=1)]
    dropped = before - len(data)

    data[config.LABEL_COLUMN] = data[config.LABEL_COLUMN].astype(int)
    valid = set(config.CLASS_NAMES)
    bad = set(data[config.LABEL_COLUMN].unique()) - valid
    if bad:
        raise DatasetError(f"Unexpected label value(s) {sorted(bad)}. "
                           f"Labels must be one of {sorted(valid)}.")

    if data.empty:
        raise DatasetError("Every row was dropped as invalid. Re-collect.")

    data.attrs["files"] = files
    data.attrs["dropped_rows"] = dropped
    return data


def class_counts(data):
    counts = data[config.LABEL_COLUMN].value_counts().to_dict()
    return {cls: int(counts.get(cls, 0)) for cls in sorted(config.CLASS_NAMES)}


def check_dataset(data, min_per_class=None):
    """Return a list of human-readable warnings. Raises if training is
    impossible."""
    min_per_class = (config.MIN_SAMPLES_PER_CLASS if min_per_class is None
                     else min_per_class)
    counts = class_counts(data)
    warnings = []

    present = [cls for cls, n in counts.items() if n > 0]
    if len(present) < 2:
        raise DatasetError(
            "Need at least two classes to train a classifier, but the data "
            f"only contains {present}. Collect samples for the other "
            "classes with collect_data.py."
        )
    if len(present) < len(config.CLASS_NAMES):
        absent = [config.CLASS_NAMES[c] for c in counts if counts[c] == 0]
        warnings.append(f"No samples at all for: {', '.join(absent)}. "
                        "The model can never predict those classes.")

    for cls, n in counts.items():
        # 2 is the minimum for a stratified split to be possible.
        if 0 < n < 2:
            raise DatasetError(
                f"Class {cls} ({config.CLASS_NAMES[cls]}) has {n} sample. "
                "A stratified train/test split needs at least 2."
            )
        if 0 < n < min_per_class:
            warnings.append(
                f"Class {cls} ({config.CLASS_NAMES[cls]}) has only {n} "
                f"samples; {min_per_class}+ recommended. Results will be "
                "unreliable."
            )

    nonzero = [n for n in counts.values() if n > 0]
    if nonzero and max(nonzero) > 3 * min(nonzero):
        warnings.append(
            "Classes are heavily imbalanced (largest is more than 3x the "
            "smallest). Accuracy will look better than the model really is - "
            "read the per-class recall instead."
        )

    dropped = data.attrs.get("dropped_rows", 0)
    if dropped:
        warnings.append(f"{dropped} invalid row(s) were dropped.")

    return warnings


def split_and_scale(data, test_size=None, random_state=None):
    """
    Stratified split, then standardise the features.

    The scaler is fitted on the training set only, then applied to the test
    set, so no test information leaks into training.

    Returns (X_train, X_test, y_train, y_test, scaler).
    """
    test_size = config.TEST_SIZE if test_size is None else test_size
    random_state = (config.RANDOM_STATE if random_state is None
                    else random_state)

    X = data[config.FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    y = data[config.LABEL_COLUMN].to_numpy(dtype=int)

    # Stratify only if every present class has enough rows to appear in both
    # halves; otherwise fall back to a plain split rather than crashing.
    counts = np.bincount(y, minlength=len(config.CLASS_NAMES))
    smallest = counts[counts > 0].min()
    stratify = y if smallest >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state,
        stratify=stratify,
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test, y_train, y_test, scaler
