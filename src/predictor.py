"""
Step 2 - real-time prediction.

The clean interface main.py calls. Wraps the saved model + scaler so callers
never have to know about joblib, feature ordering or preprocessing.

    from src.predictor import DrowsinessClassifier

    clf = DrowsinessClassifier.load()            # raises if not trained yet
    result = clf.predict(features)               # features from FeatureWindow
    result.class_id, result.label, result.confidence
"""

from collections import Counter, deque
from dataclasses import dataclass

import numpy as np

from src import config
from src import training
from src.features import features_to_array


@dataclass
class Prediction:
    """One classifier verdict."""
    class_id: int
    label: str
    confidence: float
    raw_class_id: int          # before smoothing
    probabilities: dict        # class_id -> probability

    @property
    def is_drowsy(self):
        return self.class_id == config.CLASS_DROWSY

    @property
    def is_warning(self):
        return self.class_id == config.CLASS_WARNING


class DrowsinessClassifier:
    """
    Loads a trained bundle and turns feature dicts into predictions.

    Consecutive predictions are smoothed with a majority vote over the last
    PREDICTION_SMOOTHING results, so one noisy window cannot make the verdict
    flicker.
    """

    def __init__(self, bundle, smoothing=None):
        self._model = bundle["model"]
        self._scaler = bundle["scaler"]
        self.model_name = bundle.get("model_name", "unknown")
        self.trained_at = bundle.get("trained_at", "unknown")
        self.metrics = bundle.get("metrics", {})
        self.dataset_info = bundle.get("dataset_info", {})

        smoothing = (config.PREDICTION_SMOOTHING if smoothing is None
                     else smoothing)
        self._history = deque(maxlen=max(1, int(smoothing)))

    # ------------------------------------------------------------------
    @classmethod
    def load(cls, model_dir=None, filename=None, smoothing=None):
        """Load from models/. Raises FileNotFoundError if not trained yet."""
        bundle = training.load_model(model_dir, filename)
        return cls(bundle, smoothing)

    @classmethod
    def try_load(cls, model_dir=None, filename=None, smoothing=None):
        """Same as load() but returns (classifier, error_message)."""
        try:
            return cls.load(model_dir, filename, smoothing), None
        except (FileNotFoundError, ValueError, OSError) as exc:
            return None, str(exc)

    # ------------------------------------------------------------------
    def predict(self, features):
        """
        Parameters
        ----------
        features : dict from FeatureWindow.extract(), or None if the window
                   is not ready.

        Returns
        -------
        Prediction, or None when features is None.
        """
        if features is None:
            return None

        X = features_to_array(features)
        if not np.isfinite(X).all():
            return None

        X = self._scaler.transform(X)

        raw_class = int(self._model.predict(X)[0])

        if hasattr(self._model, "predict_proba"):
            proba = self._model.predict_proba(X)[0]
            probabilities = {int(c): float(p)
                             for c, p in zip(self._model.classes_, proba)}
        else:
            probabilities = {raw_class: 1.0}

        self._history.append(raw_class)
        smoothed = Counter(self._history).most_common(1)[0][0]

        return Prediction(
            class_id=int(smoothed),
            label=config.CLASS_NAMES.get(int(smoothed), str(smoothed)),
            confidence=float(probabilities.get(int(smoothed), 0.0)),
            raw_class_id=raw_class,
            probabilities=probabilities,
        )

    def reset(self):
        """Clear the smoothing history, e.g. after the face is lost."""
        self._history.clear()

    # ------------------------------------------------------------------
    def summary(self):
        """One-line description of the loaded model, for logging."""
        accuracy = self.metrics.get("accuracy")
        f1 = self.metrics.get("f1_macro")
        parts = [f"{self.model_name} (trained {self.trained_at})"]
        if accuracy is not None:
            parts.append(f"test accuracy {accuracy:.3f}")
        if f1 is not None:
            parts.append(f"macro F1 {f1:.3f}")
        return ", ".join(parts)
