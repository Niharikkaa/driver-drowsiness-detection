"""
Geometric metrics computed from facial landmarks.

EAR = Eye Aspect Ratio  -> drops sharply when an eye closes.
MAR = Mouth Aspect Ratio -> rises sharply when the mouth opens wide.

Both are scale-invariant ratios, so they do not depend on how far the driver
is sitting from the camera.
"""

import numpy as np

from src import config

_EPS = 1e-6


def _dist(p1, p2):
    """Euclidean distance between two 2-D points."""
    return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))


def eye_aspect_ratio(points, eye_indices):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)

    eye_indices is ordered [outer, upper1, upper2, inner, lower2, lower1].
    """
    p1, p2, p3, p4, p5, p6 = (points[i] for i in eye_indices)
    vertical = _dist(p2, p6) + _dist(p3, p5)
    horizontal = _dist(p1, p4)
    return vertical / (2.0 * horizontal + _EPS)


def mouth_aspect_ratio(points):
    """
    MAR = mean(vertical lip gaps) / (mouth width)

    Uses three vertical pairs across the inner lip so a single noisy landmark
    cannot swing the result.
    """
    left, right = (points[i] for i in config.MOUTH_CORNERS)
    width = _dist(left, right)

    gaps = [_dist(points[top], points[bottom])
            for top, bottom in config.MOUTH_VERTICAL_PAIRS]

    return float(np.mean(gaps)) / (width + _EPS)


def compute_metrics(points):
    """
    Compute every per-frame metric at once.

    Returns a dict with left_ear, right_ear, ear (average) and mar.
    """
    left_ear = eye_aspect_ratio(points, config.LEFT_EYE_EAR)
    right_ear = eye_aspect_ratio(points, config.RIGHT_EYE_EAR)

    return {
        "left_ear": left_ear,
        "right_ear": right_ear,
        "ear": (left_ear + right_ear) / 2.0,
        "mar": mouth_aspect_ratio(points),
    }
