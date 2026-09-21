"""
Webcam access and FPS measurement.
"""

import time
from collections import deque

import cv2

from src import config


class CameraError(RuntimeError):
    pass


def open_camera(index=None):
    """Open the webcam and apply the configured resolution."""
    index = config.CAMERA_INDEX if index is None else index

    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise CameraError(
            f"Could not open camera index {index}. "
            "Check that a webcam is connected, that no other application is "
            "using it, and that CAMERA_INDEX in src/config.py is correct."
        )

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
    return cap


def read_frame(cap):
    """Grab one frame, mirrored if configured. Returns None on failure."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if config.FLIP_FRAME:
        frame = cv2.flip(frame, 1)
    return frame


class FPSCounter:
    """Rolling FPS over the last `window` frame intervals."""

    def __init__(self, window=30):
        self._times = deque(maxlen=window)
        self._last = None

    def tick(self):
        now = time.time()
        if self._last is not None:
            self._times.append(now - self._last)
        self._last = now
        return self.value

    @property
    def value(self):
        if not self._times:
            return 0.0
        average = sum(self._times) / len(self._times)
        return 1.0 / average if average > 0 else 0.0
