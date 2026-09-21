"""
Thin wrapper around MediaPipe face landmark detection.

Responsibility: take a BGR frame, give back an (N, 2) NumPy array of landmark
pixel coordinates, or None when no face is found. Nothing else.

MediaPipe ships two APIs and which one you get depends on the build:

  * "solutions"  - the classic mp.solutions.face_mesh.FaceMesh. This is what a
                   standard `pip install mediapipe` provides, and it needs no
                   extra model file. Preferred.
  * "tasks"      - the newer mediapipe.tasks FaceLandmarker. Some slim builds
                   ship only this one. It needs a face_landmarker.task model
                   file (see README).

This module detects what is available and uses it, so the rest of the project
never has to care.
"""

import os

import cv2
import numpy as np
import mediapipe as mp

from src import config

# Default location for the Tasks-API model file, relative to the project root.
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TASK_MODEL_PATH = os.path.join(_PROJECT_ROOT, "models", "face_landmarker.task")


def _solutions_available():
    return hasattr(mp, "solutions") and hasattr(mp.solutions, "face_mesh")


def _tasks_available():
    try:
        from mediapipe.tasks.python import vision  # noqa: F401
        return True
    except ImportError:
        return False


class FaceMeshDetector:
    """
    Parameters
    ----------
    backend : "auto" | "solutions" | "tasks"
        Leave as "auto" unless you are debugging.
    """

    def __init__(self, backend="auto"):
        self.backend = self._resolve_backend(backend)
        self._frame_index = 0

        if self.backend == "solutions":
            self._init_solutions()
        else:
            self._init_tasks()

    # ------------------------------------------------------------------
    # Backend setup
    # ------------------------------------------------------------------
    @staticmethod
    def _resolve_backend(requested):
        if requested == "solutions":
            if not _solutions_available():
                raise RuntimeError(
                    "mp.solutions.face_mesh is not available in this "
                    "mediapipe build."
                )
            return "solutions"

        if requested == "tasks":
            if not _tasks_available():
                raise RuntimeError(
                    "mediapipe.tasks is not available in this mediapipe build."
                )
            return "tasks"

        if _solutions_available():
            return "solutions"
        if _tasks_available():
            return "tasks"

        raise RuntimeError(
            "No usable MediaPipe face landmark API was found. "
            "Reinstall with: pip install --force-reinstall mediapipe"
        )

    def _init_solutions(self):
        self._mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=config.MAX_NUM_FACES,
            refine_landmarks=config.REFINE_LANDMARKS,
            min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )

    def _init_tasks(self):
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python import vision

        if not os.path.exists(TASK_MODEL_PATH):
            raise RuntimeError(
                "This MediaPipe build only provides the Tasks API, which "
                f"needs a model file at:\n  {TASK_MODEL_PATH}\n"
                "Download it once with:\n"
                "  mkdir -p models && curl -L -o models/face_landmarker.task "
                "https://storage.googleapis.com/mediapipe-models/"
                "face_landmarker/face_landmarker/float16/1/"
                "face_landmarker.task\n"
                "Alternatively install a full mediapipe build:\n"
                "  pip install --force-reinstall mediapipe==0.10.14"
            )

        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=TASK_MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=config.MAX_NUM_FACES,
            min_face_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
        )
        self._mesh = vision.FaceLandmarker.create_from_options(options)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def process(self, frame):
        """
        Returns
        -------
        np.ndarray of shape (N, 2) with x/y in pixels, or None if no face.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return None

        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self.backend == "solutions":
            landmarks = self._process_solutions(rgb)
        else:
            landmarks = self._process_tasks(rgb)

        if not landmarks:
            return None

        points = np.array(
            [(lm.x * width, lm.y * height) for lm in landmarks],
            dtype=np.float32,
        )

        # Guard against a degenerate track producing NaN/inf coordinates.
        if not np.isfinite(points).all():
            return None

        # Every index this project reads must exist.
        if points.shape[0] < 468:
            return None

        return points

    def _process_solutions(self, rgb):
        rgb.flags.writeable = False
        results = self._mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None
        return results.multi_face_landmarks[0].landmark

    def _process_tasks(self, rgb):
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        # Tasks VIDEO mode requires a strictly increasing timestamp in ms.
        self._frame_index += 1
        result = self._mesh.detect_for_video(mp_image, self._frame_index * 33)
        if not result.face_landmarks:
            return None
        return result.face_landmarks[0]

    # ------------------------------------------------------------------
    def close(self):
        if getattr(self, "_mesh", None) is not None:
            self._mesh.close()
            self._mesh = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
