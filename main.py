"""
Driver Drowsiness Detection - entry point.

This file only wires modules together. All logic lives in src/.

Pipeline:
    Webcam -> OpenCV -> MediaPipe FaceMesh -> eye/mouth landmarks
           -> EAR / MAR -> eye open-closed -> blink, closure duration, yawn
           -> overlay -> display

With --model it additionally runs the Step 2 classifier:
           -> sliding feature window -> trained model -> ALERT/WARNING/DROWSY
"""

import argparse
import sys
import time

import cv2

from src import config
from src.detectors import DrowsinessMonitor
from src.face_mesh import FaceMeshDetector
from src.features import FeatureWindow
from src.predictor import DrowsinessClassifier
from src.video import CameraError, FPSCounter, open_camera, read_frame
from src import metrics as metrics_module
from src import visualization

WINDOW_NAME = "Driver Drowsiness Detection"


def run(use_model=False):
    # ---- optional Step 2 classifier -------------------------------------
    classifier = None
    window = None
    if use_model:
        classifier, error = DrowsinessClassifier.try_load()
        if classifier is None:
            print(f"[ERROR] {error}", file=sys.stderr)
            return 1
        window = FeatureWindow()
        print(f"[INFO] Loaded model: {classifier.summary()}")

    try:
        cap = open_camera()
    except CameraError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    monitor = DrowsinessMonitor()
    fps_counter = FPSCounter()
    consecutive_read_failures = 0
    prediction = None

    print("[INFO] Camera opened. Press 'q' to quit, 'r' to reset counters.")

    with FaceMeshDetector() as face_mesh:
        while True:
            frame = read_frame(cap)
            if frame is None:
                consecutive_read_failures += 1
                if consecutive_read_failures >= 30:
                    print("[ERROR] Lost the camera feed. Exiting.",
                          file=sys.stderr)
                    break
                continue
            consecutive_read_failures = 0

            fps = fps_counter.tick()

            points = face_mesh.process(frame)

            if points is None:
                status = monitor.update_missing()
            else:
                try:
                    values = metrics_module.compute_metrics(points)
                except (IndexError, ValueError):
                    # Landmark set was incomplete for this frame.
                    status = monitor.update_missing()
                    points = None
                else:
                    status = monitor.update(values)

            # ---- Step 2: sliding window -> classifier ------------------
            if classifier is not None:
                now = time.time()
                window.update(status, now, points is not None)
                if status["face_found"] and window.ready:
                    prediction = classifier.predict(
                        window.extract(status["closure"]))
                elif not status["face_found"]:
                    classifier.reset()
                    prediction = None

            visualization.render(frame, points, status, fps)
            if classifier is not None:
                visualization.draw_prediction(
                    frame,
                    prediction.class_id if prediction else None,
                    prediction.confidence if prediction else 0.0,
                )
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("r"):
                monitor.reset()
                if classifier is not None:
                    classifier.reset()
                    window.reset()
                    prediction = None
                print("[INFO] Counters reset.")

            # Stop if the user closed the window with the title-bar X.
            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

    summary = monitor.status()
    print("\n--- Session summary ---")
    print(f"Blinks detected      : {summary['blinks']}")
    print(f"Yawns detected       : {summary['yawns']}")
    print(f"Longest eye closure  : {summary['longest_closure']:.2f}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Driver drowsiness detection.")
    parser.add_argument(
        "--model", action="store_true",
        help="also run the trained ML classifier (requires train_model.py)")
    args = parser.parse_args()
    sys.exit(run(args.model))
