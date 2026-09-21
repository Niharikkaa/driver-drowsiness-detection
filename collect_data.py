"""
Step 2 - labelled dataset collection.

Runs the exact Step 1 pipeline (webcam -> MediaPipe -> EAR/MAR -> detectors),
wraps it in a sliding feature window, and writes one labelled row to CSV every
SAMPLE_INTERVAL_SEC while you are recording.

    python collect_data.py

Controls
    0 / 1 / 2   choose the label: ALERT / WARNING / DROWSY
    SPACE       start or pause recording
    u           undo the last 10 samples of the current label
    s           save now (also happens automatically on quit)
    q / Esc     quit

Nothing is written until the window fill bar is full, so every row is built
from a complete window of real frames.
"""

import argparse
import csv
import os
import sys
import time

import cv2

from src import config
from src import metrics as metrics_module
from src import visualization
from src.detectors import DrowsinessMonitor
from src.face_mesh import FaceMeshDetector
from src.features import FeatureWindow
from src.video import CameraError, FPSCounter, open_camera, read_frame

WINDOW_NAME = "Drowsiness - Data Collection"

KEY_TO_LABEL = {
    ord("0"): config.CLASS_ALERT,
    ord("1"): config.CLASS_WARNING,
    ord("2"): config.CLASS_DROWSY,
}


def _output_path(name=None):
    os.makedirs(config.DATASET_DIR, exist_ok=True)
    if name:
        filename = name if name.endswith(".csv") else f"{name}.csv"
    else:
        filename = f"session_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    return os.path.join(config.DATASET_DIR, filename)


def save_samples(rows, path):
    """Write all collected rows to CSV. Returns the path, or None if empty."""
    if not rows:
        return None

    header = list(config.FEATURE_COLUMNS) + [config.LABEL_COLUMN]
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for features, label in rows:
            writer.writerow(
                [f"{features[name]:.6f}" for name in config.FEATURE_COLUMNS]
                + [label]
            )
    return path


def run(output_name=None):
    try:
        cap = open_camera()
    except CameraError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    monitor = DrowsinessMonitor()
    window = FeatureWindow()
    fps_counter = FPSCounter()

    rows = []
    counts = {cls: 0 for cls in config.CLASS_NAMES}
    label = None
    recording = False
    last_sample_time = 0.0
    read_failures = 0

    print("[INFO] Collection mode. 0/1/2 pick a label, SPACE records, "
          "q quits.")
    print(f"[INFO] Target: {config.MIN_SAMPLES_PER_CLASS}+ samples per class.")

    with FaceMeshDetector() as face_mesh:
        while True:
            frame = read_frame(cap)
            if frame is None:
                read_failures += 1
                if read_failures >= 30:
                    print("[ERROR] Lost the camera feed.", file=sys.stderr)
                    break
                continue
            read_failures = 0

            now = time.time()
            fps = fps_counter.tick()

            points = face_mesh.process(frame)
            if points is None:
                status = monitor.update_missing()
            else:
                try:
                    values = metrics_module.compute_metrics(points)
                except (IndexError, ValueError):
                    status = monitor.update_missing()
                    points = None
                else:
                    status = monitor.update(values, now)

            window.update(status, now, points is not None)

            # --- write a sample if we are recording and the window is full ---
            if (recording and label is not None and window.ready
                    and status["face_found"]
                    and now - last_sample_time >= config.SAMPLE_INTERVAL_SEC):
                features = window.extract(status["closure"])
                if features is not None:
                    rows.append((features, label))
                    counts[label] += 1
                    last_sample_time = now

            visualization.render(frame, points, status, fps)
            visualization.draw_collection_hud(
                frame, label, recording, counts, window.fill_ratio)
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key in KEY_TO_LABEL:
                label = KEY_TO_LABEL[key]
                print(f"[INFO] Label set to {label} "
                      f"({config.CLASS_NAMES[label]}).")
            elif key == ord(" "):
                if label is None:
                    print("[WARN] Pick a label (0/1/2) before recording.")
                else:
                    recording = not recording
                    print(f"[INFO] {'Recording' if recording else 'Paused'}.")
            elif key == ord("u"):
                removed = 0
                while rows and removed < 10 and rows[-1][1] == label:
                    counts[rows.pop()[1]] -= 1
                    removed += 1
                print(f"[INFO] Removed {removed} sample(s).")
            elif key == ord("s"):
                path = save_samples(rows, _output_path(output_name))
                print(f"[INFO] Saved {len(rows)} samples to {path}"
                      if path else "[WARN] Nothing to save yet.")

            if cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()

    path = save_samples(rows, _output_path(output_name))

    print("\n--- Collection summary ---")
    for cls in sorted(config.CLASS_NAMES):
        short = max(0, config.MIN_SAMPLES_PER_CLASS - counts[cls])
        note = "" if short == 0 else f"  (need {short} more)"
        print(f"  {cls} {config.CLASS_NAMES[cls]:<8}: {counts[cls]:>4}{note}")

    if path:
        print(f"\nSaved to: {path}")
        print("Run more sessions to add files, then train with:")
        print("  python train_model.py")
    else:
        print("\nNo samples were recorded, so no file was written.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Collect labelled samples.")
    parser.add_argument("--name", default=None,
                        help="output filename inside dataset/ "
                             "(default: session_<timestamp>.csv)")
    args = parser.parse_args()
    sys.exit(run(args.name))
