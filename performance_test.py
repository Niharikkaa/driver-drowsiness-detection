"""
Performance benchmark for the live `python main.py --model` pipeline.

STANDALONE and READ-ONLY with respect to the existing project:
  - does NOT modify main.py, src/, train_model.py, collect_data.py, or
    requirements.txt
  - does NOT modify or overwrite models/drowsiness_model.joblib (only reads
    it, to load it and to measure its file size)
  - only ever WRITES new files under reports/performance/

What it measures
-----------------
This runs the *exact* sequence main.py's --model path runs, frame by frame,
using the project's own modules rather than re-implementing any of them:

    src.video.open_camera() / read_frame()
        -> src.face_mesh.FaceMeshDetector.process()
        -> src.metrics.compute_metrics()
        -> src.detectors.DrowsinessMonitor.update() / update_missing()
        -> src.features.FeatureWindow.update() / extract()
        -> src.predictor.DrowsinessClassifier.predict()

No cv2.imshow() window is opened - this is meant to run headless (SSH, CI,
a cloud sandbox) as well as on a desktop. If stdin is an interactive
terminal, pressing 'q' or Esc stops the run early; otherwise it simply runs
for the configured duration.

Two different timings are kept separate on purpose:
  - "processed FPS"              how many frames per second made it all the
                                  way through the pipeline during the
                                  measured window (frames / elapsed seconds)
  - "frame processing latency"   how long the FaceMesh -> metrics ->
                                  detectors -> feature window -> classifier
                                  chain itself takes per frame, in
                                  milliseconds, EXCLUDING the time spent
                                  blocked waiting for the camera to hand
                                  over a frame

Usage
-----
    python performance_test.py
    python performance_test.py --duration 60 --warmup 10
    python performance_test.py --model-dir /path/to/other/models
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from src import config
from src.detectors import DrowsinessMonitor
from src.face_mesh import FaceMeshDetector
from src.features import FeatureWindow
from src.predictor import DrowsinessClassifier
from src.video import CameraError, open_camera, read_frame
from src import metrics as metrics_module

REPORT_SUBDIR = "performance"
CPU_SAMPLE_INTERVAL_SEC = 0.5   # how often to poll psutil inside the loop
PROGRESS_INTERVAL_SEC = 1.0     # how often to print a status line

try:
    import psutil
except ImportError:
    psutil = None


# ==========================================================================
# Non-blocking, GUI-free 'q'/Esc quit support
# ==========================================================================

class KeyWatcher:
    """
    Lets the user press 'q' or Esc to stop early, WITHOUT opening any
    cv2.imshow() window. Only active when stdin is a real interactive
    terminal; otherwise every check() call is a harmless no-op and the
    benchmark simply runs for the full configured duration. Terminal
    settings (POSIX) are always restored on close(), even on error.
    """

    def __init__(self):
        self._enabled = False
        self._mode = None
        self._fd = None
        self._old_settings = None
        try:
            if sys.stdin is not None and sys.stdin.isatty():
                if os.name == "nt":
                    import msvcrt  # noqa: F401  (import-time availability check)
                    self._mode = "windows"
                    self._enabled = True
                else:
                    import termios
                    import tty
                    self._fd = sys.stdin.fileno()
                    self._old_settings = termios.tcgetattr(self._fd)
                    tty.setcbreak(self._fd)
                    self._mode = "posix"
                    self._enabled = True
        except Exception:
            self._enabled = False

    def check_quit(self):
        """Return True if 'q'/Esc was pressed. Never blocks."""
        if not self._enabled:
            return False
        try:
            if self._mode == "windows":
                import msvcrt
                if msvcrt.kbhit():
                    return msvcrt.getch() in (b"q", b"Q", b"\x1b")
                return False
            import select
            ready, _, _ = select.select([sys.stdin], [], [], 0)
            if ready:
                return sys.stdin.read(1) in ("q", "Q", "\x1b")
            return False
        except Exception:
            return False

    def close(self):
        if self._enabled and self._mode == "posix":
            try:
                import termios
                termios.tcsetattr(self._fd, termios.TCSADRAIN,
                                  self._old_settings)
            except Exception:
                pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# ==========================================================================
# Stats helpers
# ==========================================================================

def _latency_stats(samples_ms):
    if not samples_ms:
        return None
    arr = np.array(samples_ms, dtype=np.float64)
    return {
        "samples": int(arr.size),
        "mean_ms": float(np.mean(arr)),
        "median_ms": float(np.median(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
    }


def _resource_stats(samples):
    if not samples:
        return None
    arr = np.array(samples, dtype=np.float64)
    return {
        "samples": int(arr.size),
        "avg": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _format_bytes(num_bytes):
    kb = num_bytes / 1024.0
    mb = kb / 1024.0
    return kb, mb


# ==========================================================================
# Benchmark
# ==========================================================================

def run(duration=30.0, warmup=5.0, model_dir=None, model_file=None,
       camera_index=None, output_dir=None):
    if warmup >= duration:
        print(f"[ERROR] --warmup ({warmup}s) must be less than --duration "
              f"({duration}s).", file=sys.stderr)
        return 1
    measured_window = duration - warmup

    print("=" * 74)
    print("Driver Drowsiness Detection - live pipeline performance benchmark")
    print("=" * 74)

    # ---- model: load + report file size, never modify it ----------------
    m_dir = config.MODEL_DIR if model_dir is None else model_dir
    m_file = config.MODEL_FILE if model_file is None else model_file
    model_path = os.path.join(m_dir, m_file)

    if os.path.exists(model_path):
        kb, mb = _format_bytes(os.path.getsize(model_path))
        print(f"\nModel file : {model_path}")
        print(f"Model size : {kb:.1f} KB ({mb:.2f} MB)")
    else:
        kb = mb = None
        print(f"\nModel file : {model_path}  (not found yet)")

    classifier, error = DrowsinessClassifier.try_load(m_dir, m_file)
    if classifier is None:
        print(f"[ERROR] {error}", file=sys.stderr)
        return 1
    print(f"Loaded     : {classifier.summary()}")

    # ---- camera -----------------------------------------------------------
    try:
        cap = open_camera(camera_index)
    except CameraError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        return 1

    requested_w, requested_h = config.FRAME_WIDTH, config.FRAME_HEIGHT
    actual_w = int(cap.get(3))   # cv2.CAP_PROP_FRAME_WIDTH  = 3
    actual_h = int(cap.get(4))   # cv2.CAP_PROP_FRAME_HEIGHT = 4
    print(f"\nCamera requested resolution : {requested_w}x{requested_h}")
    print(f"Camera actual resolution    : {actual_w}x{actual_h}")

    # ---- psutil availability ----------------------------------------------
    process = None
    if psutil is None:
        print("\n[INFO] psutil is not installed - CPU/RAM usage will be "
              "skipped. Install it separately if you want that data:\n"
              "         pip install psutil")
    else:
        process = psutil.Process(os.getpid())
        process.cpu_percent(interval=None)  # prime the counter; first read
                                            # after this is meaningful

    monitor = DrowsinessMonitor()
    window = FeatureWindow()

    process_latency_ms = []       # measured window only
    capture_latency_ms = []       # measured window only, supplementary
    cpu_samples, mem_samples_mb = [], []
    frames_processed = 0
    frames_processed_total = 0    # includes warm-up, for transparency
    face_detections = 0
    read_failures = 0
    ml_predictions = 0

    run_start = time.time()
    warmup_end = run_start + warmup
    run_end = warmup_end + measured_window
    last_progress = 0.0
    last_cpu_sample = 0.0
    quit_requested = False
    quit_reason = None

    print(f"\nRunning for {duration:.0f}s total "
          f"({warmup:.0f}s warm-up + {measured_window:.0f}s measured)."
          + (" Press 'q' or Esc to stop early." if sys.stdin.isatty()
             else ""))
    print(f"MediaPipe backend: (reported once the detector is ready below)")

    try:
        with FaceMeshDetector() as face_mesh, KeyWatcher() as keys:
            print(f"MediaPipe backend in use: {face_mesh.backend}")

            while True:
                now = time.time()
                if now >= run_end:
                    break

                if keys.check_quit():
                    quit_requested = True
                    quit_reason = "user pressed q/Esc"
                    break

                capture_start = time.perf_counter()
                frame = read_frame(cap)
                capture_elapsed_ms = (time.perf_counter() - capture_start) * 1000.0

                if frame is None:
                    read_failures += 1
                    if read_failures >= 30:
                        quit_reason = "camera feed lost (30 consecutive failed reads)"
                        print(f"\n[ERROR] {quit_reason}", file=sys.stderr)
                        break
                    continue

                in_measured_window = now >= warmup_end

                # ---- the actual pipeline, timed as one block ------------
                process_start = time.perf_counter()

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
                        status = monitor.update(values)

                frame_now = time.time()
                window.update(status, frame_now, points is not None)

                prediction = None
                if status["face_found"] and window.ready:
                    prediction = classifier.predict(
                        window.extract(status["closure"]))
                elif not status["face_found"]:
                    classifier.reset()

                process_elapsed_ms = (time.perf_counter() - process_start) * 1000.0
                # ---- end of timed block -----------------------------------

                frames_processed_total += 1
                if in_measured_window:
                    frames_processed += 1
                    process_latency_ms.append(process_elapsed_ms)
                    capture_latency_ms.append(capture_elapsed_ms)
                    if points is not None:
                        face_detections += 1
                    if prediction is not None:
                        ml_predictions += 1

                # ---- resource sampling, OUTSIDE the timed block above ---
                if process is not None and now - last_cpu_sample >= CPU_SAMPLE_INTERVAL_SEC:
                    last_cpu_sample = now
                    if in_measured_window:
                        try:
                            cpu_samples.append(process.cpu_percent(interval=None))
                            mem_samples_mb.append(
                                process.memory_info().rss / (1024.0 * 1024.0))
                        except Exception:
                            pass
                    else:
                        # Still poll during warm-up so the first measured
                        # sample reflects a properly primed counter.
                        try:
                            process.cpu_percent(interval=None)
                        except Exception:
                            pass

                # ---- progress line, no GUI required ----------------------
                if now - last_progress >= PROGRESS_INTERVAL_SEC:
                    last_progress = now
                    phase = "warm-up" if not in_measured_window else "measuring"
                    elapsed = now - run_start
                    live_fps = (frames_processed / max(1e-6, now - warmup_end)
                               if in_measured_window else 0.0)
                    sys.stdout.write(
                        f"\r  [{phase:>9}] {elapsed:5.1f}s / {duration:.0f}s "
                        f"| frames {frames_processed_total:5d} "
                        f"| live processed FPS {live_fps:5.1f}   "
                    )
                    sys.stdout.flush()

    except KeyboardInterrupt:
        quit_requested = True
        quit_reason = "KeyboardInterrupt (Ctrl+C)"
    finally:
        cap.release()

    print()  # newline after the progress line
    if quit_requested:
        print(f"[INFO] Stopped early: {quit_reason}")

    actual_measured_seconds = max(1e-6, min(time.time(), run_end) - warmup_end)
    if actual_measured_seconds <= 0 or frames_processed == 0:
        print("[WARN] No frames were recorded in the measured window - "
              "results below may be empty or unreliable. Try a longer "
              "--duration or check the camera.")

    # ---- compute summary stats --------------------------------------------
    processed_fps = frames_processed / actual_measured_seconds
    latency_stats = _latency_stats(process_latency_ms)
    capture_stats = _latency_stats(capture_latency_ms)
    face_rate_pct = (100.0 * face_detections / frames_processed
                     if frames_processed else 0.0)

    cpu_stats = _resource_stats(cpu_samples)
    mem_stats = _resource_stats(mem_samples_mb)

    # ---- print report ------------------------------------------------------
    print("\n" + "=" * 74)
    print("RESULTS (measured window only - warm-up excluded)")
    print("=" * 74)
    print(f"Measured window            : {actual_measured_seconds:.1f}s "
          f"(target {measured_window:.0f}s)")
    print(f"Frames processed           : {frames_processed}")
    print(f"Processed FPS              : {processed_fps:.2f}")
    print(f"Camera read failures       : {read_failures}")

    if latency_stats:
        print("\nFrame processing latency (FaceMesh -> metrics -> detectors "
              "-> feature window -> classifier; camera wait excluded):")
        print(f"  mean   : {latency_stats['mean_ms']:.2f} ms")
        print(f"  median : {latency_stats['median_ms']:.2f} ms")
        print(f"  p95    : {latency_stats['p95_ms']:.2f} ms")
        print(f"  min    : {latency_stats['min_ms']:.2f} ms")
        print(f"  max    : {latency_stats['max_ms']:.2f} ms")

    if capture_stats:
        print(f"\nCamera capture wait (supplementary, shown separately so it "
              f"is never confused with processing time):")
        print(f"  mean   : {capture_stats['mean_ms']:.2f} ms")

    print(f"\nFace detected              : {face_detections} / "
          f"{frames_processed} frames ({face_rate_pct:.1f}%)")
    print(f"ML predictions produced    : {ml_predictions} "
          "(frames where the feature window was ready and a face was "
          "present)")

    if psutil is None:
        print("\nCPU / RAM usage: not measured (psutil not installed).")
    elif cpu_stats is None:
        print("\nCPU / RAM usage: no samples were taken (run was too short "
              f"for the {CPU_SAMPLE_INTERVAL_SEC}s sampling interval).")
    else:
        n_cpus = psutil.cpu_count(logical=True) or 1
        print(f"\nCPU usage (this process, observed during the run - not a "
              f"hardware spec, {n_cpus} logical CPUs on this machine, "
              f"psutil's cpu_percent() is not normalized per core so >100% "
              f"is possible if more than one core is used):")
        print(f"  avg : {cpu_stats['avg']:.1f}%")
        print(f"  range : {cpu_stats['min']:.1f}% - {cpu_stats['max']:.1f}% "
              f"across {cpu_stats['samples']} samples")
        print(f"\nRAM usage (this process' resident set size, observed "
              "during the run):")
        print(f"  avg : {mem_stats['avg']:.1f} MB")
        print(f"  range : {mem_stats['min']:.1f} MB - {mem_stats['max']:.1f} MB")

    # ---- save JSON report --------------------------------------------------
    out_dir = (os.path.join(config.REPORT_DIR, REPORT_SUBDIR)
              if output_dir is None else output_dir)
    os.makedirs(out_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"performance_{stamp}.json")

    report = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "stopped_early": quit_requested,
        "stop_reason": quit_reason,
        "config": {
            "duration_sec_total": duration,
            "warmup_sec": warmup,
            "measured_window_sec_target": measured_window,
            "measured_window_sec_actual": actual_measured_seconds,
            "camera_index": (config.CAMERA_INDEX if camera_index is None
                             else camera_index),
        },
        "model": {
            "path": model_path,
            "size_kb": kb,
            "size_mb": mb,
            "model_name": classifier.model_name,
            "trained_at": classifier.trained_at,
            "reported_metrics": classifier.metrics,
        },
        "camera": {
            "requested_width": requested_w,
            "requested_height": requested_h,
            "actual_width": actual_w,
            "actual_height": actual_h,
        },
        "mediapipe_backend": face_mesh.backend,
        "frames": {
            "processed_measured_window": frames_processed,
            "processed_including_warmup": frames_processed_total,
            "read_failures": read_failures,
            "face_detections": face_detections,
            "face_detection_rate_pct": face_rate_pct,
            "ml_predictions": ml_predictions,
        },
        "processed_fps": processed_fps,
        "frame_processing_latency_ms": latency_stats,
        "camera_capture_latency_ms": capture_stats,
        "resources": {
            "psutil_available": psutil is not None,
            "cpu_percent": cpu_stats,
            "memory_mb": mem_stats,
            "logical_cpus": (psutil.cpu_count(logical=True)
                             if psutil is not None else None),
            "note": ("cpu_percent is this process only, sampled every "
                     f"{CPU_SAMPLE_INTERVAL_SEC}s, not normalized per core; "
                     "it is an observation from this run, not a hardware "
                     "specification.") if psutil is not None else
                    "psutil not installed; install separately to enable.",
        },
    }

    with open(out_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\nSaved report to: {out_path}")
    print("Nothing under models/ was created, modified or overwritten by "
          "this script.")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the live main.py --model pipeline "
                    "(webcam -> FaceMesh -> metrics -> detectors -> "
                    "feature window -> classifier). Read-only with "
                    "respect to the existing project files.")
    parser.add_argument("--duration", type=float, default=30.0,
                        help="total run time in seconds, including "
                             "warm-up (default: %(default)s)")
    parser.add_argument("--warmup", type=float, default=5.0,
                        help="seconds at the start to exclude from "
                             "measurements (default: %(default)s)")
    parser.add_argument("--model-dir", default=None,
                        help="defaults to config.MODEL_DIR ('models')")
    parser.add_argument("--model-file", default=None,
                        help="defaults to config.MODEL_FILE")
    parser.add_argument("--camera-index", type=int, default=None,
                        help="defaults to config.CAMERA_INDEX")
    parser.add_argument("--output-dir", default=None,
                        help="defaults to reports/performance/")
    args = parser.parse_args()

    sys.exit(run(
        duration=args.duration,
        warmup=args.warmup,
        model_dir=args.model_dir,
        model_file=args.model_file,
        camera_index=args.camera_index,
        output_dir=args.output_dir,
    ))
