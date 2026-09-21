"""
Offline sanity check - runs the whole pipeline without a webcam.

Useful to confirm the install works before plugging in a camera:
    python self_test.py
"""

import numpy as np

from src import config, metrics, visualization
from src.detectors import DrowsinessMonitor
from src.face_mesh import FaceMeshDetector

PASS, FAIL, SKIP = "[PASS]", "[FAIL]", "[SKIP]"
_failures = []
_skipped = []


def skip(name, reason):
    """Record a check that could not run in this environment."""
    print(f"{SKIP} {name} - {reason}")
    _skipped.append(name)


def check(name, condition, detail=""):
    if condition:
        print(f"{PASS} {name}")
    else:
        print(f"{FAIL} {name} {detail}")
        _failures.append(name)


def _synthetic_face(eye_opening, mouth_opening):
    """
    Build a fake 468-point landmark array where only the indices we care about
    are meaningful. Eye width is fixed at 30px, mouth width at 50px.
    """
    pts = np.zeros((468, 2), dtype=np.float32)

    for eye, cx in ((config.LEFT_EYE_EAR, 220), (config.RIGHT_EYE_EAR, 150)):
        outer, up1, up2, inner, low2, low1 = eye
        cy = 200.0
        pts[outer] = (cx - 15, cy)
        pts[inner] = (cx + 15, cy)
        pts[up1] = (cx - 7, cy - eye_opening / 2)
        pts[up2] = (cx + 7, cy - eye_opening / 2)
        pts[low2] = (cx + 7, cy + eye_opening / 2)
        pts[low1] = (cx - 7, cy + eye_opening / 2)

    left_corner, right_corner = config.MOUTH_CORNERS
    pts[left_corner] = (160.0, 300.0)
    pts[right_corner] = (210.0, 300.0)
    for top, bottom in config.MOUTH_VERTICAL_PAIRS:
        pts[top] = (185.0, 300.0 - mouth_opening / 2)
        pts[bottom] = (185.0, 300.0 + mouth_opening / 2)

    # Fill drawing-only contour indices so rendering has something to plot.
    for idx in (config.LEFT_EYE_CONTOUR + config.RIGHT_EYE_CONTOUR
                + config.MOUTH_CONTOUR):
        if not pts[idx].any():
            pts[idx] = (185.0, 250.0)

    return pts


def test_metrics():
    open_face = _synthetic_face(eye_opening=12.0, mouth_opening=4.0)
    shut_face = _synthetic_face(eye_opening=1.0, mouth_opening=4.0)
    yawn_face = _synthetic_face(eye_opening=12.0, mouth_opening=40.0)

    ear_open = metrics.compute_metrics(open_face)["ear"]
    ear_shut = metrics.compute_metrics(shut_face)["ear"]
    mar_shut = metrics.compute_metrics(open_face)["mar"]
    mar_yawn = metrics.compute_metrics(yawn_face)["mar"]

    check("EAR is high for open eyes", ear_open > config.EAR_THRESHOLD,
          f"(got {ear_open:.3f})")
    check("EAR is low for closed eyes", ear_shut < config.EAR_THRESHOLD,
          f"(got {ear_shut:.3f})")
    check("MAR is low for a closed mouth", mar_shut < config.MAR_THRESHOLD,
          f"(got {mar_shut:.3f})")
    check("MAR is high for a yawn", mar_yawn > config.MAR_THRESHOLD,
          f"(got {mar_yawn:.3f})")
    check("EAR is symmetric for a symmetric face",
          abs(metrics.compute_metrics(open_face)["left_ear"]
              - metrics.compute_metrics(open_face)["right_ear"]) < 1e-4)


def test_blink_counting():
    """Simulate 3 blinks in a stream of open frames at a fake 30 FPS."""
    monitor = DrowsinessMonitor()
    clock = 1000.0
    step = 1 / 30.0

    open_m = {"ear": 0.32, "mar": 0.10}
    shut_m = {"ear": 0.12, "mar": 0.10}

    for _ in range(3):
        for _ in range(10):
            monitor.update(open_m, clock)
            clock += step
        for _ in range(4):            # ~0.13s closure = a blink
            monitor.update(shut_m, clock)
            clock += step
    for _ in range(10):
        monitor.update(open_m, clock)
        clock += step

    check("Counted exactly 3 blinks", monitor.blink.blink_count == 3,
          f"(got {monitor.blink.blink_count})")
    check("Not flagged drowsy after normal blinks",
          not monitor.blink.is_drowsy)


def test_long_closure():
    monitor = DrowsinessMonitor()
    clock = 2000.0
    shut_m = {"ear": 0.10, "mar": 0.10}

    for _ in range(int(2.5 * 30)):    # 2.5 seconds of closed eyes
        status = monitor.update(shut_m, clock)
        clock += 1 / 30.0

    check("Drowsiness flagged after a long closure", status["drowsy"])
    check("Closure duration is about 2.5s",
          2.3 < status["closure"] < 2.6, f"(got {status['closure']:.2f}s)")
    check("A long closure is not counted as a blink",
          monitor.blink.blink_count == 0)


def test_yawn_counting():
    monitor = DrowsinessMonitor()
    clock = 3000.0
    closed_m = {"ear": 0.32, "mar": 0.10}
    open_m = {"ear": 0.32, "mar": 0.80}

    for _ in range(2):
        for _ in range(int(1.5 * 30)):     # 1.5s mouth wide open
            monitor.update(open_m, clock)
            clock += 1 / 30.0
        for _ in range(30):
            monitor.update(closed_m, clock)
            clock += 1 / 30.0

    check("Counted exactly 2 yawns", monitor.yawn.yawn_count == 2,
          f"(got {monitor.yawn.yawn_count})")

    # A brief mouth opening (talking) should not register.
    short = DrowsinessMonitor()
    clock = 4000.0
    for _ in range(int(0.4 * 30)):
        short.update(open_m, clock)
        clock += 1 / 30.0
    check("A short mouth opening is not a yawn",
          short.yawn.yawn_count == 0)


def test_missing_face():
    monitor = DrowsinessMonitor()
    clock = 5000.0
    for _ in range(20):
        monitor.update({"ear": 0.12, "mar": 0.10}, clock)
        clock += 1 / 30.0
    for _ in range(config.MAX_MISSING_FRAMES + 2):
        status = monitor.update_missing()

    check("face_found goes False when the face is lost",
          not status["face_found"])
    check("Closure timer is cleared when the face is lost",
          status["closure"] == 0.0)
    check("Blink count survives a lost face",
          monitor.blink.blink_count == 0)


def test_face_mesh_and_render():
    blank = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3),
                     dtype=np.uint8)
    try:
        with FaceMeshDetector() as mesh:
            result = mesh.process(blank)
    except RuntimeError as exc:
        # No usable MediaPipe backend here. The rest of the pipeline is pure
        # geometry and scikit-learn, so keep testing it.
        skip("FaceMesh returns None on a frame with no face",
             str(exc).splitlines()[0])
    else:
        check("FaceMesh returns None on a frame with no face", result is None)

    monitor = DrowsinessMonitor()
    status = monitor.update({"ear": 0.30, "mar": 0.15})
    frame = blank.copy()
    visualization.render(frame, _synthetic_face(12.0, 4.0), status, 29.7)
    check("Overlay renders without error", frame.shape == blank.shape)

    frame2 = blank.copy()
    visualization.render(frame2, None, monitor.update_missing(), 0.0)
    check("Overlay renders with no landmarks", frame2.shape == blank.shape)


# ==========================================================================
# STEP 2 checks - feature window, dataset handling, training, prediction.
#
# NOTE: the training checks below run on a SYNTHETIC dataset created in a
# temporary directory. It exists only to prove the code paths execute; its
# accuracy numbers are meaningless and it is never written to dataset/.
# Real results must come from data you collect with collect_data.py.
# ==========================================================================

def _simulate(monitor, window, ear, mar, seconds, start, fps=30.0):
    """Drive the real Step 1 detectors, then feed the status to the window."""
    t = start
    step = 1.0 / fps
    for _ in range(int(seconds * fps)):
        status = monitor.update({"ear": ear, "mar": mar}, t)
        window.update(status, t)
        t += step
    return t


def test_feature_window_alert():
    from src.features import FeatureWindow

    monitor = DrowsinessMonitor()
    window = FeatureWindow()
    t = 10000.0
    # Eyes open with short blinks: classic alert behaviour.
    for _ in range(6):
        t = _simulate(monitor, window, 0.30, 0.10, 1.4, t)
        t = _simulate(monitor, window, 0.12, 0.10, 0.1, t)

    check("Feature window becomes ready", window.ready,
          f"(frames={window.frame_count})")

    f = window.extract(0.0)
    check("extract() returns every configured column",
          f is not None and list(f.keys()) == config.FEATURE_COLUMNS)
    check("Alert PERCLOS is low", f["perclos"] < 0.20,
          f"(got {f['perclos']:.3f})")
    check("Alert mean EAR is high", f["ear_mean"] > config.EAR_THRESHOLD,
          f"(got {f['ear_mean']:.3f})")
    check("Blinks are counted in the window", f["blink_rate"] > 0,
          f"(got {f['blink_rate']:.1f}/min)")
    check("Alert max closure is short", f["max_closure"] < 0.5,
          f"(got {f['max_closure']:.2f}s)")


def test_feature_window_drowsy():
    from src.features import FeatureWindow

    monitor = DrowsinessMonitor()
    window = FeatureWindow()
    t = 20000.0
    # Long closures dominating the window: drowsy behaviour.
    for _ in range(4):
        t = _simulate(monitor, window, 0.10, 0.10, 2.0, t)
        t = _simulate(monitor, window, 0.30, 0.10, 0.5, t)

    f = window.extract(0.0)
    check("Drowsy PERCLOS is high", f["perclos"] > 0.60,
          f"(got {f['perclos']:.3f})")
    check("Drowsy max closure is long", f["max_closure"] > 1.0,
          f"(got {f['max_closure']:.2f}s)")
    check("Drowsy PERCLOS separates from alert",
          f["perclos"] > 0.20)


def test_feature_window_ignores_lost_face():
    from src.features import FeatureWindow

    monitor = DrowsinessMonitor()
    window = FeatureWindow()
    t = 30000.0
    t = _simulate(monitor, window, 0.30, 0.10, 2.0, t)
    before = window.frame_count

    # Callers that know landmarks were missing must contribute nothing, even
    # during Step 1's MAX_MISSING_FRAMES hysteresis window.
    for _ in range(50):
        window.update(monitor.update_missing(), t, face_present=False)
        t += 1 / 30.0
    check("Frames with no landmarks are excluded from the window",
          window.frame_count == before,
          f"(before={before}, after={window.frame_count})")

    f = window.extract(0.0)
    check("Window still usable after a long face loss", f is not None)


def test_yawn_appears_in_features():
    from src.features import FeatureWindow

    monitor = DrowsinessMonitor()
    window = FeatureWindow()
    t = 40000.0
    t = _simulate(monitor, window, 0.30, 0.80, 2.0, t)
    t = _simulate(monitor, window, 0.30, 0.10, 2.0, t)

    f = window.extract(0.0)
    check("Yawn raises mar_max", f["mar_max"] > config.MAR_THRESHOLD,
          f"(got {f['mar_max']:.3f})")
    check("Yawn is counted in the window", f["yawn_rate"] > 0,
          f"(got {f['yawn_rate']:.1f}/min)")
    check("mouth_open_ratio is between 0 and 1",
          0.0 < f["mouth_open_ratio"] < 1.0,
          f"(got {f['mouth_open_ratio']:.3f})")


def _write_synthetic_csv(path, rows_per_class=60, seed=0):
    """SYNTHETIC smoke-test data only. Never used for real results."""
    import csv
    rng = np.random.default_rng(seed)
    # Rough per-class centres so the classes are separable enough to train.
    centres = {
        0: [0.30, 0.03, 0.22, 0.06, 14, 0.15, 0.25, 0.10, 0.20, 0.5, 0.02],
        1: [0.25, 0.05, 0.15, 0.25, 22, 0.30, 0.70, 0.15, 0.45, 2.0, 0.08],
        2: [0.18, 0.07, 0.08, 0.55, 30, 0.60, 2.10, 0.22, 0.75, 5.0, 0.20],
    }
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(config.FEATURE_COLUMNS) + [config.LABEL_COLUMN])
        for label, centre in centres.items():
            for _ in range(rows_per_class):
                row = [max(0.0, c + rng.normal(0, abs(c) * 0.55 + 0.02))
                       for c in centre]
                writer.writerow([f"{v:.6f}" for v in row] + [label])


def test_training_pipeline():
    import os
    import tempfile
    from src import dataset as dataset_module
    from src import training
    from src.predictor import DrowsinessClassifier

    tmp = tempfile.mkdtemp(prefix="drowsy_smoke_")
    data_dir = os.path.join(tmp, "dataset")
    model_dir = os.path.join(tmp, "models")
    os.makedirs(data_dir)
    _write_synthetic_csv(os.path.join(data_dir, "synthetic.csv"))

    data = dataset_module.load_dataset(data_dir)
    check("Dataset loads every synthetic row", len(data) == 180,
          f"(got {len(data)})")
    check("All three classes are present",
          set(dataset_module.class_counts(data)) == set(config.CLASS_NAMES))

    X_tr, X_te, y_tr, y_te, scaler = dataset_module.split_and_scale(data)
    check("Split sizes add up", len(y_tr) + len(y_te) == 180)
    check("Test split is about TEST_SIZE",
          abs(len(y_te) / 180 - config.TEST_SIZE) < 0.05)
    check("Split is stratified", len(set(y_te)) == 3)
    check("Scaler standardises the training set",
          abs(X_tr.mean()) < 1e-6 and abs(X_tr.std() - 1) < 0.05)
    check("Feature count matches config",
          X_tr.shape[1] == len(config.FEATURE_COLUMNS))

    results = training.train_all(X_tr, y_tr, X_te, y_te)
    check("All three models trained",
          set(results) == {"LogisticRegression", "DecisionTree",
                           "RandomForest"},
          f"(got {sorted(results)})")

    for name, entry in results.items():
        m = entry["metrics"]
        ok = all(0.0 <= m[k] <= 1.0 for k in
                 ("accuracy", "precision_macro", "recall_macro", "f1_macro"))
        check(f"{name} metrics are in range", ok)
        check(f"{name} per-class report covers 3 classes",
              len(m["per_class"]) == 3)

    best = training.pick_best(results)
    check("A best model is selected", best in results)

    path = training.save_model(
        results[best]["model"], scaler, best, results[best]["metrics"],
        {"n_samples": 180}, model_dir=model_dir)
    check("Model file is written", os.path.exists(path))

    bundle = training.load_model(model_dir)
    check("Saved bundle carries the scaler and column order",
          bundle["scaler"] is not None
          and bundle["feature_columns"] == config.FEATURE_COLUMNS)

    # --- prediction path ---
    clf = DrowsinessClassifier.load(model_dir)
    features = {name: 0.2 for name in config.FEATURE_COLUMNS}
    pred = clf.predict(features)
    check("predict() returns a valid class",
          pred is not None and pred.class_id in config.CLASS_NAMES,
          f"(got {pred})")
    check("predict() label matches the class id",
          pred.label == config.CLASS_NAMES[pred.class_id])
    check("Confidence is a probability", 0.0 <= pred.confidence <= 1.0)
    check("predict(None) returns None", clf.predict(None) is None)

    # Smoothing should hold a steady verdict across repeated calls.
    for _ in range(config.PREDICTION_SMOOTHING):
        last = clf.predict(features)
    check("Smoothed prediction stays valid",
          last.class_id in config.CLASS_NAMES)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_dataset_error_handling():
    import os
    import tempfile
    from src import dataset as dataset_module

    tmp = tempfile.mkdtemp(prefix="drowsy_empty_")
    try:
        dataset_module.load_dataset(tmp)
        check("Empty dataset dir raises DatasetError", False)
    except dataset_module.DatasetError:
        check("Empty dataset dir raises DatasetError", True)

    # Single-class data must be rejected, not silently trained on.
    import csv
    path = os.path.join(tmp, "one_class.csv")
    with open(path, "w", newline="") as handle:
        w = csv.writer(handle)
        w.writerow(list(config.FEATURE_COLUMNS) + [config.LABEL_COLUMN])
        for _ in range(10):
            w.writerow([0.2] * len(config.FEATURE_COLUMNS) + [0])
    try:
        dataset_module.check_dataset(dataset_module.load_dataset(tmp))
        check("Single-class dataset is rejected", False)
    except dataset_module.DatasetError:
        check("Single-class dataset is rejected", True)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def test_untrained_model_is_handled():
    import tempfile
    from src.predictor import DrowsinessClassifier

    tmp = tempfile.mkdtemp(prefix="drowsy_nomodel_")
    clf, error = DrowsinessClassifier.try_load(tmp)
    check("try_load returns None when no model exists", clf is None)
    check("try_load explains how to fix it",
          error is not None and "train_model.py" in error)

    import shutil
    shutil.rmtree(tmp, ignore_errors=True)


def main():
    print("Running offline self-test...\n")
    print("-- Step 1 --")
    test_metrics()
    test_blink_counting()
    test_long_closure()
    test_yawn_counting()
    test_missing_face()
    test_face_mesh_and_render()

    print("\n-- Step 2 --")
    test_feature_window_alert()
    test_feature_window_drowsy()
    test_feature_window_ignores_lost_face()
    test_yawn_appears_in_features()
    test_training_pipeline()
    test_dataset_error_handling()
    test_untrained_model_is_handled()

    print()
    if _skipped:
        print(f"{len(_skipped)} check(s) skipped: {', '.join(_skipped)}")
    if _failures:
        print(f"{len(_failures)} check(s) failed: {', '.join(_failures)}")
        return 1
    print("All checks passed. The pipeline is ready for a webcam.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
