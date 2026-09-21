# Lightweight Edge-AI Based Driver Drowsiness Detection System

A software-only, simulation-based embedded-systems project. It runs entirely on
a laptop webcam with classical computer vision and geometric features — no
microcontroller, no cloud API, and no deep-learning training framework.

**Stage 1 (this release)** implements the full real-time perception pipeline:

```
Webcam
  -> OpenCV frame capture
  -> MediaPipe Face Landmarks
  -> Eye / mouth landmark extraction
  -> EAR (Eye Aspect Ratio) + MAR (Mouth Aspect Ratio)
  -> Eye open / closed decision
  -> Blink detection
  -> Eye closure duration
  -> Basic yawning detection
  -> On-screen overlay
```

---

## Project structure

```
drowsiness/
├── main.py              # Entry point. Integration only - no logic.
├── requirements.txt
├── README.md
├── self_test.py         # Offline pipeline check, needs no webcam
└── src/
    ├── __init__.py
    ├── config.py        # ALL thresholds, landmark indices and settings
    ├── video.py         # Webcam open/read + FPS counter
    ├── face_mesh.py     # MediaPipe wrapper -> (N, 2) landmark array or None
    ├── metrics.py       # EAR and MAR geometry
    ├── detectors.py     # Blink / closure-duration / yawn state machines
    └── visualization.py # All OpenCV drawing
```

Each module has one job. `main.py` only wires them together, and **every
tunable number lives in `src/config.py`** — no thresholds are hard-coded
anywhere else.

---

## 1. Install dependencies

From inside the project folder:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A virtual environment is recommended because `requirements.txt` pins NumPy to
the 1.x series for MediaPipe ABI compatibility.

Verify the install without plugging in a camera:

```bash
python self_test.py
```

You should see a list of `[PASS]` lines ending with
`All checks passed. The pipeline is ready for a webcam.`

## 2. Run the system

```bash
python main.py
```

Run it from the project root (the folder containing `main.py`), since the
modules are imported as `src.*`.

### Controls

| Key       | Action                                |
|-----------|---------------------------------------|
| `q` / Esc | Quit                                  |
| `r`       | Reset blink / yawn / closure counters |

On exit it prints a short session summary: total blinks, total yawns and the
longest continuous eye closure.

---

## What you should see

A single window titled **Driver Drowsiness Detection** showing your mirrored
webcam feed with:

- **Cyan outlines** around both eyes, with dots on the six EAR keypoints.
- **Magenta outline** around the inner lip, with dots on the MAR keypoints.
- A **dark panel in the top-left** listing live values:
  `EAR`, `MAR`, `Eyes: OPEN/CLOSED`, `Blinks`, `Closed` (current and longest
  closure in seconds), `Yawns`, and `FPS`.
- A **status banner along the bottom**:
  - green `DRIVER ALERT` — normal,
  - orange `YAWNING DETECTED` — mouth held wide open past the yawn duration,
  - red `DROWSINESS ALERT - EYES CLOSED` — eyes shut longer than 1.5 s,
  - orange `NO FACE DETECTED` — no face in frame.
- A coloured **border around the whole frame** during a drowsy or yawning state.

Behaviour to expect while testing:

- EAR sits around **0.25–0.35** with eyes open and drops below **0.21** when
  you close them; the `Eyes` field flips to `CLOSED`.
- A normal blink increments `Blinks` by exactly one on reopening.
- Holding your eyes shut past 1.5 s turns the banner red — a long closure is
  **not** counted as a blink.
- MAR sits near **0.05–0.15** normally and rises past **0.60** on a wide yawn;
  `Yawns` increments once per yawn after 0.8 s, so ordinary talking does not
  trigger it.
- Covering the camera or turning away shows `NO FACE DETECTED`, freezes the
  counters, and recovers automatically — it never crashes.

Expect roughly **15–30 FPS** on a typical laptop CPU.

---

## Tuning

Everything is in `src/config.py`:

| Setting                 | Default | Meaning                                    |
|-------------------------|---------|--------------------------------------------|
| `EAR_THRESHOLD`         | `0.21`  | Below this an eye counts as closed         |
| `EAR_CONSEC_FRAMES`     | `2`     | Frames below threshold before a valid blink|
| `EYE_CLOSED_ALARM_SEC`  | `1.5`   | Closure that triggers the drowsiness alert |
| `MAR_THRESHOLD`         | `0.60`  | Above this the mouth counts as wide open   |
| `YAWN_MIN_DURATION_SEC` | `0.8`   | How long the mouth must stay open          |
| `SMOOTHING_WINDOW`      | `3`     | Moving average on EAR/MAR (`1` disables)   |
| `CAMERA_INDEX`          | `0`     | Change if you have multiple cameras        |

EAR varies between people and with glasses or lighting. Watch the live `EAR`
readout with your eyes open and closed, then set `EAR_THRESHOLD` about midway
between the two.

---

## Design notes

- **Durations are measured in seconds**, not frames, so the thresholds behave
  the same whether the webcam delivers 15 or 30 FPS.
- **EAR and MAR are ratios**, so they are unaffected by how far the driver sits
  from the camera.
- **Blinks are counted on the opening edge**, which prevents a single long
  closure from being counted as many blinks.
- **Each yawn is counted once** per mouth-opening event.
- A short **moving average** suppresses single-frame landmark jitter.
- Lost-face frames are tolerated for `MAX_MISSING_FRAMES` before the counters
  are put on hold, so a brief tracking glitch does not reset your session.

---

## Troubleshooting

**`Could not open camera index 0`**
Another app is using the webcam, or the index is wrong. Close other apps, or
try `CAMERA_INDEX = 1` in `src/config.py`. On macOS, grant camera permission to
your terminal.

**`No usable MediaPipe face landmark API was found` or a Tasks-API model error**
A few slim MediaPipe builds ship only the newer Tasks API without the classic
`mp.solutions.face_mesh`. The project supports both. Either install the pinned
build:

```bash
pip install --force-reinstall mediapipe==0.10.14
```

or download the Tasks model once into `models/`:

```bash
mkdir -p models
curl -L -o models/face_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task
```

**Landmarks look jittery** — improve lighting, or raise `SMOOTHING_WINDOW` to
`5` in `src/config.py`.

**Low FPS** — lower `FRAME_WIDTH` / `FRAME_HEIGHT`, and keep
`REFINE_LANDMARKS = False`.

---

---

# Step 2 - Dataset Collection + Lightweight ML Classification

Step 1 decides drowsiness with fixed thresholds. Step 2 adds a **learned**
classifier on top of the same features, so the decision adapts to how *you*
actually behave rather than to hard-coded numbers.

```
Step 1 features -> sliding feature window -> labelled CSV (dataset/)
                -> preprocessing + split -> train 3 models -> evaluate
                -> best model saved (models/) -> real-time prediction
```

Classes: `0 = ALERT`, `1 = WARNING`, `2 = DROWSY`.

## New files

```
collect_data.py          # webcam tool that records labelled samples
train_model.py           # trains and compares the three classifiers
src/features.py          # sliding-window feature extraction
src/dataset.py           # CSV loading, validation, split, scaling
src/training.py          # model definitions, metrics, save/load
src/predictor.py         # clean prediction API used by main.py
dataset/                 # your collected CSVs
models/                  # drowsiness_model.joblib
reports/                 # confusion_matrix.png
```

## Why a feature *window*

A single frame cannot tell a blink apart from a microsleep. Every sample is
therefore aggregated over a **10-second sliding window** into 11 features:

| Feature | Meaning |
|---|---|
| `ear_mean`, `ear_std`, `ear_min` | average, variability and deepest eye closure |
| `perclos` | fraction of the window with eyes closed - the strongest single drowsiness signal |
| `blink_rate` | blinks per minute |
| `mean_closure`, `max_closure` | average and longest closure, in seconds |
| `mar_mean`, `mar_max` | average and widest mouth opening |
| `yawn_rate` | yawns per minute |
| `mouth_open_ratio` | fraction of the window with the mouth wide open |

These are computed from the Step 1 detector output. No Step 1 logic is
duplicated or modified.

## 1. Collect the dataset

```bash
python collect_data.py
```

| Key | Action |
|---|---|
| `0` / `1` / `2` | choose label ALERT / WARNING / DROWSY |
| `SPACE` | start / pause recording |
| `u` | undo the last 10 samples of the current label |
| `s` | save now |
| `q` / Esc | save and quit |

Pick a label, press `SPACE`, then **act out that state**. A row is written
every 0.5 s once the window fill bar is full, so roughly **two samples per
second**.

### How to act each class

- **`0` ALERT** - sit normally, look at the screen, blink naturally, talk a
  little. Around **60–90 seconds**.
- **`1` WARNING** - blink slowly and more often, let your eyes half-close for
  under a second at a time, yawn occasionally. Around **60–90 seconds**.
- **`2` DROWSY** - close your eyes for 1.5–3 seconds at a time, repeatedly,
  with wide yawns and a drooping head. Around **60–90 seconds**.

**Target: at least 40 samples per class** (the on-screen counter marks each
class `OK` when it gets there). 100+ per class is noticeably better. Aim for
roughly balanced classes.

Run the tool several times in different lighting and at different times of
day - each run writes a new CSV and `train_model.py` merges them all. That
variety matters far more than raw sample count.

> The tool records only what your webcam actually sees. It never generates or
> pads data.

## 2. Train

```bash
python train_model.py
```

Loads every CSV in `dataset/`, drops invalid rows, does a **stratified 75/25
split**, standardises the features with a `StandardScaler` **fitted on the
training set only** (so no test data leaks into training), then trains and
compares:

- **Logistic Regression** - linear baseline
- **Decision Tree** - `max_depth=6`, readable rules
- **Random Forest** - 120 shallow trees

It reports accuracy, macro precision, recall and F1, cross-validated F1, a
per-class report, a confusion matrix and feature importances.

The best model is chosen by **macro F1, not accuracy** - DROWSY is usually the
rarest class, and accuracy would happily reward a model that ignores it.

If a class is missing, too small, or badly imbalanced, the script warns you
instead of quietly producing a misleading score.

## 3. Where the model is saved

```
models/drowsiness_model.joblib
```

One bundle holding the classifier, the fitted scaler, the feature column order,
the class names, the test metrics and the training date. `load_model()` refuses
to load it if `FEATURE_COLUMNS` has changed since training, so a stale model can
never be fed mismatched features.

A confusion matrix image is written to `reports/confusion_matrix.png`.

## 4. Run with the classifier

```bash
python main.py --model
```

Plain `python main.py` still runs Step 1 exactly as before - the classifier is
opt-in.

With `--model` you additionally get an **`ML: <CLASS> <confidence>%` banner**
just above the Step 1 status bar, colour-coded green / orange / red. It reads
`MODEL: warming up` for the first few seconds while the window fills, which is
expected.

Predictions are smoothed by majority vote over the last 5 windows, so one noisy
reading cannot make the verdict flicker.

## Using the classifier from your own code

```python
from src.features import FeatureWindow
from src.predictor import DrowsinessClassifier

clf = DrowsinessClassifier.load()          # or .try_load() for (clf, error)
window = FeatureWindow()

window.update(status, now, face_present)   # status from DrowsinessMonitor
result = clf.predict(window.extract(status["closure"]))

if result and result.is_drowsy:
    ...                                    # result.label, result.confidence
```

`predict()` returns `None` while the window is still filling, so always
null-check it.

## Step 2 troubleshooting

**`No CSV files found in 'dataset/'`** - run `python collect_data.py` first.

**`Need at least two classes to train`** - you only recorded one label. Collect
the others.

**Scores look suspiciously perfect** - you probably recorded each class in one
continuous block, so consecutive windows overlap and near-duplicate rows land in
both the train and test halves. Record several shorter takes per class across
different sessions instead.

**The model predicts DROWSY constantly** - your ALERT and DROWSY recordings were
probably too similar. Re-record with a clearer contrast between the states.

**`The saved model expects different features than config.py defines`** - you
edited `FEATURE_COLUMNS`. Retrain: `python train_model.py`.

## Verifying the install

```bash
python self_test.py
```

Covers both steps: EAR/MAR geometry, blink and yawn state machines, feature
windowing, dataset validation, the train/split/save/load path and the
prediction API. The training checks run on a **throwaway synthetic dataset in a
temp directory** purely to prove the code paths execute - it never touches
`dataset/`, and its scores mean nothing. Real numbers only come from data you
collect.

## Scope of this stage

Deliberately **not** included yet: an audible alarm, session logging and
Matplotlib trend analytics, and any driver-specific calibration routine. Those
build on top of what Steps 1 and 2 already produce.
