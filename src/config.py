"""
Central configuration for the Edge-AI Driver Drowsiness Detection System.

Every tunable value lives here. No other module should hard-code a threshold.
"""

# --------------------------------------------------------------------------
# Camera / video
# --------------------------------------------------------------------------
CAMERA_INDEX = 0          # 0 = default webcam
FRAME_WIDTH = 640         # downscale keeps the pipeline lightweight
FRAME_HEIGHT = 480
FLIP_FRAME = True         # mirror the image so it feels like a mirror

# --------------------------------------------------------------------------
# MediaPipe FaceMesh
# --------------------------------------------------------------------------
MAX_NUM_FACES = 1
REFINE_LANDMARKS = False          # True is more accurate around eyes but slower
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5

# --------------------------------------------------------------------------
# Landmark indices (MediaPipe FaceMesh, 468-point model)
# --------------------------------------------------------------------------
# Six-point EAR sets, ordered as: [outer, upper1, upper2, inner, lower2, lower1]
RIGHT_EYE_EAR = [33, 160, 158, 133, 153, 144]
LEFT_EYE_EAR = [362, 385, 387, 263, 373, 380]

# Mouth: two corners + three vertical (upper, lower) pairs
MOUTH_CORNERS = [78, 308]
MOUTH_VERTICAL_PAIRS = [(81, 178), (13, 14), (311, 402)]

# Full rings, used only for drawing
RIGHT_EYE_CONTOUR = [33, 246, 161, 160, 159, 158, 157, 173,
                     133, 155, 154, 153, 145, 144, 163, 7]
LEFT_EYE_CONTOUR = [362, 398, 384, 385, 386, 387, 388, 466,
                    263, 249, 390, 373, 374, 380, 381, 382]
MOUTH_CONTOUR = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324,
                 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]

# --------------------------------------------------------------------------
# Detection thresholds
# --------------------------------------------------------------------------
EAR_THRESHOLD = 0.21          # below this the eye is considered closed
EAR_CONSEC_FRAMES = 2         # frames below threshold before a blink is valid
EYE_CLOSED_ALARM_SEC = 1.5    # continuous closure that counts as drowsiness

MAR_THRESHOLD = 0.60          # above this the mouth is considered wide open
YAWN_MIN_DURATION_SEC = 0.8   # mouth must stay open this long to be a yawn

SMOOTHING_WINDOW = 3          # moving-average length for EAR/MAR (1 = off)

# Frames without a face before the counters are put on hold
MAX_MISSING_FRAMES = 10

# --------------------------------------------------------------------------
# Drawing / HUD (BGR colours)
# --------------------------------------------------------------------------
COLOR_OK = (0, 200, 0)
COLOR_WARN = (0, 165, 255)
COLOR_ALERT = (0, 0, 255)
COLOR_TEXT = (255, 255, 255)
COLOR_EYE = (0, 255, 255)
COLOR_MOUTH = (255, 0, 255)

DRAW_LANDMARKS = True
FONT_SCALE = 0.55
LINE_THICKNESS = 1


# ==========================================================================
# STEP 2 - Dataset collection and ML classification
# Everything below is additive. Step 1 behaviour is unchanged.
# ==========================================================================

# --------------------------------------------------------------------------
# Paths (relative to the project root)
# --------------------------------------------------------------------------
DATASET_DIR = "dataset"
MODEL_DIR = "models"
REPORT_DIR = "reports"

MODEL_FILE = "drowsiness_model.joblib"      # classifier + scaler + metadata

# --------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------
CLASS_ALERT = 0
CLASS_WARNING = 1
CLASS_DROWSY = 2

CLASS_NAMES = {
    CLASS_ALERT: "ALERT",
    CLASS_WARNING: "WARNING",
    CLASS_DROWSY: "DROWSY",
}

CLASS_COLORS = {
    CLASS_ALERT: COLOR_OK,
    CLASS_WARNING: COLOR_WARN,
    CLASS_DROWSY: COLOR_ALERT,
}

# --------------------------------------------------------------------------
# Feature window
# --------------------------------------------------------------------------
# A single frame carries almost no drowsiness signal, so features are
# aggregated over a sliding time window.
FEATURE_WINDOW_SEC = 10.0     # length of the sliding window
MIN_WINDOW_FRAMES = 25        # window must hold this many frames to be valid
SAMPLE_INTERVAL_SEC = 0.5     # how often a labelled sample is written

# Order matters: this is the column order of the CSV and the model input.
FEATURE_COLUMNS = [
    "ear_mean",          # average eye openness
    "ear_std",           # how much the eyes fluctuate
    "ear_min",           # deepest closure in the window
    "perclos",           # fraction of frames with eyes closed (key metric)
    "blink_rate",        # blinks per minute
    "mean_closure",      # average duration of a closure, seconds
    "max_closure",       # longest closure in the window, seconds
    "mar_mean",          # average mouth openness
    "mar_max",           # widest mouth opening
    "yawn_rate",         # yawns per minute
    "mouth_open_ratio",  # fraction of frames with the mouth wide open
]

LABEL_COLUMN = "label"

# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------
TEST_SIZE = 0.25
RANDOM_STATE = 42
CV_FOLDS = 5
MIN_SAMPLES_PER_CLASS = 40    # collection target per class, also a warning gate

# --------------------------------------------------------------------------
# Real-time prediction
# --------------------------------------------------------------------------
PREDICTION_SMOOTHING = 5      # majority vote over the last N predictions
