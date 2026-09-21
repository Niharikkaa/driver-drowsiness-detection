"""
Everything that draws on the frame. No detection logic lives here.
"""

import cv2
import numpy as np

from src import config

_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _polyline(frame, points, indices, color):
    pts = np.array([points[i] for i in indices], dtype=np.int32)
    cv2.polylines(frame, [pts], isClosed=True, color=color,
                  thickness=config.LINE_THICKNESS, lineType=cv2.LINE_AA)


def draw_landmarks(frame, points):
    """Outline both eyes and the inner lip, and dot the EAR/MAR keypoints."""
    if not config.DRAW_LANDMARKS or points is None:
        return frame

    _polyline(frame, points, config.LEFT_EYE_CONTOUR, config.COLOR_EYE)
    _polyline(frame, points, config.RIGHT_EYE_CONTOUR, config.COLOR_EYE)
    _polyline(frame, points, config.MOUTH_CONTOUR, config.COLOR_MOUTH)

    key_eye = config.LEFT_EYE_EAR + config.RIGHT_EYE_EAR
    for idx in key_eye:
        x, y = points[idx]
        cv2.circle(frame, (int(x), int(y)), 2, config.COLOR_EYE, -1)

    key_mouth = list(config.MOUTH_CORNERS)
    for top, bottom in config.MOUTH_VERTICAL_PAIRS:
        key_mouth += [top, bottom]
    for idx in key_mouth:
        x, y = points[idx]
        cv2.circle(frame, (int(x), int(y)), 2, config.COLOR_MOUTH, -1)

    return frame


def _panel(frame, width, height):
    """Dim rectangle so white text stays readable over any background."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)


def draw_hud(frame, status, fps):
    """Draw the metrics panel in the top-left corner."""
    lines = [
        f"EAR   : {status['ear']:.3f}  (thr {config.EAR_THRESHOLD:.2f})",
        f"MAR   : {status['mar']:.3f}  (thr {config.MAR_THRESHOLD:.2f})",
        f"Eyes  : {status['eye_state']}",
        f"Blinks: {status['blinks']}",
        f"Closed: {status['closure']:.2f}s  (max {status['longest_closure']:.2f}s)",
        f"Yawns : {status['yawns']}",
        f"FPS   : {fps:.1f}",
    ]

    _panel(frame, 300, 24 * len(lines) + 16)

    y = 26
    for text in lines:
        cv2.putText(frame, text, (12, y), _FONT, config.FONT_SCALE,
                    config.COLOR_TEXT, 1, cv2.LINE_AA)
        y += 24

    return frame


def draw_status_banner(frame, status):
    """Big coloured verdict across the bottom of the frame."""
    height, width = frame.shape[:2]

    if not status["face_found"]:
        text, color = "NO FACE DETECTED", config.COLOR_WARN
    elif status["drowsy"]:
        text, color = "DROWSINESS ALERT - EYES CLOSED", config.COLOR_ALERT
    elif status["yawning"]:
        text, color = "YAWNING DETECTED", config.COLOR_WARN
    else:
        text, color = "DRIVER ALERT", config.COLOR_OK

    cv2.rectangle(frame, (0, height - 46), (width, height), (0, 0, 0), -1)
    cv2.putText(frame, text, (14, height - 16), _FONT, 0.7, color, 2,
                cv2.LINE_AA)

    if status["drowsy"] or status["yawning"]:
        cv2.rectangle(frame, (0, 0), (width - 1, height - 1), color, 3)

    return frame


def draw_hint(frame, text="Press 'q' to quit  |  'r' to reset counters"):
    width = frame.shape[1]
    cv2.putText(frame, text, (width - 360, 22), _FONT, 0.45,
                config.COLOR_TEXT, 1, cv2.LINE_AA)
    return frame


def render(frame, points, status, fps):
    """One call that applies the full overlay to a frame."""
    if points is not None:
        draw_landmarks(frame, points)
    draw_hud(frame, status, fps)
    draw_status_banner(frame, status)
    draw_hint(frame)
    return frame


# ==========================================================================
# STEP 2 overlays - additive, Step 1 drawing above is unchanged.
# ==========================================================================

def draw_collection_hud(frame, label, recording, counts, fill_ratio):
    """Right-hand panel for the dataset collection tool."""
    height, width = frame.shape[:2]
    x0 = width - 250

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, 0), (width, 210), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    if label is None:
        head, color = "LABEL: none", config.COLOR_TEXT
    else:
        head = f"LABEL: {config.CLASS_NAMES[label]}"
        color = config.CLASS_COLORS[label]
    cv2.putText(frame, head, (x0 + 10, 26), _FONT, 0.6, color, 2, cv2.LINE_AA)

    state = "RECORDING" if recording else "PAUSED"
    state_color = config.COLOR_ALERT if recording else config.COLOR_TEXT
    cv2.putText(frame, state, (x0 + 10, 50), _FONT, 0.5, state_color, 1,
                cv2.LINE_AA)

    y = 78
    for cls in sorted(config.CLASS_NAMES):
        n = counts.get(cls, 0)
        target = config.MIN_SAMPLES_PER_CLASS
        mark = "OK " if n >= target else "   "
        cv2.putText(frame, f"{mark}{cls} {config.CLASS_NAMES[cls]:<8}{n:>4}",
                    (x0 + 10, y), _FONT, 0.5, config.CLASS_COLORS[cls], 1,
                    cv2.LINE_AA)
        y += 22

    # Window fill bar - samples are only written once this is full.
    cv2.putText(frame, "window", (x0 + 10, y + 14), _FONT, 0.45,
                config.COLOR_TEXT, 1, cv2.LINE_AA)
    bar_x, bar_y = x0 + 80, y + 4
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 140, bar_y + 12),
                  config.COLOR_TEXT, 1)
    filled = int(138 * max(0.0, min(1.0, fill_ratio)))
    if filled > 0:
        bar_color = (config.COLOR_OK if fill_ratio >= 1.0
                     else config.COLOR_WARN)
        cv2.rectangle(frame, (bar_x + 1, bar_y + 1),
                      (bar_x + 1 + filled, bar_y + 11), bar_color, -1)

    cv2.putText(frame, "0/1/2 label  SPACE rec", (x0 + 10, y + 42), _FONT,
                0.42, config.COLOR_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, "s save   u undo   q quit", (x0 + 10, y + 60), _FONT,
                0.42, config.COLOR_TEXT, 1, cv2.LINE_AA)
    return frame


def draw_prediction(frame, class_id, confidence, smoothed=True):
    """Banner showing the ML classifier verdict (Step 2 real-time mode)."""
    height, width = frame.shape[:2]

    if class_id is None:
        text, color = "MODEL: warming up", config.COLOR_TEXT
    else:
        name = config.CLASS_NAMES.get(class_id, str(class_id))
        tag = "" if smoothed else " (raw)"
        text = f"ML: {name}{tag}  {confidence * 100:.0f}%"
        color = config.CLASS_COLORS.get(class_id, config.COLOR_TEXT)

    cv2.rectangle(frame, (0, height - 78), (width, height - 46), (0, 0, 0), -1)
    cv2.putText(frame, text, (14, height - 56), _FONT, 0.6, color, 2,
                cv2.LINE_AA)
    return frame
