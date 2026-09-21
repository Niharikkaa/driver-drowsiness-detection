"""
Sliding-window feature extraction (Step 2).

Step 1 produces per-frame values (EAR, MAR, eye state, blink count, closure
duration, yawn count). A single frame carries almost no drowsiness signal - a
low EAR could be a blink or a microsleep. Drowsiness only becomes visible over
time, so this module aggregates the Step 1 output over a sliding time window
into the feature vector the classifier consumes.

This module reads the dict returned by DrowsinessMonitor.status(). It does not
touch MediaPipe or re-implement any Step 1 logic.
"""

from collections import deque

import numpy as np

from src import config


class FeatureWindow:
    """
    Feed it one status dict per frame; ask it for a feature vector whenever
    you need one.

    Blinks and yawns are detected by watching the monotonically increasing
    counters from Step 1, so the event logic is never duplicated here.
    """

    def __init__(self, window_sec=None, min_frames=None):
        self.window_sec = (config.FEATURE_WINDOW_SEC if window_sec is None
                           else window_sec)
        self.min_frames = (config.MIN_WINDOW_FRAMES if min_frames is None
                           else min_frames)
        self.reset()

    # ------------------------------------------------------------------
    def reset(self):
        self._frames = deque()        # (t, ear, mar, eye_closed, mouth_open)
        self._blink_times = deque()   # timestamps of completed blinks
        self._closures = deque()      # (t, duration) of completed closures
        self._yawn_times = deque()    # timestamps of completed yawns
        self._prev_blinks = None
        self._prev_yawns = None
        self._prev_closure = 0.0

    # ------------------------------------------------------------------
    def update(self, status, now, face_present=None):
        """
        Add one frame.

        Parameters
        ----------
        status : dict from DrowsinessMonitor.status()
        now    : float timestamp in seconds (time.time())
        face_present : bool, optional
            Whether landmarks were actually found on *this* frame. Pass it
            when you know (e.g. ``points is not None``).

            Step 1's ``face_found`` flag deliberately has hysteresis: it stays
            True for up to MAX_MISSING_FRAMES so a brief tracking glitch does
            not reset the session. That is right for the alert logic but wrong
            here, because those frames would repeat a stale EAR/MAR into the
            window. When this argument is omitted the flag is used as a
            fallback.
        """
        present = (bool(status.get("face_found", False))
                   if face_present is None else bool(face_present))

        if not present:
            # Do not pollute the window with frames where nothing was tracked.
            self._trim(now)
            return

        self._frames.append((
            now,
            float(status["ear"]),
            float(status["mar"]),
            bool(status["eye_state"] == "CLOSED"),
            bool(status["mar"] > config.MAR_THRESHOLD),
        ))

        # --- blink events: the Step 1 counter went up ---
        blinks = int(status["blinks"])
        if self._prev_blinks is not None and blinks > self._prev_blinks:
            for _ in range(blinks - self._prev_blinks):
                self._blink_times.append(now)
            # The closure that just ended had this duration on the last frame.
            if self._prev_closure > 0:
                self._closures.append((now, self._prev_closure))
        self._prev_blinks = blinks

        # --- yawn events ---
        yawns = int(status["yawns"])
        if self._prev_yawns is not None and yawns > self._prev_yawns:
            for _ in range(yawns - self._prev_yawns):
                self._yawn_times.append(now)
        self._prev_yawns = yawns

        # A closure that is still open still counts toward max_closure below.
        self._prev_closure = float(status["closure"])

        self._trim(now)

    def _trim(self, now):
        cutoff = now - self.window_sec
        while self._frames and self._frames[0][0] < cutoff:
            self._frames.popleft()
        while self._blink_times and self._blink_times[0] < cutoff:
            self._blink_times.popleft()
        while self._closures and self._closures[0][0] < cutoff:
            self._closures.popleft()
        while self._yawn_times and self._yawn_times[0] < cutoff:
            self._yawn_times.popleft()

    # ------------------------------------------------------------------
    @property
    def ready(self):
        """True once the window holds enough frames to be meaningful."""
        return len(self._frames) >= self.min_frames

    @property
    def frame_count(self):
        return len(self._frames)

    @property
    def fill_ratio(self):
        """0.0 to 1.0 - how full the window is. Handy for a progress bar."""
        return min(1.0, len(self._frames) / max(1, self.min_frames))

    # ------------------------------------------------------------------
    def extract(self, current_closure=0.0):
        """
        Compute the feature vector.

        Returns an ordered dict keyed by config.FEATURE_COLUMNS, or None if
        the window is not ready yet.
        """
        if not self.ready:
            return None

        data = np.array([(f[1], f[2], f[3], f[4]) for f in self._frames],
                        dtype=np.float64)
        ear = data[:, 0]
        mar = data[:, 1]
        closed = data[:, 2]
        mouth_open = data[:, 3]

        span = self._frames[-1][0] - self._frames[0][0]
        minutes = max(span, 1e-6) / 60.0

        durations = [d for _, d in self._closures]
        # Include a closure still in progress so a microsleep is not missed.
        max_closure = max(durations + [float(current_closure)], default=0.0)
        mean_closure = float(np.mean(durations)) if durations else 0.0

        features = {
            "ear_mean": float(np.mean(ear)),
            "ear_std": float(np.std(ear)),
            "ear_min": float(np.min(ear)),
            "perclos": float(np.mean(closed)),
            "blink_rate": len(self._blink_times) / minutes,
            "mean_closure": mean_closure,
            "max_closure": float(max_closure),
            "mar_mean": float(np.mean(mar)),
            "mar_max": float(np.max(mar)),
            "yawn_rate": len(self._yawn_times) / minutes,
            "mouth_open_ratio": float(np.mean(mouth_open)),
        }

        # Guarantee the documented column order.
        return {name: features[name] for name in config.FEATURE_COLUMNS}


def features_to_array(features):
    """Turn a feature dict into a 1 x n_features array for scikit-learn."""
    return np.array([[features[name] for name in config.FEATURE_COLUMNS]],
                    dtype=np.float64)
