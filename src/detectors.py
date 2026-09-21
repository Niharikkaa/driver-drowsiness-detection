"""
Temporal state machines built on top of the per-frame EAR / MAR metrics.

A single frame tells you almost nothing. These classes turn a stream of ratios
into events: blinks, sustained eye closure, and yawns. All durations are
measured in seconds (via time.time) rather than frames, so the behaviour does
not change when the webcam FPS fluctuates.
"""

import time
from collections import deque

from src import config


class _Smoother:
    """Small moving average to take the jitter out of a noisy ratio."""

    def __init__(self, window):
        self._window = max(1, int(window))
        self._buffer = deque(maxlen=self._window)

    def update(self, value):
        self._buffer.append(value)
        return sum(self._buffer) / len(self._buffer)

    def reset(self):
        self._buffer.clear()


class BlinkDetector:
    """
    Tracks eye state, counts blinks and measures continuous closure time.

    A blink is registered on the *opening* edge, once the eye has been below
    the EAR threshold for at least EAR_CONSEC_FRAMES frames.
    """

    def __init__(self):
        self._smoother = _Smoother(config.SMOOTHING_WINDOW)
        self.reset()

    def reset(self):
        self._smoother.reset()
        self.blink_count = 0
        self.eye_closed = False
        self.closure_duration = 0.0
        self.longest_closure = 0.0
        self.smoothed_ear = 0.0
        self._closed_frames = 0
        self._closed_since = None

    def update(self, ear, now=None):
        """Feed one frame's EAR. Returns the smoothed EAR."""
        now = time.time() if now is None else now
        self.smoothed_ear = self._smoother.update(ear)

        if self.smoothed_ear < config.EAR_THRESHOLD:
            # --- eye is closed this frame ---
            self._closed_frames += 1
            if self._closed_since is None:
                self._closed_since = now
            self.closure_duration = now - self._closed_since
            self.longest_closure = max(self.longest_closure,
                                       self.closure_duration)
            self.eye_closed = True
        else:
            # --- eye is open: close out any closure that was in progress ---
            if self._closed_frames >= config.EAR_CONSEC_FRAMES:
                self.blink_count += 1
            self._closed_frames = 0
            self._closed_since = None
            self.closure_duration = 0.0
            self.eye_closed = False

        return self.smoothed_ear

    def pause(self):
        """Face lost: drop the in-progress closure without counting a blink."""
        self._smoother.reset()
        self._closed_frames = 0
        self._closed_since = None
        self.closure_duration = 0.0
        self.eye_closed = False

    @property
    def is_drowsy(self):
        return self.closure_duration >= config.EYE_CLOSED_ALARM_SEC

    @property
    def state_text(self):
        return "CLOSED" if self.eye_closed else "OPEN"


class YawnDetector:
    """
    Counts yawns. The mouth must stay above MAR_THRESHOLD for at least
    YAWN_MIN_DURATION_SEC, and each opening is counted only once.
    """

    def __init__(self):
        self._smoother = _Smoother(config.SMOOTHING_WINDOW)
        self.reset()

    def reset(self):
        self._smoother.reset()
        self.yawn_count = 0
        self.mouth_open = False
        self.open_duration = 0.0
        self.smoothed_mar = 0.0
        self._open_since = None
        self._counted_current = False

    def update(self, mar, now=None):
        now = time.time() if now is None else now
        self.smoothed_mar = self._smoother.update(mar)

        if self.smoothed_mar > config.MAR_THRESHOLD:
            if self._open_since is None:
                self._open_since = now
                self._counted_current = False
            self.open_duration = now - self._open_since
            self.mouth_open = True

            if (not self._counted_current
                    and self.open_duration >= config.YAWN_MIN_DURATION_SEC):
                self.yawn_count += 1
                self._counted_current = True
        else:
            self._open_since = None
            self._counted_current = False
            self.open_duration = 0.0
            self.mouth_open = False

        return self.smoothed_mar

    def pause(self):
        self._smoother.reset()
        self._open_since = None
        self._counted_current = False
        self.open_duration = 0.0
        self.mouth_open = False

    @property
    def is_yawning(self):
        return (self.mouth_open
                and self.open_duration >= config.YAWN_MIN_DURATION_SEC)


class DrowsinessMonitor:
    """
    Convenience façade that drives both detectors and exposes a flat dict of
    everything the HUD needs to draw.
    """

    def __init__(self):
        self.blink = BlinkDetector()
        self.yawn = YawnDetector()
        self.face_found = False
        self._missing_frames = 0

    def update(self, metrics, now=None):
        """metrics is the dict returned by metrics.compute_metrics()."""
        self.face_found = True
        self._missing_frames = 0
        self.blink.update(metrics["ear"], now)
        self.yawn.update(metrics["mar"], now)
        return self.status()

    def update_missing(self):
        """Called on frames where no face/landmarks were detected."""
        self._missing_frames += 1
        if self._missing_frames >= config.MAX_MISSING_FRAMES:
            self.face_found = False
            self.blink.pause()
            self.yawn.pause()
        return self.status()

    def reset(self):
        self.blink.reset()
        self.yawn.reset()
        self.face_found = False
        self._missing_frames = 0

    def status(self):
        return {
            "ear": self.blink.smoothed_ear,
            "mar": self.yawn.smoothed_mar,
            "eye_state": self.blink.state_text,
            "blinks": self.blink.blink_count,
            "closure": self.blink.closure_duration,
            "longest_closure": self.blink.longest_closure,
            "yawns": self.yawn.yawn_count,
            "drowsy": self.blink.is_drowsy,
            "yawning": self.yawn.is_yawning,
            "face_found": self.face_found,
        }
