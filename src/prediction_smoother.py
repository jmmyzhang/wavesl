"""Temporal smoothing for per-frame ASL predictions.

Buffers raw frame predictions and emits a stable label only when one
class wins a configurable majority of the rolling window.
"""

from collections import Counter, deque
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SmoothedPrediction:
    label: str
    avg_confidence: float
    is_new: bool  # True only on the first frame a new label stabilizes


class PredictionSmoother:
    """
    Rolling majority-vote buffer over per-frame (label, confidence) predictions.

    Args:
        buffer_size: Number of recent frames to consider (default 15 ~= 0.5 s at 30 fps).
        majority_threshold: Fraction of buffer a label must win to be emitted (default 0.6).
        cooldown_frames: Frames to suppress re-emission after emitting a label (default 10).
    """

    def __init__(
        self,
        buffer_size: int = 15,
        majority_threshold: float = 0.6,
        cooldown_frames: int = 10,
    ) -> None:
        self._buffer_size = buffer_size
        self._majority_threshold = majority_threshold
        self._cooldown_frames = cooldown_frames
        self._labels: deque[str] = deque(maxlen=buffer_size)
        self._confs: deque[float] = deque(maxlen=buffer_size)
        self._last_emitted: Optional[str] = None
        self._cooldown_remaining: int = 0

    def update(self, label: str, confidence: float) -> Optional[SmoothedPrediction]:
        """
        Record a new per-frame prediction.
        Returns a SmoothedPrediction when the buffer reaches consensus, else None.
        """
        self._labels.append(label)
        self._confs.append(confidence)

        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1
            return None

        if len(self._labels) < self._buffer_size:
            return None

        counts = Counter(self._labels)
        top_label, top_count = counts.most_common(1)[0]
        if top_count / self._buffer_size < self._majority_threshold:
            return None

        is_new = top_label != self._last_emitted
        self._last_emitted = top_label
        if is_new:
            self._cooldown_remaining = self._cooldown_frames

        avg_conf = (
            sum(c for lbl, c in zip(self._labels, self._confs) if lbl == top_label)
            / top_count
        )

        return SmoothedPrediction(
            label=top_label, avg_confidence=avg_conf, is_new=is_new
        )

    def clear(self) -> None:
        """Reset the buffer (call when no hands are detected)."""
        self._labels.clear()
        self._confs.clear()
        self._last_emitted = None
        self._cooldown_remaining = 0
