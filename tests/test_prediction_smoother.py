"""Unit tests for PredictionSmoother."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import pytest

from prediction_smoother import PredictionSmoother, SmoothedPrediction


# ── helpers ───────────────────────────────────────────────────────────────────

def _push(smoother: PredictionSmoother, label: str, confidence: float,
          count: int) -> list:
    """Push `count` identical predictions; return all results."""
    return [smoother.update(label, confidence) for _ in range(count)]


def _last(smoother: PredictionSmoother, label: str,
          confidence: float, count: int):
    """Push `count` predictions and return only the final result."""
    return _push(smoother, label, confidence, count)[-1]


# ── buffer fill ───────────────────────────────────────────────────────────────

class TestBufferFill:
    def test_returns_none_before_buffer_full(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        results = _push(s, 'chair', 0.9, 4)
        assert all(r is None for r in results)

    def test_emits_on_buffer_full(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        result = _last(s, 'chair', 0.9, 5)
        assert result is not None
        assert result.label == 'chair'

    def test_emitted_prediction_is_frozen(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        result = _last(s, 'book', 0.8, 5)
        with pytest.raises(Exception):
            result.label = 'other'  # type: ignore[misc]


# ── majority vote ─────────────────────────────────────────────────────────────

class TestMajorityVote:
    def test_majority_label_wins(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        # 3 × 'cat', 2 × 'dog' → cat wins (3/5 = 0.6 ≥ threshold)
        _push(s, 'dog', 0.5, 2)
        result = _last(s, 'cat', 0.9, 3)
        assert result is not None
        assert result.label == 'cat'

    def test_tie_returns_none(self):
        s = PredictionSmoother(buffer_size=4, majority_threshold=0.6, cooldown_frames=0)
        # 2 × 'a', 2 × 'b' → neither reaches 60%
        _push(s, 'a', 0.8, 2)
        result = _last(s, 'b', 0.8, 2)
        assert result is None

    def test_below_threshold_returns_none(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.8, cooldown_frames=0)
        # 4/5 = 80% of 'chair' — exactly meets threshold
        _push(s, 'other', 0.5, 1)
        result = _last(s, 'chair', 0.9, 4)
        assert result is not None  # 4/5 = 0.8 == threshold

    def test_strictly_below_threshold_returns_none(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.9, cooldown_frames=0)
        # 4/5 = 0.8 < 0.9 threshold
        _push(s, 'other', 0.5, 1)
        result = _last(s, 'chair', 0.9, 4)
        assert result is None

    def test_avg_confidence_is_for_winning_label_only(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        _push(s, 'noise', 0.1, 2)
        result = _last(s, 'word', 0.9, 3)
        assert result is not None
        assert abs(result.avg_confidence - 0.9) < 1e-5


# ── is_new flag ───────────────────────────────────────────────────────────────

class TestIsNew:
    def test_first_stable_emit_is_new(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        result = _last(s, 'hello', 0.9, 5)
        assert result is not None
        assert result.is_new is True

    def test_continued_same_label_not_new(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        _push(s, 'hello', 0.9, 5)  # fill buffer, trigger first emit
        # next frame: buffer still majority 'hello'
        result = s.update('hello', 0.9)
        assert result is not None
        assert result.is_new is False

    def test_new_label_after_old_is_new(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        _push(s, 'hello', 0.9, 5)
        # Shift buffer to 'bye' — transition happens somewhere within this push
        results = _push(s, 'bye', 0.9, 5)
        # The first non-None result for 'bye' must be flagged is_new=True
        bye_results = [r for r in results if r is not None and r.label == 'bye']
        assert len(bye_results) > 0
        assert bye_results[0].is_new is True


# ── cooldown ──────────────────────────────────────────────────────────────────

class TestCooldown:
    def test_cooldown_suppresses_re_emission(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=3)
        _push(s, 'hello', 0.9, 5)  # emits is_new=True, starts cooldown
        # next 3 frames should be suppressed
        results = _push(s, 'hello', 0.9, 3)
        assert all(r is None for r in results)

    def test_after_cooldown_emits_again(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=2)
        _push(s, 'hello', 0.9, 5)  # emit + start cooldown
        _push(s, 'hello', 0.9, 2)  # cooldown period
        result = s.update('hello', 0.9)
        assert result is not None

    def test_cooldown_only_triggers_on_new_label(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=5)
        _push(s, 'hello', 0.9, 5)  # is_new=True → cooldown starts
        # After cooldown, continued 'hello' should NOT restart cooldown
        _push(s, 'hello', 0.9, 5)  # drain cooldown
        result = s.update('hello', 0.9)
        # is_new=False, no cooldown started, should emit
        assert result is not None
        assert result.is_new is False


# ── clear ─────────────────────────────────────────────────────────────────────

class TestClear:
    def test_clear_empties_buffer(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        _push(s, 'hello', 0.9, 4)
        s.clear()
        # After clear, buffer has 0 frames — should not emit on next push
        result = s.update('hello', 0.9)
        assert result is None

    def test_clear_resets_last_emitted_so_next_is_new(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=0)
        _push(s, 'hello', 0.9, 5)  # emits is_new=True
        s.update('hello', 0.9)     # emits is_new=False
        s.clear()
        result = _last(s, 'hello', 0.9, 5)
        assert result is not None
        assert result.is_new is True

    def test_clear_resets_cooldown(self):
        s = PredictionSmoother(buffer_size=5, majority_threshold=0.6, cooldown_frames=10)
        _push(s, 'hello', 0.9, 5)  # starts cooldown of 10
        s.clear()
        result = _last(s, 'hello', 0.9, 5)
        assert result is not None  # cooldown was cleared
