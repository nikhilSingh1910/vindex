"""Candidate-selection contract for the frames stage (pure function; the cadence is
the user-facing --keyframe-interval parameter)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex.stages.frames import _candidate_indices  # noqa: E402


def test_midpoint_always_included():
    idxs = _candidate_indices(0, 899, fps=30.0, interval_s=3.0)
    assert (0 + 899) // 2 in idxs


def test_cadence_is_time_based():
    # 3 s at 30 fps -> every 90 frames; at 60 fps -> every 180.
    at30 = _candidate_indices(0, 899, fps=30.0, interval_s=3.0)
    at60 = _candidate_indices(0, 1799, fps=60.0, interval_s=3.0)
    assert 90 in at30 and 180 in at60 and 90 not in at60


def test_denser_interval_yields_more_candidates():
    coarse = _candidate_indices(0, 899, fps=30.0, interval_s=3.0)
    dense = _candidate_indices(0, 899, fps=30.0, interval_s=1.0)
    assert len(dense) > len(coarse)
    assert set(coarse) - {449} <= set(dense)  # same anchors, finer fill


def test_short_shot_still_gets_start_and_mid():
    # A 1 s shot at 30 fps is shorter than the 3 s step: start + midpoint only.
    idxs = _candidate_indices(100, 129, fps=30.0, interval_s=3.0)
    assert idxs == sorted({100, 114})


def test_interval_below_frame_duration_clamps_to_every_frame():
    # step = max(1, round(interval*fps)): a tiny interval cannot loop forever.
    idxs = _candidate_indices(0, 9, fps=30.0, interval_s=0.001)
    assert idxs == list(range(10))
