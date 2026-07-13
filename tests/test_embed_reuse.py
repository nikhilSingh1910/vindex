"""Reuse-check contract for incremental embed (pure function; pins the reviewed
dangling-source-ids defect)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex.models import Segment  # noqa: E402
from vindex.stages.embed import windows_match  # noqa: E402


def _sw(t0, t1, text, ids):
    return Segment(video_id="v", kind="speech_window", t_start_s=t0, t_end_s=t1,
                   payload={"text": text, "source_speech_ids": ids})


def test_identical_windows_match():
    existing = [_sw(0.0, 10.0, "hello world", [1, 2])]
    assert windows_match(existing, [(0.0, 10.0, "hello world", [1, 2])])


def test_new_speech_ids_force_rebuild():
    # A transcribe re-run mints new speech row ids with identical text/bounds; reuse
    # then would leave windows pointing at deleted rows (reviewed: HIGH).
    existing = [_sw(0.0, 10.0, "hello world", [1, 2])]
    assert not windows_match(existing, [(0.0, 10.0, "hello world", [4, 5])])


def test_text_change_forces_rebuild():
    existing = [_sw(0.0, 10.0, "hello world", [1, 2])]
    assert not windows_match(existing, [(0.0, 10.0, "hello there", [1, 2])])


def test_bound_shift_forces_rebuild():
    existing = [_sw(0.0, 10.0, "hello world", [1, 2])]
    assert not windows_match(existing, [(0.0, 10.5, "hello world", [1, 2])])


def test_count_change_forces_rebuild():
    existing = [_sw(0.0, 10.0, "hello world", [1, 2])]
    assert not windows_match(existing, [])
    assert windows_match([], [])
