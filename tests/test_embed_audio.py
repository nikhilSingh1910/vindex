"""Tiling contract for the CLAP audio space (pure function, no model)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex.stages.embed import audio_windows  # noqa: E402


def test_exact_multiple_tiles_cleanly():
    assert audio_windows(30.0, 10.0, 2.0) == [(0.0, 10.0), (10.0, 20.0), (20.0, 30.0)]


def test_long_remainder_is_own_window():
    assert audio_windows(25.0, 10.0, 2.0)[-1] == (20.0, 25.0)


def test_short_remainder_merges_into_previous():
    ws = audio_windows(21.0, 10.0, 2.0)
    assert ws == [(0.0, 10.0), (10.0, 21.0)]


def test_clip_shorter_than_min_still_gets_one_window():
    assert audio_windows(1.5, 10.0, 2.0) == [(0.0, 1.5)]


def test_empty_duration():
    assert audio_windows(0.0, 10.0, 2.0) == []


def test_windows_tile_without_gaps_or_overlap():
    ws = audio_windows(664.27, 10.0, 2.0)
    assert ws[0][0] == 0.0 and abs(ws[-1][1] - 664.27) < 1e-9
    assert all(a[1] == b[0] for a, b in zip(ws, ws[1:]))
