"""Hermetic tests for the parallel caption path: shot-order assembly, breaker
semantics under parallelism, and prior-caption preservation on trip. Ollama is
stubbed; a real tmp-file SQLite DB exercises the replace path."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex import storage  # noqa: E402
from vindex.config import Config  # noqa: E402
from vindex.models import Segment, Video  # noqa: E402
from vindex.stages import caption as cap  # noqa: E402


@pytest.fixture
def db(tmp_path, monkeypatch):
    cfg = Config.for_workdir(tmp_path)
    conn = storage.connect(cfg.db_path)
    storage.init_schema(conn)
    storage.upsert_video(conn, Video(
        video_id="v", media_path=str(tmp_path / "m.mp4"), media_hash="x",
        fps_num=30, fps_den=1, frame_count=300, duration_s=10.0,
        width=64, height=64, codec="h264"))
    segs = []
    for i in range(8):
        segs.append(Segment(video_id="v", kind="shot", t_start_s=float(i),
                            t_end_s=i + 1.0, frame_start=i * 30,
                            frame_end=i * 30 + 29, payload={"shot_index": i}))
        segs.append(Segment(video_id="v", kind="frame", t_start_s=float(i),
                            t_end_s=i + 0.03, frame_start=i * 30, frame_end=i * 30,
                            payload={"shot_index": i, "is_representative": True,
                                     "frame_path": f"frames/f{i}.jpg"}))
    storage.insert_segments(conn, segs)
    monkeypatch.setattr(cap, "_downscale_jpeg", lambda p, m: b"jpeg")
    monkeypatch.setattr(cap, "_ollama_unload", lambda cfg: None)
    return cfg, conn


def _reply(i):
    return {"shot_type": "WS", "people_count": 0, "setting": f"s{i}",
            "on_screen_text": "", "objects": [], "description": f"caption {i}"}


def test_parallel_rows_assemble_in_shot_order(db, monkeypatch):
    cfg, conn = db
    cfg.caption_parallel = 4
    import time

    def stub(cfg_, jpeg):
        n = int(stub_calls.pop(0))
        time.sleep(0.05 if n % 2 == 0 else 0.0)  # scramble completion order
        return _reply(n)

    stub_calls = [str(i) for i in range(8)]
    monkeypatch.setattr(cap, "_ollama_caption", stub)
    out = cap.run("v", cfg, conn)
    assert out["captions"] == 8 and not out["failures"]
    rows = storage.list_segments(conn, "v", kind="caption")
    assert [r.payload["shot_index"] for r in rows] == list(range(8))
    assert [r.payload["description"] for r in rows] == [f"caption {i}" for i in range(8)]


def test_breaker_trips_in_shot_order_and_preserves_prior_captions(db, monkeypatch):
    cfg, conn = db
    cfg.caption_parallel = 4
    cfg.caption_max_consecutive_failures = 3
    # Seed prior captions that a destroy-before-check bug would lose.
    storage.insert_segments(conn, [Segment(
        video_id="v", kind="caption", t_start_s=0.0, t_end_s=1.0,
        payload={"shot_index": 99, "description": "prior"})])

    def stub(cfg_, jpeg):
        raise cap.CaptionError("boom")

    monkeypatch.setattr(cap, "_ollama_caption", stub)
    with pytest.raises(cap.CaptionError, match="consecutive"):
        cap.run("v", cfg, conn)
    survivors = storage.list_segments(conn, "v", kind="caption")
    assert len(survivors) == 1 and survivors[0].payload["description"] == "prior"


def test_partial_failures_recorded_per_shot(db, monkeypatch):
    cfg, conn = db
    cfg.caption_parallel = 2

    def stub(cfg_, jpeg):
        i = int(calls.pop(0))
        if i == 3:
            raise cap.CaptionError("bad json")
        return _reply(i)

    # Each shot retries once on failure -> shot 3 consumes two call slots.
    calls = [str(i) for i in [0, 1, 2, 3, 3, 4, 5, 6, 7]]
    monkeypatch.setattr(cap, "_ollama_caption", stub)
    out = cap.run("v", cfg, conn)
    assert out["captions"] == 7
    assert [f["shot_index"] for f in out["failures"]] == [3]
    shot3 = [s for s in storage.list_segments(conn, "v", kind="shot")
             if s.payload["shot_index"] == 3][0]
    assert "bad json" in shot3.payload.get("caption_error", "")


def test_request_timeout_scales_with_parallelism():
    cfg = Config.for_workdir("/tmp/x")
    cfg.caption_timeout_s, cfg.caption_parallel = 120.0, 2
    assert cap._request_timeout(cfg) == 240.0
    cfg.caption_parallel = 1
    assert cap._request_timeout(cfg) == 120.0
