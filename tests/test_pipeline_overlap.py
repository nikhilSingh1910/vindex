"""Overlap-mode failure matrix: caption-fails-soft, transcribe-fails-hard, and
two-connection writes landing. Stages are stubbed; the jobs table is real."""

import sys
from contextlib import closing
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex import jobs, pipeline, storage  # noqa: E402
from vindex.config import Config  # noqa: E402
from vindex.jobs import Status  # noqa: E402
from vindex.models import Video  # noqa: E402


@pytest.fixture
def env(tmp_path, monkeypatch):
    cfg = Config.for_workdir(tmp_path)
    cfg.overlap_transcribe_caption = True
    conn = storage.connect(cfg.db_path)
    storage.init_schema(conn)
    storage.upsert_video(conn, Video(
        video_id="v", media_path=str(tmp_path / "m.mp4"), media_hash="x",
        fps_num=30, fps_den=1, frame_count=300, duration_s=10.0,
        width=64, height=64, codec="h264"))
    # Pre-complete the stages upstream of the overlap.
    for s in ("ingest", "shots", "frames"):
        jobs.set_status(conn, "v", s, Status.DONE)
    monkeypatch.setattr(pipeline.ingest, "run",
                        lambda *a, **k: pytest.fail("ingest must be skipped"))
    monkeypatch.setattr(pipeline.shots, "run",
                        lambda *a, **k: pytest.fail("shots must be skipped"))
    monkeypatch.setattr(pipeline.frames, "run",
                        lambda *a, **k: pytest.fail("frames must be skipped"))
    monkeypatch.setattr(pipeline.embed, "run", lambda vid, c, cn: {"stub": True})
    return cfg, conn


def test_overlap_happy_path_runs_both_and_embed(env, monkeypatch):
    cfg, conn = env
    ran = []
    monkeypatch.setattr(pipeline.transcribe, "run",
                        lambda vid, c, cn: ran.append("t") or {"ok": 1})
    monkeypatch.setattr(pipeline.caption, "run",
                        lambda vid, c, cn: ran.append("c") or {"ok": 1})
    vid = pipeline.index(str(cfg.workdir / "m.mp4"), cfg, video_id="v")
    assert vid == "v" and set(ran) == {"t", "c"}
    with closing(storage.connect(cfg.db_path)) as check:
        for s in ("transcribe", "caption", "embed"):
            assert jobs.get_status(check, "v", s) == Status.DONE


def test_overlap_caption_failure_is_soft(env, monkeypatch):
    cfg, conn = env
    monkeypatch.setattr(pipeline.transcribe, "run", lambda vid, c, cn: {"ok": 1})

    def bad_caption(vid, c, cn):
        raise RuntimeError("ollama dead")

    monkeypatch.setattr(pipeline.caption, "run", bad_caption)
    vid = pipeline.index(str(cfg.workdir / "m.mp4"), cfg, video_id="v")  # no raise
    assert vid == "v"
    with closing(storage.connect(cfg.db_path)) as check:
        assert jobs.get_status(check, "v", "caption") == Status.FAILED
        assert jobs.get_status(check, "v", "transcribe") == Status.DONE
        assert jobs.get_status(check, "v", "embed") == Status.DONE


def test_overlap_transcribe_failure_is_hard_but_caption_completes(env, monkeypatch):
    cfg, conn = env
    def bad_transcribe(vid, c, cn):
        raise RuntimeError("no audio")

    finished = []
    monkeypatch.setattr(pipeline.transcribe, "run", bad_transcribe)
    monkeypatch.setattr(pipeline.caption, "run",
                        lambda vid, c, cn: finished.append("c") or {"ok": 1})
    with pytest.raises(RuntimeError, match="no audio"):
        pipeline.index(str(cfg.workdir / "m.mp4"), cfg, video_id="v")
    assert finished == ["c"]  # caption's work completed and is preserved
    with closing(storage.connect(cfg.db_path)) as check:
        assert jobs.get_status(check, "v", "transcribe") == Status.FAILED
        assert jobs.get_status(check, "v", "caption") == Status.DONE


def test_partial_state_falls_back_to_sequential(env, monkeypatch):
    cfg, conn = env
    jobs.set_status(conn, "v", "caption", Status.DONE)  # only transcribe needed
    import threading

    order = []
    monkeypatch.setattr(
        pipeline.transcribe, "run",
        lambda vid, c, cn: order.append(threading.current_thread().name) or {"ok": 1})
    monkeypatch.setattr(pipeline.caption, "run",
                        lambda *a, **k: pytest.fail("caption must be skipped"))
    pipeline.index(str(cfg.workdir / "m.mp4"), cfg, video_id="v")
    # Sequential branch runs the stage inline on the main thread (no worker pool).
    assert order == ["MainThread"]
