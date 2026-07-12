"""Stage runner. Resumable via the jobs table: a done stage is skipped, a stage runs only
after its predecessor is done, and a crash leaves the failed stage marked so a rerun resumes.
"""

from __future__ import annotations

from contextlib import closing

from .config import Config
from . import storage, jobs
from .jobs import Status
from .stages import caption, embed, frames, ingest, shots, transcribe

STAGES = ("ingest", "shots", "frames", "transcribe", "caption", "embed")

# A stage re-run invalidates its consumers' persisted output: shots replaces frame rows
# (cascade delete), frames replaces the JPEGs/rows captions and image embeddings point at,
# transcribe replaces the speech rows windows are packed from, caption inserts rows whose
# embeddings only exist once embed runs. If a stage is about to run, its consumers must be
# re-marked pending — otherwise a 'done' job skips and leaves stale or unembedded
# (search-invisible) rows permanent.
DOWNSTREAM = {
    "shots": ("frames", "caption", "embed"),
    "frames": ("caption", "embed"),
    "transcribe": ("embed",),
    "caption": ("embed",),
}


def index(
    source: str,
    cfg: Config,
    video_id: str | None = None,
    force: bool = False,
    no_captions: bool = False,
) -> str:
    with closing(storage.connect(cfg.db_path)) as conn:
        storage.init_schema(conn)

        # --- ingest -----------------------------------------------------------------
        # ingest is keyed by source until it produces a video_id, so it can't use the jobs
        # gate on video_id up front; we resolve identity here, then gate the rest.
        if video_id and jobs.is_done(conn, video_id, "ingest") and not force:
            vid = video_id
            print(f"[pipeline] ingest already done for {vid}, skipping")
        else:
            print(f"[pipeline] ingest: {source}")
            try:
                out = ingest.run(source, cfg, conn, video_id=video_id)
                vid = out.video.video_id
                jobs.set_status(conn, vid, "ingest", Status.DONE)
            except Exception as e:
                # We may not have a vid; record under a best-effort key if we do.
                if video_id:
                    jobs.set_status(conn, video_id, "ingest", Status.FAILED, str(e))
                raise
            print(f"[pipeline] ingest done -> video_id={vid} "
                  f"({out.video.width}x{out.video.height} @ {out.video.fps:.3f}fps, "
                  f"{out.video.frame_count} frames)")

        # --- shots (CPU, fast) before transcribe (heavy model) ------------------------
        _cascade(conn, vid, "shots", force)
        _run_stage(conn, cfg, vid, "shots", lambda: shots.run(vid, cfg, conn), force)

        # --- frames (depends on shots) ------------------------------------------------
        _cascade(conn, vid, "frames", force)
        _run_stage(conn, cfg, vid, "frames", lambda: frames.run(vid, cfg, conn), force)

        # --- transcribe -------------------------------------------------------------
        _cascade(conn, vid, "transcribe", force)
        _run_stage(conn, cfg, vid, "transcribe", lambda: transcribe.run(vid, cfg, conn), force)

        # --- caption (advisory + individually skippable, so its failure must not cost
        # the video its entire search surface: record FAILED, warn, continue to embed.
        # The caption->embed cascade re-marks embed pending on the run where caption
        # finally succeeds, so late captions still get embedded and searchable. ---------
        caption_err: Exception | None = None
        if not no_captions:
            _cascade(conn, vid, "caption", force)
            try:
                _run_stage(conn, cfg, vid, "caption", lambda: caption.run(vid, cfg, conn), force)
            except Exception as e:
                caption_err = e
                print(f"[pipeline] WARNING: caption FAILED ({e}); continuing to embed — "
                      f"re-run indexing once Ollama is healthy to caption + re-embed")

        # --- embed (last: consumes frames + transcript + captions) --------------------
        _run_stage(conn, cfg, vid, "embed", lambda: embed.run(vid, cfg, conn), force)

        if caption_err is not None:
            print(f"[pipeline] NOTE: {vid} indexed WITHOUT captions (caption stage failed)")
        return vid


def _cascade(conn, vid, stage, force) -> None:
    if force or not jobs.is_done(conn, vid, stage):
        for downstream in DOWNSTREAM.get(stage, ()):
            jobs.set_status(conn, vid, downstream, Status.PENDING)


def _run_stage(conn, cfg, video_id, stage, fn, force) -> None:
    if jobs.is_done(conn, video_id, stage) and not force:
        print(f"[pipeline] {stage} already done for {video_id}, skipping")
        return
    print(f"[pipeline] {stage}: {video_id}")
    jobs.set_status(conn, video_id, stage, Status.RUNNING)
    try:
        result = fn()
        jobs.set_status(conn, video_id, stage, Status.DONE)
        print(f"[pipeline] {stage} done: {result}")
    except Exception as e:
        jobs.set_status(conn, video_id, stage, Status.FAILED, str(e))
        raise
