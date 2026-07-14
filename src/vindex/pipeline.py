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

        # --- transcribe + caption ------------------------------------------------------
        # Caption failure semantics (both modes): advisory + individually skippable, so
        # its failure must not cost the video its entire search surface — record FAILED,
        # warn, continue to embed. The caption->embed cascade re-marks embed pending on
        # the run where caption finally succeeds, so late captions still get embedded.
        _cascade(conn, vid, "transcribe", force)
        caption_wanted = not no_captions
        if caption_wanted:
            _cascade(conn, vid, "caption", force)
        transcribe_needed = force or not jobs.is_done(conn, vid, "transcribe")
        caption_needed = caption_wanted and (force or not jobs.is_done(conn, vid, "caption"))

        caption_err: Exception | None = None
        if cfg.overlap_transcribe_caption and transcribe_needed and caption_needed:
            # Overlap: whisper is CPU-bound, the VLM is GPU/ANE-bound — run both now,
            # in THIS process (single-process-per-DB contract unchanged), each stage on
            # its own connection (WAL + busy_timeout; disjoint kinds, short commits).
            # Cascades were already marked above, before either stage starts.
            from concurrent.futures import ThreadPoolExecutor

            def _stage_own_conn(stage_name: str, stage_fn) -> None:
                tconn = storage.connect(cfg.db_path)
                try:
                    _run_stage(tconn, cfg, vid, stage_name,
                               lambda: stage_fn(vid, cfg, tconn), force)
                finally:
                    tconn.close()

            from concurrent.futures import FIRST_EXCEPTION, wait

            with ThreadPoolExecutor(max_workers=2) as pool:
                t_fut = pool.submit(_stage_own_conn, "transcribe", transcribe.run)
                c_fut = pool.submit(_stage_own_conn, "caption", caption.run)
                # Surface the FIRST failure immediately (a transcribe preflight fails
                # in seconds; silently waiting ~2 h of captioning to report it is
                # hostile), but still let the other stage finish — its completed work
                # is preserved in its own stage either way.
                try:
                    done, _ = wait([t_fut, c_fut], return_when=FIRST_EXCEPTION)
                except KeyboardInterrupt:
                    # True cancellation of in-flight stages isn't available; the pool
                    # join on exit will wait for them. Say so instead of appearing hung.
                    print("[pipeline] interrupt received — waiting for in-flight "
                          "transcribe/caption to finish committing (resumable state); "
                          "interrupt again to abandon the join")
                    raise
                for f in done:
                    if f.exception() is not None:
                        which = "transcribe" if f is t_fut else "caption"
                        print(f"[pipeline] WARNING: {which} FAILED in overlap "
                              f"({f.exception()}); letting the other stage finish")
                try:
                    c_fut.result()
                except Exception as e:
                    caption_err = e
                    print(f"[pipeline] caption failure is non-fatal; continuing — "
                          f"re-run indexing once Ollama is healthy to caption + re-embed")
                # Transcribe failure fails the pipeline (after caption settles).
                t_fut.result()
        else:
            _run_stage(conn, cfg, vid, "transcribe",
                       lambda: transcribe.run(vid, cfg, conn), force)
            if caption_wanted:
                try:
                    _run_stage(conn, cfg, vid, "caption",
                               lambda: caption.run(vid, cfg, conn), force)
                except Exception as e:
                    caption_err = e
                    print(f"[pipeline] WARNING: caption FAILED ({e}); continuing to embed "
                          f"— re-run indexing once Ollama is healthy to caption + re-embed")

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
