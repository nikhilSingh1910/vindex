"""Stage 5 — caption. VLM structured captions for each shot's representative frame, via
Ollama's REST API (identical on macOS and Linux — the stage ports itself).

PLAN hardening, all implemented here:
- num_ctx pinned per request (the qwen2.5vl GGUF declares a 125K context window and Ollama
  over-allocates without a pin — measured ~15 GB RSS at 8k ctx for the 7B).
- Frames downscaled to <=1024 px longest side before sending (Ollama's internal 2 MP cap
  does not catch 1080p; oversized frames burn ~2,600 visual tokens).
- Per-request timeout, num_predict cap, ONE retry, then the failure is recorded on the
  shot row itself (payload.caption_error — queryable via `list --kind shot`) and the stage
  moves on — a hung/failed caption never hangs the pipeline.
- Circuit breaker: N consecutive failures abort the stage BEFORE it touches prior output
  (a wedged Ollama otherwise burns 2×timeout×shots of wall clock producing nothing).
- Structured fields are ADVISORY (7B-class VLMs mislabel ~10-20% of attributes); they are
  stored under payload.caption with model provenance, never as ground truth. Every stored
  field is hard-typed here: a malformed reply fails THIS shot, never poisons the DB for a
  later stage.

Output: kind='caption' rows (one per successfully captioned shot, sharing the shot's time
range) with structured fields + freeform description. The embed stage picks up caption
descriptions on its next run (it is already caption-aware).
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import urllib.error
import urllib.request

from ..config import Config
from ..models import Segment
from .. import storage

STAGE = "caption"

PROMPT = """Describe this video frame for a film editor's index. Reply with ONLY a JSON \
object, no other text, with exactly these keys:
- "shot_type": one of "CU" (close-up), "MS" (medium shot), "WS" (wide shot)
- "people_count": integer, number of people visible
- "setting": short phrase, where this takes place
- "on_screen_text": any readable text in the frame, empty string if none
- "objects": array of up to 8 prominent object names
- "description": 1-3 sentences, dense and specific, describing what is visible"""

REQUIRED_KEYS = {"shot_type", "people_count", "setting", "on_screen_text", "objects", "description"}


class CaptionError(RuntimeError):
    pass


def _downscale_jpeg(path: str, max_side: int) -> bytes:
    from PIL import Image

    im = Image.open(path).convert("RGB")
    w, h = im.size
    scale = max(w, h) / max_side
    if scale > 1:
        im = im.resize((int(w / scale), int(h / scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def _request_timeout(cfg: Config) -> float:
    """Client budget scales with parallelism: on a server without matching
    OLLAMA_NUM_PARALLEL slots the extra requests QUEUE, so one wall-clock budget must
    cover up to caption_parallel generations (reviewed: un-scaled, queueing silently
    halves the effective timeout and near-timeout shots start failing)."""
    return cfg.caption_timeout_s * max(1, cfg.caption_parallel)


def _ollama_caption(cfg: Config, image_jpeg: bytes) -> dict:
    """One captioning request. Raises on transport error, timeout, or unparseable output."""
    req = urllib.request.Request(
        f"{cfg.ollama_url}/api/generate",
        data=json.dumps({
            "model": cfg.caption_model,
            "prompt": PROMPT,
            "images": [base64.b64encode(image_jpeg).decode()],
            "format": "json",
            "stream": False,
            "options": {
                "num_ctx": cfg.caption_num_ctx,       # pin: GGUF declares 125K
                "num_predict": cfg.caption_num_predict,
                "temperature": 0,                      # deterministic-ish captions
            },
        }).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=_request_timeout(cfg)) as resp:
            body = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        # str(HTTPError) is just "HTTP Error 404: Not Found"; the body says WHICH problem
        # ("model not found" needs `ollama pull`, a runner crash needs a daemon restart).
        detail = e.read(300).decode("utf-8", errors="replace")
        raise CaptionError(f"HTTP {e.code}: {detail}") from None
    data = json.loads(body["response"])
    if not isinstance(data, dict):
        raise CaptionError(f"caption reply is a JSON {type(data).__name__}, not an object")
    missing = REQUIRED_KEYS - set(data)
    if missing:
        raise CaptionError(f"caption missing keys: {sorted(missing)}")
    desc = data.get("description")
    if not isinstance(desc, str) or not desc.strip():
        # description is the embedded retrieval text; a list/dict here would crash the
        # embed stage's .strip() on every later run until the row is hand-deleted.
        raise CaptionError("caption description missing, empty, or not a string")
    data["description"] = desc.strip()
    if not isinstance(data.get("objects"), list):
        data["objects"] = []
    data["objects"] = [str(o) for o in data["objects"][:8]]
    try:
        data["people_count"] = int(data.get("people_count") or 0)
    except (TypeError, ValueError):
        data["people_count"] = None  # advisory field: unknown beats invented
    for key in ("setting", "on_screen_text"):
        if not isinstance(data.get(key), str):
            data[key] = "" if data.get(key) is None else str(data[key])
    if data.get("shot_type") not in ("CU", "MS", "WS"):
        data["shot_type"] = None  # advisory field: unknown beats invented
    return data


def _ollama_unload(cfg: Config) -> None:
    """Best-effort: release the VLM now (keep_alive=0) rather than holding ~4-6 GB resident
    for Ollama's default 5 minutes while the embed stage loads SigLIP on a 16 GB box."""
    try:
        req = urllib.request.Request(
            f"{cfg.ollama_url}/api/generate",
            data=json.dumps({"model": cfg.caption_model, "keep_alive": 0}).encode(),
            headers={"Content-Type": "application/json"},
        )
        urllib.request.urlopen(req, timeout=10).close()
    except Exception:
        pass


def run(video_id: str, cfg: Config, conn) -> dict:
    video = storage.get_video(conn, video_id)
    if video is None:
        raise CaptionError(f"no ingested video {video_id}; run ingest first")
    shots = storage.list_segments(conn, video_id, kind="shot")
    if not shots:
        raise CaptionError(f"no shots for {video_id}; run shots first")
    frames = storage.list_segments(conn, video_id, kind="frame")
    rep_by_shot = {
        f.payload.get("shot_index"): f
        for f in frames
        if f.payload.get("is_representative")
    }

    def caption_shot(shot: Segment) -> tuple[Segment, dict | None, str]:
        """One shot's caption attempt: (shot, data-or-None, error). Thread-safe: pure
        request/parse work, no shared state."""
        si = shot.payload.get("shot_index")
        rep = rep_by_shot.get(si)
        if rep is None:
            return shot, None, "no representative frame"
        last_err = ""
        try:
            # frame_path is workdir-relative (absolute rows from older DBs still work:
            # joining an absolute path onto workdir yields the absolute path).
            jpeg = _downscale_jpeg(
                str(cfg.workdir / rep.payload["frame_path"]), cfg.caption_max_side_px)
        except (OSError, ValueError) as e:
            return shot, None, f"frame unreadable: {e}"[:300]  # fails THIS shot only
        for _ in range(2):  # one retry, then record failure and move on
            try:
                return shot, _ollama_caption(cfg, jpeg), ""
            except (urllib.error.URLError, TimeoutError, CaptionError, KeyError,
                    ValueError, TypeError, http.client.HTTPException, OSError) as e:
                last_err = str(e)[:300]
        return shot, None, last_err

    # Results keyed by shot id so rows assemble in SHOT order regardless of worker
    # completion order (replace_segments output must not depend on scheduling).
    outcome: dict[int, tuple[dict | None, str]] = {}
    consecutive = 0
    n_ok = 0
    try:
        workers = max(1, cfg.caption_parallel)
        if workers == 1:
            iterator = map(caption_shot, shots)
        else:
            from concurrent.futures import ThreadPoolExecutor

            pool = ThreadPoolExecutor(max_workers=workers)
            # map submits ALL futures up front but yields results in submission
            # (= shot) order, with only `workers` threads executing; on breaker trip,
            # cancel_futures drops every not-yet-started future (the tail).
            iterator = pool.map(caption_shot, shots)
        try:
            for shot, data, err in iterator:
                assert shot.id is not None
                if data is None:
                    outcome[shot.id] = (None, err)
                    if err == "no representative frame":
                        # A local data gap, not a sick VLM: recorded as a failure but
                        # never counted toward the breaker (sequential-era semantics).
                        continue
                    consecutive += 1
                    if consecutive >= cfg.caption_max_consecutive_failures:
                        # Raising here, BEFORE replace_segments, preserves prior captions.
                        raise CaptionError(
                            f"aborting after {consecutive} consecutive failures "
                            f"({n_ok} shots captioned first); last: {err}")
                    continue
                consecutive = 0
                n_ok += 1
                outcome[shot.id] = (data, "")
        finally:
            if workers > 1:
                # wait=True: bounded at <= workers in-flight requests (~workers x
                # timeout worst case, honoring the breaker's purpose), and REQUIRED —
                # orphaned in-flight generations would otherwise queue behind the
                # unload below, no-op it, and re-load the model with default
                # keep_alive, holding ~4.6 GB into embed's SigLIP load (reviewed).
                pool.shutdown(wait=True, cancel_futures=True)
    finally:
        _ollama_unload(cfg)

    rows: list[Segment] = []
    ok_shot_ids: list[int] = []
    failures: list[dict] = []
    for shot in shots:  # shot order, independent of completion order
        assert shot.id is not None
        if shot.id not in outcome:
            continue  # defensive only: trip raises before assembly, so unreachable
        data, err = outcome[shot.id]
        si = shot.payload.get("shot_index")
        if data is None:
            failures.append({"shot_id": shot.id, "shot_index": si, "error": err})
            continue
        rep = rep_by_shot[si]
        ok_shot_ids.append(shot.id)
        rows.append(Segment(
            video_id=video_id, kind="caption",
            t_start_s=shot.t_start_s, t_end_s=shot.t_end_s,
            frame_start=shot.frame_start, frame_end=shot.frame_end,
            payload={
                "shot_index": si,
                "source_frame": rep.frame_start,
                "frame_path": rep.payload["frame_path"],
                "shot_type": data["shot_type"],
                "people_count": data["people_count"],
                "setting": data["setting"],
                "on_screen_text": data["on_screen_text"],
                "objects": data["objects"][:8],
                "description": data["description"],
                "caption_model": cfg.caption_model,
                "advisory": True,  # VLM fields are hints, never ground truth
            },
        ))

    if failures and not rows:
        # Total failure (Ollama down / model missing) must fail loudly BEFORE the replace
        # below — replace-then-raise would destroy the prior good captions (and their
        # embeddings) with no replacement landing.
        raise CaptionError(f"all {len(failures)} caption requests failed; first: {failures[0]}")

    # Replace caption rows and, in the SAME transaction, record per-shot failures on the
    # shot rows (payload.caption_error — queryable via `list --kind shot`); successful
    # shots clear any stale error from a prior run.
    merges = [(f["shot_id"], {"caption_error": f["error"]})
              for f in failures if f.get("shot_id") is not None]
    merges += [(sid, {"caption_error": None}) for sid in ok_shot_ids]
    storage.replace_segments(conn, video_id, ("caption",), rows, payload_merges=merges)
    return {"captions": len(rows), "failures": failures}
