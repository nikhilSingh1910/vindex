"""Stage 6 — embed. Two embedding spaces, one column:

- Images: SigLIP (so400m) on every kept keyframe's persisted JPEG -> embedding on the
  kind='frame' row. fp16 with torch_dtype set EXPLICITLY (the HF checkpoint ships F32 —
  PLAN facts #4); MPS on Apple Silicon, CUDA/CPU elsewhere.
- Text: bge-small-en-v1.5 on sentence-aligned transcript windows (~30-60 s, built from
  Whisper segment boundaries, carrying word-timestamp anchors) -> new kind='speech_window'
  rows; and on caption descriptions when stage 5 lands (kind='caption' rows are embedded
  if present, so the stage is caption-ready without modification).

Embeddings are float32 little-endian BLOBs (sqlite-vec scalar-function format). Every
embedded row stores model_name + dim — mixed dimensions share the one column, and search
scopes each KNN by model_name (PLAN Search section).
"""

from __future__ import annotations

from dataclasses import dataclass

from ..config import Config
from ..models import Segment
from .. import storage

STAGE = "embed"

# Preferred checkpoints, in order. SigLIP 2 is the drop-in successor (PLAN: check at build
# time); fall back to SigLIP 1 if the installed transformers can't load it.
SIGLIP_CANDIDATES = (
    "google/siglip2-so400m-patch14-384",
    "google/siglip-so400m-patch14-384",
)
TEXT_MODEL = "BAAI/bge-small-en-v1.5"


class EmbedError(RuntimeError):
    pass


def _pick_device():
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class _Siglip:
    model: object
    processor: object
    name: str
    device: str
    dim: int


def _load_siglip() -> _Siglip:
    import torch
    from transformers import AutoModel, AutoProcessor

    device = _pick_device()
    last_err: Exception | None = None
    for name in SIGLIP_CANDIDATES:
        try:
            # dtype MUST be explicit: the checkpoint tensors are F32 (~3.5 GB) and a
            # default load would double the planned memory footprint (PLAN facts #4).
            model = AutoModel.from_pretrained(name, dtype=torch.float16).to(device).eval()
            processor = AutoProcessor.from_pretrained(name)
            dim = int(model.config.vision_config.hidden_size)
            return _Siglip(model=model, processor=processor, name=name, device=device, dim=dim)
        except Exception as e:  # try the next candidate
            last_err = e
    raise EmbedError(f"no SigLIP checkpoint loadable; last error: {last_err}")


def _embed_images(sig: _Siglip, paths: list[str], batch_size: int = 8) -> list[bytes]:
    import numpy as np
    import torch
    from PIL import Image

    out: list[bytes] = []
    for i in range(0, len(paths), batch_size):
        batch = [Image.open(p).convert("RGB") for p in paths[i : i + batch_size]]
        inputs = sig.processor(images=batch, return_tensors="pt").to(sig.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(torch.float16)
        with torch.no_grad():
            feats = sig.model.get_image_features(**inputs)
        # SigLIP 2 under current transformers returns BaseModelOutputWithPooling;
        # SigLIP 1 returns a raw tensor. Normalize both to a tensor.
        if not torch.is_tensor(feats):
            feats = feats.pooler_output
        feats = torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()
        out.extend(np.ascontiguousarray(v, dtype=np.float32).tobytes() for v in feats)
    return out


def _load_text_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(TEXT_MODEL, device=_pick_device())


def _embed_texts(model, texts: list[str]) -> list[bytes]:
    import numpy as np

    vecs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return [np.ascontiguousarray(v, dtype=np.float32).tobytes() for v in vecs]


def build_windows(
    speech: list[Segment], target_s: float, max_s: float
) -> list[tuple[float, float, str, list[int]]]:
    """Greedily pack consecutive Whisper segments (sentence-aligned by construction) into
    windows of ~target_s, hard-capped at max_s. Returns (t_start, t_end, text, source_ids).
    A single over-long segment becomes its own window (never split mid-segment — the word
    anchors inside it stay valid)."""
    windows: list[tuple[float, float, str, list[int]]] = []
    cur_texts: list[str] = []
    cur_ids: list[int] = []
    cur_start = cur_end = 0.0
    for seg in speech:
        text = (seg.payload.get("text") or "").strip()
        if not text:
            continue
        if not cur_texts:
            cur_start = seg.t_start_s
        elif (seg.t_end_s - cur_start) > max_s or (cur_end - cur_start) >= target_s:
            windows.append((cur_start, cur_end, " ".join(cur_texts), cur_ids))
            cur_texts, cur_ids = [], []
            cur_start = seg.t_start_s
        cur_texts.append(text)
        cur_ids.append(seg.id or -1)
        cur_end = seg.t_end_s
    if cur_texts:
        windows.append((cur_start, cur_end, " ".join(cur_texts), cur_ids))
    return windows


def run(video_id: str, cfg: Config, conn) -> dict:
    video = storage.get_video(conn, video_id)
    if video is None:
        raise EmbedError(f"no ingested video {video_id}; run ingest first")

    fps = video.fps
    last_frame = video.frame_count - 1

    # --- images: every kept keyframe ---------------------------------------------------
    frames = storage.list_segments(conn, video_id, kind="frame")
    n_images = 0
    siglip_name = None
    if frames:
        sig = _load_siglip()
        siglip_name = sig.name
        # frame_path is workdir-relative (absolute in pre-fix DBs; join is a no-op then)
        paths = [str(cfg.workdir / f.payload["frame_path"]) for f in frames]
        blobs = _embed_images(sig, paths)
        for seg, blob in zip(frames, blobs):
            assert seg.id is not None
            storage.set_embedding(conn, seg.id, blob, sig.name, sig.dim)
        n_images = len(frames)
        del sig  # 16 GB machine: release before loading the text model

    # --- text: transcript windows (+ captions when present) ----------------------------
    speech = storage.list_segments(conn, video_id, kind="speech")
    captions = storage.list_segments(conn, video_id, kind="caption")
    n_windows = n_captions = 0
    text_dim = None
    if speech or captions:
        tm = _load_text_model()
        text_dim = int(tm.get_sentence_embedding_dimension())

        windows = build_windows(speech, cfg.window_target_s, cfg.window_max_s)
        if windows:
            from .transcribe import _frame_end, _frame_start

            span_s = video.frame_count * video.fps_den / video.fps_num
            rows = []
            for t0, t1, text, ids in windows:
                fs = _frame_start(t0, fps, last_frame)
                rows.append(Segment(
                    video_id=video_id, kind="speech_window",
                    t_start_s=t0, t_end_s=t1,
                    frame_start=fs, frame_end=_frame_end(t1, fps, last_frame, fs, span_s),
                    model_name=TEXT_MODEL, dim=text_dim,
                    payload={"text": text, "source_speech_ids": ids},
                ))
            blobs = _embed_texts(tm, [r.payload["text"] for r in rows])
            storage.replace_segments(conn, video_id, ("speech_window",), rows)
            # replace_segments strips embeddings (insert path has no embedding column);
            # set them now against the freshly inserted rows, in insertion order.
            inserted = storage.list_segments(conn, video_id, kind="speech_window")
            for seg, blob in zip(inserted, blobs):
                assert seg.id is not None
                storage.set_embedding(conn, seg.id, blob, TEXT_MODEL, text_dim)
            n_windows = len(rows)

        if captions:
            texts = [(c.payload.get("description") or "").strip() for c in captions]
            keep = [(c, t) for c, t in zip(captions, texts) if t]
            if keep:
                blobs = _embed_texts(tm, [t for _, t in keep])
                for (c, _), blob in zip(keep, blobs):
                    assert c.id is not None
                    storage.set_embedding(conn, c.id, blob, TEXT_MODEL, text_dim)
                n_captions = len(keep)

    return {
        "image_embeddings": n_images,
        "image_model": siglip_name,
        "speech_windows": n_windows,
        "caption_embeddings": n_captions,
        "text_model": TEXT_MODEL if (n_windows or n_captions) else None,
    }
