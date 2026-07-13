"""Stage 6 — embed. Three embedding spaces, one column:

- Images: SigLIP (so400m) on every kept keyframe's persisted JPEG -> embedding on the
  kind='frame' row. fp16 with torch_dtype set EXPLICITLY (the HF checkpoint ships F32 —
  PLAN facts #4); MPS on Apple Silicon, CUDA/CPU elsewhere.
- Text: bge-small-en-v1.5 on sentence-aligned transcript windows (~30-60 s, built from
  Whisper segment boundaries, carrying word-timestamp anchors) -> new kind='speech_window'
  rows; and on caption descriptions when stage 5 lands (kind='caption' rows are embedded
  if present, so the stage is caption-ready without modification).
- Audio: CLAP on fixed 10 s windows (the model's native crop; audio has no shot
  boundaries) -> new kind='audio_window' rows. Source audio is a 48 kHz mono WAV
  extracted from the mezzanine on first need (backward-compatible with already-ingested
  videos; the 16 kHz ASR WAV would cap content at 8 kHz — wrong domain for CLAP).

Embeddings are float32 little-endian BLOBs (sqlite-vec scalar-function format). Every
embedded row stores model_name + dim — mixed dimensions share the one column, and search
scopes each KNN by model_name (PLAN Search section).
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

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
AUDIO_MODEL = "laion/clap-htsat-unfused"
_CLAP_SR = 48000  # CLAP's expected input rate


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


@dataclass
class _Clap:
    model: object
    processor: object
    name: str
    device: str
    dim: int


def _load_clap() -> _Clap:
    from transformers import AutoProcessor, ClapModel

    device = _pick_device()
    model = ClapModel.from_pretrained(AUDIO_MODEL).to(device).eval()
    processor = AutoProcessor.from_pretrained(AUDIO_MODEL)
    return _Clap(model=model, processor=processor, name=AUDIO_MODEL, device=device,
                 dim=int(model.config.projection_dim))


def audio_windows(duration_s: float, win_s: float, min_s: float) -> list[tuple[float, float]]:
    """Fixed tiling of [0, duration): full windows of win_s; a trailing remainder >= min_s
    becomes its own window, else it merges into the previous one (or stands alone when the
    whole clip is shorter than min_s — better one short window than none)."""
    if duration_s <= 0:
        return []
    out: list[tuple[float, float]] = []
    t = 0.0
    while t + win_s <= duration_s:
        out.append((t, t + win_s))
        t += win_s
    if duration_s - t > 1e-9:
        if duration_s - t >= min_s or not out:
            out.append((t, duration_s))
        else:
            out[-1] = (out[-1][0], duration_s)
    return out


def _ensure_audio48k(cfg: Config, video_id: str, mezz: Path) -> Path | None:
    """Extract (once) the 48 kHz mono WAV the audio space embeds. Returns None when the
    mezzanine has no audio stream — an audio-less video simply has no audio space.
    Writes temp + os.replace: a killed ffmpeg must never leave a truncated WAV that a
    later run would silently trust as the real audio (reviewed: cached-partial hazard)."""
    import os

    from .. import ffmpeg_utils as ff

    out = cfg.video_dir(video_id) / "audio_48k_mono.wav"
    if out.exists():
        return out
    if ff.probe(mezz).first("audio") is None:
        return None
    tmp = out.with_suffix(".wav.part")
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mezz), "-vn",
             "-af", "aresample=async=1:first_pts=0",  # same clock re-pin as the ASR WAV
             "-ac", "1", "-ar", str(_CLAP_SR), "-c:a", "pcm_s16le",
             # -f is mandatory: ffmpeg infers the muxer from the extension, and the
             # atomic-write temp name (.wav.part) has no recognized one (fired live).
             "-f", "wav", str(tmp)],
            check=True, capture_output=True,
        )
    except subprocess.CalledProcessError as e:
        tmp.unlink(missing_ok=True)
        raise EmbedError(
            f"48 kHz extraction failed for {video_id}: "
            f"{(e.stderr or b'').decode(errors='replace')[-500:]}") from e
    os.replace(tmp, out)
    return out


def _embed_audio(clap: _Clap, waveform, windows: list[tuple[float, float]],
                 win_s: float, batch_size: int = 8) -> list[bytes]:
    import numpy as np
    import torch

    # One window's worth of samples. Merged tail windows (win_s, win_s+min_s] are capped
    # to this: CLAP's rand_trunc path would otherwise embed an UNSEEDED random crop
    # (nondeterministic re-embeds). A capped tail embeds its first win_s seconds —
    # deterministic, and the row's t-range still reflects the window it stands for.
    max_samples = int(win_s * _CLAP_SR)
    out: list[bytes] = []
    for i in range(0, len(windows), batch_size):
        chunks = [waveform[int(t0 * _CLAP_SR): min(int(t1 * _CLAP_SR),
                                                   int(t0 * _CLAP_SR) + max_samples)]
                  for t0, t1 in windows[i: i + batch_size]]
        # No padding kwarg: `padding=True` silently overrides the checkpoint's
        # configured 'repeatpad' with zero-padding (verified against transformers
        # 5.13.1), mis-embedding every sub-10 s tail window. Each chunk is padded to
        # the model's fixed 10 s input independently — there is no pad-to-longest.
        inputs = clap.processor(
            audio=chunks, sampling_rate=_CLAP_SR, return_tensors="pt",
        ).to(clap.device)
        with torch.no_grad():
            feats = clap.model.get_audio_features(**inputs)
        # transformers 5.x wraps projected features like SigLIP 2 does (PLAN facts #31).
        if not torch.is_tensor(feats):
            feats = feats.pooler_output
        assert feats.shape[-1] == clap.dim, \
            f"CLAP audio features dim {feats.shape[-1]} != projection_dim {clap.dim}"
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

    # --- audio: CLAP over fixed windows of the 48 kHz mono extraction ------------------
    from .transcribe import _frame_end, _frame_start

    span_s = video.frame_count * video.fps_den / video.fps_num
    n_audio = 0
    wav48 = _ensure_audio48k(cfg, video_id, Path(video.media_path))
    if wav48 is not None:
        from .word_align import _load_wav  # stdlib WAV reader; validates mono s16le rate

        waveform = _load_wav(wav48, _CLAP_SR)[0].numpy()  # 1-D float32
        # Tile the timeline both streams cover: cut math is bound to the video clock,
        # and slices past the audio's end would embed nothing.
        tile_span = min(span_s, waveform.shape[0] / _CLAP_SR)
        windows = audio_windows(tile_span, cfg.audio_window_s, cfg.audio_window_min_s)
        rows = []
        blobs: list[bytes] = []
        if windows:
            clap = _load_clap()
            blobs = _embed_audio(clap, waveform, windows, cfg.audio_window_s)
            for t0, t1 in windows:
                fs = _frame_start(t0, fps, last_frame)
                rows.append(Segment(
                    video_id=video_id, kind="audio_window",
                    t_start_s=t0, t_end_s=t1,
                    frame_start=fs, frame_end=_frame_end(t1, fps, last_frame, fs, span_s),
                    model_name=clap.name, dim=clap.dim,
                    payload={"win_s": cfg.audio_window_s},
                ))
        # Unconditional replace: a zero-window outcome (e.g. an empty WAV) must clear
        # any prior rows rather than leave them as stale truth (reviewed).
        storage.replace_segments(conn, video_id, ("audio_window",), rows)
        if rows:
            inserted = storage.list_segments(conn, video_id, kind="audio_window")
            for seg, blob in zip(inserted, blobs):
                assert seg.id is not None
                storage.set_embedding(conn, seg.id, blob, clap.name, clap.dim)
            n_audio = len(rows)
            del clap  # release before SigLIP loads
        del waveform
    else:
        # No audio stream: an existing audio space would be stale truth on re-run.
        storage.replace_segments(conn, video_id, ("audio_window",), [])

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
        "audio_windows": n_audio,
        "audio_model": AUDIO_MODEL if n_audio else None,
        "speech_windows": n_windows,
        "caption_embeddings": n_captions,
        "text_model": TEXT_MODEL if (n_windows or n_captions) else None,
    }
