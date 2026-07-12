"""Stage 4 — transcribe. faster-whisper (large-v3 int8) over the clock-aligned WAV, with the
silence/ambience hallucination guards, plus persistence of the VAD speech map as first-class
speech_region / silence segments (the "where can I cut without clipping speech" primitive).

Music gate (cfg.music_gate, PLAN stage 4 — audio classification is authoritative, VAD is
not): inaSpeechSegmenter runs over the full WAV first; its music/noise intervals persist as
kind='music'/'noise' rows. Whisper segments majority-overlapped by music are excluded from
kind='speech' and has_speech (sung vocals are not spoken content); segments majority-covered
by noise keep a noise_overlap flag (demoted, never dropped). Lyrics do NOT depend on VAD: a
second whisper pass with vad_filter=False runs over music regions only (clip_timestamps) and
persists kind='lyrics' rows. Silence = complement of (padded speech OR music), so vocal or
instrumental music is never offered as a safe cut point. With the gate off, speech payloads
record music_gate_active=False so nothing silently claims a guarantee it does not provide.
"""

from __future__ import annotations

import math
from pathlib import Path

from ..config import Config
from ..models import Segment
from .. import storage
from .ingest import wav_path_for

STAGE = "transcribe"


class TranscribeError(RuntimeError):
    pass


def _load_model(cfg: Config):
    from faster_whisper import WhisperModel

    # CTranslate2 has no MPS backend; CPU int8 is the canonical Apple-Silicon path.
    return WhisperModel(cfg.asr_model, device="cpu", compute_type=cfg.asr_compute_type)


def _passes_guards(seg, cfg: Config) -> bool:
    """Silence/ambience hallucination guards. Note: these do NOT catch sung vocals — that is
    the music-gate's job (next increment)."""
    if getattr(seg, "no_speech_prob", 0.0) > cfg.no_speech_prob_threshold:
        return False
    if getattr(seg, "avg_logprob", 0.0) < cfg.avg_logprob_threshold:
        return False
    if not (seg.text or "").strip():
        return False
    return True


def _segment_music(wav: Path) -> tuple[list[tuple[float, float]], list[tuple[float, float]], dict]:
    """inaSpeechSegmenter over the full WAV. Returns (music_regions, noise_regions,
    provenance). Male/female labels are its speech classes; noEnergy is neither. FAILS
    LOUDLY if the 'music' extra is not installed — a silently skipped gate would let sung
    vocals pose as spoken content."""
    try:
        from inaSpeechSegmenter import Segmenter
    except ModuleNotFoundError as e:
        raise TranscribeError(
            "music_gate=True but inaSpeechSegmenter is not installed; "
            "run `uv sync --extra music` (or set music_gate=False explicitly)"
        ) from e
    import importlib.metadata

    intervals = Segmenter()(str(wav))  # [(label, start_s, stop_s)]
    music = _merge([(s, e) for lab, s, e in intervals if lab == "music"])
    noise = _merge([(s, e) for lab, s, e in intervals if lab == "noise"])
    prov = {
        "segmenter": "inaSpeechSegmenter",
        "segmenter_version": importlib.metadata.version("inaSpeechSegmenter"),
    }
    return music, noise, prov


def _overlap_frac(t0: float, t1: float, regions: list[tuple[float, float]]) -> float:
    """Fraction of [t0, t1] covered by `regions` (merged, sorted)."""
    span = max(t1 - t0, 1e-9)
    covered = sum(max(0.0, min(t1, e) - max(t0, s)) for s, e in regions)
    return min(1.0, covered / span)


def _complement(regions: list[tuple[float, float]], total: float) -> list[tuple[float, float]]:
    """Gaps in [0, total] not covered by `regions` (assumed sorted, non-overlapping)."""
    gaps: list[tuple[float, float]] = []
    cursor = 0.0
    for start, end in regions:
        if start > cursor:
            gaps.append((cursor, start))
        cursor = max(cursor, end)
    if cursor < total:
        gaps.append((cursor, total))
    return gaps


# Frame convention: an interval [t_start_s, t_end_s) maps to the INCLUSIVE frame range
# [frame_start, frame_end], both valid indices in [0, last_frame]. frame_start is the frame
# that CONTAINS t_start (floor); frame_end is the last frame the interval touches
# (ceil(t_end*fps)-1). This matches PLAN's kind='frame' rule (frame_start==frame_end for a
# single instant) and never emits an index past the mezzanine's last frame. The seconds
# columns remain the authoritative anchor; frames are the clamped, cut-actionable view.
#
# _FRAME_EPS absorbs float error at exact frame boundaries: t stored as k*fps_den/fps_num
# can evaluate to k-1e-15 when multiplied back by fps (e.g. floor(0.5005*29.97) = 14.999...
# -> 14 instead of 15). Sub-microframe epsilon; harmless at any real timeline length.
_FRAME_EPS = 1e-6


def _frame_start(t: float, fps: float, last_frame: int) -> int:
    return min(last_frame, max(0, math.floor(t * fps + _FRAME_EPS)))


def _frame_end(t_end: float, fps: float, last_frame: int, frame_start: int, span_s: float) -> int:
    idx = math.ceil(min(t_end, span_s) * fps - _FRAME_EPS) - 1
    return min(last_frame, max(frame_start, idx))


def run(video_id: str, cfg: Config, conn) -> dict:
    video = storage.get_video(conn, video_id)
    if video is None:
        raise TranscribeError(f"no ingested video {video_id}; run ingest first")

    wav = wav_path_for(cfg, video_id)
    if not wav.exists():
        # No audio stream at ingest time -> nothing to transcribe. Record the fact.
        storage.set_has_speech(conn, video_id, False)
        return {"has_speech": False, "speech_segments": 0, "reason": "no audio stream"}

    fps = video.fps
    # Cut math is bound to the VIDEO timeline (frame_count/fps), not the audio-inclusive
    # container duration — an AAC tail running past the last video frame must not push
    # frame indices out of range.
    video_span_s = video.frame_count * video.fps_den / video.fps_num
    last_frame = video.frame_count - 1

    # --- music gate: classify the full WAV BEFORE whisper (authoritative over VAD) -------
    music_regions: list[tuple[float, float]] = []
    noise_regions: list[tuple[float, float]] = []
    gate_prov: dict = {}
    if cfg.music_gate:
        music_regions, noise_regions, gate_prov = _segment_music(wav)

    model = _load_model(cfg)
    segments_iter, info = model.transcribe(
        str(wav),
        language=cfg.asr_language,
        word_timestamps=True,
        vad_filter=cfg.vad_filter,
    )

    speech_rows: list[Segment] = []
    speech_regions: list[tuple[float, float]] = []
    kept = excluded_as_music = 0
    for seg in segments_iter:
        if not _passes_guards(seg, cfg):
            continue
        music_frac = _overlap_frac(float(seg.start), float(seg.end), music_regions)
        if music_frac >= cfg.music_overlap_exclude_frac:
            # Sung content: not spoken speech, not a has_speech contributor. The lyrics
            # pass below owns this region.
            excluded_as_music += 1
            continue
        kept += 1
        noise_frac = _overlap_frac(float(seg.start), float(seg.end), noise_regions)
        words = [
            {"w": w.word, "t0": round(w.start, 3), "t1": round(w.end, 3),
             "prob": round(w.probability, 3)}
            for w in (seg.words or [])
        ]
        fs = _frame_start(seg.start, fps, last_frame)
        speech_rows.append(Segment(
            video_id=video_id, kind="speech",
            t_start_s=float(seg.start), t_end_s=float(seg.end),
            frame_start=fs,
            frame_end=_frame_end(seg.end, fps, last_frame, fs, video_span_s),
            model_name=cfg.asr_model,
            payload={
                "text": seg.text.strip(),
                "words": words,
                "no_speech_prob": round(getattr(seg, "no_speech_prob", 0.0), 4),
                "avg_logprob": round(getattr(seg, "avg_logprob", 0.0), 4),
                "language": info.language,
                "music_gate_active": cfg.music_gate,
                **({"music_overlap_frac": round(music_frac, 3)} if music_frac > 0 else {}),
                # Majority-noise segments are demoted (excluded as cut anchors until
                # validated), never dropped.
                **({"noise_overlap": True} if noise_frac >= 0.5 else {}),
            },
        ))
        speech_regions.append((float(seg.start), float(seg.end)))

    # --- lyrics: whisper over music regions only, VAD OFF (Silero is bidirectionally
    # unreliable on sung vocals — lyrics must not depend on it firing) -------------------
    lyric_rows: list[Segment] = []
    if music_regions:
        clips = [t for s, e in music_regions for t in (s, e)]
        lyric_iter, _ = model.transcribe(
            str(wav),
            language=cfg.asr_language,
            word_timestamps=True,
            vad_filter=False,
            clip_timestamps=clips,
        )
        for seg in lyric_iter:
            text = (seg.text or "").strip()
            if not text:
                continue
            fs = _frame_start(seg.start, fps, last_frame)
            lyric_rows.append(Segment(
                video_id=video_id, kind="lyrics",
                t_start_s=float(seg.start), t_end_s=float(seg.end),
                frame_start=fs,
                frame_end=_frame_end(seg.end, fps, last_frame, fs, video_span_s),
                model_name=cfg.asr_model,
                payload={
                    "text": text,
                    "music_overlap": round(
                        _overlap_frac(float(seg.start), float(seg.end), music_regions), 3),
                    "advisory": True,  # unvalidated lyric WER; never spoken content
                    **gate_prov,
                },
            ))

    # Persist the VAD-derived speech map as first-class segments. Pad each merged speech
    # interval outward, then RE-MERGE: padding can make adjacent utterances overlap, and a
    # sub-2*pad gap between two utterances must not be offered as a safe cut. Silence is the
    # complement of these PADDED intervals, so no frame is ever labeled both 'speech_region'
    # (protected) and 'silence' (safe-to-cut) — honoring PLAN criterion 4.
    pad = cfg.speech_region_pad_s
    merged = _merge(speech_regions)
    padded = _merge([
        (max(0.0, start - pad), min(video_span_s, end + pad)) for start, end in merged
    ])
    region_rows: list[Segment] = []
    for s, e in padded:
        fs = _frame_start(s, fps, last_frame)
        region_rows.append(Segment(
            video_id=video_id, kind="speech_region",
            t_start_s=s, t_end_s=e,
            frame_start=fs, frame_end=_frame_end(e, fps, last_frame, fs, video_span_s),
            payload={"pad_s": pad, "source": "faster_whisper.vad_filter"},
        ))
    # --- music/noise as first-class segments (music is a protected region: silence must
    # never overlap it, so instrumental beds are never offered as safe cut points) --------
    music_noise_rows: list[Segment] = []
    for kind, regions in (("music", music_regions), ("noise", noise_regions)):
        for s, e in regions:
            s, e = max(0.0, s), min(video_span_s, e)
            if e <= s:
                continue
            fs = _frame_start(s, fps, last_frame)
            music_noise_rows.append(Segment(
                video_id=video_id, kind=kind,
                t_start_s=s, t_end_s=e,
                frame_start=fs, frame_end=_frame_end(e, fps, last_frame, fs, video_span_s),
                payload={**gate_prov,
                         **({"advisory": True} if kind == "noise" else {})},
            ))

    protected = _merge(padded + [(max(0.0, s), min(video_span_s, e))
                                 for s, e in music_regions])
    silence_rows: list[Segment] = []
    for start, end in _complement(protected, video_span_s):
        fs = _frame_start(start, fps, last_frame)
        silence_rows.append(Segment(
            video_id=video_id, kind="silence",
            t_start_s=start, t_end_s=end,
            frame_start=fs, frame_end=_frame_end(end, fps, last_frame, fs, video_span_s),
            payload={"note": ("complement of padded speech OR music" if cfg.music_gate else
                              "complement of padded speech only; music gate OFF")},
        ))

    # Atomic replace: DELETE of prior output + INSERT in one transaction, so a crash
    # mid-stage never leaves the index missing its previous good transcript. music/noise/
    # lyrics are included even when the gate is off, so a prior gated run's rows never
    # survive an ungated re-run as stale truth.
    storage.replace_segments(
        conn, video_id, ("speech", "speech_region", "silence", "music", "noise", "lyrics"),
        speech_rows + region_rows + silence_rows + music_noise_rows + lyric_rows,
    )

    has_speech = kept > 0
    storage.set_has_speech(conn, video_id, has_speech)

    return {
        "has_speech": has_speech,
        "speech_segments": kept,
        "speech_regions": len(region_rows),
        "silence_regions": len(silence_rows),
        "music_gate_active": cfg.music_gate,
        "music_regions": len(music_regions),
        "noise_regions": len(noise_regions),
        "lyrics_segments": len(lyric_rows),
        "excluded_as_music": excluded_as_music,
        "language": info.language,
    }


def _merge(regions: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Sort and merge overlapping/adjacent intervals."""
    if not regions:
        return []
    ordered = sorted(regions)
    out = [list(ordered[0])]
    for start, end in ordered[1:]:
        if start <= out[-1][1]:
            out[-1][1] = max(out[-1][1], end)
        else:
            out.append([start, end])
    return [(a, b) for a, b in out]
