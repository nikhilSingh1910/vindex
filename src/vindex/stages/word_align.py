"""Stage-4 helper — wav2vec2 CTC forced alignment refines whisper word timestamps.

Whisper's cross-attention word times drift (criterion 4 fired: start>=end word rows on
real media); forced alignment against the audio is the standard fix. Implemented on
torchaudio's official forced_align API — the whisperx package does the same alignment
but would drag pyannote-audio into the venv for a feature we don't use.

Scope guards, by construction:
- English-only (the bundle's character vocabulary); the caller gates on detected language.
- Only word t0/t1 move. Segment bounds, text, probs, and every downstream artifact
  (speech_region padding, silence complement, embed windows) are untouched, and words are
  clamped inside their segment so the criterion-4 "no word overlaps silence" contract
  cannot regress.
- Any per-segment failure (unalignable tokens, too-short audio, CTC infeasibility) keeps
  that segment's whisper words — refinement, never a new failure mode.
"""

from __future__ import annotations

from pathlib import Path

# Word timings shorter than this are clamped up; also the minimum audio slice we try to
# align (CTC needs a few emission frames per token to be meaningful).
_MIN_WORD_S = 0.02
_MIN_SLICE_S = 0.05
# Context around the whisper segment bounds: absorbs segment-level drift without letting
# a word wander into a neighboring utterance.
_PAD_S = 0.25

# Alignment slices longer than this fall back to whisper timing: wav2vec2 memory grows
# ~22 MB/s of audio (measured), and VAD-restored whisper segments can span arbitrarily
# long gaps — an uncapped forward pass is a swap-storm risk on the 16 GB dev box.
_MAX_SLICE_S = 30.0

_ALIGNER = None  # (model, tokenize label->id dict, separator id, sample_rate)


def preflight() -> None:
    """Load (downloading on first use) the alignment model. Call BEFORE the expensive
    whisper pass: a missing extra or a failed download must not cost a full
    transcription (same pattern as the music-gate preflight)."""
    _load_aligner()


def release() -> None:
    """Drop the ~1 GB resident model once the stage is done with it, mirroring the
    codebase's model-hygiene convention (embed's `del sig`, the ollama unload)."""
    global _ALIGNER
    _ALIGNER = None


def _load_aligner():
    global _ALIGNER
    if _ALIGNER is None:
        try:
            import torchaudio
        except ModuleNotFoundError as e:
            raise RuntimeError(
                "word_align=True but torchaudio is not installed; "
                "run `uv sync --extra align` (or set word_align=False in config.py)"
            ) from e
        bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
        model = bundle.get_model().eval()
        all_labels = list(bundle.get_labels())
        sep = all_labels.index("|")
        # Tokenize vocabulary must exclude the CTC blank (index 0 — its placeholder
        # char '-' would map ordinary hyphenated words onto blank, which forced_align
        # rejects) and the word separator (inserted structurally, never from text).
        labels = {ch: i for i, ch in enumerate(all_labels) if i != 0 and ch != "|"}
        _ALIGNER = (model, labels, sep, bundle.sample_rate)
    return _ALIGNER


def _tokenize(word: str, labels: dict[str, int]) -> list[int]:
    """Characters of `word` present in the CTC vocabulary (drops digits/punctuation).
    Empty result means the word is unalignable and keeps its whisper timing."""
    return [labels[ch] for ch in word.strip().upper() if ch in labels]


def _load_wav(wav: Path, expect_sr: int):
    """Read the ingest-produced WAV (pinned 16 kHz mono s16le) without audio-IO deps —
    torchaudio 2.11 delegates load() to the separate torchcodec package we don't need."""
    import wave

    import numpy as np
    import torch

    with wave.open(str(wav), "rb") as f:
        if f.getnchannels() != 1 or f.getsampwidth() != 2 or f.getframerate() != expect_sr:
            raise RuntimeError(
                f"word_align expects the ingest WAV format (mono s16le {expect_sr} Hz), "
                f"got {f.getnchannels()}ch {8 * f.getsampwidth()}bit {f.getframerate()} Hz"
            )
        pcm = np.frombuffer(f.readframes(f.getnframes()), dtype=np.int16)
    return torch.from_numpy(pcm.astype(np.float32) / 32768.0).unsqueeze(0)


def align_speech_words(wav: Path, speech_rows) -> dict:
    """Refine payload['words'] t0/t1 in place for each row; returns summary stats."""
    model, labels, sep, model_sr = _load_aligner()  # curated error before bare imports
    import torch
    import torchaudio

    waveform = _load_wav(wav, model_sr)
    sr = model_sr
    total_s = waveform.shape[1] / sr

    rows_aligned = rows_fallback = words_moved = words_kept = 0
    with torch.inference_mode():
        for row in speech_rows:
            words = row.payload.get("words") or []
            if not words:
                continue
            try:
                if row.t_end_s - row.t_start_s < 2 * _MIN_WORD_S:
                    raise ValueError("segment too short for word bounds")
                s = max(0.0, row.t_start_s - _PAD_S)
                e = min(total_s, row.t_end_s + _PAD_S)
                if e - s < _MIN_SLICE_S:
                    raise ValueError("slice too short")
                if e - s > _MAX_SLICE_S:
                    # VAD-restored segments can span long gated-out gaps; an uncapped
                    # forward pass is a memory hazard, not a quality win.
                    raise ValueError("slice too long")
                chunk = waveform[:, int(s * sr): int(e * sr)]

                # Flat CTC target: word tokens joined by the separator; remember which
                # target indices belong to which word (separators map to no word).
                targets: list[int] = []
                owner: list[int | None] = []
                for wi, w in enumerate(words):
                    toks = _tokenize(w["w"], labels)
                    if toks and targets:
                        targets.append(sep)
                        owner.append(None)
                    targets.extend(toks)
                    owner.extend([wi] * len(toks))
                if not targets:
                    raise ValueError("no alignable tokens")

                emission, _ = model(chunk)
                emission = torch.log_softmax(emission, dim=-1)
                if emission.shape[1] < len(targets) + 2:
                    raise ValueError("audio too short for target length")

                frames, scores = torchaudio.functional.forced_align(
                    emission, torch.tensor([targets]), blank=0)
                spans = torchaudio.functional.merge_tokens(frames[0], scores[0])
                sec_per_frame = (e - s) / emission.shape[1]

                # First/last emission frame per word -> seconds, clamped into the segment.
                bounds: dict[int, list[float]] = {}
                for ti, span in enumerate(spans):
                    wi = owner[ti]
                    if wi is None:
                        continue
                    t0 = s + span.start * sec_per_frame
                    t1 = s + span.end * sec_per_frame
                    if wi in bounds:
                        bounds[wi][1] = t1
                    else:
                        bounds[wi] = [t0, t1]
                for wi, (t0, t1) in bounds.items():
                    t0 = min(max(t0, row.t_start_s), row.t_end_s - _MIN_WORD_S)
                    t1 = min(max(t1, t0 + _MIN_WORD_S), row.t_end_s)
                    words[wi]["t0"], words[wi]["t1"] = round(t0, 3), round(t1, 3)
                    words_moved += 1
                # Unalignable words (digits/symbols) kept whisper timing and may still
                # carry its degenerate start>=end defect inside an otherwise-aligned
                # row; sanitize so every persisted aligned row has sane word spans.
                for w in words:
                    if w["t1"] <= w["t0"]:
                        w["t0"] = min(max(w["t0"], row.t_start_s),
                                      row.t_end_s - _MIN_WORD_S)
                        w["t1"] = round(min(w["t0"] + _MIN_WORD_S, row.t_end_s), 3)
                        w["t0"] = round(w["t0"], 3)
                words_kept += len(words) - len(bounds)
                row.payload["word_align"] = "wav2vec2_asr_base_960h"
                rows_aligned += 1
            except Exception:
                rows_fallback += 1  # whisper words stand, exactly as they were
                words_kept += len(words)

    return {
        "word_align_rows": rows_aligned,
        "word_align_fallback_rows": rows_fallback,
        "word_align_words_moved": words_moved,
        "word_align_words_kept_whisper": words_kept,
    }
