"""Regression net for the adversarially-reviewed word_align defects (no model load:
bundle labels are static metadata, and _load_wav is pure stdlib)."""

import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex.stages.word_align import _load_wav, _tokenize  # noqa: E402

import torchaudio  # noqa: E402

_ALL = list(torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H.get_labels())
# The tokenize vocabulary, exactly as _load_aligner builds it.
LABELS = {ch: i for i, ch in enumerate(_ALL) if i != 0 and ch != "|"}


def test_tokenize_never_emits_blank_or_separator():
    # '-' is the blank placeholder at index 0; forced_align rejects blank in targets,
    # so a hyphenated word must tokenize to its letters only (reviewed: major #1).
    toks = _tokenize("twenty-five", LABELS)
    assert toks and 0 not in toks and _ALL.index("|") not in toks


def test_tokenize_unalignable_word_is_empty():
    assert _tokenize("2005", LABELS) == []
    assert _tokenize("€50", LABELS) == []


def test_tokenize_keeps_apostrophe():
    assert len(_tokenize("don't", LABELS)) == 5


def _write_wav(path, sr, n=1600, channels=1, width=2):
    with wave.open(str(path), "wb") as f:
        f.setnchannels(channels)
        f.setsampwidth(width)
        f.setframerate(sr)
        f.writeframes(b"\x00" * n * width * channels)


def test_load_wav_accepts_ingest_format(tmp_path):
    p = tmp_path / "ok.wav"
    _write_wav(p, sr=16000)
    wf = _load_wav(p, expect_sr=16000)
    assert wf.shape == (1, 1600) and float(wf.abs().max()) == 0.0


def test_load_wav_rejects_wrong_rate(tmp_path):
    p = tmp_path / "bad.wav"
    _write_wav(p, sr=8000)
    with pytest.raises(RuntimeError, match="16000"):
        _load_wav(p, expect_sr=16000)
