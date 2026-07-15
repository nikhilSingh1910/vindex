"""Central configuration: all paths, model names, thresholds, and encode parameters.

Everything the pipeline tunes lives here so no stage hard-codes a path or a magic number,
and so the Mac->Linux move is a matter of swapping a config, not editing stage code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# Rational frame-rate whitelist (num, den). Probed source rates are snapped to the nearest
# of these so the CFR mezzanine has a deterministic, cross-machine-reproducible timebase.
# Includes legacy/low rates (12/15 fps) — early YouTube uploads really use them ("Me at the
# zoo" is 15 fps; discovered by the fail-loud rejection working as designed).
FPS_WHITELIST: tuple[tuple[int, int], ...] = (
    (12, 1),
    (15, 1),
    (24000, 1001),  # 23.976
    (24, 1),
    (25, 1),
    (30000, 1001),  # 29.97
    (30, 1),
    (48, 1),
    (50, 1),
    (60000, 1001),  # 59.94
    (60, 1),
)

# ffprobe color_transfer values that indicate HDR and therefore require tonemapping.
HDR_TRANSFERS: frozenset[str] = frozenset({"smpte2084", "arib-std-b67"})


@dataclass
class Config:
    """Resolved runtime configuration. Construct via Config.for_workdir()."""

    workdir: Path
    db_path: Path
    media_dir: Path

    # --- ingest / mezzanine encode ---
    fps_tolerance: float = 0.002  # 0.2% relative; else hard-fail
    crf: int = 16
    preset: str = "slow"
    # Fixed N, recorded on the videos row (PLAN sanctions any fixed N): same N -> same
    # bytes, verified by execution (t=4 twice and t=8 three times hash-identical).
    # Measured on M4 Air, preset slow: t4 = 1.6x over t1, t8 = 2.3x (sublinear).
    encode_threads: int = 8
    audio_bitrate: str = "192k"
    # audio/video start_time must agree within one AAC frame (~21ms @ 48kHz) or we treat
    # it as an offset to persist and correct.
    max_start_time_skew_s: float = 0.021
    asr_sample_rate: int = 16000

    # --- transcribe ---
    asr_model: str = "large-v3"
    asr_compute_type: str = "int8"
    asr_language: str | None = None  # None = autodetect
    vad_filter: bool = True
    # Hallucination guards for the silence/ambience failure mode.
    no_speech_prob_threshold: float = 0.6
    avg_logprob_threshold: float = -1.0
    # Padding applied around persisted speech regions (seconds).
    speech_region_pad_s: float = 0.1
    # Music-authoritative gate (inaSpeechSegmenter): music/noise become first-class
    # segments, silence = complement of (padded speech OR music), lyrics transcribed over
    # music regions. Requires the 'music' extra; transcribe FAILS LOUDLY if the flag is on
    # without it.
    music_gate: bool = True
    # A whisper segment whose fraction-overlapped-by-music is >= this is excluded from
    # kind='speech' and has_speech (it is sung content, covered by the lyrics pass).
    music_overlap_exclude_frac: float = 0.5
    # wav2vec2 CTC forced alignment refines whisper word t0/t1 (criterion-4 fix; English
    # only, per-segment fallback). Requires the 'align' extra; FAILS LOUDLY without it.
    word_align: bool = True

    # --- shots ---
    # Scenes shorter than this merge into their neighbor (phantom micro-shot guard).
    min_shot_frames: int = 8
    # Allowed disagreement between PySceneDetect's decoded frame count and ingest's
    # authoritative ffprobe count before the stage refuses to trust its boundaries.
    max_decode_mismatch_frames: int = 2

    # --- frames ---
    keyframe_interval_s: float = 3.0
    # dHash Hamming distance at or below which a candidate is a near-duplicate.
    dedup_hamming_max: int = 4
    frame_jpeg_quality: int = 90
    person_score_min: float = 0.5
    # Camera-motion classification thresholds (per-frame-pair, normalized by frame size).
    motion_still_px_frac: float = 0.002   # mean displacement below this => static
    motion_zoom_scale_eps: float = 0.005  # |scale-1| above this => zoom
    motion_axis_dominance: float = 2.0    # dx/dy (or dy/dx) ratio above this => pan/tilt
    # Featureless pair (flow can't run): mean abs pixel diff below this => static.
    motion_flat_diff_max: float = 0.5

    # --- embed ---
    window_target_s: float = 45.0  # transcript window packing target (~30-60 s)
    window_max_s: float = 60.0
    # CLAP audio space: fixed tiling at the model's native 10 s training crop. Audio has
    # no shot boundaries; the read contract still refines nothing here (a window IS the
    # cut range) — coarse but honest discovery granularity.
    audio_window_s: float = 10.0
    # Trailing remainder shorter than this merges into the previous window.
    audio_window_min_s: float = 2.0

    # --- caption (Ollama) ---
    ollama_url: str = "http://localhost:11434"
    # Shootout winner (PLAN 2026-07-15, 46-shot cross-content fixture, pre-registered
    # retrieval benchmark): qwen3-vl:2b-instruct beat qwen2.5vl:3b on EVERY axis —
    # 2.0x faster (6.05 vs 12.17 s/caption), better retrieval (MRR .770 vs .689),
    # zero sample hallucinations, smaller (1.9 vs 3.2 GB), and it captions the frame
    # class that deterministically broke 2.5vl. MUST be the -instruct tag: Ollama's
    # bare qwen3-vl tags are THINKING variants that return empty under format=json.
    # (4b-instruct measured: slower than baseline AND worst MRR — verbosity hurts
    # retrieval. florence-2: best MRR/speed, tracked as a future dedicated backend.)
    caption_model: str = "qwen3-vl:2b-instruct"
    caption_num_ctx: int = 4096      # pin: the GGUF declares 125K and Ollama over-allocates
    caption_num_predict: int = 512
    caption_timeout_s: float = 120.0
    caption_max_side_px: int = 1024  # downscale before sending; 2 MP internal cap misses 1080p
    # Circuit breaker: this many consecutive per-shot failures abort the stage (a wedged
    # Ollama otherwise costs 2 x timeout x shots of wall clock before failing).
    # Semantics identical under caption_parallel (results consumed in shot order).
    caption_max_consecutive_failures: int = 5
    # Concurrent caption requests. >1 needs the server started with
    # OLLAMA_NUM_PARALLEL >= this value: without it Ollama QUEUES the extra request,
    # which silently halves the effective per-request timeout (the client budget below
    # is therefore scaled by this value). Do NOT combine >1 with OLLAMA_KEEP_ALIVE=0
    # (reload per request makes queued timeouts systematic). Parity gate: 30-shot A/B,
    # 29/30 byte-identical, 1 quality-equal rewording (PLAN). Measured gain on M4
    # Metal: ~1.1x (compute-saturated); worth more on GPU servers.
    caption_parallel: int = 2
    # Run transcribe (CPU-bound whisper) and caption (Ollama GPU/ANE) concurrently in
    # the ONE pipeline process, each on its own SQLite connection (WAL). The
    # single-process-per-DB contract is unchanged. Default True on the strength of a
    # MEASURED receipt (PLAN 2026-07-14): full overlap window on the 16 GB dev box,
    # min free memory 24%, peak swap 0 MB, watchdog silent; transcribe (+32% under
    # CPU contention) hides entirely under caption. Requires disk headroom for swap
    # growth — on a nearly-full disk prefer sequential (False / no --overlap).
    overlap_transcribe_caption: bool = True

    hdr_transfers: frozenset[str] = field(default_factory=lambda: HDR_TRANSFERS)
    fps_whitelist: tuple[tuple[int, int], ...] = FPS_WHITELIST

    @classmethod
    def for_workdir(cls, workdir: str | Path) -> "Config":
        wd = Path(workdir).expanduser().resolve()
        return cls(
            workdir=wd,
            db_path=wd / "index.db",
            media_dir=wd / "media",
        )

    def video_dir(self, video_id: str) -> Path:
        return self.media_dir / video_id
