"""Stage 1 — ingest. Produces the canonical SDR CFR mezzanine (the edit master every frame
number references), a clock-aligned 16 kHz mono WAV for ASR, and the videos-table row.

Fails loudly rather than silently corrupting the index. Raises on: sources needing color
normalization (PQ/HLG HDR, BT.2020 wide gamut, or metadata-stripped 10-bit) — since no
tonemap chain is wired into the encode yet, these are refused regardless of ffmpeg's
capabilities; frame rates that don't snap to a whitelisted rational within tolerance; and
post-encode A/V container start_time skew beyond ~one AAC frame. The WAV is re-pinned to the
video clock via aresample=first_pts=0, but that correction is assumed from filter semantics,
not runtime-verified — hence the skew ceiling.
"""

from __future__ import annotations

import hashlib
import shutil
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

from .. import ffmpeg_utils as ff
from ..config import Config
from ..models import Video

STAGE = "ingest"


class IngestError(RuntimeError):
    pass


@dataclass
class IngestOutput:
    video: Video
    wav_path: Path | None  # None if the source had no audio stream


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _tonemap_available() -> bool:
    """The plan's zscale-based HDR->SDR chain needs libzimg; check the running ffmpeg."""
    try:
        out = subprocess.run(
            ["ffmpeg", "-hide_banner", "-filters"], capture_output=True, text=True
        ).stdout
    except FileNotFoundError:
        return False
    return " zscale " in out


def _source_bit_depth(pix_fmt: str | None) -> int:
    pf = (pix_fmt or "").lower()
    if "12le" in pf or "12be" in pf:
        return 12
    if "10le" in pf or "10be" in pf:
        return 10
    return 8


def _needs_color_normalization(transfer: str | None, primaries: str | None, pix_fmt: str | None) -> bool:
    """True if a naive libx264 yuv420p (bt709, 8-bit) encode would distort this source.
    Broader than a two-string HDR check: catches BT.2020 wide-gamut and metadata-stripped
    10-bit sources that would be crushed by an assume-bt709 downconvert."""
    t = (transfer or "").lower()
    p = (primaries or "").lower()
    if t in {"smpte2084", "arib-std-b67"}:  # PQ / HLG => HDR
        return True
    if p in {"bt2020", "bt2020nc", "bt2020c"}:  # wide gamut, clipped by bt709
        return True
    if t in {"", "unknown"} and _source_bit_depth(pix_fmt) >= 10:  # stripped metadata, 10-bit+
        return True
    return False


def _is_url(source: str) -> bool:
    return source.startswith(("http://", "https://", "www."))


def _download(source: str, staging_root: Path) -> Path:
    # Stage PER SOURCE (keyed by URL hash): a shared staging file made yt-dlp skip the
    # download when a DIFFERENT video's staged file was present, silently ingesting the
    # wrong video under this video_id (observed live on the round-1 corpus run). Within
    # one source's own dir, an existing complete file is a valid re-run cache.
    dest_dir = staging_root / hashlib.sha256(source.encode()).hexdigest()[:16]
    dest_dir.mkdir(parents=True, exist_ok=True)
    out_tmpl = str(dest_dir / "source.%(ext)s")
    # Best <=1080p video + best audio, preferring h264 (SDR, cv2-friendly, and the
    # mezzanine re-encode normalizes anyway) — an uncapped bv* can pull 4K AV1/HDR
    # masters the 16 GB dev box has no business decoding. Falls back progressively.
    fmt = ("bv*[height<=1080][vcodec^=avc1]+ba/"
           "bv*[height<=1080]+ba/b[height<=1080]/b")
    cmd = ["yt-dlp", "-f", fmt, "--merge-output-format", "mp4",
           "--retries", "10", "--fragment-retries", "10", "--retry-sleep", "5",
           "-o", out_tmpl, source]
    # YouTube 403s mid-stream when a media URL expires or throttles (observed live at 8.5%
    # of a re-download); a fresh invocation re-extracts fresh URLs and yt-dlp resumes from
    # the .part file, so a full-command retry is both cheap and effective.
    for attempt in (1, 2, 3):
        try:
            subprocess.run(cmd, check=True)
            break
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            print(f"[ingest] download attempt {attempt} failed; re-extracting and resuming")
    candidates = list(dest_dir.glob("source.*"))
    if not candidates:
        raise IngestError(f"yt-dlp produced no file for {source}")
    return candidates[0]


def run(source: str, cfg: Config, conn, video_id: str | None = None) -> IngestOutput:
    # --- resolve source to a local file ---------------------------------------------
    staging = cfg.workdir / "_staging"
    if _is_url(source):
        src_path = _download(source, staging)
        source_url: str | None = source
    else:
        src_path = Path(source).expanduser().resolve()
        if not src_path.exists():
            raise IngestError(f"source file not found: {src_path}")
        source_url = None

    source_hash = _sha256(src_path)
    vid = video_id or source_hash[:12]
    vdir = cfg.video_dir(vid)
    vdir.mkdir(parents=True, exist_ok=True)

    # --- probe -----------------------------------------------------------------------
    p = ff.probe(src_path)
    v = p.first("video")
    a = p.first("audio")
    if v is None:
        raise IngestError(f"no video stream in {src_path}")

    # --- color guard (fail loudly; see facts register #14) ---------------------------
    # No HDR->SDR tonemap chain is wired into the encode yet, so ANY source needing color
    # normalization is refused regardless of whether this ffmpeg could tonemap — proceeding
    # would silently emit washed-out / clipped frames. The zscale-absent case gets a more
    # actionable message (install ffmpeg-full).
    transfer = v.color_transfer
    src_bit_depth = _source_bit_depth(v.pix_fmt)
    if _needs_color_normalization(transfer, v.color_primaries, v.pix_fmt):
        hint = (
            "this ffmpeg lacks zscale/libzimg; install ffmpeg-full or supply an SDR source"
            if not _tonemap_available()
            else "HDR->SDR tonemapping is not yet implemented in the mezzanine encode"
        )
        raise IngestError(
            f"source needs color normalization (transfer={transfer}, "
            f"primaries={v.color_primaries}, pix_fmt={v.pix_fmt}): {hint}. "
            f"Refusing to encode washed-out frames."
        )
    color_normalized = False  # no normalization applied; SDR bt709 passes through as-is

    # --- snap frame rate -------------------------------------------------------------
    probed_rate = ff.parse_rate(v.avg_frame_rate)
    if probed_rate is None:
        raise IngestError(f"could not determine source frame rate (avg_frame_rate={v.avg_frame_rate})")
    fps_num, fps_den = ff.snap_fps(probed_rate, cfg.fps_whitelist, cfg.fps_tolerance)

    # --- disk preflight (fail in seconds, not 20 minutes into a 1-thread encode) ------
    # Peak transient need: the mezzanine (~source size at CRF 16) plus its +faststart
    # rewrite copy. Observed live: a 99%-full disk killed the encode at finalize.
    src_bytes = src_path.stat().st_size
    free_bytes = shutil.disk_usage(vdir).free
    need_bytes = 2 * src_bytes + 1_000_000_000
    if free_bytes < need_bytes:
        raise IngestError(
            f"insufficient disk space for mezzanine encode: {free_bytes / 1e9:.1f} GB free, "
            f"need ~{need_bytes / 1e9:.1f} GB (2x source size + 1 GB headroom). Free space "
            f"and re-run; the stage is resumable."
        )

    # --- encode mezzanine ------------------------------------------------------------
    mezz = vdir / "master.mp4"
    cmd = ["ffmpeg", "-y", "-i", str(src_path), "-map", "0:v:0"]
    if a is not None:
        cmd += ["-map", "0:a:0"]
    cmd += [
        "-fps_mode", "cfr", "-r", f"{fps_num}/{fps_den}",
        "-c:v", "libx264", "-crf", str(cfg.crf), "-preset", cfg.preset,
        "-pix_fmt", "yuv420p", "-threads", str(cfg.encode_threads),
    ]
    if a is not None:
        cmd += ["-c:a", "aac", "-b:a", cfg.audio_bitrate]
    cmd += ["-movflags", "+faststart", str(mezz)]
    subprocess.run(cmd, check=True)

    # --- probe mezzanine + clock-skew check ------------------------------------------
    pm = ff.probe(mezz)
    mv = pm.first("video")
    ma = pm.first("audio")
    assert mv is not None
    v_start = mv.start_time or 0.0
    a_start = (ma.start_time if ma else 0.0) or 0.0
    # Raw measured container skew, kept as provenance. Not applied to timestamps: the WAV
    # extraction below re-pins audio to the video clock, and any skew beyond the ceiling
    # raises rather than being trusted.
    audio_offset_s = a_start - v_start

    frame_count = ff.count_frames(mezz)
    # Store the VIDEO-timeline duration (frame-exact and reproducible from fps+frame_count),
    # not the container duration, which includes any AAC audio tail past the last frame.
    duration_s = frame_count * fps_den / fps_num

    # --- extract clock-aligned ASR WAV ----------------------------------------------
    # aresample async=1:first_pts=0 is intended to pin WAV t=0 to media t=0 regardless of
    # container start_time/edit-list offsets. This alignment is assumed from filter
    # semantics (and demonstrated in review round 2), not runtime-verified per video — the
    # skew ceiling above is what bounds our trust in it. transcribe uses the WAV directly
    # and does NOT re-apply audio_offset_s (that would double-correct).
    wav_path: Path | None = None
    if a is not None:
        wav_path = vdir / "audio_16k_mono.wav"
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(mezz), "-vn",
             "-af", "aresample=async=1:first_pts=0",
             "-ac", "1", "-ar", str(cfg.asr_sample_rate),
             "-c:a", "pcm_s16le", str(wav_path)],
            check=True,
        )

    if abs(audio_offset_s) > cfg.max_start_time_skew_s:
        # Fail loudly (PLAN option A). The aresample=first_pts=0 extraction re-pins the WAV
        # to the video clock for container start_time offsets, but that correction is NOT
        # runtime-verified here, so we refuse to commit a possibly-skewed transcript as
        # authoritative. (We deliberately do NOT persist+apply audio_offset_s to timestamps:
        # for container-offset skew the WAV is already re-pinned, so applying it would
        # double-correct.)
        raise IngestError(
            f"mezzanine A/V start_time skew {audio_offset_s:+.3f}s exceeds "
            f"{cfg.max_start_time_skew_s}s (~one AAC frame): v_start={v_start:.3f} "
            f"a_start={a_start:.3f}. Re-mux the source to zero container start_time, or add "
            f"runtime external-clock verification (criterion 4) before committing the "
            f"transcript."
        )

    media_hash = _sha256(mezz)

    video = Video(
        video_id=vid,
        source_url=source_url,
        source_hash=source_hash,
        media_path=str(mezz),
        media_hash=media_hash,
        fps_num=fps_num, fps_den=fps_den,
        frame_count=frame_count, duration_s=float(duration_s),
        width=mv.width or 0, height=mv.height or 0,
        codec=mv.codec_name or "h264",
        audio_offset_s=audio_offset_s,
        source_color_transfer=transfer,
        source_color_primaries=v.color_primaries,
        source_pix_fmt=v.pix_fmt,
        source_bit_depth=src_bit_depth,
        color_normalized=color_normalized,
        encode_threads=cfg.encode_threads,
    )
    from .. import storage
    storage.upsert_video(conn, video)

    # Clean the staging download once the mezzanine (our real identity) exists.
    if _is_url(source) and staging.exists():
        shutil.rmtree(staging, ignore_errors=True)

    return IngestOutput(video=video, wav_path=wav_path)


def wav_path_for(cfg: Config, video_id: str) -> Path:
    return cfg.video_dir(video_id) / "audio_16k_mono.wav"
