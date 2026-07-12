"""Thin, well-tested wrappers over ffprobe/ffmpeg. All timing- and identity-critical
behavior of stage 1 lives here so it can be unit-tested in isolation.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path


class FfmpegError(RuntimeError):
    pass


def _run(cmd: list[str]) -> str:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise FfmpegError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stderr[-2000:]}")
    return proc.stdout


@dataclass
class StreamInfo:
    codec_type: str
    codec_name: str | None
    start_time: float | None
    # video-only fields
    width: int | None = None
    height: int | None = None
    avg_frame_rate: str | None = None
    color_transfer: str | None = None
    color_primaries: str | None = None
    pix_fmt: str | None = None


@dataclass
class ProbeResult:
    duration_s: float | None
    streams: list[StreamInfo]

    def first(self, codec_type: str) -> StreamInfo | None:
        for s in self.streams:
            if s.codec_type == codec_type:
                return s
        return None


def probe(path: str | Path) -> ProbeResult:
    out = _run([
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", str(path),
    ])
    data = json.loads(out)
    streams: list[StreamInfo] = []
    for s in data.get("streams", []):
        streams.append(StreamInfo(
            codec_type=s.get("codec_type", ""),
            codec_name=s.get("codec_name"),
            start_time=_to_float(s.get("start_time")),
            width=s.get("width"),
            height=s.get("height"),
            avg_frame_rate=s.get("avg_frame_rate"),
            color_transfer=s.get("color_transfer"),
            color_primaries=s.get("color_primaries"),
            pix_fmt=s.get("pix_fmt"),
        ))
    duration = _to_float(data.get("format", {}).get("duration"))
    return ProbeResult(duration_s=duration, streams=streams)


def count_frames(path: str | Path, stream: str = "v:0") -> int:
    """Exact frame count by decoding. Slow but deterministic — used on our own mezzanine,
    where an exact count is required for frame-accurate cut math."""
    out = _run([
        "ffprobe", "-v", "error", "-select_streams", stream,
        "-count_frames", "-show_entries", "stream=nb_read_frames",
        "-print_format", "json", str(path),
    ])
    data = json.loads(out)
    streams = data.get("streams", [])
    if not streams or streams[0].get("nb_read_frames") in (None, "N/A"):
        raise FfmpegError(f"could not count frames for {path}")
    return int(streams[0]["nb_read_frames"])


def _to_float(v: object) -> float | None:
    if v is None or v == "N/A":
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_rate(avg_frame_rate: str | None) -> Fraction | None:
    """Parse an ffprobe 'num/den' rate string into a Fraction. Returns None if unusable
    (e.g. '0/0', which ffprobe emits when it can't determine a rate)."""
    if not avg_frame_rate or "/" not in avg_frame_rate:
        return None
    num_s, den_s = avg_frame_rate.split("/", 1)
    try:
        num, den = int(num_s), int(den_s)
    except ValueError:
        return None
    if den == 0 or num == 0:
        return None
    return Fraction(num, den)


def snap_fps(
    probed: Fraction,
    whitelist: tuple[tuple[int, int], ...],
    tolerance: float,
) -> tuple[int, int]:
    """Snap a probed rate to the nearest whitelisted rational within `tolerance` (relative).
    Raises FfmpegError naming the probed value if nothing is close enough."""
    probed_f = float(probed)
    best: tuple[float, tuple[int, int]] | None = None
    for num, den in whitelist:
        cand = num / den
        rel = abs(cand - probed_f) / probed_f
        if best is None or rel < best[0]:
            best = (rel, (num, den))
    assert best is not None
    rel, (num, den) = best
    if rel > tolerance:
        raise FfmpegError(
            f"probed frame rate {probed_f:.5f} ({probed}) is not within {tolerance:.1%} "
            f"of any whitelisted rate; nearest is {num}/{den} ({num / den:.5f}, "
            f"rel diff {rel:.3%}). Refusing to guess a timebase."
        )
    return num, den
