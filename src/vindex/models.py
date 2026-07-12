"""Typed Pydantic contracts. These are the wire contract an editing agent consumes —
search() and list_segments() both return Segment models, so the future FastAPI layer
serializes exactly these.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

# The open set of segment kinds. New kinds are just new string values; this tuple documents
# the ones round 1 produces and is used to validate CLI --kind filters.
SEGMENT_KINDS: tuple[str, ...] = (
    "shot",
    "frame",
    "speech",
    "speech_window",
    "speech_region",
    "silence",
    "music",
    "noise",
    "lyrics",
    "caption",
    "audio_window",
)


class Video(BaseModel):
    """A row of the videos table: the media-identity anchor. All frame numbers in Segment
    reference the file at media_path (verified by media_hash) and its rational fps."""

    video_id: str
    source_url: str | None = None
    source_hash: str | None = None
    media_path: str
    media_hash: str
    fps_num: int
    fps_den: int
    frame_count: int
    duration_s: float
    width: int
    height: int
    codec: str
    audio_offset_s: float = 0.0
    source_color_transfer: str | None = None
    source_color_primaries: str | None = None
    source_pix_fmt: str | None = None
    source_bit_depth: int = 8
    color_normalized: bool = False
    encode_threads: int = 1

    @property
    def fps(self) -> float:
        return self.fps_num / self.fps_den


class Segment(BaseModel):
    """One timestamp-anchored, cut-actionable index element. frame_start/frame_end are
    valid only against the owning Video's media_path + fps."""

    id: int | None = None
    video_id: str
    kind: str
    t_start_s: float
    t_end_s: float
    frame_start: int | None = None
    frame_end: int | None = None
    model_name: str | None = None
    dim: int | None = None
    payload: dict[str, Any] = Field(default_factory=dict)
    # embedding is stored as a raw BLOB in SQLite; excluded from the typed contract that
    # agents read (they get vectors via search ranking, not raw bytes).
