"""All SQL lives here, behind typed functions. Boring, Postgres-portable SQL so the eventual
SQLite->Postgres move touches only this module. sqlite-vec is loaded lazily (only search/embed
need it); stages 1 and 4 use plain columns.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Iterable

from .models import Segment, Video

SCHEMA = """
CREATE TABLE IF NOT EXISTS videos (
    video_id               TEXT PRIMARY KEY,
    source_url             TEXT,
    source_hash            TEXT,
    media_path             TEXT NOT NULL,
    media_hash             TEXT NOT NULL,
    fps_num                INTEGER NOT NULL,
    fps_den                INTEGER NOT NULL,
    frame_count            INTEGER NOT NULL,
    duration_s             REAL NOT NULL,
    width                  INTEGER NOT NULL,
    height                 INTEGER NOT NULL,
    codec                  TEXT NOT NULL,
    audio_offset_s         REAL NOT NULL DEFAULT 0,
    source_color_transfer  TEXT,
    source_color_primaries TEXT,
    source_pix_fmt         TEXT,
    source_bit_depth       INTEGER NOT NULL DEFAULT 8,
    color_normalized       INTEGER NOT NULL DEFAULT 0,
    encode_threads         INTEGER NOT NULL DEFAULT 1,
    -- set by transcribe; NULL until that stage runs. Kept off the Video contract for now.
    has_speech             INTEGER
);

CREATE TABLE IF NOT EXISTS segments (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    video_id      TEXT NOT NULL REFERENCES videos(video_id),
    kind          TEXT NOT NULL,
    t_start_s     REAL NOT NULL,
    t_end_s       REAL NOT NULL,
    frame_start   INTEGER,
    frame_end     INTEGER,
    model_name    TEXT,
    dim           INTEGER,
    -- reserved for deferred stages; present from day 1 so adding them is a backfill:
    speaker               TEXT,
    alignment_confidence  REAL,
    payload       TEXT NOT NULL DEFAULT '{}',
    embedding     BLOB
);

CREATE INDEX IF NOT EXISTS idx_segments_video_kind ON segments(video_id, kind);
CREATE INDEX IF NOT EXISTS idx_segments_time ON segments(video_id, t_start_s);

CREATE TABLE IF NOT EXISTS jobs (
    video_id   TEXT NOT NULL,
    stage      TEXT NOT NULL,
    status     TEXT NOT NULL DEFAULT 'pending',
    error      TEXT,
    updated_at TEXT NOT NULL DEFAULT (datetime('now')),
    PRIMARY KEY (video_id, stage)
);
"""


def connect(db_path: str | Path, load_vec: bool = False) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    if load_vec:
        import sqlite_vec  # imported lazily; only search/embed need it

        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
    return conn


def init_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(SCHEMA)
    conn.commit()


# --- videos ---------------------------------------------------------------------------

def upsert_video(conn: sqlite3.Connection, v: Video) -> None:
    conn.execute(
        """
        INSERT INTO videos (video_id, source_url, source_hash, media_path, media_hash,
            fps_num, fps_den, frame_count, duration_s, width, height, codec,
            audio_offset_s, source_color_transfer, source_color_primaries,
            source_pix_fmt, source_bit_depth, color_normalized, encode_threads)
        VALUES (:video_id, :source_url, :source_hash, :media_path, :media_hash,
            :fps_num, :fps_den, :frame_count, :duration_s, :width, :height, :codec,
            :audio_offset_s, :source_color_transfer, :source_color_primaries,
            :source_pix_fmt, :source_bit_depth, :color_normalized, :encode_threads)
        ON CONFLICT(video_id) DO UPDATE SET
            source_url=excluded.source_url, source_hash=excluded.source_hash,
            media_path=excluded.media_path, media_hash=excluded.media_hash,
            fps_num=excluded.fps_num, fps_den=excluded.fps_den,
            frame_count=excluded.frame_count, duration_s=excluded.duration_s,
            width=excluded.width, height=excluded.height, codec=excluded.codec,
            audio_offset_s=excluded.audio_offset_s,
            source_color_transfer=excluded.source_color_transfer,
            source_color_primaries=excluded.source_color_primaries,
            source_pix_fmt=excluded.source_pix_fmt,
            source_bit_depth=excluded.source_bit_depth,
            color_normalized=excluded.color_normalized,
            encode_threads=excluded.encode_threads
        """,
        {**v.model_dump(), "color_normalized": int(v.color_normalized)},
    )
    conn.commit()


def get_video(conn: sqlite3.Connection, video_id: str) -> Video | None:
    row = conn.execute("SELECT * FROM videos WHERE video_id = ?", (video_id,)).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["color_normalized"] = bool(d["color_normalized"])
    d.pop("has_speech", None)  # not part of the Video contract yet
    return Video(**d)


def set_has_speech(conn: sqlite3.Connection, video_id: str, value: bool) -> None:
    conn.execute("UPDATE videos SET has_speech = ? WHERE video_id = ?", (int(value), video_id))
    conn.commit()


def get_has_speech(conn: sqlite3.Connection, video_id: str) -> bool | None:
    row = conn.execute(
        "SELECT has_speech FROM videos WHERE video_id = ?", (video_id,)
    ).fetchone()
    if row is None or row["has_speech"] is None:
        return None
    return bool(row["has_speech"])


# --- segments -------------------------------------------------------------------------

def insert_segments(conn: sqlite3.Connection, segments: Iterable[Segment]) -> int:
    rows = list(segments)
    conn.executemany(
        """
        INSERT INTO segments (video_id, kind, t_start_s, t_end_s, frame_start, frame_end,
            model_name, dim, payload)
        VALUES (:video_id, :kind, :t_start_s, :t_end_s, :frame_start, :frame_end,
            :model_name, :dim, :payload)
        """,
        [
            {
                "video_id": s.video_id, "kind": s.kind,
                "t_start_s": s.t_start_s, "t_end_s": s.t_end_s,
                "frame_start": s.frame_start, "frame_end": s.frame_end,
                "model_name": s.model_name, "dim": s.dim,
                "payload": json.dumps(s.payload),
            }
            for s in rows
        ],
    )
    conn.commit()
    return len(rows)


def replace_segments(
    conn: sqlite3.Connection,
    video_id: str,
    kinds: tuple[str, ...],
    segments: Iterable[Segment],
    payload_merges: list[tuple[int, dict[str, Any]]] | None = None,
) -> int:
    """Atomically replace all segments of the given kinds for a video: DELETE + INSERT in
    one transaction, so a crash mid-stage never destroys the previous good output without
    its replacement landing. payload_merges (segment_id, updates) are applied to OTHER
    rows inside the same transaction — e.g. frames annotating shot payloads — so a crash
    can't leave rows half-annotated."""
    rows = list(segments)
    with conn:  # one transaction; commits on success, rolls back on exception
        for kind in kinds:
            conn.execute(
                "DELETE FROM segments WHERE video_id = ? AND kind = ?", (video_id, kind)
            )
        conn.executemany(
            """
            INSERT INTO segments (video_id, kind, t_start_s, t_end_s, frame_start, frame_end,
                model_name, dim, payload)
            VALUES (:video_id, :kind, :t_start_s, :t_end_s, :frame_start, :frame_end,
                :model_name, :dim, :payload)
            """,
            [
                {
                    "video_id": s.video_id, "kind": s.kind,
                    "t_start_s": s.t_start_s, "t_end_s": s.t_end_s,
                    "frame_start": s.frame_start, "frame_end": s.frame_end,
                    "model_name": s.model_name, "dim": s.dim,
                    "payload": json.dumps(s.payload),
                }
                for s in rows
            ],
        )
        for seg_id, updates in payload_merges or []:
            _merge_payload(conn, seg_id, updates)
    return len(rows)


def delete_segments(conn: sqlite3.Connection, video_id: str, kind: str | None = None) -> int:
    """Used when re-running a stage: clear its prior output for this video so a re-run is
    idempotent rather than duplicating rows."""
    if kind is None:
        cur = conn.execute("DELETE FROM segments WHERE video_id = ?", (video_id,))
    else:
        cur = conn.execute(
            "DELETE FROM segments WHERE video_id = ? AND kind = ?", (video_id, kind)
        )
    conn.commit()
    return cur.rowcount


def list_segments(
    conn: sqlite3.Connection,
    video_id: str,
    kind: str | None = None,
    payload_filters: dict[str, Any] | None = None,
    time_range: tuple[float, float] | None = None,
) -> list[Segment]:
    """Exhaustive, ordered full scan — the completeness contract for 'all X' / time-range
    queries (distinct from ranked top-k search). time_range matches by interval overlap."""
    sql = "SELECT * FROM segments WHERE video_id = ?"
    params: list[Any] = [video_id]
    if kind is not None:
        sql += " AND kind = ?"
        params.append(kind)
    if time_range is not None:
        start, end = time_range
        sql += " AND t_start_s < ? AND t_end_s > ?"  # overlap
        params.extend([end, start])
    sql += " ORDER BY t_start_s, id"
    rows = conn.execute(sql, params).fetchall()
    out: list[Segment] = []
    for row in rows:
        d = dict(row)
        payload = json.loads(d.pop("payload") or "{}")
        d.pop("embedding", None)
        d.pop("speaker", None)
        d.pop("alignment_confidence", None)
        seg = Segment(**{k: v for k, v in d.items() if k in Segment.model_fields}, payload=payload)
        if payload_filters and not _matches(payload, payload_filters):
            continue
        out.append(seg)
    return out


def _matches(payload: dict[str, Any], filters: dict[str, Any]) -> bool:
    return all(payload.get(k) == v for k, v in filters.items())


def _merge_payload(conn: sqlite3.Connection, segment_id: int, updates: dict[str, Any]) -> None:
    """Non-committing payload merge; caller owns the transaction."""
    row = conn.execute("SELECT payload FROM segments WHERE id = ?", (segment_id,)).fetchone()
    if row is None:
        raise KeyError(f"no segment id {segment_id}")
    payload = json.loads(row["payload"] or "{}")
    payload.update(updates)
    conn.execute(
        "UPDATE segments SET payload = ? WHERE id = ?", (json.dumps(payload), segment_id)
    )


def merge_segment_payload(conn: sqlite3.Connection, segment_id: int, updates: dict[str, Any]) -> None:
    """Merge keys into one segment's JSON payload (read-modify-write; last writer wins).
    Standalone form for callers outside a stage transaction."""
    _merge_payload(conn, segment_id, updates)
    conn.commit()


def knn_segments(
    conn: sqlite3.Connection,
    model_name: str,
    query: bytes,
    k: int,
    video_id: str | None = None,
    kinds: tuple[str, ...] | None = None,
) -> list[tuple[Segment, float]]:
    """Exact KNN within ONE embedding space (scoped by model_name so mixed dimensions in
    the BLOB column never meet), via sqlite-vec's scalar distance function — brute force,
    exact, maps directly to pgvector's `<=>` later. Returns (segment, cosine_distance)
    ascending. Requires a connection opened with load_vec=True."""
    sql = (
        "SELECT *, vec_distance_cosine(embedding, ?) AS dist FROM segments "
        "WHERE embedding IS NOT NULL AND model_name = ?"
    )
    params: list[Any] = [query, model_name]
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)
    if kinds:
        sql += f" AND kind IN ({','.join('?' * len(kinds))})"
        params.extend(kinds)
    sql += " ORDER BY dist ASC LIMIT ?"
    params.append(k)
    out: list[tuple[Segment, float]] = []
    for row in conn.execute(sql, params).fetchall():
        d = dict(row)
        dist = d.pop("dist")
        payload = json.loads(d.pop("payload") or "{}")
        for drop in ("embedding", "speaker", "alignment_confidence", "has_speech"):
            d.pop(drop, None)
        seg = Segment(**{kk: v for kk, v in d.items() if kk in Segment.model_fields}, payload=payload)
        out.append((seg, float(dist)))
    return out


def get_segments_by_ids(conn: sqlite3.Connection, ids: list[int]) -> list[Segment]:
    if not ids:
        return []
    sql = f"SELECT * FROM segments WHERE id IN ({','.join('?' * len(ids))}) ORDER BY t_start_s"
    out: list[Segment] = []
    for row in conn.execute(sql, ids).fetchall():
        d = dict(row)
        payload = json.loads(d.pop("payload") or "{}")
        for drop in ("embedding", "speaker", "alignment_confidence"):
            d.pop(drop, None)
        out.append(Segment(**{k: v for k, v in d.items() if k in Segment.model_fields},
                           payload=payload))
    return out


def get_shot(conn: sqlite3.Connection, video_id: str, shot_index: int) -> Segment | None:
    for seg in list_segments(conn, video_id, kind="shot"):
        if seg.payload.get("shot_index") == shot_index:
            return seg
    return None


def embedded_model_names(
    conn: sqlite3.Connection, kinds: tuple[str, ...], video_id: str | None = None
) -> set[str]:
    """Distinct model_names that actually have embeddings for these kinds — lets search
    verify its query encoder matches the space the rows were embedded in."""
    sql = (
        "SELECT DISTINCT model_name FROM segments WHERE embedding IS NOT NULL "
        f"AND kind IN ({','.join('?' * len(kinds))})"
    )
    params: list[Any] = list(kinds)
    if video_id is not None:
        sql += " AND video_id = ?"
        params.append(video_id)
    return {row["model_name"] for row in conn.execute(sql, params).fetchall()}


def unembedded_ids(
    conn: sqlite3.Connection, video_id: str, kind: str, model_name: str | None = None
) -> list[int]:
    """Row ids of this kind still needing an embedding: NULL embedding, or — when
    model_name is given — embedded under a different model (stale space). Upstream stage
    re-runs replace their rows (new ids, NULL embeddings), so this is the exact
    incremental-embed work list."""
    sql = "SELECT id FROM segments WHERE video_id = ? AND kind = ? AND (embedding IS NULL"
    params: list[Any] = [video_id, kind]
    if model_name is not None:
        # IS NOT, not !=: a NULL model_name must read as stale, not as SQL-unknown.
        sql += " OR model_name IS NOT ?"
        params.append(model_name)
    sql += ")"
    return [row["id"] for row in conn.execute(sql, params).fetchall()]


def set_embedding(
    conn: sqlite3.Connection, segment_id: int, blob: bytes, model_name: str, dim: int
) -> None:
    """Attach an embedding (float32 LE bytes) + its space identity to an existing row."""
    cur = conn.execute(
        "UPDATE segments SET embedding = ?, model_name = ?, dim = ? WHERE id = ?",
        (blob, model_name, dim, segment_id),
    )
    if cur.rowcount != 1:
        raise KeyError(f"no segment id {segment_id}")
    conn.commit()
