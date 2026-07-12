"""Hybrid search — the ranked, agent-facing read surface (top-k discovery).

One KNN per embedding space, each scoped by model_name; results merged with Reciprocal
Rank Fusion (rank-based, because SigLIP sigmoid scores and bge cosines are on incomparable
scales). Structured filters (video, kind) apply BEFORE fusion. The exhaustive counterpart
for "all X" tasks is `vindex list` / storage.list_segments — search never guarantees
completeness (PLAN: destructive edits must be driven by list, not top-k).

Two query routes per PLAN:
- query -> SigLIP text tower -> image space (kind='frame' rows)
- query -> bge (with its retrieval prefix) -> text space (speech_window + caption rows)
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

from .config import Config
from .models import Segment
from . import storage


@functools.lru_cache(maxsize=1)
def _text_model():
    """Search-lifetime cache. The embed STAGE deliberately reloads/releases its models to
    respect the 16 GB sequential-stage budget — do not move this cache there."""
    from .stages.embed import _load_text_model

    return _load_text_model()

# Standard RRF constant (Cormack et al.); rank is 1-based.
RRF_K = 60

# bge-small is asymmetric: queries need this prefix, passages are embedded bare.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

IMAGE_KINDS = ("frame",)
TEXT_KINDS = ("speech_window", "caption")


@dataclass
class SearchResult:
    segment: Segment
    rrf_score: float
    per_space: dict[str, tuple[int, float]]  # space -> (rank, raw cosine distance)
    # The CUT-ACTIONABLE range (PLAN end goal: exact ranges, not containers to spelunk):
    # speech_window hits refine to their best-matching constituent speech segment (which
    # carries the word anchors); frame hits expand to their enclosing shot. Falls back to
    # the segment's own range.
    cut_range: tuple[float, float] = (0.0, 0.0)
    cut_frames: tuple[int | None, int | None] = (None, None)


def _encode_query_image_space(query: str) -> tuple[bytes, str]:
    import numpy as np
    import torch

    from .stages.embed import _load_siglip

    sig = _load_siglip()
    inputs = sig.processor(
        text=[query], return_tensors="pt", padding="max_length", max_length=64
    ).to(sig.device)
    with torch.no_grad():
        feats = sig.model.get_text_features(**inputs)
    if not torch.is_tensor(feats):
        feats = feats.pooler_output
    v = torch.nn.functional.normalize(feats.float(), dim=-1).cpu().numpy()[0]
    return np.ascontiguousarray(v, dtype=np.float32).tobytes(), sig.name


def _encode_query_text_space(query: str) -> tuple[bytes, str]:
    import numpy as np

    from .stages.embed import TEXT_MODEL

    tm = _text_model()
    v = tm.encode([BGE_QUERY_PREFIX + query], normalize_embeddings=True)[0]
    return np.ascontiguousarray(v, dtype=np.float32).tobytes(), TEXT_MODEL


def _require_stored_space(conn, model: str, kinds: tuple[str, ...],
                          video_id: str | None) -> None:
    """The query encoder must be the model the rows were embedded with — a SigLIP fallback
    flip at query time would KNN against zero rows and read as 'no matches'. An index with
    no embeddings at all for these kinds is fine (legitimately empty)."""
    stored = storage.embedded_model_names(conn, kinds, video_id)
    if stored and model not in stored:
        raise RuntimeError(
            f"query encoder {model!r} does not match stored embeddings "
            f"{sorted(stored)} for kinds {kinds}; re-run embed or fix the model environment")


def search(
    cfg: Config,
    query: str,
    video_id: str | None = None,
    kinds: tuple[str, ...] | None = None,
    k: int = 10,
) -> list[SearchResult]:
    """kinds, when given, restricts which segment kinds may appear (pre-fusion filter);
    spaces whose kinds are entirely filtered out are skipped (their model never loads)."""
    searchable = IMAGE_KINDS + TEXT_KINDS
    if kinds is not None and not any(kk in searchable for kk in kinds):
        # A silently empty result here is indistinguishable from a genuine miss — and an
        # editing agent would act on it. Transcript text lives on 'speech_window' rows.
        raise ValueError(
            f"kinds {kinds} contain no searchable kind; searchable kinds: {searchable}")
    conn = storage.connect(cfg.db_path, load_vec=True)
    try:
        ranked: dict[int, dict] = {}  # segment id -> {segment, per_space}

        def admit(space: str, hits: list[tuple[Segment, float]]) -> None:
            for rank, (seg, dist) in enumerate(hits, start=1):
                assert seg.id is not None
                entry = ranked.setdefault(seg.id, {"segment": seg, "per_space": {}})
                entry["per_space"][space] = (rank, dist)

        image_kinds = tuple(kk for kk in IMAGE_KINDS if kinds is None or kk in kinds)
        text_kinds = tuple(kk for kk in TEXT_KINDS if kinds is None or kk in kinds)

        if image_kinds:
            qblob, model = _encode_query_image_space(query)
            _require_stored_space(conn, model, image_kinds, video_id)
            admit("image", storage.knn_segments(
                conn, model_name=model, query=qblob, k=k,
                video_id=video_id, kinds=image_kinds,
            ))
        if text_kinds:
            qblob, model = _encode_query_text_space(query)
            _require_stored_space(conn, model, text_kinds, video_id)
            admit("text", storage.knn_segments(
                conn, model_name=model, query=qblob, k=k,
                video_id=video_id, kinds=text_kinds,
            ))

        results = [
            SearchResult(
                segment=e["segment"],
                rrf_score=sum(1.0 / (RRF_K + rank) for rank, _ in e["per_space"].values()),
                per_space=e["per_space"],
            )
            for e in ranked.values()
        ]
        results.sort(key=lambda r: r.rrf_score, reverse=True)
        results = results[:k]
        for r in results:
            r.cut_range, r.cut_frames = _cut_target(conn, query, r.segment)
        return results
    finally:
        conn.close()


def _cut_target(conn, query: str, seg: Segment) -> tuple[tuple[float, float],
                                                         tuple[int | None, int | None]]:
    """Resolve a hit to the range an editor would actually cut on."""
    if seg.kind == "speech_window":
        ids = [i for i in seg.payload.get("source_speech_ids", []) if i and i > 0]
        parts = storage.get_segments_by_ids(conn, ids)
        texts = [(p, (p.payload.get("text") or "").strip()) for p in parts]
        texts = [(p, t) for p, t in texts if t]
        if texts:
            import numpy as np

            tm = _text_model()
            vecs = tm.encode([BGE_QUERY_PREFIX + query] + [t for _, t in texts],
                             normalize_embeddings=True)
            sims = np.asarray(vecs[1:]) @ np.asarray(vecs[0])
            best = texts[int(np.argmax(sims))][0]
            return (best.t_start_s, best.t_end_s), (best.frame_start, best.frame_end)
    elif seg.kind == "frame":
        si = seg.payload.get("shot_index")
        if si is not None:
            shot = storage.get_shot(conn, seg.video_id, si)
            if shot is not None:
                return (shot.t_start_s, shot.t_end_s), (shot.frame_start, shot.frame_end)
    return (seg.t_start_s, seg.t_end_s), (seg.frame_start, seg.frame_end)
