"""Stage 3 — frames. For each shot: candidate keyframes (midpoint + every ~3 s), person
detection on every candidate BEFORE dedup, shot-scoped dHash dedup with the representative
frame exempt, persisted JPEGs, and deterministic camera-motion classification via sparse
optical flow (never asked of a VLM — a single still cannot encode motion).

Outputs kind='frame' rows (one per kept keyframe; frame_start==frame_end, spanning one
frame's duration with exclusive t_end so half-open time_range queries can reach them) and
annotates each shot row's payload with camera_motion + provenance.

Requires shots to have run. Frame decoding uses OpenCV on the mezzanine, so indices share
the index timebase (shots stage already reconciles decoder-vs-ffprobe counts).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from ..config import Config
from ..models import Segment
from .. import storage

STAGE = "frames"


class FramesError(RuntimeError):
    pass


# --- person detection behind a protocol (PLAN portability discipline) -------------------

@dataclass
class PersonResult:
    count: int
    max_area_frac: float  # largest person bbox area / frame area, 0.0 if none


class PersonDetector(Protocol):
    name: str
    version: str

    def detect(self, bgr_frame) -> PersonResult: ...


class TorchvisionPersonDetector:
    """ssdlite320_mobilenet_v3_large (COCO, BSD). CPU, ~10 ms/frame class."""

    def __init__(self, score_min: float):
        import torch
        import torchvision
        from torchvision.models.detection import (
            SSDLite320_MobileNet_V3_Large_Weights,
            ssdlite320_mobilenet_v3_large,
        )

        self._torch = torch
        weights = SSDLite320_MobileNet_V3_Large_Weights.COCO_V1
        self._model = ssdlite320_mobilenet_v3_large(weights=weights).eval()
        self._score_min = score_min
        self.name = "ssdlite320_mobilenet_v3_large"
        self.version = torchvision.__version__

    def detect(self, bgr_frame) -> PersonResult:
        import numpy as np

        rgb = bgr_frame[:, :, ::-1].copy()
        tensor = self._torch.from_numpy(np.transpose(rgb, (2, 0, 1))).float() / 255.0
        with self._torch.no_grad():
            out = self._model([tensor])[0]
        h, w = bgr_frame.shape[:2]
        frame_area = float(h * w)
        count, max_area = 0, 0.0
        for label, score, box in zip(out["labels"], out["scores"], out["boxes"]):
            if int(label) != 1 or float(score) < self._score_min:  # COCO label 1 = person
                continue
            count += 1
            x1, y1, x2, y2 = (float(v) for v in box)
            max_area = max(max_area, (x2 - x1) * (y2 - y1) / frame_area)
        return PersonResult(count=count, max_area_frac=round(max_area, 4))


# --- camera motion ----------------------------------------------------------------------

def classify_motion(gray_pairs, cfg: Config) -> tuple[str, dict]:
    """Classify shot camera motion from consecutive grayscale frame pairs using sparse
    optical flow + a partial affine fit. Returns (label, evidence). Deterministic, CPU."""
    import cv2
    import numpy as np

    per_pair: list[str] = []
    evidence: list[dict] = []
    for a, b in gray_pairs:
        pts = cv2.goodFeaturesToTrack(a, maxCorners=200, qualityLevel=0.01, minDistance=8)
        if pts is None or len(pts) < 12:
            # Featureless (solid color, sky, water): flow can't run, but pixel difference
            # still distinguishes "truly unchanged" from "content moved but untrackable".
            d = float(np.mean(cv2.absdiff(a, b)))
            per_pair.append("static" if d < cfg.motion_flat_diff_max else "complex")
            evidence.append({"features": 0, "mean_abs_diff": round(d, 3)})
            continue
        nxt, status, _ = cv2.calcOpticalFlowPyrLK(a, b, pts, None)
        good_a = pts[status.flatten() == 1]
        good_b = nxt[status.flatten() == 1]
        if len(good_a) < 12:
            # Features existed but tracking lost most of them: displacement beyond LK's
            # range, i.e. fast motion — NOT static (observed: a hard pan classified static
            # because far-apart samples broke tracking).
            per_pair.append("complex")
            evidence.append({"features": int(len(pts)), "tracked": int(len(good_a))})
            continue
        m, _ = cv2.estimateAffinePartial2D(good_a, good_b, method=cv2.RANSAC)
        if m is None:
            per_pair.append("complex")
            evidence.append({"affine": None})
            continue
        # Partial affine: [[s*cos, -s*sin, tx], [s*sin, s*cos, ty]]
        scale = float(np.hypot(m[0, 0], m[1, 0]))
        tx, ty = float(m[0, 2]), float(m[1, 2])
        h, w = a.shape[:2]
        dx, dy = abs(tx) / w, abs(ty) / h
        mag = max(dx, dy)
        if abs(scale - 1.0) > cfg.motion_zoom_scale_eps and mag < cfg.motion_still_px_frac * 10:
            label = "zoom"
        elif mag < cfg.motion_still_px_frac:
            label = "static"
        elif dx > dy * cfg.motion_axis_dominance:
            label = "pan"
        elif dy > dx * cfg.motion_axis_dominance:
            label = "tilt"
        else:
            label = "complex"
        per_pair.append(label)
        evidence.append({"scale": round(scale, 4), "dx": round(dx, 4), "dy": round(dy, 4)})

    # Vote: unanimous wins; a static+X mix means the camera moved (X); otherwise a strict
    # majority of measured pairs wins, else complex. Majority (not veto) matters because a
    # single featureless sample reads 'complex' by pixel diff even during a clean pan —
    # observed: [pan, complex, pan] on a true pan must classify pan.
    from collections import Counter

    known = [p for p in per_pair if p != "unknown"]
    if not known:
        overall = "complex"  # defensive: no measurable pair (should be unreachable now)
    else:
        counts = Counter(known)
        moving = Counter({k: c for k, c in counts.items() if k != "static"})
        if not moving:
            overall = "static"
        elif len(moving) == 1:
            overall = next(iter(moving))
        else:
            # Vote among MOVING labels only — static pairs must not outvote motion.
            label, top = moving.most_common(1)[0]
            overall = label if top * 2 > sum(moving.values()) else "complex"
    return overall, {"pairs": per_pair, "evidence": evidence}


# --- stage ------------------------------------------------------------------------------

def _candidate_indices(f_start: int, f_end: int, fps: float, interval_s: float) -> list[int]:
    """Midpoint + one frame every ~interval_s across [f_start, f_end] (inclusive)."""
    mid = (f_start + f_end) // 2
    idxs = {mid}
    step = max(1, int(round(interval_s * fps)))
    for f in range(f_start, f_end + 1, step):
        idxs.add(f)
    return sorted(idxs)


def run(video_id: str, cfg: Config, conn) -> dict:
    import cv2
    import imagehash
    from PIL import Image

    video = storage.get_video(conn, video_id)
    if video is None:
        raise FramesError(f"no ingested video {video_id}; run ingest first")
    shots = storage.list_segments(conn, video_id, kind="shot")
    if not shots:
        raise FramesError(f"no shots for {video_id}; run shots first")

    detector: PersonDetector = TorchvisionPersonDetector(cfg.person_score_min)
    frames_dir = cfg.video_dir(video_id) / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video.media_path)
    if not cap.isOpened():
        raise FramesError(f"OpenCV cannot open mezzanine {video.media_path}")

    def read_frame(idx: int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            raise FramesError(f"failed to decode frame {idx} of {video.media_path}")
        return frame

    fps = video.fps
    fps_num, fps_den = video.fps_num, video.fps_den
    rows: list[Segment] = []
    shot_motion: list[tuple[int, str, dict]] = []  # (shot segment id, label, evidence)
    kept_total = dropped_total = 0

    try:
        for shot in shots:
            f_start, f_end = shot.frame_start or 0, shot.frame_end or 0
            mid = (f_start + f_end) // 2
            candidates = _candidate_indices(f_start, f_end, fps, cfg.keyframe_interval_s)
            rep_idx = min(candidates, key=lambda i: abs(i - mid))

            def emit(idx: int, frame, h, person: PersonResult) -> None:
                path = frames_dir / f"f{idx:08d}.jpg"
                if not cv2.imwrite(str(path), frame,
                                   [int(cv2.IMWRITE_JPEG_QUALITY), cfg.frame_jpeg_quality]):
                    # An unchecked failed write (disk full, permissions) would commit a row
                    # pointing at a missing JPEG that crashes caption/embed much later.
                    raise FramesError(f"cv2.imwrite failed for {path}")
                # One-frame extent with exclusive end (same convention as shots): a
                # zero-width row would be unreachable by half-open time_range queries —
                # verified: the t=0 frame never matched any range.
                t = idx * fps_den / fps_num
                rows.append(Segment(
                    video_id=video_id, kind="frame",
                    t_start_s=t, t_end_s=(idx + 1) * fps_den / fps_num,
                    frame_start=idx, frame_end=idx,
                    payload={
                        # Workdir-relative so the index survives a workdir move/restore
                        # (consumers join cfg.workdir; absolute rows from older DBs still
                        # resolve because joining an absolute path is a no-op).
                        "frame_path": str(path.relative_to(cfg.workdir)),
                        "shot_index": shot.payload.get("shot_index"),
                        "is_representative": idx == rep_idx,
                        "dhash": str(h),
                        "person_count": person.count,
                        "person_max_area_frac": person.max_area_frac,
                        "person_detector": detector.name,
                        "person_detector_version": detector.version,
                    },
                ))

            # The representative is decoded first (always kept; its hash also serves as a
            # dedup reference), then candidates STREAM one at a time — decoding a whole
            # long shot's candidates at once would hold gigabytes for a 20-min single-take.
            #
            # Person detection runs on EVERY candidate, before the drop decision, and a
            # near-dup is dropped only if its person state also matches the reference:
            # dHash is a background hash — a talking head leaving frame can dhash-match
            # the empty room (verified empirically), which would silently break the ~3 s
            # person-visibility timeline the plan promises.
            def person_eq(p1: PersonResult, p2: PersonResult) -> bool:
                return p1.count == p2.count

            rep_frame = read_frame(rep_idx)
            rep_hash = imagehash.dhash(Image.fromarray(rep_frame[:, :, ::-1]))
            rep_person = detector.detect(rep_frame)
            emit(rep_idx, rep_frame, rep_hash, rep_person)
            del rep_frame
            kept_count = 1
            last_kept_hash: imagehash.ImageHash | None = None
            last_kept_person: PersonResult | None = None
            for idx in candidates:
                if idx == rep_idx:
                    last_kept_hash, last_kept_person = rep_hash, rep_person
                    continue
                frame = read_frame(idx)
                h = imagehash.dhash(Image.fromarray(frame[:, :, ::-1]))
                person = detector.detect(frame)
                # Near-dup of the last kept frame OR of the (always-kept) representative —
                # the rep check prevents pre-rep candidates duplicating it (observed).
                dup_of_last = (
                    last_kept_hash is not None
                    and (h - last_kept_hash) <= cfg.dedup_hamming_max
                    and last_kept_person is not None
                    and person_eq(person, last_kept_person)
                )
                dup_of_rep = (
                    (h - rep_hash) <= cfg.dedup_hamming_max and person_eq(person, rep_person)
                )
                if dup_of_last or dup_of_rep:
                    dropped_total += 1
                    continue
                emit(idx, frame, h, person)
                kept_count += 1
                last_kept_hash, last_kept_person = h, person
            kept_total += kept_count

            # Camera motion: flow must be measured between temporally CLOSE frames —
            # optical flow cannot track displacements from samples seconds apart (observed:
            # a hard pan read as static because tracking broke). Sample short (frame,
            # frame+gap) pairs at the shot's start, middle, and end; gap ~0.2 s.
            span = f_end - f_start
            gap = max(1, int(round(fps * 0.2)))
            pair_starts = sorted({
                f_start,
                f_start + max(0, (span - gap) // 2),
                max(f_start, f_end - gap),
            })
            pairs = []
            for p0 in pair_starts:
                p1 = min(f_end, p0 + gap)
                if p1 <= p0:
                    continue
                ga = cv2.cvtColor(read_frame(p0), cv2.COLOR_BGR2GRAY)
                gb = cv2.cvtColor(read_frame(p1), cv2.COLOR_BGR2GRAY)
                pairs.append((ga, gb))
            label, evidence = classify_motion(pairs, cfg) if pairs else ("static", {"pairs": []})
            assert shot.id is not None
            shot_motion.append((shot.id, label, evidence))
    finally:
        cap.release()

    # Rows + shot annotations land in ONE transaction: a crash can't leave shots
    # half-annotated against a frame set that never committed.
    storage.replace_segments(
        conn, video_id, ("frame",), rows,
        payload_merges=[
            (seg_id, {
                "camera_motion": label,
                "motion_method": "sparse_flow_partial_affine",
                "motion_evidence": evidence,
            })
            for seg_id, label, evidence in shot_motion
        ],
    )

    # Prune orphaned JPEGs: DB rows were replaced atomically above, but files from prior
    # runs with different candidates/config would otherwise accumulate and be visible to
    # anything that lists frames_dir (verified: config changes left disowned stills).
    referenced = {Path(r.payload["frame_path"]).name for r in rows}
    pruned = 0
    for jpg in frames_dir.glob("*.jpg"):
        if jpg.name not in referenced:
            jpg.unlink(missing_ok=True)
            pruned += 1

    return {
        "frames_kept": kept_total,
        "near_duplicates_dropped": dropped_total,
        "shots_annotated": len(shot_motion),
        "orphaned_jpegs_pruned": pruned,
    }
