"""Stage 2 — shots. PySceneDetect (AdaptiveDetector, alone) over the mezzanine.

Emits one kind='shot' segment per detected shot with detector provenance. Runs on the
mezzanine (never the source) so boundaries share the index's frame timebase.

Detector choice (from adversarial review, verified against PySceneDetect 0.7 source):
SceneManager UNIONS the cut lists of multiple registered detectors — it never filters one
detector's cuts by another's — so pairing ContentDetector with AdaptiveDetector strictly
ADDS false positives. AdaptiveDetector alone is used: it scores the same content values as
ContentDetector but thresholds against a rolling average, which is itself the fast-motion
false-positive suppression. Known limitation (PLAN): threshold detectors don't fire on
gradual within-take change; the TransNetV2 promotion trigger is acceptance criterion 6.
"""

from __future__ import annotations

from ..config import Config
from ..models import Segment
from .. import storage

STAGE = "shots"

DETECTOR_NAME = "pyscenedetect"


class ShotsError(RuntimeError):
    pass


def _merge_micro_shots(
    scenes: list[tuple[int, int]], min_frames: int
) -> list[tuple[int, int]]:
    """Collapse scenes shorter than min_frames into their successor by dropping the later
    of the two bounding cuts (the earlier cut is where the transition was first detected).
    Guards against near-coincident boundaries minting phantom 1-2 frame shots — detectors
    enforce min_scene_len only on their own cut stream, and any future multi-source cut
    list would reintroduce the risk. The final scene, if short, merges backward instead."""
    if not scenes:
        return scenes
    out: list[list[int]] = [list(scenes[0])]
    for start, end in scenes[1:]:
        if out[-1][1] - out[-1][0] < min_frames:
            out[-1][1] = end  # previous scene too short: absorb this one (drop later cut)
        else:
            out.append([start, end])
    if len(out) > 1 and out[-1][1] - out[-1][0] < min_frames:
        tail = out.pop()
        out[-1][1] = tail[1]  # short final scene merges backward
    return [(a, b) for a, b in out]


def run(video_id: str, cfg: Config, conn) -> dict:
    video = storage.get_video(conn, video_id)
    if video is None:
        raise ShotsError(f"no ingested video {video_id}; run ingest first")

    # Imported here so the core package works without the shots extra installed.
    from scenedetect import AdaptiveDetector, open_video, SceneManager
    import scenedetect

    v = open_video(video.media_path)
    manager = SceneManager()
    manager.add_detector(AdaptiveDetector())
    decoded_frames = manager.detect_scenes(v, show_progress=False)
    scene_list = manager.get_scene_list()

    # Reconcile the cv2 decode count against ingest's authoritative ffprobe count in BOTH
    # branches. All emitted indices are clamped to the count both decoders can produce:
    # on a negative mismatch, pinning frame_end to frame_count-1 would hand stage 3 an
    # index cv2 provably cannot decode (verified: read() at that index returns ok=False),
    # dead-ending the pipeline on a video this stage deliberately accepted.
    decode_mismatch = (decoded_frames or video.frame_count) - video.frame_count
    if abs(decode_mismatch) > cfg.max_decode_mismatch_frames:
        raise ShotsError(
            f"PySceneDetect decoded {decoded_frames} frames but ingest counted "
            f"{video.frame_count} (diff {decode_mismatch:+d} > "
            f"{cfg.max_decode_mismatch_frames}); shot boundaries would not share the "
            f"index timebase. Investigate the mezzanine before trusting this stage."
        )
    usable = min(decoded_frames or video.frame_count, video.frame_count)
    last_frame = usable - 1
    if not scene_list:
        # No boundaries detected -> the whole video is one shot. An empty scene list must
        # not mean "no shots" (every video has at least one).
        scenes = [(0, usable)]
    else:
        # PySceneDetect scenes are [start, end) in frames; end == next start.
        scenes = [(s.get_frames(), e.get_frames()) for s, e in scene_list]
        scenes[-1] = (scenes[-1][0], usable)

    scenes = _merge_micro_shots(scenes, cfg.min_shot_frames)

    fps_num, fps_den = video.fps_num, video.fps_den
    rows: list[Segment] = []
    for i, (f_start, f_end_excl) in enumerate(scenes):
        f_start = max(0, min(last_frame, f_start))
        # Convert PySceneDetect's exclusive end to our inclusive convention.
        f_end = max(f_start, min(last_frame, f_end_excl - 1))
        # Derive seconds from frames exactly (frames are authoritative here — the detector
        # works in frame space), keeping both domains consistent by construction.
        t_start = f_start * fps_den / fps_num
        # t_end is the exclusive-end instant of the inclusive last frame.
        t_end = (f_end + 1) * fps_den / fps_num
        rows.append(Segment(
            video_id=video_id, kind="shot",
            t_start_s=t_start, t_end_s=t_end,
            frame_start=f_start, frame_end=f_end,
            model_name=DETECTOR_NAME,
            payload={
                "shot_index": i,
                "detector": DETECTOR_NAME,
                "detector_version": scenedetect.__version__,
                "detectors": ["AdaptiveDetector"],
                "decode_mismatch_frames": decode_mismatch,
            },
        ))

    # Cascade: frame rows join shots via shot_index, so replacing shots atomically deletes
    # any frame rows built against the old segmentation — a frames crash then leaves an
    # honest absence, never a stale cross-generation join. Pipeline re-runs frames after.
    storage.replace_segments(conn, video_id, ("shot", "frame"), rows)
    return {
        "shots": len(rows),
        "detector_version": scenedetect.__version__,
        "decode_mismatch_frames": decode_mismatch,
    }
