"""vindex CLI. Round 1 surfaces: `index`, `list` (exhaustive read), and `info`.
`search` (ranked KNN) arrives with the embed stage.
"""

from __future__ import annotations

import argparse
import json
import sys
from contextlib import closing
from pathlib import Path

from .config import Config
from .models import SEGMENT_KINDS
from . import storage, pipeline

DEFAULT_WORKDIR = Path.cwd() / "vindex_data"


def _cfg(args) -> Config:
    return Config.for_workdir(args.workdir)


def cmd_index(args) -> int:
    cfg = _cfg(args)
    vid = pipeline.index(args.source, cfg, video_id=args.id, force=args.force,
                         no_captions=args.no_captions)
    from .jobs import get_status, Status

    with closing(storage.connect(cfg.db_path)) as conn:
        failed = [s for s in pipeline.STAGES
                  if get_status(conn, vid, s) == Status.FAILED]
    if failed:
        print(f"\nindexed: {vid} (WITH FAILED STAGES: {', '.join(failed)})", file=sys.stderr)
        return 1
    print(f"\nindexed: {vid}")
    return 0


def cmd_list(args) -> int:
    cfg = _cfg(args)
    filters = {}
    for kv in args.filter or []:
        if "=" not in kv:
            print(f"bad --filter {kv!r}, expected key=value", file=sys.stderr)
            return 2
        k, v = kv.split("=", 1)
        # Payloads hold typed JSON; `"2" == 2` is False, so string-only filters would
        # silently match nothing for numeric/boolean fields (person_count=0, advisory=true).
        try:
            filters[k] = json.loads(v)
        except json.JSONDecodeError:
            filters[k] = v  # plain string filter
    time_range = None
    if args.range:
        a, b = args.range.split(":", 1)
        time_range = (float(a), float(b))
    with closing(storage.connect(cfg.db_path)) as conn:
        segs = storage.list_segments(
            conn, args.video, kind=args.kind, payload_filters=filters or None,
            time_range=time_range,
        )
    for s in segs:
        print(json.dumps({
            "kind": s.kind, "t": [round(s.t_start_s, 3), round(s.t_end_s, 3)],
            "frames": [s.frame_start, s.frame_end],
            "payload": s.payload,
        }))
    print(f"\n{len(segs)} segments", file=sys.stderr)
    return 0


def cmd_search(args) -> int:
    from . import search as search_mod

    cfg = _cfg(args)
    kinds = tuple(args.kind) if args.kind else None
    try:
        results = search_mod.search(cfg, args.query, video_id=args.video, kinds=kinds, k=args.k)
    except (ValueError, RuntimeError) as e:
        print(f"search error: {e}", file=sys.stderr)
        return 2
    for r in results:
        s = r.segment
        print(json.dumps({
            "video": s.video_id, "kind": s.kind,
            "t": [round(s.t_start_s, 3), round(s.t_end_s, 3)],
            "frames": [s.frame_start, s.frame_end],
            "cut_t": [round(r.cut_range[0], 3), round(r.cut_range[1], 3)],
            "cut_frames": list(r.cut_frames),
            "rrf": round(r.rrf_score, 5),
            "spaces": {sp: {"rank": rk, "dist": round(d, 4)} for sp, (rk, d) in r.per_space.items()},
            "preview": (s.payload.get("text") or s.payload.get("description")
                        or s.payload.get("frame_path") or "")[:120],
        }))
    print(f"\n{len(results)} results (ranked top-k; use 'vindex list' for exhaustive reads)",
          file=sys.stderr)
    return 0


def cmd_info(args) -> int:
    cfg = _cfg(args)
    with closing(storage.connect(cfg.db_path)) as conn:
        v = storage.get_video(conn, args.video)
        if v is None:
            print(f"no such video: {args.video}", file=sys.stderr)
            return 1
        has_speech = storage.get_has_speech(conn, args.video)
    print(json.dumps({**v.model_dump(), "has_speech": has_speech}, indent=2))
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="vindex")
    p.add_argument("--workdir", default=str(DEFAULT_WORKDIR),
                   help="index + media directory (default ./vindex_data)")
    sub = p.add_subparsers(dest="cmd", required=True)

    pi = sub.add_parser("index", help="index a video file or YouTube URL")
    pi.add_argument("source")
    pi.add_argument("--id", default=None, help="explicit video_id (default: source hash)")
    pi.add_argument("--force", action="store_true", help="re-run all stages")
    pi.add_argument("--no-captions", action="store_true",
                    help="skip the VLM caption stage (slowest stage)")
    pi.set_defaults(func=cmd_index)

    pl = sub.add_parser("list", help="exhaustive, ordered segment listing")
    pl.add_argument("--video", required=True)
    pl.add_argument("--kind", choices=SEGMENT_KINDS, default=None)
    pl.add_argument("--filter", action="append", help="payload key=value (repeatable)")
    pl.add_argument("--range", default=None, help="start_s:end_s overlap filter")
    pl.set_defaults(func=cmd_list)

    ps = sub.add_parser("search", help="ranked hybrid top-k search (discovery)")
    ps.add_argument("query")
    ps.add_argument("--video", default=None)
    # Only embedded kinds are searchable; offering the full enum here made
    # `--kind speech` return silently-empty results (transcript text is on speech_window).
    ps.add_argument("--kind", action="append",
                    choices=("frame", "speech_window", "caption"),
                    help="restrict result kinds (repeatable)")
    ps.add_argument("-k", type=int, default=10)
    ps.set_defaults(func=cmd_search)

    pf = sub.add_parser("info", help="show a video's media-identity row")
    pf.add_argument("--video", required=True)
    pf.set_defaults(func=cmd_info)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
