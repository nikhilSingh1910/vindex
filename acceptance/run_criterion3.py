"""Criterion 3 scorer: each pre-registered query must return the ground-truth range in its
top 3 (IoU >= 0.5, or start within +/-0.5 s). Run: uv run python acceptance/run_criterion3.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex.config import Config
from vindex.search import search

WORKDIR = Path(__file__).resolve().parents[1] / "vindex_data"


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def main() -> int:
    spec = json.loads((Path(__file__).parent / "criterion3_queries.json").read_text())
    cfg = Config.for_workdir(WORKDIR)
    passed = failed = skipped = 0
    for q in spec["queries"]:
        if q.get("gt_ranges") is None:
            skipped += 1
            print(f"SKIP {q['id']} ({q['video']}): ground truth not yet labeled")
            continue
        query = q.get("query_final") or q["query"]
        results = search(cfg, query, video_id=q["video"], k=3)
        gts = [tuple(g) for g in q["gt_ranges"]]
        containment_ok = bool(q.get("containment_ok"))
        hit = None
        for rank, r in enumerate(results, 1):
            cand = r.cut_range  # the cut-actionable range, per the end goal
            for gt in gts:
                contained = containment_ok and cand[0] >= gt[0] and cand[1] <= gt[1]
                if iou(cand, gt) >= 0.5 or abs(cand[0] - gt[0]) <= 0.5 or contained:
                    hit = (rank, cand, round(iou(cand, gt), 3))
                    break
            if hit:
                break
        if hit:
            passed += 1
            print(f"PASS {q['id']} ({q['mode']}): rank={hit[0]} cut={[round(x,1) for x in hit[1]]} iou={hit[2]}")
        else:
            failed += 1
            tops = [(r.segment.kind, [round(x, 1) for x in r.cut_range]) for r in results]
            print(f"FAIL {q['id']} ({q['mode']}): gt={gts} top3_cuts={tops}")
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (unlabeled)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
