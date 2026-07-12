"""Criterion 3 scorer.

Positives: each pre-registered query must return a ground-truth range in its top 3
(IoU >= 0.5, start within +/-0.5 s, or containment where flagged). expected_fail
queries invert the exit-code contribution: their failure is the documented steady
state (XFAIL), and an unexpected pass (XPASS) demands a spec update — so the exit
code signals REGRESSIONS, not known roadmap gaps.

Negatives (gt_ranges == []): nothing in the corpus matches. Every top-3 result is
compared against the floor of ITS OWN embedding space — the loosest distance any
passing positive's true hit scored there (data-calibrated, printed). A result at or
inside its space's floor is a breach (plausible poisoned filter). A result whose
space has no calibrated floor makes the negative UNCALIBRATED (skipped loudly),
never a vacuous pass.

Per-class results expose capability classes an aggregate pass rate would hide.

Run: uv run python acceptance/run_criterion3.py
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from vindex.config import Config
from vindex.search import search

WORKDIR = Path(__file__).resolve().parents[1] / "vindex_data"


def iou(a: tuple[float, float], b: tuple[float, float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def find_hit(results, gts, containment_ok):
    """First top-3 result whose cut range matches any gt range, else None."""
    for rank, r in enumerate(results, 1):
        cand = r.cut_range  # the cut-actionable range, per the end goal
        for gt in gts:
            contained = containment_ok and cand[0] >= gt[0] and cand[1] <= gt[1]
            if iou(cand, gt) >= 0.5 or abs(cand[0] - gt[0]) <= 0.5 or contained:
                return rank, r, round(iou(cand, gt), 3)
    return None


def main() -> int:
    spec = json.loads((Path(__file__).parent / "criterion3_queries.json").read_text())
    cfg = Config.for_workdir(WORKDIR)
    by_class: dict[str, list[tuple[str, str]]] = defaultdict(list)  # id -> +/-/x mark
    passed = failed = xfailed = skipped = 0
    # Per-space floor: the loosest distance a true hit scored there. Negatives must
    # score strictly worse than this in their result's own space.
    floor: dict[str, float] = {}
    negatives = []

    for q in spec["queries"]:
        cls = q.get("class", "?")
        if q.get("gt_ranges") is None:
            skipped += 1
            print(f"SKIP {q['id']} ({q['video']}): ground truth not yet labeled")
            continue
        if q["gt_ranges"] == []:
            negatives.append(q)  # scored after positives calibrate the floors
            continue
        query = q.get("query_final") or q["query"]
        results = search(cfg, query, video_id=q["video"], k=3)
        hit = find_hit(results, [tuple(g) for g in q["gt_ranges"]], bool(q.get("containment_ok")))
        expect_fail = bool(q.get("expected_fail"))
        if hit:
            rank, r, iou_v = hit
            for space, (_, dist) in r.per_space.items():
                floor[space] = max(floor.get(space, 0.0), dist)
            if expect_fail:
                failed += 1  # strict xpass: the spec says this cannot pass yet
                by_class[cls].append((q["id"], "-"))
                print(f"XPASS {q['id']} ({cls}): rank={rank} — expected_fail no longer "
                      f"holds; update the spec")
            else:
                passed += 1
                by_class[cls].append((q["id"], "+"))
                print(f"PASS {q['id']} ({cls}): rank={rank} "
                      f"cut={[round(x, 1) for x in r.cut_range]} iou={iou_v}")
        else:
            tops = [(r.segment.kind, [round(x, 1) for x in r.cut_range]) for r in results]
            if expect_fail:
                xfailed += 1
                by_class[cls].append((q["id"], "x"))
                print(f"XFAIL {q['id']} ({cls}): known gap, top3_cuts={tops}")
            else:
                failed += 1
                by_class[cls].append((q["id"], "-"))
                print(f"FAIL {q['id']} ({cls}): gt={q['gt_ranges']} top3_cuts={tops}")

    for q in negatives:
        cls = q.get("class", "negative")
        results = search(cfg, q.get("query_final") or q["query"], video_id=q["video"], k=3)
        breaches, uncalibrated = {}, set()
        for r in results:
            for space, (_, dist) in r.per_space.items():
                if space not in floor:
                    uncalibrated.add(space)
                elif dist <= floor[space]:
                    breaches[f"{space}@{r.segment.kind}"] = round(dist, 4)
        if breaches:
            failed += 1
            by_class[cls].append((q["id"], "-"))
            print(f"FAIL {q['id']} (negative): result within true-hit range {breaches} "
                  f"vs floor { {s: round(f, 4) for s, f in floor.items()} } "
                  f"— plausible poisoned filter")
        elif uncalibrated:
            skipped += 1
            by_class[cls].append((q["id"], "?"))
            print(f"SKIP {q['id']} (negative): space(s) {sorted(uncalibrated)} have no "
                  f"passing-positive floor — uncalibrated, not a pass")
        else:
            passed += 1
            by_class[cls].append((q["id"], "+"))
            print(f"PASS {q['id']} (negative): no top-3 result within its space's "
                  f"true-hit range")

    print(f"\nnegative-floor calibration (loosest true-hit distance per space): "
          f"{ {s: round(f, 4) for s, f in floor.items()} }")
    for cls in sorted(by_class):
        entries = by_class[cls]
        n_ok = sum(mark == "+" for _, mark in entries)
        marks = " ".join(f"{qid}{mark}" for qid, mark in entries)
        print(f"  {cls:>13}: {n_ok}/{len(entries)}  {marks}")
    print(f"\n{passed} passed, {failed} failed, {xfailed} xfailed (known gaps), "
          f"{skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
