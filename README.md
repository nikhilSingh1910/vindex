# vindex

A local-first video indexer that produces **timestamp-anchored, cut-actionable indexes for
AI editing agents**. An agent goes from a query — *"the drone shot of the coastline"*,
*"where he says stay hungry"* — to exact time ranges and frame numbers it can cut,
reorder, and overlay.

This is not a generic video search index. Every artifact must survive contact with an
edit decision: search results carry a `cut_range`/`cut_frames` refined to shot and
sentence boundaries against a deterministic, pinned mezzanine encode. Anything that
cannot become an edit decision is dead weight — and a poisoned filter is worse than
dead weight.

**Constraints**: free, fully local, open-source models only. No paid APIs. Accuracy over
speed (overnight batch is fine). Dev on a 16 GB MacBook Air; Linux prod trajectory.

## Status

Round 1 complete (2026-07-12): six-stage pipeline built, adversarially reviewed, and
verified end-to-end on a 3-video corpus (talking-head / skate montage with vocal music
bed / no-dialogue ambience). Pre-registered acceptance: **8/10 queries pass at
top-3, IoU ≥ 0.5**; the two failures are measured, understood, and drive the round-2
roadmap (action-class queries; lyric embedding). Full design rationale, verified facts,
and the research ledger live in [PLAN.md](PLAN.md).

## Pipeline

```
ingest → shots → frames → transcribe → caption → embed
```

| Stage | What it does | Backed by |
|---|---|---|
| ingest | deterministic CFR mezzanine (pinned libx264), fps whitelist, HDR refusal, audio-clock check | ffmpeg |
| shots | frame-exact shot boundaries | PySceneDetect |
| frames | keyframes every 3 s, dHash+person-state dedup, camera-motion labels | OpenCV, torchvision |
| transcribe | sentence segments + word timestamps, hallucination guards, music-authoritative gate (music/noise/silence regions, advisory lyrics) | faster-whisper large-v3, inaSpeechSegmenter |
| caption | one advisory VLM caption per shot (typed, guarded, per-shot failure isolation) | Qwen2.5-VL via Ollama |
| embed | image space + text space in one SQLite BLOB column, scoped by model/dim | SigLIP 2, bge-small |

Everything lands in a single SQLite database. The pipeline is resumable (jobs table),
idempotent per stage, and single-process by contract.

## Read contracts

Two deliberately different surfaces:

- `vindex search "query"` — ranked top-k **discovery** (hybrid RRF over both embedding
  spaces), each hit carrying a cut-actionable range.
- `vindex list --video ID --kind K --filter k=v --range a:b` — the **exhaustive**
  contract: exact full scan for "all X" tasks. Destructive edits must be driven by
  `list`, never by top-k search.

## Usage

```bash
uv sync --extra embed --extra frames --extra shots --extra music --extra align
uv run vindex index <file-or-youtube-url> [--id NAME] [--no-captions]
uv run vindex search "waves crashing on rocks" --video coast -k 5
uv run vindex list --video credits --kind lyrics --range 70:85
uv run vindex info --video coast
```

Captions need a local [Ollama](https://ollama.com) with `qwen2.5vl:3b` pulled.
Acceptance suite: `uv run python acceptance/run_criterion3.py`.

## Design notes worth stealing

- **Fail loud, never silently degrade**: unknown frame rates, HDR sources, and
  unparseable VLM replies refuse the input rather than poison the index.
- **Advisory vs authoritative**: model-generated labels (captions, lyrics, motion) carry
  provenance and never gate primary artifacts.
- **Pre-registered acceptance**: queries and ground truth are fixed before searching;
  ground truth is labeled from the media, never from the index's own output.

Built with [Claude Code](https://claude.com/claude-code).
