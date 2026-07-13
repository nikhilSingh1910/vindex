# vindex

A local-first video indexer that produces **timestamp-anchored, cut-actionable indexes for
AI editing agents**. An agent goes from a query — *"the drone shot of the coastline"*,
*"where he says stay hungry"*, *"the crowd applauding"* — to exact time ranges and frame
numbers it can cut, reorder, and overlay.

This is not a generic video search index. Every artifact must survive contact with an
edit decision: search results carry a `cut_range`/`cut_frames` refined to shot and
sentence boundaries against a deterministic, pinned mezzanine encode. Anything that
cannot become an edit decision is dead weight — and a poisoned filter is worse than
dead weight.

**Constraints**: free, fully local, open-source models only. No paid APIs. Accuracy over
speed (overnight batch is fine). Dev on a 16 GB MacBook Air; Linux prod trajectory.

## Status

Round 1 complete + first round-2 features (2026-07-12): six-stage pipeline built,
adversarially reviewed, verified end-to-end on a 3-video corpus (talking-head / skate
montage with vocal music bed / no-dialogue ambience). Word timestamps forced-aligned
(criterion 4: 24 defects → 0). Three embedding spaces. Acceptance suite: **14 passed /
3 xfailed (pre-registered known gaps) / exit 0** across seven query classes. Full design
rationale, verified-fact ledger, and research notes live in [PLAN.md](PLAN.md).

## Pipeline

```
ingest → shots → frames → transcribe → caption → embed
```

| Stage | What it does | Backed by |
|---|---|---|
| ingest | deterministic CFR mezzanine (pinned libx264), fps whitelist, HDR refusal, audio-clock check | ffmpeg |
| shots | frame-exact shot boundaries | PySceneDetect |
| frames | keyframes every 3 s, dHash+person-state dedup, camera-motion labels | OpenCV, torchvision |
| transcribe | sentence segments + forced-aligned word timestamps, hallucination guards, music-authoritative gate (music/noise/silence regions, advisory lyrics) | faster-whisper large-v3, inaSpeechSegmenter, wav2vec2 CTC |
| caption | one advisory VLM caption per shot (typed, guarded, per-shot failure isolation) | Qwen2.5-VL via Ollama |
| embed | image + text + audio spaces in one SQLite BLOB column, scoped by model/dim | SigLIP 2, bge-small, CLAP |

Everything lands in a single SQLite database. The pipeline is resumable (jobs table),
idempotent per stage, and single-process by contract.

## Read contracts

- `vindex search "query"` — ranked top-k **discovery** (hybrid RRF over three embedding
  spaces), each hit carrying a cut-actionable range.
- `vindex list --video ID --kind K --filter k=v --range a:b` — the **exhaustive**
  contract: exact full scan for "all X" tasks. Destructive edits must be driven by
  `list`, never by top-k search.

## Install

Requires **Python 3.12** (the music extra's dependency set pins the upper bound) and
two system tools: **ffmpeg** (all stages) and **[Ollama](https://ollama.com)** with
`qwen2.5vl:3b` pulled (captions only — skippable with `--no-captions`).

```bash
# as a user (no clone):
pip install "vindex[all] @ git+https://github.com/nikhilSingh1910/vindex.git"

# or per-stage extras instead of [all]:
#   embed, frames, shots, music, align, urls (YouTube ingest via yt-dlp)
```

First runs download the models (SigLIP 2, whisper large-v3, CLAP, bge, wav2vec2 —
roughly 10 GB cached under ~/.cache/huggingface).

## Development

```bash
uv sync --extra embed --extra frames --extra shots --extra music --extra align
uv run vindex index <file-or-youtube-url> [--id NAME] [--no-captions]
uv run vindex search "crowd applauding and cheering" --video jobs2005 --kind audio_window
uv run vindex list --video credits --kind lyrics --range 70:85
uv run python acceptance/run_criterion3.py   # pre-registered acceptance suite
```

Captions need a local [Ollama](https://ollama.com) with `qwen2.5vl:3b` pulled.
Low-disk machines: run Ollama with `OLLAMA_KEEP_ALIVE=0` so the VLM unloads between
caption requests (~35% slower, flat memory ceiling — measured on a 16 GB M4 Air).

---

# Learnings

Everything below was learned by building, measuring, or being bitten — the receipts are
in [PLAN.md](PLAN.md). Two sessions of adversarial review (independent reviewers per
diff) produced ~46 findings; every "verified" below means verified by execution.

## What worked

**Content-aligned segmentation over fixed chunks.** Shots, sentences, and speech regions
as the units — never a fixed clock grid. A cut at a fixed boundary lands mid-shot and
mid-word; our acceptance metric (top-3 hit, IoU ≥ 0.5) was only reachable at all because
hits refine to shot/sentence bounds. The one place we DO tile on a clock (CLAP's native
10 s audio windows) is documented as coarse discovery granularity, not cut truth.

**Cut-actionability as a product requirement, discovered by measurement.** Early results
were *correct but uncuttable* — a 45 s window "containing" an 8 s answer can never reach
IoU 0.5. That failing metric forced `cut_range`: speech-window hits refine to their
best-matching sentence (IoU 0.98–1.0 observed), frame hits expand to their enclosing
shot. The metric designed to catch this gap caught it. That's what pre-registered
acceptance is for.

**Pre-registered queries, ground truth from the media, never from the index.** Queries
were fixed before searching; ground truth came from direct frame inspection (a full
31-shot census for one video), transcripts cross-verified by an *independent* second ASR
model, and upstream-classifier anchors (applause = segmenter noise regions that start
0.3 s after "Thank you all very much"). When a query was true for 37% of the timeline we
narrowed the query rather than accept an unscorable test.

**expected_fail semantics.** Two known capability gaps (action queries, lyric search) are
pre-registered XFAILs — the suite's exit code means *regression*, and an unexpected pass
demands a spec update. Steady-state green with documented gaps beats a permanently red
suite nobody reads.

**Fail loud, never silently degrade.** The fps whitelist hard-rejected a real 2005-era
15 fps video — the rejection was correct and the whitelist grew deliberately. HDR input
is refused outright until a tonemap path exists (the naive guard was *inverted* — it
would have silently let HDR through on capable machines). Unknown VLM reply shapes are
rejected per-shot, never coerced into the index.

**Advisory vs authoritative, with provenance.** Model opinions (captions, lyrics, motion
labels, person counts) carry `advisory` flags plus model/version provenance and never
gate primary artifacts. When whisper hallucinated subtitle credits over ocean surf
("Teksting av Nicolai Winther" — a training-data ghost), the contract held: it landed in
advisory lyrics, zero contamination of speech, silence, or search.

**Resumable-by-contract pipeline.** One jobs row per (video, stage); stages replace their
own outputs atomically (delete + insert in one transaction) and re-mark their consumers
pending (shots→frames→caption→embed cascade). This survived two laptop shutdowns, a
mid-run dependency crash, and let single stages re-run after every model upgrade.

**Two read contracts.** Top-k search for discovery; exhaustive `list` for "all X" tasks.
A destructive "remove all X" edit driven by top-k would silently miss instances — the
split exists so an agent can't make that mistake.

**One BLOB column, many spaces.** Embeddings store `model_name` + `dim` per row; adding
the third (audio) space needed zero schema change, and a query-encoder/stored-model
mismatch raises loudly instead of KNN-ing against nothing.

**The music gate.** Audio classification is authoritative over VAD: sung vocals are not
speech. On the skate montage it recovered 113 timestamped lyric lines from the music bed
and cost **zero** genuine speech (its one excluded segment was itself a hallucination —
verified). Silence is the complement of (padded speech ∪ music), so a music bed is never
offered as a safe cut point.

**Minimal-dependency choices, twice.** Word alignment via `torchaudio.functional.
forced_align` instead of the whisperx package (which drags pyannote-audio); reading our
own pinned-format WAVs with stdlib `wave` instead of adding torchcodec. Both chosen after
the heavier option failed or bloated; both survived review.

**Adversarial review as a stage, not a gesture.** Every substantive diff got two
independent reviewers (correctness lens; hostile-media/ops lens) who verify claims
against installed source and live data before reporting. They caught, pre-ship: an
inverted HDR guard, caption failure destroying prior captions before raising, a hyphen
mapping to CTC-blank that silently disabled alignment on ordinary English, a partial-WAV
cache that would permanently amputate audio coverage (both failure modes live-reproduced),
`padding=True` silently overriding a checkpoint's trained pad mode, and a negative-control
check that only ever tested one embedding space. None of these were visible in tests that
passed.

**Machine guardrails.** Thread caps + `nice` + `caffeinate`, watchdogs that kill the
pipeline below 8% free memory or 1.5 GB free disk (the pipeline is resumable; the laptop
is not), models loaded sequentially and released after use, download/dependency
preflights *before* the expensive compute they could invalidate.

## What didn't work (and what it taught)

**Whisper word timestamps drift.** 24 words with `start >= end` on one real video.
Fixed by CTC forced alignment (24 → 0, all 330 segments, zero fallbacks). Whisper's
cross-attention times are estimates, not measurements.

**Whisper hallucinates over non-speech.** Silence guards (`no_speech_prob`, `avg_logprob`)
are mandatory on every decode path — our lyrics pass initially lacked them and inherited
the subtitle-credit ghost. Any whisper output without guards is a poisoned-filter risk.

**VLM captions cannot label.** Caption-grep recall on known-positive shots was **1/3** —
the VLM described a camcorder close-up without ever saying "camera". Captions are a
recall *aid*; visual verification of every candidate plus a negative audit is the floor
for ground-truth work. (The audit caught a miss the grep never surfaced.)

**Single-frame embeddings can't see actions.** A wallride is indexed, captioned, and
embedded — and unrankable, because SigLIP sees per-frame appearance and the action lives
in motion. Pre-registered as the standing action-class XFAIL; the fix direction is
temporal grounding / video-native embeddings, not more frames.

**CLAP distances are not confidence.** Measured: a nonsense query's nearest audio window
(0.334 cosine distance) scores *closer* than the loosest true hit (0.676). Cross-modal
distances don't calibrate across queries — the negative-floor guard exempts the audio
space (documented in the scorer), and raw audio `dist` in output must never be read as
confidence. Rank within a query is meaningful; the number is not.

**Acoustically homogeneous content can't be discriminatively tested.** "A man singing
over music" against a video that is ~90 windows of one song measures labeling luck, not
retrieval — the verified window ranked 19/93 among its own true siblings. Honest xfail;
the discriminative audio test (applause vs 14 min of speech) passed at rank 1.

**RRF with kind-disjoint spaces has a structural quirk.** Fused top-3 is always the three
per-space rank-1s (identical RRF score, admit-order tiebreak) — a space's rank-2 result
can never appear. Fine at three spaces; needs per-space k or weighted fusion beyond that.

**`uv sync` prunes what you didn't declare.** Two runtime deps (sentence-transformers,
scenedetect) were installed-but-undeclared; the first honest sync removed them and a
stage that "always worked" crashed at import. Every stage import must be declared, and
the README's own sync line must include every extra — ours didn't, and would have
uninstalled the feature it documented.

**The modern ML stack shifts under you.** TensorFlow 2.21 declares CUDA deps with no
platform markers (breaks macOS resolution; fixed with uv `required-environments`).
transformers 5.x renamed `audios=`→`audio=`, wraps features in output objects
(`pooler_output` unwrap needed for SigLIP 2 *and* CLAP), and `padding=True` silently
replaces a checkpoint's trained `repeatpad` with zero-padding — off-distribution
embeddings with no error. torchaudio 2.11 delegated its own `load()` to a package we
didn't need. Pin, verify empirically, and assert output dims.

**Nondeterminism hides in preprocessing defaults.** CLAP's `rand_trunc` crops >10 s
inputs at an *unseeded* random offset — re-embedding the same video produced different
vectors. Determinism is a property you must check per-component, not assume.

**Partial files poison caches forever.** An interrupted ffmpeg extract leaves a file that
`exists()` trusts on every later run — one mode crash-loops, the other silently truncates
1200 s of audio to 49 s *and replaces previously good coverage*. Temp + `os.replace`,
always, for anything cached by existence.

**Ops reality on a 16 GB fanless laptop.** The two mystery shutdowns were swap storms
against a ~99% full disk (macOS grew 1 GB swapfiles every ~6 minutes under whisper + TF).
Memory scales with audio length in more places than you expect (wav2vec2: ~22 MB/s of
audio; full-WAV float32 materialization is ~5× file size transient) — cap slice lengths,
stream where possible, and treat "file size" and "memory budget" as different numbers.

**YouTube is a hostile source.** Mid-stream 403s at 8.5%, shared staging directories
silently ingesting the *wrong video* under a video_id, datacenter-IP blocks. Per-source
staging keyed by URL hash, full-invocation retries, format caps (an uncapped `bv*` pulls
4K HDR AV1 masters), and files-as-canonical-input for anything beyond a dev laptop.

**Metadata lies; measure instead.** ffprobe returned `color_transfer=None` on a real
10-bit clip (transfer-string matching alone can't gate HDR — bit depth must be part of
the rule). PySceneDetect *unions* multiple detectors' cuts (no voting — pairing detectors
adds phantom 1–2 frame scenes). cv2 decodes 1–2 fewer frames than ffprobe counts. dHash
matches a person-left-the-frame to the empty background. Every one of these was found by
testing against real media, not by reading docs.

**License traps in the "open" ecosystem.** CrisperWhisper: CC-BY-NC. insightface's
pretrained weights: non-commercial. edit-mind: proprietary with a free tier and a
no-competing-products clause. Apache/MIT-verified alternatives existed for all three —
check the *weights* license, not just the repo badge.

## What we'd tell someone building this

1. Decide what the index is *for* and make the acceptance metric measure exactly that.
   Ours is "can an agent cut on this" — that single decision drove the mezzanine, the
   read contracts, and every refinement.
2. Ground truth is the hard part. Budget for it, protect its independence, and label
   *all* occurrences or the metric punishes the labeler instead of the index.
3. Guards and provenance are the product. The models are commodities; the machinery that
   keeps their mistakes out of the cut is not.
4. Review adversarially with independent lenses, and make reviewers verify against
   installed source and live data. "Plausible" findings are cheap; verified ones change
   the ship decision.
5. Run the failure paths. Kill the process mid-write. Feed it a 2005 video, an HDR
   master, a music bed, silence. Every one of those found something the happy path hid.

## License

Apache-2.0 — see [LICENSE](LICENSE).

Built with [Claude Code](https://claude.com/claude-code).
