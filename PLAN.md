# Video Indexer — Round 1 Plan (v5, updated from implementation)

> **v2** (⚑): first gauntlet — 3 reviewers → 10 findings debated → 6 upheld, 4 modified.
> **v3** (⚑2): second gauntlet on v2 — 2 independent fact-checkers + 4 red-team lenses
> (editing-agent simulation, hostile real-world media, resource audit, internal consistency)
> → 21 raw → 12 debated → 3 upheld, 9 modified, 0 rejected. Two v2 "facts" were refuted by
> experiment (audio clock skew; VLM runtime memory) and are corrected below.
> **v4** (⚑3): stages 1 (ingest) + 4 (transcribe) BUILT and adversarially code-reviewed
> (2 reviewers → 10 findings → independent verification → 8 confirmed incl. 2 blockers,
> 1 rejected; all confirmed fixes applied and re-verified by execution). This version folds
> the implementation learnings back into the spec — see "Implementation status".
> **v5** (2026-07-12, third session): round-1 corpus run COMPLETE — all three videos
> indexed, music gate enabled corpus-wide, criterion 3 5/5 before and after re-embed.
> New facts: TF 2.21 CUDA-marker fix (required-environments), undeclared-deps lesson
> (sentence-transformers/scenedetect), whisper subtitle-credit hallucination in the lyrics
> pass (round-2 guard item), and the dev-Mac shutdown post-mortem + run guardrails.

## End goal (the north star)

Index videos so that **AI agents can later edit those videos**. The index is not a generic
search index: every artifact must be **timestamp-anchored and cut-actionable**. An editing
agent must go from a query ("the drone shot of the coastline", "where she explains pricing")
to exact time ranges and frame numbers it can cut, reorder, and overlay. Anything that
cannot become an edit decision is dead weight — and a poisoned filter is worse than dead
weight.

## Constraints

- **Cost**: free — fully local, open-source models only. No paid APIs.
- **Quality**: accuracy is the priority.
- **Speed**: explicitly compromisable. Overnight batch is acceptable.
- **Dev hardware**: M4 MacBook Air, 16 GB RAM (stages run sequentially, never concurrently).
- **Prod trajectory**: staging/prod on Linux servers later. All runtime backends must be
  cross-platform or trivially swappable.
- **Round-1 corpus**: 10–20 min YouTube videos: one talking-head, one action/montage clip
  **with a vocal-music bed**, one scenery/ambience clip (no dialogue). ⚑2 At least one clip
  (or a checked-in PQ fixture) must be **genuinely HDR** so the tonemap path is exercised.

## Architecture

Single installable Python package **`vindex`** (managed with `uv`). CLI:

```
vindex index <file>                  # canonical, CI-tested entry point
vindex index <youtube-url>           # dev-laptop convenience wrapper (yt-dlp)
vindex search <query> [--kind frame|speech_window|caption] [--video ID]   # ⚑3 embedded kinds only; other kinds would be silently-empty results, so the CLI rejects them
vindex list --video ID [--kind ...] [--filter key=value ...] [--range start_s:end_s]   # ⚑2
```

⚑2 **Two read contracts**: `search` is ranked top-k for *discovery*; `list` is the
**exhaustive** contract for "all X" and time-range/context tasks — an exact full scan
(no KNN, no k) ordered by `t_start_s`, via one typed storage function
`list_segments(video_id, kind=None, payload_filters=None, time_range=None)` returning the
same typed Pydantic segment models as search (time ranges match by interval overlap).
Recall on payload filters is bounded by the advisory VLM label accuracy documented below.
A destructive "remove all X" edit must be driven by `list`, never by top-k search.

No API server in round 1; the future FastAPI layer **wraps this existing typed read
surface** (search + list), it does not introduce new read semantics.

**Batch stage pipeline, resumable — single-process by contract.** ⚑3 A `jobs` table holds
one row per (video, stage) with `pending / running / done / failed` + error text. Stages
write outputs then mark done; a crash resumes; any stage can be re-run alone after a model
upgrade (stages delete their own prior output kinds before inserting, so re-runs are
idempotent — verified by execution). The `running` state is **not a cross-process lock**:
two concurrent pipeline processes on one DB were observed to produce duplicate segment rows.
One pipeline process per index DB is a documented round-1 constraint, enforced by operator
discipline, not code. No Temporal/Conductor — revisit (with real locking) only when multiple
machines share a queue.

**Portability discipline**:
- Every model behind a small protocol: `ASRBackend`, `Captioner`, `Embedder`,
  `ShotDetector`, ⚑2 `PersonDetector` — selected by config, never `platform.system()`.
- All paths, model names, devices, thresholds in one config object.
- `linux/amd64` Dockerfile from week 1; CI and Docker smoke tests run on **pre-downloaded
  fixture media**, never network downloads.

## Pipeline stages

### 1. ingest ⚑⚑2⚑3 (media identity + clock alignment + color guard + reproducibility) — BUILT

Produces the **canonical edit master** (mezzanine) that every frame number references.

- **File input is canonical.** URL ingest via yt-dlp is a dev convenience only: YouTube
  blocks datacenter IPs and no free/local workaround exists. Staging/prod receives files.
- **Pinned rational frame rate.** ⚑2⚑3 Probe `avg_frame_rate`, snap to the nearest rational
  in the whitelist `{12, 15, 24000/1001, 24, 25, 30000/1001, 30, 48, 50, 60000/1001, 60}`
  within 0.2% relative tolerance; **hard-fail with the probed value named** if nothing
  matches. Encode with `-fps_mode cfr -r <num>/<den>`. (Without explicit `-r`, ffmpeg
  derives the rate from a probe that varies across builds.) ⚑3 The 12/15/48 entries were
  added after the fail-loud rejection fired on a real legacy upload — "Me at the zoo"
  (2005) is 15 fps; early-YouTube corpus content genuinely uses low rates.
- **Declared deterministic mezzanine encode.** ⚑2 `libx264 CRF 16, preset slow,
  -pix_fmt yuv420p` (pinned unconditionally), `-threads 1` (or a fixed N recorded in the
  videos row); ffmpeg/libx264 versions pinned by the week-1 container image. Retained and
  hashed — the file all `frame_start`/`frame_end` reference. Audio codec pinned
  (`-c:a aac -b:a 192k`) for reproducibility (not as an alignment measure — see below).
- **Color guard — broadened detection, unconditional refusal until tonemap is wired.** ⚑3
  (Code review found the naive form of this guard *inverted*: gating only on "ffmpeg lacks
  zscale" would let HDR through silently on any tonemap-capable machine, because no tonemap
  filter was in the encode command. And live testing showed ffprobe reporting
  `color_transfer=None` on a real 10-bit clip — transfer-string matching alone is
  insufficient.) The implemented rule: a source **needs color normalization** if transfer is
  PQ/HLG, OR primaries are BT.2020 (any variant), OR transfer is unknown/absent on a
  ≥10-bit `pix_fmt`. Any such source is **refused loudly** — regardless of ffmpeg
  capability — until the zscale tonemap chain
  (`zscale=t=linear:npl=100,tonemap=hable,zscale=p=bt709:t=bt709:m=bt709:r=tv,format=yuv420p`,
  output tagged bt709) is actually implemented in the encode, at which point the mezzanine
  becomes a documented **SDR BT.709 edit master** (pixels change; timing never does) and
  `color_normalized` is set truthfully. Source `color_transfer`/`color_primaries`/
  `pix_fmt`/`bit_depth` stored as provenance.
- **Clock alignment — fail-loud policy (option A), implemented.** ⚑3 (Experimentally
  verified in review round 2: a DASH merge with nonzero audio `start_time` survives the
  mezzanine encode and silently skews a naive WAV extraction by ~150 ms — past our own
  120 ms threshold — while criterion 4's waveform measurement, sharing the same WAV, reads
  zero.) As built:
  - After encode, ffprobe asserts video and audio `start_time` agree within one AAC frame
    (~21 ms); beyond that, **ingest raises** — we do NOT persist-and-apply the offset,
    because the WAV re-pinning below would make that a double-correction.
  - The 16 kHz mono ASR WAV is extracted from the mezzanine with
    `-af aresample=async=1:first_pts=0`, re-pinning WAV t=0 to media t=0. This alignment is
    assumed from filter semantics (demonstrated once in round 2), **not runtime-verified
    per video** — the skew ceiling is what bounds our trust; criterion 4's external-clock
    check is the per-corpus verification.
  - `audio_offset_s` is stored as measured provenance only; nothing downstream applies it.
- **duration_s is the video-timeline duration** ⚑3: `frame_count × fps_den / fps_num` —
  frame-exact and reproducible — NOT the container duration, which includes any AAC audio
  tail past the last video frame and was observed pushing frame indices out of range.
- A `videos` table row records both identities (see Storage).

### 2. shots ⚑3 — BUILT, adversarially reviewed (5 confirmed findings fixed)
- **PySceneDetect `AdaptiveDetector` — alone, not paired with ContentDetector.** ⚑3 The
  v2/v3 prescription "ContentDetector + AdaptiveDetector" was wrong: review verified
  against PySceneDetect 0.7 source that SceneManager **unions** multiple detectors' cut
  lists (no filtering/voting exists), so the pair strictly adds fast-motion false
  positives. AdaptiveDetector subsumes ContentDetector's hard-cut detection while its
  rolling-average threshold suppresses motion spikes.
- Runs on the mezzanine; converts exclusive frame ends to the inclusive convention;
  derives seconds FROM frames (detector works in frame space; both domains consistent by
  construction); emits one whole-video shot when no boundaries are detected.
- **Guards** ⚑3: scenes shorter than `min_shot_frames` (default 8) merge into a neighbor
  (phantom micro-shot guard — near-coincident cuts would otherwise mint 1–2-frame shots
  that stage 3 would dutifully keyframe/caption/embed); the detector's decoded frame count
  is reconciled against ingest's authoritative count and the stage **refuses** beyond a
  small mismatch, so boundaries always share the index timebase.
- Output: `t_start_s`, `t_end_s`, `frame_start`, `frame_end`, `detector`, `score`.
- Weak on gradual transitions; accepted for round 1. **Quantitative promotion trigger**:
  TransNetV2 is promoted if cut-detection F1 on the labeled montage clip falls below the
  criterion-6 bar. Note ⚑2: an unedited single-take talking-head video legitimately yields
  shots minutes long — within-shot visual state is covered by per-keyframe rows (stage 3),
  never by stretching one caption across the span.

### 3. frames ⚑⚑2 (persistence, shot-scoped dedup, person detection, deterministic motion)
- Candidate keyframes: shot midpoint + one frame every ~3 s within long shots.
- **Person detection on every candidate frame (before dedup)** ⚑2: a deterministic CPU
  detector (torchvision `ssdlite320_mobilenet_v3_large`, COCO weights, BSD license) behind
  the `PersonDetector` protocol emits `person_count`, max person bbox-area fraction, and
  detector name+version into each keyframe row's payload. This is the structured
  person-visibility signal ("speaker off-screen") — sampled every ~3 s, not once per shot.
- **Shot-scoped perceptual-hash dedup** ⚑2: dHash via `imagehash`; **within each shot**,
  drop candidates within a small Hamming distance of the *shot's* last kept frame — the
  comparison **resets at every shot boundary**, so cross-shot near-duplicates are never
  dropped. **The representative frame (kept frame nearest the shot midpoint) is exempt from
  dedup** — every shot is structurally guaranteed at least one persisted frame, one caption
  row, and one image embedding.
- **Kept keyframes are persisted** as image files in a per-video directory; every
  frame-derived segment stores `frame_path` in its payload; each shot's caption row
  references its representative frame's path. Enables caption/embed re-runs without
  re-decoding video, and lets an agent visually verify a candidate before cutting.
- **Camera motion computed deterministically here** — never asked of the VLM (a single
  still cannot encode motion; a VLM will confabulate it). Sparse optical flow / global
  homography (OpenCV, CPU) between 2–3 separated frames per shot →
  `camera_motion` ∈ {static, pan, tilt, zoom, complex}, with provenance.
- **Validity rule for frame-derived attributes** ⚑2: all frame-derived structured fields
  (person-detector fields AND per-shot VLM caption fields) are valid only at their source
  frame's timestamp ± half the keyframe interval (~±1.5 s). Query filters over wider ranges
  must intersect the per-keyframe rows, never the per-shot caption row alone.
- Two-tier policy: **embed many, caption one** — all kept frames get embeddings; exactly
  one representative frame per shot goes to the VLM.

### 4. transcribe ⚑⚑2⚑3 (music-authoritative gating; persisted audio map; verbatim caveat) — BUILT (music gate pending)
- **faster-whisper** (CTranslate2), **Whisper large-v3 int8** — canonical engine, identical
  on M4 CPU and Linux CPU/GPU. (mlx-whisper: dev-only backend later, never the reference.)
- `word_timestamps=True`, `vad_filter=True` (bundled Silero VAD — currently v6; re-validate
  VAD assumptions against the actually-bundled model on the round-1 corpus ⚑2).
- **Audio classification is authoritative, VAD is not** ⚑2 (corrected fact: Silero is
  *bidirectionally unreliable* on sung vocals — it may score singing as speech OR return
  no speech at all on clear singing; see facts register):
  - **inaSpeechSegmenter** (CPU, free) runs over the full WAV first; its speech/music/noise
    intervals are **persisted as first-class segments**: `kind='music'`, `kind='noise'`
    (advisory — no published precision for the noise class), with model version + params as
    provenance.
  - `kind='silence'` rows are the complement of **(speech OR music)** — never just
    VAD-complement, so vocal or instrumental music is never offered as a safe cut point.
  - **Lyrics do not depend on Silero firing**: faster-whisper runs with `vad_filter=False`
    explicitly over segmenter-detected music regions; resulting segments are persisted
    directly as `kind='lyrics'` (timestamp-anchored, text + `music_overlap` provenance;
    never returned as spoken content; lyric *retrieval* deferred — see Deferred).
  - Whisper segments overlapping music are excluded from `has_speech` and from
    `kind='speech'` embedding windows. Whisper segments majority-covered by *noise*
    intervals get a `noise_overlap` flag — reduced confidence, excluded as cut anchors
    until validated (flag and demote, never drop).
  - *Silence/ambience guards* (unchanged): drop segments with high `no_speech_prob` or very
    poor `avg_logprob`; per-video `has_speech` flag.
- **VAD speech map persisted — padded tiling rule** ⚑3: speech regions become
  `kind='speech_region'` rows padded outward, **then re-merged** (padding can bridge
  sub-2×pad gaps between utterances, which must not be offered as cuts), and
  `kind='silence'` is the complement of the **padded** set — code review found that
  complementing the *unpadded* set let boundary frames be labeled both "protected" and
  "safe-to-cut". Target state adds music: silence = complement of (padded speech OR music).
  VAD params + model version in payload. "Where can I cut without clipping speech" is
  answered directly from the index.
- **The transcript is non-verbatim** ⚑2: Whisper-family ASR systematically omits fillers,
  stutters, and false starts (intended-speech style). This caveat is part of the
  agent-facing contract — word rows are **not** token-complete coverage of the audio.
  Mitigation: any `speech_region` whose overlapping-word coverage ratio falls below a
  threshold gets `has_untranscribed_speech=true` in its payload and is surfaced as a
  candidate region for agent audio verification. (Mid-utterance fillers without flanking
  pauses: documented out of round-1 scope.)
- Words adjacent to VAD-removed gaps get reduced `alignment_confidence`; excluded as cut
  points until validated.
- Diarization (pyannote) and WhisperX alignment: **deferred**; reserved columns
  (`speaker`, `alignment_confidence`); measured promotion trigger in criterion 4.

### 5. caption ⚑⚑2 (memory budget corrected; hardening extended)
- **Qwen2.5-VL 7B q4_K_M via Ollama**. ⚑2 **6 GB is the GGUF *download* size, not the
  budget**: documented runtime footprint is **~15 GB RSS at 8k context**
  (ollama/ollama#14312) because the GGUF declares a 125K context window and Ollama
  over-allocates for this model. On a 16 GB machine this stage is feasible **only with
  explicit memory controls**:
  - Pin `num_ctx` ≈ 4096 via a **checked-in Modelfile**.
  - **Downscale each representative frame to ≤1024 px longest side** before sending
    (Ollama's internal 2 MP cap does not catch 1080p frames, which otherwise cost ~2,600
    visual tokens).
  - Pin and validate the Ollama version (compute-graph estimation regression since 0.13.4,
    ollama/ollama#13687).
  - **Record actual peak RSS on the first corpus run**; fail stage setup if the pinned
    config is not in effect. Documented fallback: `qwen2.5vl:3b` (3.2 GB GGUF).
- One representative frame per shot (guaranteed to exist by shot-scoped dedup).
- Structured + freeform output: `shot_type` (CU/MS/WS), `people_count`, `setting`,
  `on_screen_text` (one frame per shot — burned-in subtitles are ASR's job; multi-frame OCR
  deferred), `objects[]`, plus freeform `description` for text embedding. **No
  `camera_motion`** (optical flow owns it).
- Structured fields are **advisory** (estimated 10–20% attribute error) — filters treat
  them as hints, and the ±1.5 s validity rule from stage 3 applies.
- Ollama hardening: per-request timeout (~120 s), `num_predict` cap, one retry then mark
  the jobs row failed — never hang; `OLLAMA_FLASH_ATTENTION` on.
- Stage is last and individually skippable (`--no-captions`).

### 6. embed
- **Images**: SigLIP `so400m-patch14-384` — 0.9 B params full model ("400M" names only the
  vision tower); **~1.8 GB fp16; HF checkpoint ships F32 (~3.5 GB) — the Embedder must pass
  `torch_dtype=torch.float16` explicitly** and verify fp16 numerics on MPS. Check for
  SigLIP 2 checkpoints at build time. Embeds every kept keyframe (`kind='frame'` rows).
- **Text**: **BAAI/bge-small-en-v1.5** (384-dim, ~130 MB; English-only — see Deferred) on
  (a) sentence-aligned transcript windows ~30–60 s with word-timestamp anchors — chunking
  drives recall more than encoder choice — and (b) caption descriptions.
- Every embedding row stores `model_name` + `dim`; re-embedding is a versioned re-run.

## Storage ⚑⚑2

**SQLite + sqlite-vec**, one file per index.

- **`videos` table — media identity + clock + color provenance** ⚑2⚑3:
  `videos(video_id, source_url, source_hash, media_path, media_hash, fps_num, fps_den,
  frame_count, duration_s, width, height, codec, audio_offset_s, source_color_transfer,
  source_color_primaries, source_pix_fmt, source_bit_depth, color_normalized,
  encode_threads, has_speech)`. `duration_s` is the video-timeline duration
  (`frame_count × fps_den / fps_num`), not the container duration.
  All `frame_start`/`frame_end` in `segments` reference the file at `media_path` (verified
  by `media_hash`) and its rational fps. Source download is provenance; mezzanine is the
  edit master.
- **Core table (abridged sketch — reserved columns like `speaker`/`alignment_confidence`
  live in migrations)** ⚑2:
  `segments(video_id, kind, t_start_s, t_end_s, frame_start, frame_end, model_name, dim,
  payload JSON, embedding BLOB)`.
- **Frame-number convention (normative)** ⚑3 — code review found no convention was declared
  anywhere, making frame columns ambiguous to an editing agent; and the naive `round()`
  mapping both emitted out-of-range indices (frame_count, one past the last frame) and
  rounded starts UP past the frame containing the content. The convention, now implemented
  and unit-verified: an interval `[t_start_s, t_end_s)` maps to the **inclusive** frame
  range `[frame_start, frame_end]`, where `frame_start = floor(t_start × fps)` (the frame
  that CONTAINS the start) and `frame_end = ceil(t_end × fps) − 1` (the last frame the
  interval touches), both **clamped to `[0, frame_count − 1]`** and bounded by the video
  timeline, never the container duration. `frame_start == frame_end` for a single-instant
  segment, matching the `kind='frame'` rule. Seconds columns remain the authoritative
  anchor; frames are the clamped, cut-actionable view. ⚑3 A sub-microframe epsilon is
  applied inside floor/ceil (`floor(t×fps + ε)`, `ceil(t×fps − ε)`): seconds stored as
  `k×fps_den/fps_num` can evaluate to `k − 1e-15` when multiplied back by fps (verified:
  `floor(0.5005 × 29.97…) = 14.999…` → 14 instead of 15), so boundary times would
  otherwise round-trip to the wrong frame.
- **Stage output replacement is atomic** ⚑3: each stage replaces its prior output kinds
  via a single DELETE+INSERT transaction (`replace_segments`) — a crash mid-stage can no
  longer destroy the previous good output without its replacement landing.
- **Segment kinds** ⚑2: `shot`, **`frame`** (one row per kept keyframe; `t_start_s =
  t_end_s`, `frame_start = frame_end`, `frame_path` + person-detector fields in payload,
  SigLIP embedding), `speech` (words + windows), `speech_region`, `silence`, **`music`**,
  **`noise`**, `lyrics`, `caption` — and future kinds are just new values.
- **`embedding` is a plain BLOB** queried with sqlite-vec scalar distance functions
  (`vec_distance_cosine` + ORDER BY), *not* a vec0 virtual table — permits mixed dimensions
  (SigLIP 1152-d, bge 384-d) in one column and maps directly to pgvector `<=>`. Brute-force
  scan is exact and takes milliseconds at round-1 scale.
- All SQL in one `storage/` module behind typed functions (`search_segments`,
  ⚑2 `list_segments`); schema in migration files; boring Postgres-portable SQL.
- Known risk: sqlite-vec pre-1.0. Accepted; exactly two planned implementations.

## Search (what the editing agent consumes)

- **Hybrid with a defined merge rule**: one KNN per embedding space, scoped
  `WHERE model_name = :space`; top-k per space; merged via **Reciprocal Rank Fusion**
  (rank-based — SigLIP and bge scores are incomparable; extends unchanged when CLAP lands).
  Structured filters apply before fusion.
- ⚑2 `list` is the exhaustive counterpart (see Architecture); agent-facing docs state the
  split explicitly, plus the non-verbatim-transcript caveat and the advisory status of VLM
  fields.
- Results are typed segments with seconds AND frame numbers, cut-actionable against the
  declared media master.
- **Edit representation**: OTIO deliberately not in the schema (it models timelines — the
  *output* of editing); its semantics adopted: frame-exact columns, rational fps,
  CFR SDR BT.709 mezzanine. Acceptance: toy OTIO export from index rows.

## Acceptance criteria ⚑⚑2

1. All three corpus videos index end-to-end on the M4 with one command each, resumably
   (kill mid-run, rerun, completes).
2. Scenery video (zero dialogue): zero hallucinated transcript segments; findable via
   visual queries.
3. Ten predefined queries (spoken + visual) return the correct range in top 3 — "correct"
   = IoU ≥ 0.5 with the pre-registered ground-truth range, or start within ±0.5 s.
4. **Measured word timestamps** (talking-head): mark true boundaries on the waveform for
   20 word starts AND 20 word ends; record signed offsets. Promote WhisperX if
   p95 |offset| > 120 ms, any sampled word end audibly clips, or any word row has
   `start >= end`. Cuts placed inside persisted `silence` regions clip zero words.
   ⚑2 **External-clock check** (criterion 4 must not share its clock with the artifact
   under test): one clapboard-style A/V event per corpus video verified against frame
   numbers, or one rendered cut from a silence region audited by ear from the mezzanine.
   ⚑2 Annotation: this criterion measures timing of *emitted* words only; disfluency
   coverage is a documented, untested limitation.
5. Toy OTIO export references `videos.media_path` with stored rational fps and round-trips
   frame-exactly. ⚑2 **Reproducibility, two tiers**: byte-identical `media_hash` required
   when ingest re-runs under the pinned container image + pinned thread count; across
   unpinned builds (M4 native vs Linux), require equality of
   `(fps_num, fps_den, frame_count, duration_s, width, height)` plus a hash over decoded
   frame content. `media_hash` verification gates any cross-machine use of frame numbers.
6. **Shot-boundary accuracy**: hand-label cuts on the montage video (spot-check others);
   cut-detection F1 ≥ 0.9, matches within ±2 frames (thresholds provisional until first
   labeling pass). Below bar → TransNetV2 promotes.
7. **Lyrics gate, two-sided** ⚑2: the montage clip yields (a) zero lyric-as-speech segments
   in spoken-content results, (b) at least one `kind='lyrics'` segment covering the vocal
   bed, and (c) zero `kind='silence'` rows overlapping segmenter-detected music.
8. **Person visibility** ⚑2 (talking-head): hand-label person presence for 20 sampled
   keyframes; detector matches ≥ 18/20; a "speaker off-screen" interval query returns no
   interval contradicted by the labels.

## Deferred (explicitly, with re-entry conditions)

- **WhisperX alignment** — trigger in criterion 4. **TRIGGER FIRED 2026-07-12** on
  jobs2005 (first talking-head corpus video): 24/2297 word rows with start >= end
  (degenerate zero-duration alignment clusters, e.g. three words at t=84.36). Promotion is
  now a queued round-1 item (after the corpus run frees the venv). The other criterion-4
  results on the same video: 0 words overlap persisted silence (2297 x 188 regions — the
  cut-safety contract holds); waveform signed-offset measurement was INCONCLUSIVE because
  RMS-energy onset detection counts applause/crowd noise as speech (offsets railed at the
  search-window cap around ovations) — needs a speech-band-aware detector or manual
  labeling; do NOT read the first-pass numbers as alignment error.
- **pyannote diarization** — when multi-speaker content enters the corpus.
- **Audio-event layer (CLAP / beat analysis)** — round 1.5 ("cut on the beat"); the
  persisted `music`/`noise` map is its foundation.
- **Lyric search** ⚑2 — round 1.5: add `lyrics` to the `--kind` enum and embed lyric text
  (versioned re-run). Re-entry condition: lyric WER spot-check vs hand-labeled lyrics on
  the montage clip (done during the criterion-7 pass) meets an advisory bar; results ship
  flagged advisory.
- **Verbatim/filler-aware ASR** ⚑2 — re-entry: a filler-removal feature is actually
  scheduled AND (license posture permits NC OR a permissively-licensed verbatim ASR
  appears). CrisperWhisper is CC BY-NC 4.0 — fails the free/open constraint today.
- **TransNetV2 shots** — trigger in criterion 6.
- **Multi-frame OCR** — round 1.5 if one-frame-per-shot coverage proves insufficient.
- **Multilingual text embeddings (bge-m3)** — when non-English content enters the corpus.
- **Postgres/pgvector; FastAPI service (wrapping search+list); lazy captioning;
  Temporal-class orchestration** — at server/scale time; no schema changes required.

## Implementation status ⚑3 (as of 2026-07-12)

**Built, code-reviewed, and execution-verified**: package scaffold (`vindex`, Python 3.12
via uv), storage layer (schema + typed functions incl. `list_segments`), jobs table,
pipeline runner, CLI (`index`/`list`/`info`/`search`), **stage 1 ingest**, **stage 2
shots**, **stage 3 frames**, **stage 4 transcribe** (minus the music segmenter — see
below), **stage 5 caption**, **stage 6 embed**, and the **hybrid search module** (RRF over
both embedding spaces). ALL SIX stages run end-to-end on real media with captions enabled
(`ingest → shots → frames → transcribe → caption → embed`) and search is verified live in
both spaces via the CLI, including `--kind` restriction and `--video` scoping.

**Stage-5/search adversarial review (2026-07-12, second session)**: 2 reviewers
(correctness lens; hostile-media/ops lens) → 20 raw findings → 2 blockers + 8 majors
confirmed and FIXED, re-verified by execution (`verify_fixes.py` suite: dead-Ollama runs,
cascade re-embedding, typed filters). The fixes, briefly:
- **Blocker — destroy-before-check**: caption total failure committed `replace_segments`
  (deleting all prior captions + their embeddings) before raising. Raise now precedes the
  replace; verified prior captions survive a dead-Ollama re-run intact.
- **Blocker — missing caption→embed cascade**: captions created after embed was `done`
  (e.g. a `--no-captions` run, or today's real Ollama outage) stayed unembedded and
  permanently invisible to search. Pipeline now has a general `DOWNSTREAM` cascade
  (shots→frames/caption/embed, frames→caption/embed, transcribe→embed, caption→embed):
  any stage about to run re-marks its consumers pending. Verified: `--no-captions` then a
  plain re-run yields an embedded, searchable caption.
- **Caption failure no longer blocks embed**: caption failure is recorded (jobs.error),
  the pipeline warns and continues, so an Ollama outage costs captions, not the whole
  search surface; `vindex index` exits 1 with the failed-stage list. Per-shot failures are
  persisted on the shot rows (`payload.caption_error`, cleared on success) in the same
  transaction as the caption replace.
- **VLM reply hard-typing**: description must be a non-empty string (it is the embedded
  retrieval text — a list here used to crash embed on every later run); people_count
  coerces or becomes None; objects[] stringified; setting/on_screen_text coerced;
  non-dict replies rejected. All failures are per-shot, retried once, then recorded.
  HTTPError bodies are surfaced (404 "model not found" is now actionable). Circuit
  breaker: 5 consecutive failures abort BEFORE touching prior output.
- **`frame_path` is now workdir-relative** (consumers join `cfg.workdir`; absolute paths
  in older DBs still resolve since joining an absolute path is a no-op) — the index
  survives a workdir move. `cv2.imwrite` return is checked. `videos.media_path` is still
  absolute — deferred to the OTIO-export pass.
- **Search read-contract honesty**: unembeddable `--kind` values are rejected (CLI enum
  narrowed to frame/speech_window/caption; `search()` raises on an empty searchable set);
  the query encoder must match the stored embedding space (a silent SigLIP-fallback flip
  at query time read as "no matches"); `list --filter` values parse as JSON so numeric/
  boolean payload filters (`person_count=0`, `is_representative=true`) actually match.

**Ops fact (dev Mac)**: Ollama's Metal shader-compiler XPC connection can wedge after long
idle (`XPC_ERROR_CONNECTION_INVALID`; every request 500s). Fix: restart `ollama serve`.
The caption stage now degrades per the above instead of taking the pipeline down.

**Music gate (stage 4) — BUILT behind `cfg.music_gate` (2026-07-12, second session)**,
execution-tested in an isolated venv on a composite clip (19 s zoo speech + ~70 s of a
commercial vocal track): music region detected [14.3, 89.0]; real lyrics transcribed to
`kind='lyrics'` (advisory, `music_overlap=1.0`, clip_timestamps keeps the global
timeline); spoken speech survived with `music_gate_active=true`; **no silence row
overlaps any music row**; `excluded_as_music` correctly removed a whisper segment the
segmenter classed as music. Known precision trade, observed live: low-energy trailing
speech can be absorbed into an adjacent music block (it lands in lyrics, not speech) —
criterion 7 measures this on the montage clip. Dependency facts: inaSpeechSegmenter 0.8.0
still imports tensorflow/keras AND declares onnxruntime-gpu (no macOS wheels) — pyproject
now carries the working macOS dep set (tensorflow, scikit-image, soundfile, onnxruntime
via darwin marker) plus a uv override restricting onnxruntime-gpu to Linux. Runtime ~8x
realtime on M4 CPU; models (~9 MB) download to the keras cache on first use.
~~Default is music_gate=False until `uv sync --extra music` runs~~ **DONE (2026-07-12,
third session): music_gate=True is the default**; the extra is installed and the whole
corpus was re-transcribed + re-embedded under the gate (results below). The deferred `uv
sync` risk was real, twice over: (a) the lock had resolved **tensorflow 2.21.0, whose CUDA
deps carry no platform markers** — macOS install demanded nonexistent nvidia wheels; fixed
with `tool.uv.required-environments = [darwin/arm64, linux]` in pyproject (uv then forks
resolution per-platform: still TF 2.21, zero nvidia packages on the Mac). (b) `uv sync`
prunes undeclared packages, which exposed that **sentence-transformers (bge runtime) and
scenedetect (stage 2) were installed but never declared** — embed crashed at import on the
first re-run (at the import, before its replace: no rows lost; the failed-job resume
worked as designed). Both are now declared (`embed`/`shots` extras). Lesson: every stage
import must be declared, or the next honest sync breaks a stage that "always worked".

**Acceptance progress (2026-07-12, jobs2005 = first corpus video)**:
- **Criterion 3 (queries)**: 5/5 pre-registered jobs2005 queries pass
  (`acceptance/criterion3_queries.json` + `run_criterion3.py`; coast/credits queries
  pre-registered, ground truth pending their media). Getting here surfaced a real product
  gap the criterion was designed to catch: raw hits were correct but NOT cut-actionable
  (a 45 s speech_window containing an 8 s target can never reach IoU 0.5; a single frame
  can't overlap a range). Search now returns a **cut_range/cut_frames** per result —
  speech_window hits refine to their best-matching constituent speech segment (bge cosine
  over `source_speech_ids` texts; sentence-exact in practice, IoU 0.98-1.0 observed) and
  frame hits expand to their enclosing shot. CLI prints `cut_t`/`cut_frames`.
  Ground-truth lesson recorded: label ALL occurrences (Jobs says "stay hungry" three
  times) and use exact sentence bounds, or the metric punishes the labeler, not the index.
- **Criterion 4 (word timestamps)**: see Deferred — WhisperX trigger FIRED (24 start>=end
  rows); silence-clipping contract PASSED (0/2297 words overlap silence); waveform offset
  measurement inconclusive (applause confounds RMS onset detection).

**Round-1 corpus run — COMPLETE (2026-07-12, third session)**: all three videos indexed
end-to-end and searchable (jobs2005 1051 segments / credits 1199 / coast 300 pre-gate),
then re-transcribed + re-embedded corpus-wide with the music gate on. Third-session facts:
- **Music gate on the corpus** (transcribe jobs reset to pending; caption skipped as done;
  transcribe→embed cascade re-embedded everything):
  - jobs2005: clean regression — 330 speech segments (identical to pre-gate), 0 excluded
    as music, + 5 music / 9 noise regions and 5 lyrics segments (ceremony bed).
  - credits (the motivating clip): **113 lyrics segments** over the vocal track, 16 music
    + 26 noise regions as first-class segments, silence 3 coarse → 18 honest regions;
    speech 2 → 1 with `excluded_as_music: 1` (the documented low-energy-trailing-speech
    precision trade — criterion 7's subject).
  - coast: 0 speech (unchanged), but the segmenter called two ~2 s blips of surf "music"
    and whisper emitted its training-data subtitle-credit hallucination over both
    ("Teksting av Nicolai Winther" — also the source of the bogus language=nn). Contract
    held (advisory lyrics only; nothing entered speech/search/has_speech), but **round-2
    item: the lyrics pass needs the hallucination guards the speech pass has** (and/or a
    min-duration floor on music regions before lyric transcription).
- **Criterion 3 re-verified post-re-embed: 5/5 pass with identical ranks and IoUs** —
  also validates that the sentence-transformers 4.x→5.6.0 bump (forced by the dependency
  fix above) produces an equivalent bge text space. North-star query verified live on
  coast ("aerial drone shot of the coastline" → both spaces rank the aerial shot #1,
  35 s cut-actionable range).
- **Ops: the dev-Mac shutdowns were (best theory) swap storms against a ~99%-full disk**
  — macOS grew 1 GB swapfiles every ~6 min under whisper/TF while the boot volume
  starved. All heavy runs now use: thread caps (OMP=4), `nice -n 10`, `caffeinate -i`,
  PYTHONUNBUFFERED=1, and a watchdog killing the pipeline at <8% free memory or <1.5 GB
  free disk (pipeline is resumable; the laptop is not). Reclaimed ~6 GB of regenerable
  caches (uv/brew/vscode-shipit/go-build) mid-run when swap growth outpaced headroom.
  Unloading the Ollama VLM immediately after caption (`ollama stop <model>`) released
  ~4.6 GB before embed loaded SigLIP — worth keeping as a between-stages hygiene step.
- **Round-2 candidates from the edit-mind study** (repo + maker's Multimodal Weekly 104
  talk, youtube k_aesDa3sFw@21:00; proprietary license bars code reuse — ideas only,
  implemented independently): (a) **CLAP audio embeddings as a third RRF space** — the
  coast clip is pure ambience; "waves crashing" should hit in audio space, not only via
  caption text; fits the existing model_name/dim BLOB scoping unchanged. (b) **EasyOCR
  on-screen-text pass** (cheaper + more reliable than the VLM's on_screen_text field).
  (c) **Face identity clustering with user labeling** as the PersonDetector evolution.
  Contrast kept on record: edit-mind "scenes" are fixed 2–2.5 s chunks (cuts land
  mid-shot/mid-word), fusion is self-described "basic scoring", editing handoff is
  zip-download → manual Premiere/Resolve ("a starting point") — the cut-actionable agent
  contract remains vindex's differentiator, per its own maker's demo. Its face stack
  (VGG-Face @ cosine 0.45, running-mean unknown-cluster centroids) is the documented
  anti-pattern: when faces come, use an ArcFace-class model and clustering over fixed
  embeddings (no drifting centroids).
- **Round-2 candidates from TiViBench (arXiv 2511.13704) + code audit**:
  (d) **negative-control acceptance queries** — pre-registered queries with NO correct
  answer in the corpus; pass = nothing returned above a confidence floor. Directly
  measures poisoned-filter resistance (the north-star risk criterion 3 can't see), plus
  per-space ablation (image-only vs text-only vs RRF) to quantify fusion value.
  (e) **VideoTPO-style test-time critic**: local-LLM pass verifying each top-k candidate
  (caption/transcript vs query) before an editing agent acts on it — training-free
  reranking at the moment of highest blast radius. (f) **source metadata at ingest**:
  persist creation_time/device/GPS from `ffprobe -show_format` tags on the videos row
  (currently dropped; matters at corpus scale). Deliberately NOT adopted from the field
  study: fixed-interval chunking, built-in NL query planning/chat (the agent is the query
  planner; FastAPI stays a typed read surface), precomputed collections, multi-service
  architecture.
- **Adoption shortlist (GitHub sweep, 2026-07-12)** — licenses verified where noted:
  (1) **browser-use/video-use** (MIT, 16.7k★): agentic transcript→EDL→ffmpeg editor with
  NO retrieval indexing and a paid ElevenLabs ASR dependency — vindex supplies exactly
  both; the round-2 flagship is an integration demo (agent query → our cut_frames → their
  EDL/render). (2) **AcademySoftwareFoundation/OpenTimelineIO** (Apache-2.0): the export
  format; WyattBlue/auto-editor (public domain) as EDL/FCPXML mechanics reference.
  (3) **CLAP** for the audio RRF space via transformers (already a dep; LAION or msclap
  MIT checkpoints). (4) **line/lighthouse** (Apache-2.0, maintained, CPU inference):
  temporal-grounding model zoo (query→span) as an advisory `moment` producer — needs a
  compute spike (CLIP+SlowFast+PANNs extraction) before committing. (5) **m-bain/whisperX**
  (BSD) for the already-fired criterion-4 word-timestamp trigger; jianfch/stable-ts (MIT)
  fallback. License traps to avoid: CrisperWhisper (CC-BY-NC), insightface pretrained
  weights (NC), edit-mind (proprietary).
- **Ops: ingest has no `running` job state** (it only writes done/failed at stage end),
  so a re-run over a previously failed ingest shows the stale `failed` row while ffmpeg
  is actively encoding — read the process, not the row. The per-source staging cache
  enables pre-downloading OUTSIDE the pipeline process (yt-dlp sees the complete file and
  skips) — how coast's flaky download was retried without violating the one-process-per-DB
  contract. Coast's original yt-dlp exit-1 was transient (clean retry succeeded hours
  later).

Corpus: coast = pNe25miQ3-M (11 min coastal ambience, no
dialogue), jobs2005 = UF8uR6Z6KLc (15 min Stanford commencement, talking-head), credits =
IQbZFFpK4kE (15 min Vans skate montage, vocal music bed). Ingest hardening from live
failures on this run:
- **Shared-staging contamination (blocker, fired live)**: all sources staged to the same
  `_staging/source.mp4`; after one video's ingest failed, the NEXT video's yt-dlp saw the
  leftover file, skipped its download, and began encoding the wrong video under its own
  video_id. Staging is now per-source (`_staging/<sha256(url)[:16]>/`), which also makes
  an existing complete file a *valid* re-run cache.
- **Disk preflight**: a 99%-full disk killed a 20-minute encode at the `+faststart`
  finalize (which rewrites the whole mezzanine). Ingest now requires ~2x source + 1 GB
  free up front and fails in seconds with a clear message.
- **Download hardening**: yt-dlp format capped at 1080p/avc1-preferred (an uncapped `bv*`
  can pull 4K AV1/HDR masters); YouTube 403'd mid-stream on a re-download (observed at
  8.5%) — the command now retries up to 3 full invocations (fresh URL extraction, resumes
  from `.part`) plus `--retries/--fragment-retries 10`. Stage-1/4 adversarial
code review: 2 reviewers → 10 findings → 8 confirmed (2 blockers: out-of-range `frame_end`;
inverted HDR guard), 1 rejected; all fixes applied and re-verified. Stage-2 adversarial
review: 6 findings → 5 confirmed (detector union, phantom micro-shots, decoded-frame-count
reconciliation, float round-trip epsilon, non-atomic replace), 1 benign; all fixes applied,
regression re-run frame-exact. Stage-3 adversarial review (verifiers ran empirical tests
against real media): 8 findings → 7 confirmed, 1 modified; all fixed:
- **Negative decode-mismatch dead-end**: shots tolerated cv2 decoding ≤2 fewer frames than
  ffprobe but still pinned the last shot to `frame_count−1` — an index cv2 provably cannot
  read (verified `ok=False`), permanently failing stage 3 on a video stage 2 accepted.
  Fixed: all indices clamp to `min(decoded, ffprobe)`; empty-scene-list branch now records
  the mismatch too. cv2 seek exactness itself was verified: 229 seeks, 0 mismatches.
- **Orphaned JPEGs on re-run** (reproduced empirically): rows were replaced atomically but
  files accumulated across config changes. Fixed: post-replace prune of unreferenced JPEGs.
- **Zero-width frame rows unreachable** by half-open time_range queries (the t=0 frame
  never matched any range — verified). Fixed: frame rows span one frame duration,
  exclusive end, same convention as shots.
- **dHash drops person-state changes** (verified: a person leaving frame can dhash-match
  the empty background): dedup now requires person-state equality in addition to hash
  proximity — detection runs on every candidate, before the drop decision, as PLAN said.
- **Featureless pairs asserted static**: now measured by pixel difference (static if
  near-identical, complex if content moved untrackably), with **majority voting among
  moving labels** so one featureless sample can't veto a clean pan ([pan, complex, pan] →
  pan; verified on the synthetic pan clip).
- **Shot annotations + frame rows now commit in one transaction**; **shots re-run
  cascade-deletes frame rows** (shot_index join can never go stale) and the pipeline
  re-marks frames pending whenever shots will run.

**Embed stage facts** ⚑3: SigLIP 2 (`google/siglip2-so400m-patch14-384`) loads via current
transformers and is the drop-in successor as planned (dim 1152); `get_image_features`
returns `BaseModelOutputWithPooling` under SigLIP 2 (code handles tensor and pooled forms);
**fp16-on-MPS numerics verified** — cosine ≥ 0.99999 against fp32-CPU reference on real
keyframes (the plan's explicit check). Text: bge-small (384-dim) windows pack Whisper
segments (sentence-aligned) to ~45 s, never splitting mid-segment. Both spaces coexist in
the one BLOB column with `model_name`/`dim` scoping, as designed.

**Real-media validation (first real clip, "Me at the zoo", 19 s / 15 fps / 320×240)** ⚑3:
transcript came back verbatim-correct with plausible word timestamps; `has_speech=True`,
language=en; speech/silence tiling exact (no overlap, no gap); frame math spot-checked
(0.94 s → frame 14 at 15 fps; final silence ends at frame_count−1); shots correctly
returned one continuous shot. On the 3-scene synthetic clip, shot boundaries were
**frame-exact against ground truth** (cuts at 90/180 → [0,89],[90,179],[180,269]).

**Verified by execution** (synthetic 29.97 fps clip, 299 frames, 440 Hz tone):
- fps snap 29.97 → 30000/1001; stored `frame_count` = decoded count exactly; whitelist
  rejection of 48 fps works.
- Frame convention unit-verified: full-span silence → frames [0, 298]; [1 s, 2 s] →
  [29, 59]; instants → equal frames; past-EOF clamps.
- Non-speech path: `has_speech=False`, zero hallucinated segments — guards held even though
  Whisper's language autodetect confidently returned a bogus language code on a pure tone
  (do not trust the `language` field as a speech signal).
- Broadened color guard fired end-to-end on a real 10-bit clip whose `color_transfer` came
  back None — the two-string transfer check would have missed it.
- Idempotent re-runs (within one process); interrupted stage resumes from `running` ≠ `done`.

**Not yet exercised**: real YouTube downloads, long-form word-timestamp validation on a
talking-head clip (criterion 4's measured pass), the music/speech segmenter
(`kind='music'`/`'lyrics'` paths — code honestly records `music_gate_active=False` in
speech payloads until wired), multi-shot caption runs (the zoo clip has one shot — the
per-shot failure tolerance and circuit breaker are execution-tested only via single-shot
and simulated paths), all acceptance criteria (need the real round-1 corpus).

**Dev-environment facts** (portability-relevant):
- Homebrew's slim `ffmpeg` (8.1.2) ships with **neither libzimg (`zscale`) nor libplacebo**
  — facts register #14 confirmed live; HDR corpus work needs `brew install ffmpeg-full`
  (keg-only). Until the tonemap chain is implemented, ingest refuses color-abnormal sources
  unconditionally.
- **Python pinned to 3.12**: 3.14 (Homebrew default) lacks wheels for the ML stack
  (ctranslate2/torch).
- Whisper large-v3 download is ~3 GB (HF cache); first transcribe run pays it once.

## Facts register (checkable claims; ⚑2 = corrected/added in round 2, ⚑3 = from implementation)

1. faster-whisper runs Whisper large-v3 int8 in ~3–4 GB RAM, CPU-only on Apple Silicon
   (CTranslate2 has no MPS backend), with `word_timestamps=True`, `vad_filter=True`, and a
   callable `get_speech_timestamps`.
2. ⚑2 **Corrected**: Whisper hallucinates on non-speech audio without VAD gating; but
   Silero VAD is **bidirectionally unreliable on sung vocals** — it may score singing as
   speech OR return no speech on clear singing (silero-vad discussion #546; maintainer
   confirms singing/music is a weak domain). Neither VAD nor logprob guards handle lyrics;
   only a music segmenter does. faster-whisper currently bundles Silero **v6** — VAD
   assumptions must be re-validated against the bundled model.
3. ⚑2 **Corrected**: qwen2.5vl:7b q4_K_M is a **6.0 GB download but ~15 GB runtime RSS at
   8k context** (ollama/ollama#14312; 125K declared context causes over-allocation;
   compute-graph regression since 0.13.4, #13687). Runtime footprint, not file size, is the
   budget — controlled via `num_ctx` pinning + frame downscaling; hang/repetition defects
   motivate per-request timeouts. Fallback: qwen2.5vl:3b (3.2 GB GGUF).
4. SigLIP so400m-patch14-384: "so400m" names the ~400M vision tower; full model 0.9 B
   params, ~1.8 GB fp16; HF checkpoint tensors are F32 (~3.5 GB) → `torch_dtype` must be
   set explicitly. SigLIP 2 checkpoints exist as drop-in successors.
5. bge-small-en-v1.5: 384-dim, ~130 MB, strong MTEB retrieval for size; English-only.
6. sqlite-vec is pre-1.0; scalar distance functions operate on plain BLOB columns (no fixed
   dimension) enabling mixed embedding spaces in one column; vec0 virtual tables would
   require one table per dimension.
7. PySceneDetect provides `ContentDetector`/`AdaptiveDetector`; both threshold adjacent-
   frame deltas, so gradual within-take changes produce no boundary (by design);
   TransNetV2 is more accurate on gradual transitions.
8. YouTube/yt-dlp output is frequently VFR; `-fps_mode cfr` re-encodes, but without
   explicit `-r` the rate comes from a probed estimate that varies across builds → pin a
   whitelisted rational rate.
9. `imagehash` provides dHash/pHash; all dedup semantics (scoping, thresholds) are the
   caller's responsibility.
10. ⚑2 **Corrected**: with `num_ctx` pinned to ~4096 and frames downscaled, the caption
    stage fits 16 GB sequentially; peak RSS must be measured on the first corpus run —
    file sizes are not memory budgets.
11. yt-dlp from datacenter IPs routinely hits YouTube's bot wall; no free/local workaround
    → file input is the canonical prod path.
12. Camera motion is not inferable from a single still; optical flow between frames
    classifies it deterministically on CPU.
13. ⚑2 **Verified by experiment** (defender + judge independently, ffmpeg 7.x): a DASH
    merge with nonzero audio `start_time` survives the mezzanine encode and skews naive WAV
    extraction by ~150 ms, invisible to same-WAV measurement; AAC priming/edit-list delay
    is NOT the mechanism (ffmpeg's default `ignore_editlist=false` trims it — measured
    0.1 ms error);
    `-af aresample=async=1:first_pts=0` recovers true media time exactly.
14. ⚑2 yt-dlp's default format sorting prefers HDR over SDR at equal res/fps → HDR sources
    will be fetched; naive libx264 encode of PQ/HLG yields washed-out output. zscale
    tonemapping requires ffmpeg built with libzimg — **Homebrew's slim `ffmpeg` formula
    does not include it** (verify `ffmpeg -filters | grep zscale` in dev setup and in the
    Dockerfile; `tonemap`+libplacebo is the documented fallback).
15. ⚑2 Whisper-family models systematically omit fillers/disfluencies (faster-whisper
    #901 — reproduced across sizes/temperatures, no working fix); the transcript is
    intended-speech, not verbatim. CrisperWhisper (verbatim) is CC BY-NC 4.0.
16. ⚑2 Lossy x264 CRF output is not bit-identical across encoder builds/versions/thread
    counts → byte-level reproducibility only under a pinned container + pinned threads;
    cross-build equivalence is checked on decoded frame content + stream parameters.
17. ⚑2 inaSpeechSegmenter's default model natively emits speech/music/noise intervals
    (singing → music); no published precision for the noise class → advisory.
18. ⚑2 torchvision `ssdlite320_mobilenet_v3_large` (COCO, BSD) runs ~10 ms/frame CPU-class
    for person detection — negligible in the overnight budget.
19. ⚑3 **Confirmed live**: Homebrew's slim `ffmpeg` formula (8.1.2) includes neither
    libzimg (`zscale`) nor libplacebo — the plan's tonemap chain cannot run on it;
    `ffmpeg-full` (keg-only) is required on macOS.
20. ⚑3 **Confirmed live**: ffprobe can report `color_transfer=None` on a real 10-bit
    source — HDR detection must also key on `pix_fmt` bit depth and BT.2020 primaries,
    never transfer strings alone.
21. ⚑3 **Confirmed live**: MP4 container duration includes the AAC audio tail and exceeds
    the video timeline (`frame_count × fps_den / fps_num`); cut math bound to container
    duration emits frame indices past the last frame.
22. ⚑3 **Confirmed live**: Whisper's language autodetect returns a confident arbitrary
    language code on pure non-speech audio (440 Hz tone → "nn") — the `language` field is
    not evidence of speech; only the segment guards are.
23. ⚑3 **Confirmed live**: two concurrent pipeline processes on one SQLite DB duplicate
    segment rows — the jobs `running` state is not a lock. Single process per index DB
    until real locking lands with the multi-machine milestone.
24. ⚑3 Python 3.14 has no wheels for the ML stack (ctranslate2/torch as of 2026-07);
    the project pins `>=3.12,<3.13`.
25. ⚑3 **Confirmed live**: legacy YouTube uploads use low frame rates (15 fps observed on
    a 2005 video); the fps whitelist must include 12/15/48 or real corpus content is
    rejected at ingest.
26. ⚑3 Environment risk, observed: `opencv-python` and PyAV (a scenedetect dependency)
    each bundle their own libav dylibs on macOS — objc warns about duplicate
    AVFFrameReceiver/AVFAudioReceiver classes. Review verdict: benign in practice (device
    layer, unused by decode paths); not eliminable from our code. The Linux Docker image
    should still install a single ffmpeg lineage.
27. ⚑3 **Verified against source**: PySceneDetect 0.7's SceneManager UNIONS multiple
    registered detectors' cut lists (`_cutting_list += cuts`, then `sorted(set(...))`) —
    no voting, intersection, or suppression exists. Pairing detectors adds false
    positives; `min_scene_len` is enforced per-detector only, so unioned near-coincident
    cuts can produce 1–2-frame phantom scenes. AdaptiveDetector subclasses ContentDetector
    (alternative thresholding, not a filter).
28. ⚑3 **Verified by reproduction**: float seconds↔frames round-trips fail at exact
    boundaries without an epsilon (`floor(0.5005 × 30000/1001⁻¹…) → 14, not 15`); the
    frame convention requires `floor(t×fps + ε)` / `ceil(t×fps − ε)`.
29. ⚑3 **Verified empirically**: cv2.CAP_PROP_POS_FRAMES seeking is frame-exact on the
    mezzanine recipe (H.264, B-frames — 229 seeks, 0 mismatches), BUT cv2 may decode 1–2
    fewer frames than ffprobe counts; any frame index handed downstream must be clamped
    to the count BOTH decoders can produce, or reads at the tail fail (`ok=False`).
30. ⚑3 **Verified empirically**: dHash is a background hash — a person entering/leaving
    frame can hash within dedup distance of the same background; perceptual dedup must be
    conditioned on structured content state (person count), not hash distance alone.
31. ⚑3 SigLIP 2 (`siglip2-so400m-patch14-384`) confirmed as drop-in: loads fp16 on MPS,
    dim 1152, cosine vs fp32-CPU ≥ 0.99999 on real keyframes. transformers returns
    `BaseModelOutputWithPooling` from `get_image_features` (SigLIP 2) vs raw tensor
    (SigLIP 1) — handle both. `torch_dtype=` is deprecated for `dtype=`.
32. ⚑3 Zero-width segment rows (t_start == t_end) are unreachable by half-open
    time_range overlap queries — instant artifacts must span one frame duration with
    exclusive end (verified: the t=0 keyframe matched no range until fixed).
