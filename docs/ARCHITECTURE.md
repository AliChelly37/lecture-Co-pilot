# Lecture Co-Pilot — Architecture (v0.2, local-only)

Status: **build baseline.** v0.1 (hosted backend, phone pairing, live gated
extraction, Opus recap) was stress-tested on 2026-09-16 and replaced by this
version under D13: *runs on the student's laptop, as cheap as possible, kill
features before adding cloud compute.* Decision history is in
[DECISIONS.md](DECISIONS.md). Cost figures are planning estimates until
replaced by measured `usage` rows.

## 1. What this is

A **local-first, model-minimal pipeline**. During a lecture, nothing leaves the
laptop: the mic is captured by the backend process, transcribed by a
self-hosted Whisper model on the laptop GPU, and stored in a local SQLite file.
After the lecture, a language model is called a handful of times to extract
deadlines, write the recap, and answer follow-up questions. By default that
model is **`gemma3:4b` running locally through Ollama** (D21), so nothing
leaves the laptop after class either; Cloudflare Workers AI (free tier) and
Claude are optional providers behind the same interface. Confirmed deadlines
are written to Google Calendar and Notion on a tap.

Four guarantees carry the ethics and engineering story:
1. **Audio never leaves the laptop** (D12, D13), and with the default local provider **nothing leaves the laptop at all**. With a cloud provider, only derived text and downscaled board photos leave, and the UI shows which provider is active.
2. **The model cannot write anywhere** (D5). It returns structured suggestions; only a user tap reaches `ActionTarget.confirm()`.
3. **Raw media is never persisted by the app** (D8, D19). Audio exists only in memory between VAD and Whisper; imported photos are dropped after OCR (the originals stay in the phone's camera roll, under the user's control).
4. **One module talks to any model provider** (`LlmGateway`): one schema-validated call shape for Ollama, Cloudflare and Anthropic, and one ledger row per call, so every token and dollar is attributed to a stage.

Portfolio claim, stated honestly: **provider-agnostic, model-minimal design**:
every stage that can be deterministic is; the model is called only where
judgment is needed; the default runs on a 4B model on the student's own GPU at
zero cost; and the same eval measures what a free-tier 70B or a paid Claude
model would have bought.

## 2. Stage table

| Stage | Runs where | Model / tech | Effort | Why this, and what breaks if changed |
|---|---|---|---|---|
| Mic capture + VAD | Laptop, backend process | `sounddevice` + Silero VAD (bundled with faster-whisper) | — | The backend owns the mic (D14), so a backgrounded browser tab can't stop recording. VAD drops silence before inference: less compute, less battery, and it starves Whisper's hallucination-on-silence failure mode. |
| Speech-to-text | Laptop GPU (RTX 4050), CPU fallback | faster-whisper, model chosen by measured real-time factor (`large-v3-turbo` int8 target; `small`/`distil-small.en` on CPU or battery) | — | No audio leaves the device. If the worker falls behind real time it downgrades the model, never lets the transcript lag grow unbounded. |
| Live "possible deadline" badge | Laptop | Deterministic `TriggerFilter` (date expressions + lexicon) | — | Zero cost, zero network. The live signal survived D15 as a badge; the model confirms after class. |
| Confusion flag | Laptop | Global hotkey (backend `pynput` listener) + UI button; lookback window 90 s before / 15 s after | — | Works while typing in any app. A tap costs nothing and does nothing live, by design. |
| Deck ingestion | Laptop | `pymupdf` / `python-pptx` text extraction, then one model call to build the `SlideIndex` (default gemma3:4b via Ollama) | `low` | Text extraction is free and avoids PPTX-to-PDF conversion. One call per deck, cached per deck hash; $0 locally. |
| Board photo OCR | Laptop, one model call per photo at import | Vision model (default gemma3:4b via Ollama; llama-3.2-11b-vision on Cloudflare; Sonnet 5 on Anthropic), photo downscaled to ~1280 px | `low` | Measured on a synthetic board: every term read, 26 s, $0. Output schema `{content_kind, text, latex[], diagram_notes, legibility}` with enum-constrained categorical fields. Non-board content is discarded. |
| Slide alignment | Laptop | Deterministic lexical overlap between transcript windows and `SlideIndex` | — | Off-slide detection and recap structure without a model. |
| Deadline extraction | Laptop, after the lecture | Structured output, **no tools**. Local provider: ~10-minute transcript chunks within the 16K context, merged and de-duplicated; cloud providers: one pass | `low` | Measured: gemma3:4b found every real event on the fixture in 14 s (D21). No tools means no write path exists (D5). |
| Date resolution | Laptop | Deterministic `DateResolver` (`dateparser` + course term calendar + course timezone) | — | "Next Thursday" and "week 7" are testable code, not a model guess (D9). |
| Recap | Laptop, after the lecture | Structured output with per-bullet source pointers; map-reduce over chunks on the local provider | `high` | The hero output. Cloud 70B and Claude are compared **in the eval only** (D21). |
| Ask-the-lecture | Laptop, per question | Relevant chunks retrieved by lexical overlap, then one model call | `medium` | $0 locally; the cached-bundle trick (D18) only applies on Anthropic. |
| Calendar / Notion write | Laptop, on tap | REST clients behind `ActionTarget` | — | Idempotent, undoable, never automatic. |
| Dashboard, MCP server | Laptop | No model; read-only stdio MCP as last milestone | — | Aggregates structured data. |

## 3. Components

```
 LAPTOP (everything below runs here; no network is needed during class)
 +----------------------------------------------------------------------+
 | Browser UI (React/Vite, http://localhost:8765)                       |
 |  live transcript | possible-deadline badge | flag/pause buttons      |
 |  photo import | suggestion inbox (confirm/dismiss/undo) | recap      |
 |  ask-the-lecture | dashboard | export/delete                         |
 +-----------------------------+----------------------------------------+
                               | HTTP + WebSocket, localhost only
 +-----------------------------v----------------------------------------+
 | Backend: one Python process (FastAPI + uvicorn), D17                 |
 |                                                                      |
 |  DURING CLASS                                                        |
 |  MicCapture -> VAD -> WhisperWorker (GPU, CPU fallback) -> Timeline  |
 |  HotkeyListener (flag, pause) -----------------------------> Timeline|
 |  TriggerFilter (deterministic) <---------------------------- Timeline|
 |  SleepGuard (blocks system sleep while recording)                    |
 |                                                                      |
 |  AFTER CLASS                                                         |
 |  PhotoImport (EXIF time) -> BoardReader ------+                      |
 |  DeckIngestor -> SlideIndex -> SlideAligner   |                      |
 |  LectureBundle (cached prefix, D18) <---------+                      |
 |     -> Extractor -> DateResolver -> SuggestionService                |
 |                                        -> ActionTargets [tap only]   |
 |     -> Distiller -> Recap (versioned, source pointers)               |
 |     -> AskLecture (Q&A)                                              |
 |                                                                      |
 |  LlmGateway (sole Anthropic caller, cache breakpoints, UsageLedger)  |
 |  Store (SQLite; export, retention, hard delete) | Metrics            |
 |  McpServer (read-only, stdio; last milestone)                        |
 +----------------------------------------------------------------------+
        | outbound only, after class: Claude API (text + small images),
        v Google Calendar / Notion on confirmed writes
```

| Component | Responsibility |
|---|---|
| `MicCapture` | Opens the default input device via `sounddevice`, 16 kHz mono float32, pushes frames to a queue. Emits `Gap{mic_lost}` on device loss. |
| `WhisperWorker` | Thread owning the faster-whisper model. Runs Silero VAD on the rolling buffer, cuts at speech pauses (decode-once chunks of 2–20 s), transcribes with `hotwords` from the SlideIndex, applies hallucination filters (no-speech probability, average log-prob, compression ratio; `condition_on_previous_text=False`). Reports real-time factor; downgrades model when behind. |
| `Timeline` | Append-only per-lecture event log on one clock: segments, gaps, flags, pauses, board captures, alignment. Persisted as it arrives (crash-safe). |
| `TriggerFilter` | Scores each final segment for date expressions and deadline lexicon; increments the live badge; its hits are also passed to the extractor as hints. |
| `HotkeyListener` | Global hotkeys (default F9 flag, F10 pause/resume), registered by the backend. |
| `DeckIngestor` | PDF/PPTX -> per-slide text (local) -> Sonnet 5 `SlideIndex` (title, key terms, stated dates, summary per slide). Keyed by deck hash. |
| `PhotoImport` + `BoardReader` | Reads EXIF `DateTimeOriginal`, assigns photos to the lecture whose window contains them, downscales, calls Sonnet 5 vision, stores derived text, drops the image. |
| `SlideAligner` | Maps transcript windows to a slide or `off_slide`; coverage score; warns when the deck doesn't match the lecture. |
| `LectureBundle` | Assembles course context + SlideIndex + slide text + transcript (with low-confidence spans marked) + board text + flags into the cached system prefix. |
| `Extractor` | One Sonnet 5 call returning `CandidateEvent[]` with `intent` and `evidence_quote`. |
| `DateResolver` | Expression + lecture datetime + timezone + term calendar -> date or `unresolved`. |
| `SuggestionService` | Policy (one-tap vs maybe tray vs log-only), dedup, supersede, state machine. |
| `ActionTargets` | `GoogleCalendarTarget`, `NotionTarget`: `propose / confirm / undo / healthcheck`, idempotent. |
| `Distiller` | One Sonnet 5 `high` call returning the structured `Recap`. |
| `LlmGateway` | Routing table, prompt assembly with fixed breakpoints, retry chain, one `UsageLedger` row per call. |
| `Store` | SQLite schema, JSON + Markdown export, transcript retention job, hard delete. |

## 4. Data flow

### 4a. During the lecture (no network required)
1. Start lecture (UI or hotkey). `SleepGuard` blocks system sleep; the capture indicator turns on.
2. `MicCapture` -> VAD -> `WhisperWorker` -> `TranscriptSegment{t0, t1, text, conf, model}` -> `Timeline` -> UI over WebSocket.
3. `TriggerFilter` runs on each final segment; the badge counts "possible deadline mentions".
4. F9 writes `Flag{t, window}`; F10 toggles `Pause` (nothing is transcribed while paused, so classmates' Q&A can be excluded at the user's discretion).
5. Stop lecture. Nothing has been sent anywhere.

### 4b. After the lecture
1. Import board photos (drag-drop or file picker); EXIF time assigns them to the lecture; `BoardReader` derives text; images dropped.
2. `SlideAligner` finalises alignment and marks off-slide stretches.
3. `LectureBundle` is assembled and placed behind a cache breakpoint.
4. `Extractor` -> `DateResolver` -> `SuggestionService` -> inbox. Policy: only `commitment` + confidence above threshold + resolved future date + not inside a low-confidence ASR span reaches the **one-tap tray**; `tentative`/`unresolved` reach the **maybe tray** (editable date, no one-tap); `hypothetical`/`joke`/`past_reference` are log-only.
5. `Distiller` -> `Recap` (highlights, importance-ranked key concepts, one explanation per flag tied to its window, review questions, off-slide notes, detected events), every bullet with source pointers for "tap to see source".
6. User confirms suggestions; `ActionTarget.confirm()` writes with an idempotency key; undo available.
7. Ask-the-lecture questions reuse the cached bundle.

## 5. Model providers (D21)

`LlmGateway.structured(stage, schema, system, user, images)` is the only way
the app calls a model. It validates the reply against a pydantic schema (one
corrective retry, then an error), never passes tools, and writes a ledger row
per call. The provider is a setting (`LC_LLM_PROVIDER`):

| Provider | Models (default) | Where data goes | Cost | Structured output | Notes |
|---|---|---|---|---|---|
| **ollama** (default) | `gemma3:4b` for text and vision | Nowhere: runs on the laptop GPU | $0 | Grammar-enforced JSON schema | 16K context on 6 GB; lecture is chunked. Measured 54 tok/s. |
| cloudflare | `llama-3.3-70b-instruct-fp8-fast`; `llama-3.2-11b-vision-instruct` | Cloudflare Workers AI | Free tier 10k neurons/day, then ~$0.29 / $2.25 per M tokens (70B) | JSON mode on text models; vision models prompted for JSON and validated | Token needs "Workers AI - Read/Edit" |
| anthropic | `claude-sonnet-5` | Anthropic | $2 / $10 per M tokens | `messages.parse` | Prompt caching on the last system block (D18) |

Schema rules that came out of measurement (docs/LLM_BENCHMARK.md): every
categorical field is an enum, every list has `maxItems`, and every call has
an output-token cap. Small models confuse free-text categorical fields and can
loop inside an unbounded grammar.

## 6. Cost model (D21)

On the default local provider the marginal cost of a lecture is **$0**: Whisper
and gemma3:4b both run on the laptop GPU. The tracked "cost" metrics are
therefore **battery drain per lecture-hour** and **wall-clock time to recap**
(measured: ~14–20 s per extraction chunk, ~26 s per board photo).

On Cloudflare Workers AI a lecture-hour is roughly 25K input + 8K output
tokens across all calls: ~$0.03 at paid rates for the 70B model, and free
within the 10k-neurons/day tier. On Anthropic (Sonnet 5 with the cached
bundle) the earlier estimate of ~$0.20 per lecture-hour stands; it is now an
eval configuration, not the product default.

The eval (section 10) is the only place paid providers are used, under a hard
spend cap that is printed before each run.

## 7. Data model

| Entity | Key fields | Notes |
|---|---|---|
| `Course` | name, timezone, term_calendar, `capture_enabled`, policy_ack_at, transcript_retention_days (default 30) | D2, D19 |
| `Lecture` | course_id, started_at, ended_at, status, deck_id, asr_model, asr_rtf_p95, power_state, battery_start/end | Battery fields feed the drain metric |
| `TranscriptSegment` | lecture_id, t0, t1, text, asr_conf, asr_model | Purged by the retention job; recaps keep their quoted spans |
| `Gap` | lecture_id, t0, t1, cause (`mic_lost` / `asr_backlog` / `paused` / `crash`) | Recaps must name gaps |
| `Flag` | lecture_id, t, window_t0, window_t1, reason (nullable), resolved | |
| `Deck` / `SlideIndex` / `SlideAlignment` | deck hash, per-slide text, index JSON / window -> slide or off_slide | Index cached by deck hash |
| `BoardCapture` | lecture_id, t_shutter, content_kind, text, latex[], legibility | No image stored |
| `CandidateEvent` | lecture_id, type, title, date_expression, intent, evidence_quote, t0, confidence, resolved_date, course_hint | All kept, including rejected (eval data) |
| `Suggestion` | candidate_id, target, payload, state, idempotency_key, external_id, supersedes_id | State machine below |
| `Recap` | lecture_id, version, model, effort, sections JSON, rating, flag_helpful[] | Versioned; regenerate never overwrites |
| `UsageLedger` | call_id, lecture_id, **stage**, model, effort, input, cache_read, cache_write, output, cost_usd, latency_ms, stop_reason | One row per Anthropic call |
| `KnownConcept` | concept, course_id, marked_at | v2 stub |

Suggestion states: `proposed -> confirmed -> written -> (undone)`; `proposed -> dismissed`; `confirmed -> failed_retryable -> written | failed_permanent`; `proposed | written -> superseded` (a correction produces an update to the existing external event, never a duplicate).

## 8. Metrics

| Metric | Definition | Source |
|---|---|---|
| Claude cost per lecture-hour, by stage | sum of `cost_usd` / duration | `UsageLedger` |
| Cache hit rate | `cache_read / (cache_read + cache_write + input)` per stage | `UsageLedger` |
| ASR real-time factor (p50/p95) and model downgrades | worker telemetry | `Lecture` |
| Battery drain per lecture-hour | battery % at start/end, on battery only | `Lecture` |
| Suggestion precision | confirmed / (confirmed + dismissed), plus labelled-eval precision | `Suggestion`, eval |
| Recap-ready latency | lecture end -> recap visible | spans |
| Recap usefulness | 1–5 rating + per-flag "did this help" | `Recap` |
| OCR accuracy | character error rate on spot-checked photos | eval |

## 9. Error handling and edge cases

Principle: **degrade visibly, never silently.**

### 9.1 Capture and speech-to-text
| Case | Behaviour |
|---|---|
| Mic lost / device switched | `Gap{mic_lost}`, red banner, one-tap resume; never silently re-acquire a different device |
| Laptop sleeps (lid closed) | `SleepGuard` blocks sleep while recording; if it sleeps anyway, `Gap{crash}` on resume and the recap names it |
| Whisper hallucinates on silence/noise | VAD first; drop segments failing no-speech / log-prob / compression-ratio thresholds; dedicated eval case asserts zero candidate events from silence and HVAC noise |
| Worker falls behind (RTF > 1) | Downgrade to the next-smaller model; if still behind, drop the oldest queued audio as `Gap{asr_backlog}` rather than serve stale transcript; badge shows "running behind" |
| On battery | Default to a smaller model (configurable); record battery drain; warn below 20% |
| Low-confidence span | Marked in the bundle; any date inside it caps intent at `tentative` (never one-tap) |
| Jargon / names | `hotwords` from the SlideIndex (prompt biasing is documented as unreliable, so its effect is measured, not assumed); slides and board text back up spellings in the recap |
| Other students' speech | Pause hotkey; no attribution in recaps; transcript retention limit; documented limitation (no diarisation) |
| Disk nearly full (26 GB free today) | Model cache is checked before download; SQLite growth is small (text only) |

### 9.2 Dates: ambiguous, sarcastic, hypothetical, corrected
1. The extractor classifies **intent** (`commitment | tentative | hypothetical | joke | past_reference | correction`) with a verbatim `evidence_quote`; prompt examples cover the known traps.
2. Code resolves dates (D9); `unresolved` is a legitimate output.
3. Policy gates surfacing (section 4b).
4. The confirm card shows the quote, the timestamp, the resolved date **and how it was resolved**.
5. Corrections supersede (dedup key = normalised type + title within a course); a written event gets an **update**, never a duplicate.
6. Slide-vs-speech conflicts show both on one card and ask.
7. Other-course mentions carry `course_hint` and go to the maybe tray.
8. Course timezone, never browser timezone; DST covered by table tests.

### 9.3 Photos and slides
| Case | Behaviour |
|---|---|
| Photo has no EXIF time / wrong clock | Assign manually in the import UI; photos outside any lecture window are held unassigned |
| Illegible photo | Stored with low `legibility`; recap treats its text as low-trust; the user can import a better shot later and regenerate (bundle re-cached) |
| Vision call fails | Retry chain; then `BoardCapture{status: failed}` so the recap says "1 photo couldn't be read"; image dropped |
| Accidental non-board photo | `content_kind != board` -> discarded, no derived text stored |
| Professor off-slide for a long stretch | `off_slide` windows; the distiller is told off-slide material can be high importance and gets an "Off-slide material" section |
| Wrong or no deck | Low coverage -> warning, deck down-weighted; no deck -> transcript + board mode |

### 9.4 Integrations
| Case | Behaviour |
|---|---|
| Token expired | `healthcheck()` at app start and before the inbox is shown, not at confirm time |
| Google consent screen in Testing | Refresh tokens die after 7 days; D19 sets the app to In production (unverified, personal use) |
| Workspace account blocks the app | Detected via `admin_policy_enforced`; UI suggests a personal account |
| Double tap / lost response | Idempotency key; Google: client-generated event `id`; Notion: hidden `copilot_id` property queried before create |
| Notion schema drift | `healthcheck()` validates the property mapping; mismatch -> `failed_permanent` with a "fix mapping" link |
| Rate limits | Notion <=3 req/s with `Retry-After`; Google exponential backoff |

### 9.5 Model and platform
| Case | Behaviour |
|---|---|
| 429 / 5xx / network | SDK retries on retryable errors only; the post-lecture flow is resumable per stage |
| `stop_reason: refusal` | Checked before reading content; logged; the user sees "the model declined; retry or edit inputs" |
| Truncated structured output | Fails validation; never shown as a complete recap |
| Prompt injection in lecture content | Content is data, not instructions; D5 caps the blast radius at a wrong suggestion that still needs a tap; red-team case in the eval |
| Offline after class | Post-lecture flow queues until online; nothing is lost |

## 10. Testing and evaluation

### Datasets (D11, D19)
- **Public set:** MIT OCW lectures with transcript + slides (6.0001 Fall 2016 first), CC BY-NC-SA 4.0, in a separately licensed `eval/data` folder with third-party-excluded slide content stripped; transcripts normalised before WER.
- **Synthetic deadline inserts:** labelled deadline, hypothetical, sarcastic and correction utterances inserted into real transcripts (public lectures rarely contain admin talk).
- **Private set:** own lectures, local only, never committed.
- Dev / held-out test split; labels (key-point checklists, flag notes) written **before** any recap is seen.

### L0: deterministic tests (every commit, free)
`DateResolver` tables (relative expressions x lecture dates x timezones x DST x term calendar); `TriggerFilter` recall on labelled segments (its misses are invisible in production); suggestion state machine, dedup, supersede and idempotency against mocked Google/Notion; schema validation on recorded responses; the cache probe.

### L1: replay harness
Recorded Timelines replayed through the real pipeline with fault injection (mic loss, late photo, contradicting deadline, wrong deck). Produces latency numbers and the fair cost comparison: the default config vs an **"Opus 5 everywhere"** baseline on identical inputs.

### L2: component evals
| Component | Metric | Configs |
|---|---|---|
| Speech-to-text | WER and jargon error rate vs OCW transcripts; RTF on this laptop (AC and battery); phantom segments per hour of silence (target 0); **downstream date-extraction precision per model size** | Whisper sizes, int8 vs float16, GPU vs CPU |
| Extraction | Precision of the one-tap tier (target >= 0.95), recall of commitments (>= 0.90), intent confusion matrix, adversarial pass rate | Sonnet 5 `low`/`medium`, Haiku 4.5, Opus 5 `low` |
| Board OCR | CER vs hand transcription; equation exact-match | Sonnet 5 `low` vs Opus 5 `low`; phone Live Text on a subset, for honesty |
| Alignment | Window accuracy; off-slide F1 | Threshold sweep |

### L3: recap quality (five separate scores, never one blended number)
1. **Faithfulness:** atomic claims; source pointer checked deterministically, then a judge verifies the *pointed span* supports the claim; `elaboration` claims (flag explanations) are judged for correctness, not support.
2. **Coverage:** recall of the pre-written key-point checklist.
3. **Flag responsiveness:** right concept, adds beyond the transcript, correct.
4. **Review questions:** answerable from the lecture, not verbatim, spread across key points.
5. **Overall usefulness:** blind, position-swapped pairwise preference.

Judge: Opus 5 with narrow binary/3-point rubrics; ~40 decisions hand-labelled and Cohen's kappa reported next to every judge-based number; self-preference bias stated in the README. Configs: Sonnet 5 `medium`/`high`/`xhigh`, Opus 5 `high` (the "was it worth it" comparison), slide text vs slide page images. Runs via the Batch API under a hard cap.

### L4: real-use pilot (n = 1, reported as such)
48-hour delayed recall questions, alternating recap-reviewed vs notes-only lectures; confirm/dismiss rates; recap ratings; open flags after a week.

## 11. Build milestones

| # | Milestone | Proves |
|---|---|---|
| M0 | Whisper benchmark on this laptop (GPU and CPU, AC and battery) | Model default and fallback are measured, not guessed |
| M1 | Record a lecture: mic -> VAD -> Whisper -> Timeline -> live transcript, hotkey flag, badge, start/stop, sleep guard | The local capture loop works for 90 minutes |
| M2 | Deck ingestion + photo import + OCR + alignment | Multimodal inputs on one timeline |
| M3 | Bundle + extraction + date resolution + suggestion inbox (in-app only) | Trust-critical path, with the cache probe passing |
| M4 | Recap + ask-the-lecture + ratings | The hero output |
| M5 | Google Calendar target, then Notion target, behind `ActionTarget` | Confirm-before-write on real APIs |
| M6 | Eval harness L0–L3 on OCW data; cost chart | The claims have numbers |
| M7 | Dashboard, export/delete, retention job, read-only MCP server, README | Complete v1 |
