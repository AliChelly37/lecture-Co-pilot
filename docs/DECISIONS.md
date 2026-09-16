# Lecture Co-Pilot — Decision Log

Working title; rename is still open. This log is the running record of design
decisions, the options considered, and the reasoning. Entries are append-only —
if a decision is reversed, add a new entry that supersedes the old one rather
than editing history.

Status legend: **DECIDED** (user-confirmed) / **PROPOSED** (drafted, pending stress-test) / **OPEN** / **SUPERSEDED**

---

## D1 — Calendar & task integrations for v1
**Date:** 2026-09-16
**Status:** DECIDED

**Question:** Which calendar/task services should v1 actually write to (always
behind the confirm-tap)?

**Options considered:**
| Option | Trade-off |
|---|---|
| Google Calendar only | One OAuth path, one write schema. Notion deferred behind an adapter. Lowest demo fragility. |
| **Google Calendar + Notion** | Two OAuth flows, two write schemas, two failure modes. Notion needs DB selection + property mapping. Strongest dogfooding story. |
| In-app suggestions only (.ics / Markdown export) | No external OAuth at all. Cheapest, but drops the agentic tool-use story that carries the portfolio pitch. |

**Decision: Google Calendar + Notion.**

**Reasoning:** User dogfoods Notion daily; a tool about sustaining personal
focus is only credible if it writes into the workflow the user actually lives
in. The extra integration cost was raised explicitly and accepted.

**Flagged cost (accepted, not dismissed):** doubles OAuth surface (two token
stores, two refresh paths), doubles write-schema mapping, and Notion's
database-property mapping (pick/create DB, map title/date/status props) is the
fiddliest part. Mitigation is D1a.

**D1a — consequence:** both targets sit behind a single `ActionTarget` adapter
interface (`propose()` / `confirm()` / `undo()` / `healthcheck()`). With two
targets this is load-bearing, not speculative generality. Adding a third
(Todoist, Apple Calendar) must be a new adapter, never a change to the
extraction agent.

**Environment note:** the Notion connector in the current Claude Code session is
unauthenticated, so Notion writes cannot be prototyped or tested from this
session until authorized via claude.ai connector settings. Does not affect the
decision; does affect what can be validated cheaply during build.

---

## D2 — Recording & consent context
**Date:** 2026-09-16
**Status:** DECIDED

**Question:** Is v1 for the builder's own classes (personal accessibility use)
or for distribution to other students?

**Options considered:**
| Option | Trade-off |
|---|---|
| **Own classes only** | Single user, participant-captured, personal study use. Near-zero consent machinery. Ships fastest. |
| Personal now + ConsentContext scaffolding | ~10-15% more design work, buys later beta without migration. Speculative against stated non-goals. |
| Small closed beta | Requires professor notification/opt-in, per-course opt-out registry, jurisdiction-aware consent, data-controller story. Legal exposure sits with the builder. |

**Decision: own classes only — personal accessibility use.**

**Reasoning:** Matches the stated non-goal ("single user or small beta only")
and keeps the consent story defensible without inventing a compliance layer for
users who do not exist yet. The tool is fully demoable to recruiters in this
form; the ethics story is carried by the ephemeral-media and
confirm-before-write guarantees, not by a consent registry.

**What this means we build:**
- Visible capture-in-progress indicator whenever the mic is live (no silent recording, including from the user's own perspective).
- A one-time per-course acknowledgement that the user has checked their institution's recording policy for that course.
- A per-course `capture_enabled` flag, so if a specific professor or course policy disallows it, capture is switched off for that course without disabling the app. Cheap now, and the honest alternative to hard-coding "recording is fine".
- Existing hard requirements stand: raw audio/photos discarded after derivation, only derived text persists, full view/export/delete.

**What this means we explicitly do NOT build in v1:**
- No professor opt-in or notification flow.
- No jurisdiction / two-party-consent logic.
- No multi-tenant data-controller model or per-institution policy engine.

**Standing caveat:** the builder should confirm their institution's recording
policy, and check whether the disability-services accommodation route applies —
in many institutions that converts this from a grey area to explicitly
permitted, and is worth doing before the first real lecture capture.

**Supersession trigger:** any distribution beyond the builder's own use reopens
D2 as a new entry. It is not a config change; it is a different system.

---

## D3 — Target platform
**Date:** 2026-09-16
**Status:** DECIDED

**Options considered:** web app / PWA single device; **web app + QR-paired
phone capture**; native mobile; desktop (Tauri/Electron).

**Decision: web app + QR-paired phone capture page.**

**Reasoning:** matches real lecture-hall behaviour (laptop open for audio,
phone raised for the board), stays one codebase and one deploy, and a recruiter
can try it from a URL.

**Consequences:**
- The paired phone page is a **remote**, not just an uploader: `[Flag]` and
  `[Photo]` buttons. This removes the browser-tab focus problem, since a laptop
  tab can't register a global hotkey but the phone is always in hand.
- **A hosted backend is required.** Lecture-hall Wi-Fi (e.g. eduroam) commonly
  isolates clients, so the phone can't reach a server running on the laptop.
  That's in tension with "local-first" (see D8).
- Pairing token: short-lived, scoped to one lecture session, and limited to
  upload + flag. It can't read anything.
- Photo and flag timestamps come from the phone's clock at the moment of
  capture, corrected by a clock offset measured at pairing, so late uploads
  still line up with the transcript.

---

## D4 — Pipeline of specialised calls, not autonomous agents; no MCP in v1
**Status:** PROPOSED

Every stage has a fixed, code-controlled flow (extract → propose → confirm;
ingest → distil). None needs open-ended, model-driven tool loops, so this is a
**multi-model workflow**, and the pitch should call it that. Tools are plain
typed functions behind module interfaces.

**MCP rejected for v1:** MCP pays off when a *second client* consumes the tools.
Here there is exactly one caller, so an MCP server would add a process, a
transport and a schema surface for nothing. **Revisit trigger:** a read-only
MCP server over the lecture store (search transcript, get recap, list open
flags) so the student can query past lectures from Claude Desktop / Claude
Code. That's the first defensible MCP use, because a second client exists.

## D5 — The live extractor has no write capability
**Status:** PROPOSED

Live extraction returns **structured output** (a list of candidate events),
not tool calls to calendar/Notion. Writes exist only in
`ActionTarget.confirm()`, which is reachable only from a user tap.
Confirm-before-write is therefore an architectural guarantee, not a prompt
instruction. It also caps prompt-injection blast radius: hostile text in slides
or speech can at worst produce a wrong *suggestion*.

## D6 — Gated live extraction
**Status:** SUPERSEDED by D15 (extraction moved after the lecture; the Stage A filter survives as the free live badge)

Stage A, deterministic (no model): a trigger filter over each finalised
transcript segment (date/time expressions via a date-parsing library,
lexicon: due, deadline, exam, midterm, quiz, submit, assignment, "on the test",
"make sure you"...), plus the professor-question heuristic. Stage B: Sonnet 5 at
`low` effort, called on a trigger **or** a sweep every ~3 min. The sweep catches
what the lexicon misses and keeps the 5-minute cache warm. Haiku 4.5 is an eval
candidate, not the default. Its 4,096-token minimum cacheable prefix (vs 1,024
on Sonnet 5) means the static prefix may not cache at all. The model sweep in the
eval decides.

## D7 — SlideIndex for the live path; full deck only for distillation
**Status:** PROPOSED

The deck is ingested once into a compact structured `SlideIndex` (per slide:
title, key terms, stated dates, summary). The live extractor carries the index
(a few K tokens), never the full deck. The full PDF goes only into the
distillation call. PPTX is converted to PDF server-side, since Claude accepts
PDF natively but not PPTX. Side benefit: SlideIndex key terms seed the ASR
model as faster-whisper `hotwords` (D12).

## D8 — Storage: single-user hosted backend, raw media never touches disk
**Status:** SUPERSEDED by D13 (no hosted backend; everything runs on the laptop)

Forced by D3. Derived data lives in a single-user SQLite DB the user can
export/delete. API keys and OAuth tokens stay server-side (never in the
browser). Raw audio is streamed straight through to ASR and never persisted.
Photos are held in memory until derivation succeeds, then dropped. "Local-first"
is honestly downgraded to "**minimal-retention, single-tenant**", and the README
should say so rather than overclaim.

## D9 — Relative dates are resolved in code, never by the model
**Status:** PROPOSED

The model extracts the *expression* ("next Thursday", "end of week 7") and its
intent. A deterministic resolver turns it into a date using the lecture's
timestamp, timezone and the course term calendar. It's unit-testable and removes
a known LLM weak spot.

## D10 — Distillation: Opus 5 at `high` default; Fable 5.1 premium
**Status:** SUPERSEDED by D16 (Sonnet 5 only; Opus 5 eval-only; Fable 5.1 dropped)

Default `high`, not `xhigh`. Promote only if the eval sweep shows measurable
headroom. Both models share one request shape: structured output,
**no forced `tool_choice`** (400 on Fable 5.1), and server-side refusal
fallbacks enabled. **Privacy disclosure required on the premium toggle:** Fable
5.1 requires 30-day data retention and isn't available under zero data
retention.

## D11 — Eval set = openly licensed public lectures + private set
**Status:** PROPOSED

The public, shareable eval set comes from openly licensed courseware (e.g. MIT
OpenCourseWare, CC BY-NC-SA; license to be verified per course), so the eval is
reproducible by anyone reading the repo **without exposing real classroom
content**. The private set (own lectures) is used locally and never committed.

---

## D12 - Speech-to-text: self-hosted Whisper on the backend
**Date:** 2026-09-16
**Status:** DECIDED (resolves O1)

**Options considered:** cloud streaming ASR under no-retention terms;
**self-hosted Whisper-family model on the backend**; in-browser on-device
Whisper; cloud for v1 plus a self-hosted comparison in the eval.

**Decision: self-hosted Whisper on the backend.**

**Reasoning:** raw classroom audio was the largest remaining privacy exposure.
Self-hosting gives a clean, checkable claim: *no third party ever receives
audio*. Anthropic only ever sees derived text.

**Accepted costs:** lower accuracy on jargon and names than top cloud
providers; no native diarisation; a hosting bill and ops owned solo; chunked
pseudo-streaming with a ~2-5 s delay (fine for this use).

**Consequences:**
- **Two runtimes.** TypeScript app + a separate ASR worker (Python
  faster-whisper or a whisper.cpp server). O3 is amended accordingly.
- **Whisper's failure modes are now ours.** It is known to hallucinate text on
  silence/noise and to fall into repetition loops. That matters here because a
  hallucinated "due Friday" would enter the extraction path. Mitigation: VAD
  before inference; drop segments on no-speech probability, average log-prob
  and compression-ratio thresholds; don't condition on previous text.
  Hallucination-on-silence gets its own eval case.
- **Vocabulary biasing** through faster-whisper `hotwords` (re-applied every window, unlike `initial_prompt`), seeded from SlideIndex
  key terms (replaces cloud keyword boosting).
- **Capacity is a correctness issue, not just a performance one.** The worker
  must run faster than real time. If it falls behind, degrade (smaller model)
  rather than let transcript delay grow without bound. Real-time factor (RTF) is
  a tracked metric.
- **Hosting shape:** lectures are a few hours a week, so scale-to-zero or
  on-demand compute, started ahead of scheduled lectures using the course
  timetable (cold start happens before class, not during). Model size is chosen
  by eval (word error rate vs RTF vs $/hr), not guessed.
- **The other-students'-voices limitation gets worse** (no diarisation). v1
  accepts and documents it; a diarisation model is a v2 option.
- **The eval gains a ground truth.** Public lectures with human transcripts give
  word error rate directly, plus a portfolio chart: downstream
  date-extraction precision per Whisper model size.

---

## Open items
- ~~O1 - ASR provider~~ -> resolved as D12.
- **O2 - Photo retention for OCR eval:** default applied, not yet explicitly
  confirmed: discard; per-photo opt-in "keep for accuracy check", auto-deleted
  after 7 days.
- **O3 - Stack (amended by D12):** TypeScript web app + Node backend using the
  official `@anthropic-ai/sdk`, plus a separate ASR worker in Python
  (faster-whisper) or whisper.cpp. **Reopened by D13** (pending stack question).

---

## D13 - Governing constraint: runs locally on the student's laptop, cost-first
**Date:** 2026-09-16
**Status:** DECIDED (user directive). Supersedes D8; amends D3, D10, D12.

**Decision:** the whole system runs on the student's own laptop (Lenovo,
i5-12450HX 8C/12T, RTX 4050 Laptop 6 GB, 16 GB RAM, Windows 11, ~26 GB free
disk). No hosted backend, no rented GPU, no relay server. The only network
dependency is the Claude API (derived text + downscaled board photos), paid from
the student's existing credit. **Affordability and local execution outrank
features**: anything that needs cloud compute is cut or deferred.

**Facts that shaped it (research, 2026-09-16):**
- Campus Wi-Fi commonly isolates clients (ASU, CU Boulder, Cambridge; NDSS 2026
  AirSnitch paper on an eduroam network), so phone-to-laptop pairing over
  campus Wi-Fi is not viable.
- Cloud GPU for ~6 h/week: Modal ~$20-27/mo (within its $30 free credit),
  RunPod pod ~$8-17/mo, Cloud Run ~$27-30/mo; every option can drop a 60-90 min
  connection (Cloud Run hard-cuts at 60 min). Cost was small, but none is local.
- large-v3-turbo int8 needs ~1.5 GB VRAM on GPU; the RTX 4050 has 6 GB.
  Laptop-specific benchmarks are unverified, so the model size is chosen by a
  measured real-time factor on this machine, with a CPU / small-model fallback.
- Toolchain already present: Python 3.11, uv, Node 24, ffmpeg, NVIDIA driver.

**Consequences:**
- D8 superseded: local SQLite + localhost server. The "local-first" claim is
  restored honestly: **audio never leaves the laptop**.
- D12 amended: the Whisper worker runs on the laptop GPU; model size and CPU
  fallback are picked from measured RTF and the power state (AC vs battery).
- D3 amended: the QR-paired phone remote over campus Wi-Fi is dead; the
  replacement is D14 (pending).
- D10 amended: Fable 5.1 premium mode is out of v1 (eval-only at most); the
  recap default is D15 (pending).
- Every cost figure in ARCHITECTURE.md section 6 is to be re-baselined after
  D15; planning target is well under $0.50 per lecture-hour of Claude spend.

---

## D14 - Capture: laptop-only; the backend owns the mic; photos imported after class
**Date:** 2026-09-16
**Status:** DECIDED (amends D3)

**Options considered:** laptop-only capture with post-class photo import;
phone hotspot during class; Tailscale overlay; drop whiteboard photos.

**Decision: laptop-only capture.** The Python backend captures the mic
directly (`sounddevice`), so the browser is only a view: no tab-focus, tab
suspension or Wake Lock problems. The confusion flag is a **global hotkey**
registered by the backend (works while typing in any app) plus a button in the
UI. Whiteboard photos are taken with the phone's normal camera and **imported
after class**; EXIF `DateTimeOriginal` aligns them to the transcript. Nothing
is paired and no network is needed during class.

**Lost, knowingly:** the live "retake?" legibility check and the phone flag
button. **Kept for later:** a LAN upload page (home Wi-Fi) so the phone can
push photos without a cable.

## D15 - Deadline extraction runs after the lecture, in one pass
**Date:** 2026-09-16
**Status:** DECIDED (supersedes D6)

One Sonnet 5 call over the whole lecture bundle at lecture end: better recall
than any windowed scheme (the model sees everything), ~5-7x cheaper, and it
deletes the gating/sweep/rolling-cache machinery. The live signal survives for
free: the deterministic `TriggerFilter` ticks a silent **"possible deadline
mentions"** badge during class; the model confirms afterwards. Lost: model-
confirmed suggestions mid-lecture and the live conflicting-date marker. The
pitch changes from "live extraction" to "live capture and flagging, zero-cost
live signal, model judgment after class".

## D16 - Recap model: Sonnet 5 `high`, no toggle
**Date:** 2026-09-16
**Status:** DECIDED (supersedes D10)

Sonnet 5 at `high` effort for the recap; no premium mode. Opus 5 stays in the
**eval only**, to answer "what would the expensive model have bought", which is
evidence for the choice rather than a feature. Fable 5.1 is dropped entirely.
Consequence: every Claude call in the product is Sonnet 5, so the portfolio
claim becomes **model-minimal routing** (deterministic wherever possible, the
cheapest effort that holds quality, one cached bundle) rather than multi-model
routing. That is the honest version.

## D17 - Stack: Python backend with faster-whisper in-process, React/Vite UI
**Date:** 2026-09-16
**Status:** DECIDED (resolves O3)

FastAPI + uvicorn on localhost, faster-whisper in a worker thread of the same
process, SQLite via the standard library, `pynput` for the global hotkey,
Anthropic Python SDK, React + Vite + TypeScript frontend served by the backend
in production. One command: `uv run lecture-copilot`. Rejected: Node + Python
sidecar (two runtimes), HTMX (weaker live UI), Tauri (most new tech).

## D18 - LectureBundle: one cached prefix shared by every post-lecture call
**Status:** PROPOSED

The lecture materials (course context, SlideIndex, slide text, transcript,
board text, flags) are assembled once into a `LectureBundle` placed in the
**system prompt behind a cache breakpoint**; each task (extraction, recap,
follow-up questions) is a short user message after it. Calls run back-to-back,
so the 5-minute TTL is enough. Effect: the ~25K-token bundle is billed once at
1.25x and then read at 0.1x, which makes "ask this lecture a question" and
"regenerate recap" nearly free. Cache reads verified in a standing test.

## D19 - Remaining defaults, set under D13 (user may veto any line)
**Date:** 2026-09-16
**Status:** DECIDED by default

- **Recruiter demo:** read-only replay of a public lecture with pre-generated
  outputs and the cost chart; no live public instance.
- **Other voices:** a global "pause capture" hotkey for Q&A stretches, and
  transcripts are retained 30 days by default (configurable; recaps, flags and
  confirmed events are kept). Documented limitation: no diarisation in v1.
- **Deck input to the recap:** extracted slide text + SlideIndex (local
  `pymupdf` / `python-pptx`, no PPTX-to-PDF conversion, no page images). Page
  images for diagram-heavy slides are an eval config, not a v1 feature.
- **SlideIndex:** built by one Sonnet 5 `low` call over locally extracted text
  (~$0.03 per deck), not over the PDF.
- **Labelling:** staged; deadline labels and grader calibration first.
- **MCP:** a local read-only stdio server (`search_transcripts`, `get_recap`,
  `list_open_flags`) as the last v1 milestone, for Claude Desktop / Claude Code.
- **Google Calendar:** scope `calendar.app.created` (the app writes only to a
  calendar it creates), consent screen set to In production without
  verification (personal use, <100 users) to avoid 7-day token expiry, personal
  Google account recommended (Workspace admins may block the app), client-
  generated event ids for de-duplication.
- **Notion:** mapping to an existing database via a config file, API version
  2025-09-03+ with `data_source_id` parents, a hidden `copilot_id` property
  queried before every create (no native idempotency), <=3 requests/s.
- **O2 photo retention:** the app keeps no images; originals stay in the
  phone's camera roll under the user's control, so OCR spot-checks use them on
  request.
- **Eval data:** MIT OCW lectures (6.0001 F16 first) under CC BY-NC-SA 4.0 in a
  separately licensed `eval/data` folder, third-party-excluded slide content
  stripped, transcripts normalised before WER.
- **Sleep:** the backend blocks system sleep while a lecture is recording
  (Windows `SetThreadExecutionState`); lid-close behaviour is documented.

---

## D20 - ASR defaults set from the M0 measurement
**Date:** 2026-09-16
**Status:** DECIDED (measured; see docs/ASR_BENCHMARK.md)

On this laptop `large-v3-turbo` (int8_float16, GPU) runs at RTF 0.077 with the
best accuracy, so it is the AC default. `small` is the battery default and the
slow-down fallback; `base` is excluded (WER 0.65). Language is pinned per course
because auto-detect chose French on English audio three times. The replay
harness also found that a queue built during the cold start tripped the
downgrade, so downgrade now requires both a backlog *and* a measured RTF above
0.7; a fast model drains a transient queue on its own.

---

## D21 - No Anthropic key: provider-agnostic gateway; local Ollama first, Cloudflare Workers AI free tier second
**Date:** 2026-09-16
**Status:** DECIDED (user: no Claude API key, "so expensive"; use the
Cloudflare token, and if that fails a local model via Ollama, already
installed). Supersedes D16; D13's "Claude API is the only network dependency"
becomes "no network dependency at all on the local provider"; D18's cached
prefix is Anthropic-only.

**Measured on 2026-09-16:**
- Cloudflare token: active, account-owned, expires 2027-03-16, but **no
  Workers AI permission** (401/403). Fix is in the dashboard: add
  "Workers AI - Read" and "Workers AI - Edit", or create a token from the
  Workers AI template. Free tier 10,000 neurons/day, then $0.011 per 1k;
  llama-3.3-70b at $0.293 / $2.253 per M tokens; JSON mode on the 70b/8b text
  models, not on vision models.
- Ollama 0.34, `llama3.1` (8B): structured output via `format` schema was
  valid and found all six real events on the fixture, with the correction
  merged into one corrected date. It **confused the `type` and `intent`
  fields**, so categorical fields are now enum-constrained in every schema
  (the grammar then forbids the confusion). At `num_ctx` 16384 it ran at
  9 tok/s with a 43 s load: the KV cache spilled off the 6 GB GPU. Context
  budget, not model quality, is the binding local constraint.
- No local vision model was installed; `gemma3:4b` (3.3 GB, vision, 128K
  context) is the only one that fits comfortably in 6 GB and was pulled.
- **gemma3:4b measured** (docs/LLM_BENCHMARK.md): 100% GPU at 8K and 16K
  context, 54 tok/s, 14–20 s per extraction, schema-valid, best semantics of
  the local candidates; spills to CPU at 32K. Vision path read every term on
  a synthetic board in 26 s. `llama3.2` (3B) hung >10 min without an output
  cap. **Local defaults: gemma3:4b for text and vision, 16K context.**

**Consequences:**
- `LlmGateway.structured()` keeps one signature; providers are
  `ollama | cloudflare | anthropic`, chosen by `LC_LLM_PROVIDER`
  (default `ollama`). Invalid output gets one corrective retry, then raises.
- Local calls run with an ~8K context, so post-lecture steps must **chunk the
  lecture (map-reduce)** on the local provider; single-pass stays for cloud
  providers.
- Cost model: $0 on Ollama; effectively $0 on Workers AI within the daily free
  tier. Every cost figure in ARCHITECTURE.md section 6 is superseded.
- Privacy: on Ollama **nothing leaves the laptop, even after class**, stronger
  than D13 promised. On Cloudflare, derived text and photos go to Cloudflare
  instead of Anthropic; the README must say which provider is active.
- Portfolio claim shifts to: provider-agnostic pipeline, measured across a
  local 8B, a free-tier 70B and (optionally) Claude, with the same eval.

---

## D22 - Suggestion trust policy (M3)
**Date:** 2026-09-16
**Status:** DECIDED by default (thresholds are settings-free constants until
the eval says otherwise)

- **Tiers.** `one_tap` requires all of: intent `commitment` or `correction`,
  model confidence >= 0.7, ASR confidence of the evidence segment >= 0.5,
  a resolved date with resolver confidence >= 0.6, the date not in the past,
  and no `course_hint`. Everything else that isn't log-only is `maybe`: the
  card shows the quote and asks for a date, and can't be one-tapped.
  `hypothetical`, `joke`, `past_reference`, duplicates and superseded
  candidates are log-only (kept in `candidate_events` for the eval).
- **Dates in code (D9), future-preferring.** A bare "September 1" said in
  mid-September resolves to next year; "next Thursday" said on a Thursday
  means the following week; "week 7" needs the term calendar and "next time"
  needs the course schedule, otherwise the item is `maybe` with an honest
  note. Every resolution carries the arithmetic as text for the card.
- **Corrections supersede.** A `correction` with a matching type and title
  hides the commitment it corrects (measured on the fixture: the midterm
  surfaces as Oct 21, never Oct 14). Duplicates across windows keep the more
  confident one. Title matching normalises number words ("four" = "4").
- **Idempotent.** Suggestions are keyed by course + type + normalised title +
  resolved date; re-running extraction never duplicates a card.
- **Local chunking.** 10-minute windows with 60 s overlap on the local
  provider (16K context); one window on cloud providers. Keyword-filter hits
  are passed as hints, not as decisions.
- **Export before integrations.** Until M5, a confirmed suggestion exports
  as `.ics`; confirming never writes anywhere by itself.

---

## D23 - UI design system: dimmed hall, glass panels, one highlighter
**Date:** 2026-09-16
**Status:** DECIDED (user asked for liquid-glass, deliberate typography, nothing that reads as generated)

- **Material:** a dimmed lecture hall (ink `#0A0F1E`) with a warm projector beam
  top-left and faint chalk-dust grain; translucent glass panels with a specular
  top edge and backdrop blur; a floating glass toolbar with a segmented tab
  control.
- **Colour is a whiteboard-marker set**, not one accent: highlighter yellow
  `#FFD54A` for the flag and anything the student did, marker blue `#6FA0FF`
  for timestamps and actions, marker red `#FF6A5C` for recording and delete,
  marker green `#58C99A` for ok.
- **Type:** Bricolage Grotesque for display (wordmark, section titles, big
  numbers), IBM Plex Sans for body, IBM Plex Mono for anything measured
  (timestamps, dates, pills). Loaded from Google Fonts with system fallbacks
  so an offline lecture hall still works.
- **Signature:** the highlighter. The flag button is a yellow liquid-glass
  lozenge whose refraction follows the pointer; flagged transcript is drawn
  as a highlighter stroke across the words; the flagged moment in the recap
  sits in a yellow glass panel.
- **Copy rules:** sentence case, plain verbs that name what happens ("Find
  deadlines", "Write recap", "Ready", "Needs a date"); empty states say what
  to do next; errors say what went wrong.
- **Routes:** `#lectures`, `#lectures/<id>`, `#inbox`, `#dashboard` (also lets
  the redesign be screenshot-checked headlessly).
- Reviewed against headless screenshots; recap output from the local model is
  normalised in code (markdown stripped, inline `[01:36]` and bare `t=..`
  citations folded into source chips, invented slide/board sources dropped,
  `gaps_note` kept only when the recording had gaps).

---

## Open items
- Cloudflare token permissions (user action) before the cloud provider can be tested.
- ~~Local context budget~~ resolved: 16K on gemma3:4b; lecture chunking is part of M3/M4.
- Tier thresholds (D22) are unvalidated constants until the OCW eval (M6).
- M5 targets (Google Calendar via `calendar.app.created`, Notion via data-source
  pages with a hidden `copilot_id`) are implemented and unit-tested against
  fakes; real-service calls need the user's own credentials and are untested
  until then.
