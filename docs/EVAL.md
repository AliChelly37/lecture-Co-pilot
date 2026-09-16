# Evaluation (M6)

How the claims in the README are checked. The harness lives in
[eval/harness](../eval/harness) and is documented in [eval/README.md](../eval/README.md).
Every run uses a throwaway database and writes a JSON record plus a row in
`eval/results/summary.md`.

## What is measured, and why these metrics

| Layer | Question | How |
|---|---|---|
| L0 deterministic | Do the code paths that never touch a model behave? | 61 unit tests: date resolution tables (weekdays, relative dates, term weeks, DST-free timezone handling), trigger-filter hits and misses, merge/supersede rules, suggestion state machine, targets against fakes, recap normalisation, the scorer itself |
| L2 extraction | Can a fake deadline reach the one-tap tier? How much of the real ones does it find, with the right date? | `harness.run_extraction` on labelled cases (below) through the **real** pipeline: chunking, model call, evidence location, date resolution, merge, tiers |
| L3 recap (part 1) | Do the recap's citations point at real, relevant moments? | `harness.recap_citations`: every `t=MM:SS` must exist and its 90-second window must share content words with the claim |
| ASR | Does the speech-to-text keep up, and how accurate is it? | `scripts/bench_asr.py` (docs/ASR_BENCHMARK.md); real WER needs public lectures |

### The trust metric
The one number that matters most is **traps → one-tap**: the count of
non-events (a hypothetical "if this were due tomorrow", a joke, last year's
midterm, a superseded date) that reached the tier where one tap would write
them to the calendar. Its target is zero. Precision and recall come after it.

## Cases

Five synthetic cases (`eval/cases/synthetic-*.json`, ~22 minutes each), built
by construction: filler sentences from the fixture transcript with labelled
utterances inserted at known times. Each case carries 4 commitments with
resolvable dates, one correction pair (the later date must win, the earlier
must not surface), and traps (hypothetical, joke, past reference), plus a
tentative item, another course's deadline and an unresolvable "reading for
next time" that should land in the "needs a date" tray rather than vanish.

Because the labels are constructed, they are exact, and the false-one-tap
count is trustworthy. The limitation is equally clear: the filler is
synthetic, so the model is tested on intent and date handling, not on the
noise of a real 90-minute lecture with a real Whisper transcript. Public
lecture cases (MIT OCW, CC BY-NC-SA 4.0, downloaded by the user) use the same
format and replace the filler.

## Results

### Recap citations (demo lecture, gemma3:4b, 2026-09-16)
12 claims; 10 carried timestamps; 10/10 timestamps exist in the transcript and
10/10 cited windows support their claim (support rate 1.0). The two claims
without a source were the recap's generic opening highlights. This check ran
after recap normalisation folded the model's inline citations into source
chips.

### Extraction (gemma3:4b, local, 5 synthetic cases, 25 gold one-tap events, 2026-09-16)

Each row is the same model on the same cases; only the deterministic rules
between the model and the one-tap tier changed. Every rule was added because
the previous run showed the failure (D25).

| run | rules | one-tap precision | one-tap recall | traps → one-tap | false one-taps | dates exact | time |
|---|---|---|---|---|---|---|---|
| 1–2 | none (model + resolver + merge) | 0.60–0.71 | 0.48 | **1** (a joke) | 4–7 | 9/12 | ~9 min |
| 3 | grounding, joke/hypothetical markers on the quote, resolver precedence | 0.85 | 0.44 | **1** | 1 | 9/11 | 7.7 min |
| 4 | markers read the transcript line, spoken-only course hints, no-date → log, date rescue | **1.00** | 0.68 | **0** | **0** | 13/17 | 7.9 min |
| 5 | other-course phrase, plain-statement override, keyword-filter fallback, scorer exact-date-first matching | 1.00 | 0.60 | 0 | 0 | 13/15 | 7.6 min |
| 6 | fallback runs after merge; superseded cards count as covered; other-course cards kept out of merge | **1.00** | **0.68** | **0** | **0** | **16/17** | 7.6 min |

Run 6 also reports **surfaced recall 0.92** (found in either tray) and
**needs-a-date → one-tap 0** (no tentative or other-course item promoted).

What the numbers say: the trust metric reached zero once the rules stopped
trusting the model's own quote and hint fields, and stayed at zero through
the recall work. One-tap recall is bounded by the 4B model skipping or
mislabelling plain deadline sentences; the keyword-filter fallback turns most
of those into "needs a date" cards (surfaced recall 0.92), which is the
honest place for them: a card the student glances at, never one tap from the
calendar. The two items still unsurfaced are a "reading for next time" that
cannot be dated without a course schedule and one other-course mention the
model skipped. Two runs with identical settings differed (rows 1–2): Ollama
at temperature 0 is not bit-for-bit deterministic across loads, so single
runs are indicative, not exact.

The same harness runs unchanged against Cloudflare's free-tier 70B or Claude
(`--provider cloudflare|anthropic`), which is how the "what would a bigger
model buy" comparison will be made once a token with the right permission
exists.

<!-- RESULTS -->

## What the eval does not yet cover
- Recap **correctness** (whether a flag explanation is right) needs labelled
  key-point checklists and a judge; with no paid API the judge would be the
  same local model, so that number is deliberately not claimed.
- Real lecture audio end to end: the ASR benchmark uses synthetic speech.
- OCR accuracy on handwritten boards: only a rendered synthetic board so far.
