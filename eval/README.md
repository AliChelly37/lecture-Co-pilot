# Evaluation

The claims in the top-level README are only as good as these numbers. Every
run uses a throwaway database; nothing here touches your lectures.

## Layout

```
eval/
  fixtures/      synthetic lecture (Windows TTS) used by the ASR benchmark and the seed
  cases/         labelled extraction cases (JSON: segments + gold events), committed
  harness/       the code: case generation, runners, scorers
  results/       one JSON per run + summary.md (append-only)
  data/          (gitignored) public lectures you download yourself; see licence note
```

## Cases

`harness/make_cases.py` builds five synthetic cases: ~22 minutes of filler
sentences with administrative utterances inserted at known positions, each
with a gold intent and (where resolvable) a gold date computed from the case's
lecture date and term calendar. Each case contains 4 commitments, one
correction pair (the second date must win), and traps: a hypothetical, a
joke, a past reference, a tentative mention, another course's deadline, and a
reading "for next time" that is unresolvable without a schedule.

Synthetic filler is a limitation: the model is tested on intent and date
handling, not on real lecture noise. Real cases replace the filler when public
lecture transcripts are added (below), using the same JSON format.

## Runs

```powershell
cd eval
uv run python -m harness.make_cases
uv run python -m harness.run_extraction --provider ollama --model gemma3:4b
uv run python -m harness.run_extraction --provider cloudflare --model @cf/meta/llama-3.3-70b-instruct-fp8-fast
uv run python -m harness.recap_citations --lecture <lecture id>
```

### Extraction metrics
- **traps → one-tap** (the trust metric): non-events that reached the one-tap
  tier. The target is 0. A single one would put a fake deadline one tap away
  from your calendar.
- **one-tap precision / recall** against gold commitments (corrections count
  with their corrected date; the superseded date must not appear).
- **dates exact**: of the one-tap hits, how many resolved to the gold date.
- **maybe surfaced**: tentative, other-course and unresolvable items should
  appear in the "needs a date" tray, not vanish.
- time and cost per run, from the usage ledger.

### Recap citations
`recap_citations.py` checks a stored recap deterministically: every claim's
`t=MM:SS` sources must exist in the transcript, and the cited 90-second window
must share content words with the claim. It reports the support rate and lists
the claims that fail. It cannot judge whether an explanation is *correct*; that
needs the labelled checklists and a judge (L3 in docs/ARCHITECTURE.md).

## Public lecture data (not included)

MIT OpenCourseWare lectures (e.g. 6.0001 Fall 2016) come with transcripts and
slides under **CC BY-NC-SA 4.0**. Anything derived from them belongs in
`eval/data/` under that licence, separate from the code licence, with
attribution to MIT OCW and no implied endorsement; slide content marked
"excluded from our Creative Commons license" must be stripped. OCW transcripts
are cleaned text, so normalise before computing WER.

To add one: download the lecture's transcript, save it as
`eval/data/<course>/<lecture>.txt`, then build a case with `harness.inserts`
(insert the labelled utterances into the real transcript) so the gold labels
stay exact.
