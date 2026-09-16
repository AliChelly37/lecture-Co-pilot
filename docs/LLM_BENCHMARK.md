# Model-provider benchmark (D21)

Same task for every model: extract deadlines/exams/readings with intent labels
from `eval/fixtures/tts_lecture.txt` (473 prompt tokens, six real events, one
hypothetical, one joke, one correction, one past reference) as
schema-constrained JSON. Scripts: `scripts/probe_llm.py`, `scripts/probe_vision.py`.
Machine: RTX 4050 Laptop 6 GB, i5-12450HX, Ollama 0.34. All on battery.

## Text extraction, 2026-09-16

| Provider / model | Context | Placement | tok/s | Total | Schema valid | Semantics |
|---|---|---|---|---|---|---|
| ollama / llama3.1 (8B Q4) | 16384 | 43% CPU / 57% GPU | 9.0 | 128 s (43 s load) | yes | Found all 6 events; **confused `type` and `intent`** (free-text fields) |
| ollama / llama3.1 (8B Q4) | 8192 | 32% CPU / 68% GPU | 12.4 | 85 s | yes | same |
| ollama / llama3.2 (3B) | 8192 | — | — | **hung > 10 min** | — | No output cap + enum grammar: generated until timeout |
| **ollama / gemma3:4b** | 8192 | **100% GPU** | **54.3** | **14 s** | yes | Best: correction, past_reference and joke labelled correctly; one duplicate event |
| ollama / gemma3:4b | 16384 | **100% GPU** | 54.3 | 20 s | yes | same |
| ollama / gemma3:4b | 32768 | 19% CPU / 81% GPU | 47.2 | 26 s | yes | same (prompt processing would slow on a full-size prompt) |
| cloudflare / llama-3.3-70b-fp8-fast | — | cloud | — | **blocked** | — | Token lacks the Workers AI permission (401/403) |

## Board photo reading, 2026-09-16

Synthetic whiteboard (rendered text: a title, three equations, "PS4 due Thu
5pm"), through the real `LlmGateway`:

| Provider / model | Result |
|---|---|
| ollama / gemma3:4b | 26 s, $0, `content_kind=board`, legibility 1.0, **all expected terms read**; one equation slightly mangled; `latex[]` came back as plain lines |

Real handwritten boards will be harder; this only shows the path works.

## What the numbers decided

1. **Local default: `gemma3:4b` for text and vision.** An 8B model does not
   fit the 6 GB GPU with any useful context; the 4B does at 16K.
2. **Local context budget: 16K tokens** (`LC_OLLAMA_NUM_CTX=16384`). A 90-min
   lecture is ~15–25K tokens with slides, so post-lecture steps **chunk the
   lecture** on the local provider (about 10-minute windows) and merge.
3. **Schemas must be grammar-friendly:** enums for every categorical field
   (the 8B model put quotes into a free-text `intent`), `maxItems` on every
   list, and an output-token cap (`num_predict`) on every call. Without the
   cap a 3B model looped for more than 10 minutes.
4. **Cloudflare Workers AI** stays the cloud option for quality (70B, free
   tier 10k neurons/day) once the token has "Workers AI - Read/Edit".

## Recap and ask on the fixture, 2026-09-16 (gemma3:4b, real API path)

Whole pipeline for a 140 s lecture with one flag: 1 notes call + 1 flag call +
1 recap call + 1 question = **~70 s, $0**. The recap had a title, >=3 highlights,
>=2 concepts (fugacity among them) with `t=MM:SS` sources, a >80-character
flag explanation and >=2 review questions.

Two behaviours worth keeping in the design:
- **A repetition loop inside a string** ("198 198 198 ...") hit the 900-token
  output cap on the flag call; the JSON was invalid, the gateway's corrective
  retry produced a valid one. The cap and the retry are load-bearing.
- **Booleans are unreliable on the 4B model:** it answered a question
  correctly from the excerpts and still set `not_covered: true`. Categorical
  fields are now enums everywhere, including that one.

## Next measurements
- llama3.2 (3B) re-run with the output cap, for a speed floor.
- gemma3:4b on real transcript chunks (Whisper output, not clean text) and on
  real board photos.
- Cloudflare 70B on the same fixture once the token is fixed; then the
  OCW eval (M6) compares local 4B vs cloud 70B on precision/recall.
