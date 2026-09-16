# Lecture Co-Pilot

A local-first lecture assistant for students who lose focus mid-lecture. It
records and transcribes the lecture **on your own laptop** (no audio ever
leaves the machine), lets you flag "I didn't get that" with one key, and after
class uses a language model (local by default) to turn the transcript, slides
and whiteboard photos into a recap that explains exactly what you flagged, plus deadline suggestions you
confirm with a tap before they reach Google Calendar or Notion.

Assistive, not diagnostic. Single user. Built as a portfolio project with the
design written down before the code: see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md),
the decision log in [docs/DECISIONS.md](docs/DECISIONS.md) and the measured
speech-to-text benchmark in [docs/ASR_BENCHMARK.md](docs/ASR_BENCHMARK.md)
and the model-provider benchmark in [docs/LLM_BENCHMARK.md](docs/LLM_BENCHMARK.md).

## Status

- [x] M0 Whisper benchmark on the target laptop
- [x] M1 Local capture: mic, VAD, Whisper (GPU/CPU), live transcript, hotkey flag, deadline badge
- [x] M2 Deck ingestion (PDF/PPTX text), photo import (EXIF), board OCR, slide alignment, Lectures view
- [x] M3 Post-lecture deadline extraction (chunked for the local model), date resolution in code, suggestion inbox with one-tap / check-me tiers, .ics export
- [x] M4 Recap (map-reduce for the local model, per-flag explanations, source pointers, versions, ratings) and ask-the-lecture
- [ ] M5 Google Calendar and Notion targets (confirm-before-write)
- [ ] M6 Eval harness on MIT OCW lectures, cost chart
- [ ] M7 Dashboard, export/delete, retention, read-only MCP server

## Run it

Requirements: Python 3.11+, [uv](https://docs.astral.sh/uv/), Node 20+, an
NVIDIA GPU is optional (CPU fallback is automatic).

```powershell
uv sync                      # backend + Whisper runtime
cd frontend; npm install; npm run build; cd ..
uv run lecture-copilot       # opens http://127.0.0.1:8765
```

First start downloads the Whisper model (~1.6 GB for `large-v3-turbo`).

The model that reads slides and photos and writes the recap runs **locally**
through [Ollama](https://ollama.com) by default, so nothing leaves the laptop:

```powershell
ollama pull gemma3:4b        # 3.3 GB; text + vision, fits a 6 GB GPU
```

Optional cloud providers (set `LC_LLM_PROVIDER` in a gitignored `.env`):
`cloudflare` (Workers AI free tier; needs `CLOUDFLARE_API_TOKEN` with the
"Workers AI - Read/Edit" permission and `CLOUDFLARE_ACCOUNT_ID`) or
`anthropic` (needs `ANTHROPIC_API_KEY`). Capture and transcription work with
no model provider at all.
Hotkeys while recording: **F9** flag, **F10** pause/resume.

After a lecture: open **Lectures**, attach the slide deck and import board photos,
then **Extract deadlines**. Detected items land in **Inbox**: ready ones confirm
with one tap, uncertain ones ask you for a date first. Nothing is ever written
anywhere without that tap; confirmed items export as `.ics` (calendar and
Notion targets come in M5). **Generate recap** builds the study recap: highlights,
ranked concepts, an explanation for every moment you flagged, review questions,
and off-slide material, each with a timestamp you can tap to jump to the
transcript. Then ask the lecture questions.

Settings are environment variables prefixed `LC_` (or a `.env` file), e.g.
`LC_ASR_MODEL_AC=small`. See `src/lecture_copilot/config.py`.

## Develop

```powershell
uv run pytest -q
uv run ruff check src tests scripts
cd frontend; npm run dev      # Vite dev server proxying to the backend
uv run python scripts/simulate_lecture.py eval/fixtures/tts_lecture.wav --speed 4
uv run python scripts/probe_llm.py ollama --model gemma3:4b     # extraction probe
uv run python scripts/probe_vision.py                           # board-reading probe
```

## Privacy

- Audio is processed in memory and never written to disk or sent anywhere.
- Only derived text (transcript, OCR text, recaps) is stored, in a local SQLite file you can export or delete.
- With the default local model provider, nothing leaves the laptop at any point. With a cloud provider, only text and downscaled board photos leave, after class.
- Check your institution's recording policy before recording a lecture.
