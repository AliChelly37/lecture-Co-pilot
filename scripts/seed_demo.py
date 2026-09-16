"""Seed a demo lecture from the synthetic fixture so the UI has content:
a course, a lecture with the fixture transcript on a plausible timeline, one
confusion flag, then (unless --no-model) deadline extraction and a recap on
the configured provider.

  uv run python scripts/seed_demo.py            # uses the app's data/ database
  uv run python scripts/seed_demo.py --no-model # transcript + flag only
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lecture_copilot import pipeline  # noqa: E402
from lecture_copilot.config import settings  # noqa: E402
from lecture_copilot.llm import LlmGateway  # noqa: E402
from lecture_copilot.store import Store  # noqa: E402
from lecture_copilot.trigger import score_segment  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-model", action="store_true")
    ap.add_argument("--course", default="Thermodynamics II")
    args = ap.parse_args()

    store = Store(settings.db_path)
    course = next((c for c in store.list_courses() if c["name"] == args.course), None) or store.create_course(
        args.course, "Europe/Paris", {"week1_start": "2026-09-07"}
    )
    lecture = store.start_lecture(course["id"], "gemma3:4b (demo)", "cuda", "battery", 47)

    text = (ROOT / "eval/fixtures/tts_lecture.txt").read_text(encoding="utf-8")
    t = 0.0
    flag_t = None
    badge = 0
    for sentence in [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]:
        dur = max(2.0, len(sentence.split()) / 2.5)
        trig = score_segment(sentence)
        badge += int(trig.hit)
        store.add_segment(lecture["id"], t, t + dur, sentence, 0.85, "gemma3:4b (demo)", trig.score, trig.terms if trig.hit else [])
        if "residual Gibbs energy divided by" in sentence:
            flag_t = t + dur
        t += dur + 0.5
    if flag_t is not None:
        store.add_flag(lecture["id"], flag_t, max(0.0, flag_t - 90), flag_t + 15)
    store.end_lecture(lecture["id"], 0.077, 44)
    print(f"seeded lecture {lecture['id']} in course '{course['name']}': {t:.0f}s of transcript, badge hits {badge}, flag at {flag_t}")

    if args.no_model:
        store.close()
        return
    llm = LlmGateway(settings, store)
    print(f"provider: {llm.describe()}")
    summary = pipeline.extract_lecture(store, llm, lecture["id"], settings.extract_window_s, settings.extract_overlap_s)
    print("extraction:", summary)
    recap = pipeline.recap_lecture(
        store,
        llm,
        lecture["id"],
        settings.extract_window_s,
        progress=lambda ev: print(f"  recap {ev['done']}/{ev['total']} ({ev['stage']})"),
    )
    print(
        "recap:",
        recap["sections"]["title"],
        "| highlights:",
        len(recap["sections"]["highlights"]),
        "| concepts:",
        len(recap["sections"]["concepts"]),
    )
    store.close()


if __name__ == "__main__":
    main()
