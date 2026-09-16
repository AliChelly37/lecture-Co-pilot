"""Generate the synthetic labelled cases (deterministic; committed so results
are reproducible).

  uv run python -m harness.make_cases            # from eval/
"""

from __future__ import annotations

import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from harness.inserts import build_case, save_case  # noqa: E402

FIXTURE = ROOT / "eval/fixtures/tts_lecture.txt"
OUT = ROOT / "eval/cases"


def base_sentences() -> list[str]:
    text = FIXTURE.read_text(encoding="utf-8").replace("\n", " ")
    sentences = [s.strip() + "." for s in text.split(". ") if s.strip()]
    # Drop the fixture's own admin lines so the only labelled events are the inserts.
    admin = ("due", "midterm", "reading for", "problem set four", "before Thursday", "make sure you can derive")
    return [s for s in sentences if not any(a.lower() in s.lower() for a in admin)]


def main() -> None:
    base = base_sentences()
    tz = "Europe/Paris"
    week1 = date(2026, 9, 7)
    starts = [
        datetime(2026, 9, 16, 10, 0, tzinfo=ZoneInfo(tz)),
        datetime(2026, 9, 21, 14, 0, tzinfo=ZoneInfo(tz)),
        datetime(2026, 10, 2, 9, 0, tzinfo=ZoneInfo(tz)),
    ]
    for i, seed in enumerate([11, 23, 37, 41, 59]):
        case = build_case(f"synthetic-{seed}", base, starts[i % len(starts)], tz, week1, seed, n_commitments=4, target_minutes=22)
        save_case(case, OUT / f"synthetic-{seed}.json")
        gold = case["gold"]
        print(
            f"{case['name']}: {len(case['segments'])} segments, {len(gold)} gold events ({sum(g['expect'] == 'one_tap' for g in gold)} one-tap)"
        )


if __name__ == "__main__":
    main()
