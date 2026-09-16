"""Labelled cases by construction (D11): administrative utterances with known
intent and date are inserted into a base transcript at known positions. The
gold labels are exact because we wrote them, which is what makes the
false-one-tap metric trustworthy.

A case is JSON: {"name", "lecture_start", "timezone", "term_calendar",
"segments": [{t0, t1, text, asr_conf}], "gold": [GoldEvent...]}.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday"]
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"]
TIMES = [("5 p.m.", "17:00"), ("noon", "12:00"), ("9 a.m.", "09:00"), ("midnight", "23:59")]
COURSES = ["Statistics", "Linear Algebra", "Organic Chemistry"]


@dataclass
class GoldEvent:
    title: str
    type: str
    intent: str  # commitment | correction | tentative | hypothetical | joke | past_reference | other_course
    date: str | None  # ISO date the resolver should produce, or None if unresolvable by design
    time: str | None
    expect: str  # one_tap | maybe | absent
    t0: float
    text: str


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _future_weekday(base: date, rng: random.Random) -> tuple[str, date]:
    name = rng.choice(WEEKDAYS)
    target = WEEKDAYS.index(name)
    ahead = (target - base.weekday()) % 7 or 7
    return name, base + timedelta(days=ahead)


def _future_month_day(base: date, rng: random.Random) -> tuple[str, date]:
    d = base + timedelta(days=rng.randint(10, 60))
    return f"{MONTHS[d.month - 1]} {_ordinal(d.day)}", d


def make_inserts(base: date, term_week1: date, rng: random.Random, n: int) -> list[tuple[str, GoldEvent]]:
    """Return (sentence, gold) pairs. `n` commitments plus a fixed set of traps."""
    out: list[tuple[str, GoldEvent]] = []
    k = rng.randint(2, 6)
    # commitments (one-tap expected)
    for i in range(n):
        kind = rng.choice(["due_weekday", "due_month", "quiz", "reading_week", "project_month"])
        if kind == "due_weekday":
            wd, d = _future_weekday(base, rng)
            tt, tv = rng.choice(TIMES)
            s = f"Problem set {k + i} is due next {wd} at {tt}, submitted through the course website."
            out.append((s, GoldEvent(f"Problem set {k + i}", "assignment", "commitment", d.isoformat(), tv, "one_tap", 0, s)))
        elif kind == "due_month":
            md, d = _future_month_day(base, rng)
            s = f"Please hand in the lab report by {md}, no extensions this time."
            out.append((s, GoldEvent("lab report", "assignment", "commitment", d.isoformat(), None, "one_tap", 0, s)))
        elif kind == "quiz":
            wd, d = _future_weekday(base, rng)
            s = f"There will be a short quiz on {wd} covering everything up to today."
            out.append((s, GoldEvent("quiz", "quiz", "commitment", d.isoformat(), None, "one_tap", 0, s)))
        elif kind == "reading_week":
            w = rng.randint(3, 9)
            d = term_week1 + timedelta(weeks=w - 1)
            s = f"The reading for week {w} is chapter {rng.randint(3, 14)}; make sure it is done before that week's lecture."
            out.append((s, GoldEvent(f"reading week {w}", "reading", "commitment", d.isoformat(), None, "one_tap", 0, s)))
        else:
            md, d = _future_month_day(base, rng)
            s = f"Project proposals are due on {md}; one page, submitted as a PDF."
            out.append((s, GoldEvent("project proposal", "project", "commitment", d.isoformat(), None, "one_tap", 0, s)))

    # a correction pair: the second mention wins
    md1, d1 = _future_month_day(base, rng)
    md2 = f"{MONTHS[d1.month - 1]} {_ordinal(min(d1.day + 7, 28))}"
    d2 = d1.replace(day=min(d1.day + 7, 28))
    s1 = f"The midterm exam is on {md1}."
    s2 = f"Actually, let me correct that, the midterm has moved to {md2}, same room, same time."
    out.append((s1, GoldEvent("midterm exam", "exam", "superseded", d1.isoformat(), None, "absent", 0, s1)))
    out.append((s2, GoldEvent("midterm exam", "exam", "correction", d2.isoformat(), None, "one_tap", 0, s2)))

    # traps: must never become one-tap
    s = "Now, if this problem set were due tomorrow, you would all be panicking, but it is not, so relax."
    out.append((s, GoldEvent("hypothetical problem set", "assignment", "hypothetical", None, None, "absent", 0, s)))
    s = "The final exam is tonight at midnight. I'm joking, of course, it's in December like always."
    out.append((s, GoldEvent("joke exam", "exam", "joke", None, None, "absent", 0, s)))
    s = "Last year the midterm was in week six, but this year we pushed it back."
    out.append((s, GoldEvent("last year's midterm", "exam", "past_reference", None, None, "absent", 0, s)))
    s = "We might do a second quiz around week eight, but I haven't decided yet."
    out.append((s, GoldEvent("second quiz", "quiz", "tentative", None, None, "maybe", 0, s)))
    other = rng.choice(COURSES)
    md, d = _future_month_day(base, rng)
    s = f"By the way, I hear your {other} midterm is on {md}, so plan your week."
    out.append((s, GoldEvent(f"{other} midterm", "exam", "other_course", d.isoformat(), None, "maybe", 0, s)))
    s = "The reading for next time is chapter eleven, sections one through four."
    out.append((s, GoldEvent("reading chapter 11", "reading", "commitment", None, None, "maybe", 0, s)))  # unresolvable without a schedule
    return out


def build_case(
    name: str,
    base_sentences: list[str],
    lecture_start: datetime,
    timezone: str,
    term_week1: date,
    seed: int,
    n_commitments: int = 4,
    target_minutes: float = 25.0,
) -> dict:
    rng = random.Random(seed)
    inserts = make_inserts(lecture_start.date(), term_week1, rng, n_commitments)
    # Filler: shuffle base sentences and repeat until the target length.
    filler: list[str] = []
    while sum(len(s.split()) for s in filler) / 150 < target_minutes:
        block = list(base_sentences)
        rng.shuffle(block)
        filler += block
    # Place inserts at spread-out positions; keep the correction after the original.
    positions = sorted(rng.sample(range(5, len(filler) - 5), len(inserts)))
    ordered = list(inserts)
    orig = next(i for i, (_, g) in enumerate(ordered) if g.intent == "superseded")
    corr = next(i for i, (_, g) in enumerate(ordered) if g.intent == "correction")
    if orig > corr:
        ordered[orig], ordered[corr] = ordered[corr], ordered[orig]
    plan: dict[int, tuple[str, GoldEvent]] = dict(zip(positions, ordered, strict=True))

    segments: list[dict] = []
    gold: list[GoldEvent] = []
    t = 0.0
    for i, sentence in enumerate(filler):
        for text, g in [plan[i]] if i in plan else []:
            dur = max(2.0, len(text.split()) / 2.5)
            g.t0 = round(t, 1)
            segments.append({"t0": round(t, 1), "t1": round(t + dur, 1), "text": text, "asr_conf": 0.85})
            gold.append(g)
            t += dur + 0.5
        dur = max(2.0, len(sentence.split()) / 2.5)
        segments.append({"t0": round(t, 1), "t1": round(t + dur, 1), "text": sentence, "asr_conf": 0.85})
        t += dur + 0.5
    return {
        "name": name,
        "seed": seed,
        "lecture_start": lecture_start.isoformat(),
        "timezone": timezone,
        "term_calendar": {"week1_start": term_week1.isoformat()},
        "segments": segments,
        "gold": [asdict(g) for g in gold],
    }


def save_case(case: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(case, indent=1, ensure_ascii=False), encoding="utf-8")


def load_case(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))
