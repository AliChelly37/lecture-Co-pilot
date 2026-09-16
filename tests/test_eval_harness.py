"""The eval harness must be trustworthy before its numbers are: case
generation is deterministic and correctly labelled, and the scorer credits
and penalises the right things."""

import random
import sys
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from harness.inserts import build_case, make_inserts
from harness.recap_citations import check_recap
from harness.score import score_case

BASE = [
    "Fugacity is the effective pressure of a real gas.",
    "For an ideal gas the fugacity coefficient is one.",
    "We compute Z from the equation of state.",
]


def test_inserts_have_exact_labels() -> None:
    rng = random.Random(1)
    inserts = make_inserts(date(2026, 9, 16), date(2026, 9, 7), rng, 3)
    intents = [g.intent for _, g in inserts]
    assert intents.count("commitment") == 4  # 3 generated + the "next time" reading (expect maybe)
    assert {"correction", "superseded", "hypothetical", "joke", "past_reference", "tentative", "other_course"} <= set(intents)
    corr = next(g for _, g in inserts if g.intent == "correction")
    sup = next(g for _, g in inserts if g.intent == "superseded")
    assert corr.date > sup.date and corr.expect == "one_tap" and sup.expect == "absent"
    for _, g in inserts:
        if g.expect == "one_tap":
            assert g.date is not None and date.fromisoformat(g.date) > date(2026, 9, 16)


def test_build_case_is_deterministic_and_ordered() -> None:
    start = datetime(2026, 9, 16, 10, 0, tzinfo=ZoneInfo("Europe/Paris"))
    a = build_case("c", BASE, start, "Europe/Paris", date(2026, 9, 7), seed=5, n_commitments=2, target_minutes=3)
    b = build_case("c", BASE, start, "Europe/Paris", date(2026, 9, 7), seed=5, n_commitments=2, target_minutes=3)
    assert a == b
    gold = a["gold"]
    t_sup = next(g["t0"] for g in gold if g["intent"] == "superseded")
    t_corr = next(g["t0"] for g in gold if g["intent"] == "correction")
    assert t_sup < t_corr  # the correction is spoken after the original
    texts = {s["text"] for s in a["segments"]}
    assert all(g["text"] in texts for g in gold)
    assert all(a["segments"][i]["t0"] < a["segments"][i + 1]["t0"] for i in range(len(a["segments"]) - 1))


def sugg(title: str, tier: str, date_: str | None, quote: str = "", sid: str | None = None) -> dict:
    return {
        "id": sid or f"{title}-{tier}-{date_}",
        "tier": tier,
        "state": "proposed",
        "payload": {"title": title, "date": date_, "evidence_quote": quote},
    }


def test_scorer_credits_hits_and_flags_traps() -> None:
    gold = [
        {
            "title": "Problem set 4",
            "type": "assignment",
            "intent": "commitment",
            "date": "2026-09-17",
            "time": "17:00",
            "expect": "one_tap",
            "t0": 10,
            "text": "Problem set 4 is due next Thursday at 5 p.m.",
        },
        {
            "title": "midterm exam",
            "type": "exam",
            "intent": "superseded",
            "date": "2026-10-14",
            "time": None,
            "expect": "absent",
            "t0": 20,
            "text": "The midterm exam is on October 14th.",
        },
        {
            "title": "midterm exam",
            "type": "exam",
            "intent": "correction",
            "date": "2026-10-21",
            "time": None,
            "expect": "one_tap",
            "t0": 30,
            "text": "Actually the midterm moved to October 21st.",
        },
        {
            "title": "hypothetical problem set",
            "type": "assignment",
            "intent": "hypothetical",
            "date": None,
            "time": None,
            "expect": "absent",
            "t0": 40,
            "text": "If this problem set were due tomorrow you would panic.",
        },
        {
            "title": "second quiz",
            "type": "quiz",
            "intent": "tentative",
            "date": None,
            "time": None,
            "expect": "maybe",
            "t0": 50,
            "text": "We might do a second quiz around week eight.",
        },
    ]
    preds = [
        sugg("Problem set four", "one_tap", "2026-09-17"),
        sugg("Midterm exam", "one_tap", "2026-10-21"),
        sugg("second quiz", "maybe", None),
    ]
    s = score_case("t", gold, preds)
    assert s.one_tap_hits == 2 and s.gold_one_tap == 2 and s.precision == 1.0 and s.recall == 1.0
    assert s.trap_one_taps == 0 and s.false_one_taps == 0 and s.date_exact == 2 and s.maybe_hits == 1

    # A hypothetical promoted to one-tap, and the superseded date surfacing, are both trust failures.
    bad = [
        *preds,
        sugg("Problem set", "one_tap", "2026-09-17", quote="If this problem set were due tomorrow you would panic.", sid="hypo"),
        sugg("midterm exam", "one_tap", "2026-10-14", sid="old"),
    ]
    s2 = score_case("t", gold, bad)
    assert s2.trap_one_taps == 2 and s2.precision < 1.0


def test_recap_citation_check() -> None:
    segments = [
        {"t0": 0, "t1": 10, "text": "fugacity is the effective pressure of a real gas"},
        {"t0": 60, "t1": 70, "text": "the residual gibbs energy integral vanishes for an ideal gas"},
    ]
    sections = {
        "highlights": ["Fugacity is an effective pressure (t=00:05)", "Unsupported claim about entropy (t=01:05)", "No source at all"],
        "concepts": [{"name": "Residual Gibbs energy", "explanation": "integral that vanishes for an ideal gas", "sources": ["t=01:02"]}],
        "review_questions": [],
        "flag_explanations": [{"what_was_confusing": "x", "explanation": "y", "sources": ["t=09:00"]}],
    }
    r = check_recap(sections, segments)
    assert r["claims"] == 5 and r["with_source"] == 4 and r["valid_timestamp"] == 3 and r["supported_by_window"] == 2
    assert any("beyond the lecture" in p for p in r["problems"]) and any("no source" in p for p in r["problems"])
