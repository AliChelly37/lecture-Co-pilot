from datetime import date, datetime, time
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from lecture_copilot.dates import DateResolver, Resolution
from lecture_copilot.extract import Candidate, chunk_segments, locate, merge
from lecture_copilot.store import Store
from lecture_copilot.suggest import SuggestionService, idempotency_key, tier_for

TZ = "Europe/Paris"
LECTURE_DT = datetime(2026, 9, 16, 10, 0, tzinfo=ZoneInfo(TZ))


def seg(t0: float, t1: float, text: str, conf: float = 0.9) -> dict:
    return {"t0": t0, "t1": t1, "text": text, "asr_conf": conf}


def cand(
    title: str,
    intent: str = "commitment",
    typ: str = "assignment",
    expr: str = "next Thursday",
    conf: float = 0.95,
    asr: float = 0.9,
    hint: str = "",
) -> Candidate:
    r = DateResolver(LECTURE_DT, TZ, {"week1_start": "2026-09-07"}).resolve(expr)
    return Candidate(typ, title, expr, "", intent, f"quote about {title}", conf, hint, 10.0, asr, r)


def test_chunking_windows_overlap() -> None:
    segments = [seg(i * 30, i * 30 + 25, f"segment {i}") for i in range(60)]  # 30 min
    chunks = chunk_segments(segments, window_s=600, overlap_s=60)
    assert [c.index for c in chunks] == [0, 1, 2, 3]
    assert chunks[0].t0 == 0 and chunks[1].t0 == 540  # second window starts 60 s early
    assert "segment 19" in chunks[0].text and "segment 19" in chunks[1].text  # boundary seen twice
    assert chunk_segments(segments, window_s=0)[0].t1 == 1795  # single window mode


def test_locate_tolerates_asr_noise() -> None:
    segments = [
        seg(0, 5, "welcome back everyone"),
        seg(47.2, 53.5, "Problem Set 4 is due next Thursday at 5 p.m. Submit it through the website", 0.84),
    ]
    t0, conf = locate("Problem set four is due next Thursday at five p.m., submitted through the course website.", segments)
    assert t0 == 47.2 and conf == 0.84
    assert locate("something completely unrelated to anything", segments) == (None, 1.0)


def test_merge_duplicates_corrections_and_log() -> None:
    ps_a = cand("Problem set 4", conf=0.9)
    ps_b = cand("Problem set four", conf=0.98)  # duplicate from an overlapping window
    mid_old = cand("Midterm exam", typ="exam", expr="October 14")
    mid_new = cand("Midterm exam", intent="correction", typ="exam", expr="October 21")
    joke = cand("Navier-Stokes joke", intent="joke", typ="other", expr="")
    past = cand("Last year's midterm", intent="past_reference", typ="exam", expr="week 6")
    hypo = cand("Problem set 4", intent="hypothetical", expr="tomorrow")
    mid_new.t0 = 70.0  # the correction is spoken after the original (t0=10)
    out = merge([ps_a, ps_b, mid_old, mid_new, joke, past, hypo])
    assert ps_a.status == "duplicate" and out[ps_a.superseded_by] is ps_b and ps_b.status == "surfaced"
    assert mid_old.status == "superseded" and out[mid_old.superseded_by] is mid_new and mid_new.status == "surfaced"
    assert joke.status == "log" and past.status == "log" and hypo.status == "log"


def test_later_mention_wins_even_without_a_correction_label() -> None:
    # Measured on gemma3:4b: the corrected date sometimes comes back as a plain commitment.
    first = cand("Midterm exam", typ="exam", expr="October 14", conf=0.98)
    later = cand("midterm exam", typ="exam", expr="October 21", conf=0.9)
    first.t0, later.t0 = 64.5, 68.5
    merge([first, later])
    assert first.status == "superseded" and later.status == "surfaced"


def test_tier_policy() -> None:
    today = LECTURE_DT.date()
    assert tier_for(cand("Problem set 4"), today) == "one_tap"
    assert tier_for(cand("Midterm", intent="correction", typ="exam", expr="October 21"), today) == "one_tap"
    assert tier_for(cand("Problem set 4", intent="tentative"), today) == "maybe"
    assert tier_for(cand("Problem set 4", conf=0.5), today) == "maybe"  # low model confidence
    assert tier_for(cand("Problem set 4", asr=0.3), today) == "maybe"  # unclear audio
    assert tier_for(cand("Reading", typ="reading", expr="next time"), today) == "maybe"  # unresolved without schedule
    assert tier_for(cand("Stats midterm", typ="exam", expr="October 21", hint="Statistics"), today) == "maybe"
    assert tier_for(cand("Old thing", expr="September 1, 2026"), today) == "maybe"  # explicitly past
    # A bare "September 1" said mid-September means next year under future preference.
    assert tier_for(cand("Next year thing", expr="September 1"), today) == "one_tap"
    assert tier_for(cand("Joke", intent="joke", typ="other", expr=""), today) == "log"


def test_suggestion_lifecycle_and_ics(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.sqlite3")
    course = store.create_course("Thermo", TZ, {"week1_start": "2026-09-07"})
    lecture = store.start_lecture(course["id"], "gemma", "cuda", "ac", None)
    store.end_lecture(lecture["id"], 0.1, None)
    svc = SuggestionService(store)

    c1 = cand("Problem set 4", expr="next Thursday at 5 pm")
    c2 = cand("Reading chapter 11", typ="reading", expr="next time")  # maybe tier
    counts = svc.propose([c1, c2], lecture, course)
    assert counts == {"one_tap": 1, "maybe": 1, "log": 0, "existing": 0, "retired": 0}
    # Re-running extraction is idempotent.
    assert svc.propose([c1, c2], lecture, course)["existing"] == 2
    # A re-run that no longer produces c2 retires its still-proposed card.
    assert svc.propose([c1], lecture, course)["retired"] == 1
    assert len(store.suggestions(state="proposed")) == 1
    svc.propose([c1, c2], lecture, course)  # bring c2 back for the rest of the test

    one_tap = next(s for s in store.suggestions(state="proposed") if s["tier"] == "one_tap")
    maybe = next(s for s in store.suggestions(state="proposed") if s["tier"] == "maybe")
    assert one_tap["payload"]["date"] == "2026-09-17" and one_tap["payload"]["time"] == "17:00"
    assert "Wed 16 Sep 2026" in one_tap["payload"]["resolution_note"]

    with pytest.raises(ValueError):
        svc.confirm(maybe["id"])  # no date yet
    svc.set_date(maybe["id"], "2026-09-18")
    assert store.get_suggestion(maybe["id"])["tier"] == "one_tap"
    svc.confirm(maybe["id"])
    svc.confirm(one_tap["id"])
    svc.undo(one_tap["id"])
    assert store.get_suggestion(one_tap["id"])["state"] == "proposed"
    svc.dismiss(one_tap["id"])
    with pytest.raises(ValueError):
        svc.confirm(one_tap["id"])

    ics = svc.ics(store.suggestions(state="confirmed"))
    assert "BEGIN:VEVENT" in ics and "DTSTART;VALUE=DATE:20260918" in ics and "Thermo: Reading chapter 11" in ics
    assert idempotency_key(course["id"], c1) == idempotency_key(course["id"], cand("problem set 4", expr="next Thursday at 5 pm"))
    store.close()


def test_clean_hint_and_rerun_refreshes_proposed(tmp_path: Path) -> None:
    from lecture_copilot.extract import clean_hint

    assert clean_hint("None", "Thermodynamics II") == ""
    assert clean_hint("thermodynamics", "Thermodynamics II") == ""
    assert clean_hint("Statistics", "Thermodynamics II") == "Statistics"
    assert clean_hint(None, "X") == ""

    store = Store(tmp_path / "t.sqlite3")
    course = store.create_course("Thermo", TZ, {"week1_start": "2026-09-07"})
    lecture = store.start_lecture(course["id"], "g", "cuda", "ac", None)
    store.end_lecture(lecture["id"], 0.1, None)
    svc = SuggestionService(store)
    demoted = cand("Problem set 4", hint="None")  # what a small model produced
    assert svc.propose([demoted], lecture, course) == {"one_tap": 0, "maybe": 1, "log": 0, "existing": 0, "retired": 0}
    fixed = cand("Problem set 4", hint="")
    assert svc.propose([fixed], lecture, course)["existing"] == 1
    assert store.suggestions(state="proposed")[0]["tier"] == "one_tap"  # refreshed in place, same row
    assert len(store.suggestions()) == 1
    store.close()


def test_resolution_iso() -> None:
    assert Resolution(date(2026, 9, 17), time(17, 0), "", 1.0).iso() == "2026-09-17T17:00"
    assert Resolution(None, None, "", 0.0).iso() is None
