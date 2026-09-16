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
    return Candidate(typ, title, expr, "", intent, f"{title} is due {expr}", conf, hint, 10.0, asr, r)


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


def test_grounding_and_lexical_guards() -> None:
    from lecture_copilot.extract import date_grounded
    from lecture_copilot.suggest import tier_and_reason

    today = LECTURE_DT.date()
    # The date words must be in the quote or the located line (the model sprayed dates onto filler).
    assert date_grounded("next Thursday", "Problem set 4 is due next Thursday at 5 p.m.", None)
    assert date_grounded("October twenty first", "", "the midterm has moved to October 21st")
    assert not date_grounded("next Thursday", "Fugacity is the effective pressure of a real gas.", "Fugacity is the effective pressure.")
    ungrounded = cand("Fugacity")
    ungrounded.evidence_quote = "Fugacity is the effective pressure of a real gas."
    ungrounded.extras["segment_text"] = ungrounded.evidence_quote
    assert tier_and_reason(ungrounded, today) == ("maybe", "date words not in the quoted sentence")
    grounded = cand("Problem set 4")
    grounded.evidence_quote = "Problem set 4 is due next Thursday at 5 p.m."
    assert tier_and_reason(grounded, today)[0] == "one_tap"
    joke = cand("Final exam", typ="exam", expr="December")
    joke.evidence_quote = "The final exam is tonight at midnight."  # the model's quote omits the punchline...
    joke.extras["segment_text"] = "The final exam is tonight at midnight. I'm joking, of course, it's in December like always."
    assert tier_and_reason(joke, today) == ("log", "the line says it was a joke")  # ...but the transcript line has it
    hypo = cand("Problem set 4", expr="tomorrow")
    hypo.evidence_quote = "Now, if this problem set were due tomorrow, you would all be panicking."
    assert tier_and_reason(hypo, today)[0] == "maybe"
    tentative = cand("second quiz", typ="quiz", expr="week 8")
    tentative.evidence_quote = "We might do a second quiz around week eight, but I haven't decided yet."
    assert tier_and_reason(tentative, today) == ("maybe", "the wording sounds tentative")
    # An invented course code that nobody said is ignored; a spoken one is honoured.
    invented = cand("Midterm exam", typ="exam", expr="October 21", hint="CHEM 300")
    invented.evidence_quote = "The midterm exam is on October 21."
    assert tier_and_reason(invented, today)[0] == "one_tap"
    spoken = cand("Statistics midterm", typ="exam", expr="October 15", hint="Statistics")
    spoken.evidence_quote = "By the way, I hear your Statistics midterm is on October 15th."
    assert tier_and_reason(spoken, today) == ("maybe", "mentioned for another course: Statistics")
    # A "commitment" with no date anywhere is noise, not a card.
    noise = cand("fugacity", expr="None")
    noise.evidence_quote = "Fugacity is the effective pressure of a real gas."
    noise.extras["segment_text"] = noise.evidence_quote
    assert tier_and_reason(noise, today) == ("log", "no date mentioned")


def test_plain_statement_override_and_spoken_other_course() -> None:
    from lecture_copilot.suggest import tier_and_reason

    today = LECTURE_DT.date()
    mislabelled = cand("Problem set 6", intent="tentative", expr="next Monday")
    mislabelled.evidence_quote = "Problem set 6 is due next Monday at 9 a.m., submitted through the course website."
    tier, reason = tier_and_reason(mislabelled, today)
    assert tier == "one_tap" and "the model said tentative" in reason
    hedged = cand("second quiz", intent="tentative", typ="quiz", expr="week 8")
    hedged.evidence_quote = "We might do a second quiz around week eight."
    assert tier_and_reason(hedged, today)[0] == "maybe"
    other = cand("Organic Chemistry midterm", typ="exam", expr="November 8th", hint="")  # the model dropped the hint
    other.evidence_quote = "By the way, I hear your Organic Chemistry midterm is on November 8th, so plan your week."
    other.extras["course_name"] = "Thermodynamics II"
    assert tier_and_reason(other, today) == ("maybe", "mentioned for another course: Organic Chemistry")
    junk = cand("example", expr="three hundred kelvin and fifty bar")
    junk.evidence_quote = "Okay, let us work through an example with nitrogen at three hundred kelvin and fifty bar."
    junk.extras["segment_text"] = junk.evidence_quote
    assert tier_and_reason(junk, today) == ("log", "no date mentioned")


def test_find_date_phrases_for_rescue() -> None:
    from lecture_copilot.dates import find_date_phrases

    assert find_date_phrases("The reading for week 6 is chapter 11; make sure it is done before that week's lecture.") == ["week 6"]
    assert find_date_phrases("Problem set 4 is due next Thursday at 5 p.m.")[0] == "next thursday"
    assert find_date_phrases("Project proposals are due on November 14th; one page.") == ["november 14"]
    assert find_date_phrases("Fugacity is the effective pressure of a real gas.") == []


def test_tier_policy() -> None:
    today = LECTURE_DT.date()
    assert tier_for(cand("Problem set 4"), today) == "one_tap"
    assert tier_for(cand("Midterm", intent="correction", typ="exam", expr="October 21"), today) == "one_tap"
    hedged = cand("Problem set 4", intent="tentative")
    hedged.evidence_quote = "Problem set 4 might be due next Thursday, we'll see."
    assert tier_for(hedged, today) == "maybe"
    assert tier_for(cand("Problem set 4", conf=0.5), today) == "maybe"  # low model confidence
    assert tier_for(cand("Problem set 4", asr=0.3), today) == "maybe"  # unclear audio
    assert tier_for(cand("Reading", typ="reading", expr="next time"), today) == "maybe"  # unresolved without schedule
    assert tier_for(cand("Statistics midterm", typ="exam", expr="October 21", hint="Statistics"), today) == "maybe"  # hint is in the quote
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
    demoted = cand("Problem set 4", conf=0.5)  # first run: the model was unsure
    assert svc.propose([demoted], lecture, course) == {"one_tap": 0, "maybe": 1, "log": 0, "existing": 0, "retired": 0}
    fixed = cand("Problem set 4", conf=0.95)  # a re-run is confident: same key, refreshed tier
    assert svc.propose([fixed], lecture, course)["existing"] == 1
    assert store.suggestions(state="proposed")[0]["tier"] == "one_tap"  # refreshed in place, same row
    assert len(store.suggestions()) == 1
    store.close()


def test_resolution_iso() -> None:
    assert Resolution(date(2026, 9, 17), time(17, 0), "", 1.0).iso() == "2026-09-17T17:00"
    assert Resolution(None, None, "", 0.0).iso() is None
