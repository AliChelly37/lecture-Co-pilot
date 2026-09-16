from datetime import date, datetime, time
from zoneinfo import ZoneInfo

import pytest

from lecture_copilot.dates import DateResolver, normalise_numbers

TZ = "Europe/Paris"
# Wednesday 16 September 2026, 10:00 Paris. Week 1 started Monday 7 September.
LECTURE = datetime(2026, 9, 16, 10, 0, tzinfo=ZoneInfo(TZ))
CAL = {"week1_start": "2026-09-07", "next_lecture": "2026-09-18"}


@pytest.fixture
def r() -> DateResolver:
    return DateResolver(LECTURE, TZ, CAL)


def test_normalise_numbers() -> None:
    assert normalise_numbers("October twenty first") == "october 21"
    assert normalise_numbers("October fourteenth") == "october 14"
    assert normalise_numbers("five p.m.") == "5 pm"
    assert normalise_numbers("week seven") == "week 7"
    assert normalise_numbers("the 21st") == "the 21"


@pytest.mark.parametrize(
    "expr, expected, min_conf",
    [
        ("next Thursday", date(2026, 9, 17), 0.8),  # said on a Wednesday
        ("Thursday", date(2026, 9, 17), 0.8),
        ("by Friday", date(2026, 9, 18), 0.8),
        ("tomorrow", date(2026, 9, 17), 0.9),
        ("October twenty first", date(2026, 10, 21), 1.0),
        ("October 14th", date(2026, 10, 14), 1.0),
        ("14 October", date(2026, 10, 14), 1.0),
        ("week 7", date(2026, 10, 19), 0.6),
        ("end of week seven", date(2026, 10, 23), 0.6),
        ("in two weeks", date(2026, 9, 30), 0.8),
        ("next week", date(2026, 9, 21), 0.6),
        ("next time", date(2026, 9, 18), 0.8),  # from the course schedule
    ],
)
def test_resolves(r: DateResolver, expr: str, expected: date, min_conf: float) -> None:
    res = r.resolve(expr)
    assert res.date == expected, (expr, res.note)
    assert res.confidence >= min_conf


def test_explicit_weekday_beats_today_in_the_same_phrase(r: DateResolver) -> None:
    # Found by the eval: "quiz on Thursday covering everything up to today" resolved to today.
    assert r.resolve("Thursday covering everything up to today").date == date(2026, 9, 17)
    assert r.resolve("today").date == date(2026, 9, 16)


def test_weekday_said_on_same_weekday_means_next_week() -> None:
    thursday = datetime(2026, 9, 17, 9, 0, tzinfo=ZoneInfo(TZ))
    assert DateResolver(thursday, TZ).resolve("Thursday").date == date(2026, 9, 24)


def test_time_is_carried(r: DateResolver) -> None:
    res = r.resolve("next Thursday at five p.m.")
    assert res.date == date(2026, 9, 17) and res.time == time(17, 0)
    assert res.iso() == "2026-09-17T17:00"
    res = r.resolve("Friday", time_expression="noon")
    assert res.time == time(12, 0)


def test_unresolved_is_honest() -> None:
    bare = DateResolver(LECTURE, TZ, None)
    assert not bare.resolve("week 7").resolved
    assert "term calendar" in bare.resolve("week 7").note
    assert not bare.resolve("next time").resolved
    assert not bare.resolve("soon-ish").resolved
    assert not bare.resolve("").resolved


def test_notes_explain_the_arithmetic(r: DateResolver) -> None:
    note = r.resolve("next Thursday").note
    assert "Wed 16 Sep 2026" in note and "Thu 17 Sep 2026" in note


def test_past_dates_prefer_future_year() -> None:
    # A lecture in December mentioning "January 10" means next year.
    dec = datetime(2026, 12, 1, 10, 0, tzinfo=ZoneInfo(TZ))
    assert DateResolver(dec, TZ).resolve("January 10").date == date(2027, 1, 10)
