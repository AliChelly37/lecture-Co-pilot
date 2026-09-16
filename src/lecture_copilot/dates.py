"""DateResolver (D9): turn a date expression the model quoted verbatim into a
concrete date, in code, relative to the lecture's start time and the course's
term calendar. `unresolved` is a legitimate answer; guessing is not.

The resolution note explains the arithmetic so the confirm card can show it:
"'next Thursday' from lecture date Tue 16 Sep 2026 -> Thu 24 Sep 2026".
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from zoneinfo import ZoneInfo

_ORDINALS = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
    "thirtieth": 30,
}
_CARDINALS = {
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
}
_WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]


def normalise_numbers(text: str) -> str:
    """'October twenty first' -> 'October 21', 'five p.m.' -> '5 pm', 'week seven' -> 'week 7'."""
    t = text.lower().replace("-", " ")
    # compound ordinals: twenty first, twenty-second, thirty first
    t = re.sub(
        r"\b(twenty|thirty)\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth)\b",
        lambda m: str(_CARDINALS[m.group(1)] + _ORDINALS[m.group(2)]),
        t,
    )
    t = re.sub(
        r"\b(twenty|thirty)\s+(one|two|three|four|five|six|seven|eight|nine)\b",
        lambda m: str(_CARDINALS[m.group(1)] + _CARDINALS[m.group(2)]),
        t,
    )
    t = re.sub(r"\b(" + "|".join(_ORDINALS) + r")\b", lambda m: str(_ORDINALS[m.group(1)]), t)
    t = re.sub(r"\b(" + "|".join(_CARDINALS) + r")\b", lambda m: str(_CARDINALS[m.group(1)]), t)
    t = re.sub(r"\b(\d{1,2})\s*(st|nd|rd|th)\b", r"\1", t)
    # "5 p.m." -> "5 pm" (a trailing dot defeats \b, so match the dotted form explicitly)
    t = re.sub(r"\b([ap])\.m\.?(?!\w)", r"\1m", t)
    return re.sub(r"\s+", " ", t).strip()


@dataclass
class Resolution:
    date: date | None
    time: time | None
    note: str
    confidence: float  # 1.0 exact date; 0.8 relative; 0.5 fuzzy ("end of week"); 0 unresolved

    @property
    def resolved(self) -> bool:
        return self.date is not None

    def iso(self) -> str | None:
        if self.date is None:
            return None
        if self.time is not None:
            return datetime.combine(self.date, self.time).isoformat(timespec="minutes")
        return self.date.isoformat()


class DateResolver:
    def __init__(self, lecture_start: datetime, timezone: str, term_calendar: dict | None = None) -> None:
        tz = ZoneInfo(timezone)
        self.base = lecture_start.astimezone(tz) if lecture_start.tzinfo else lecture_start.replace(tzinfo=tz)
        self.tz = tz
        cal = term_calendar or {}
        self.week1_start: date | None = date.fromisoformat(cal["week1_start"]) if cal.get("week1_start") else None
        self.next_lecture: date | None = date.fromisoformat(cal["next_lecture"]) if cal.get("next_lecture") else None

    def resolve(self, expression: str | None, time_expression: str | None = None) -> Resolution:
        if not expression or not expression.strip():
            return Resolution(None, None, "no date expression", 0.0)
        raw = expression.strip()
        text = normalise_numbers(raw)
        t = self._time(normalise_numbers(time_expression)) if time_expression else self._time(text)
        base_label = self.base.strftime("%a %d %b %Y")

        if re.search(r"\b(next (time|lecture|class|session|tutorial|lab))\b", text):
            if self.next_lecture:
                return Resolution(self.next_lecture, t, f"'{raw}' = next scheduled lecture {self.next_lecture:%a %d %b}", 0.8)
            return Resolution(None, t, f"'{raw}' needs the course schedule (next lecture date unknown)", 0.0)

        m = re.search(r"\b(end of |start of |beginning of )?week (\d{1,2})\b", text)
        if m:
            if self.week1_start is None:
                return Resolution(None, t, f"'{raw}' needs the term calendar (week 1 start unknown)", 0.0)
            n = int(m.group(2))
            monday = self.week1_start + timedelta(weeks=n - 1)
            if m.group(1) and m.group(1).startswith("end"):
                return Resolution(
                    monday + timedelta(days=4), t, f"'{raw}' = Friday of week {n} (week 1 starts {self.week1_start:%d %b})", 0.6
                )
            return Resolution(monday, t, f"'{raw}' = Monday of week {n} (week 1 starts {self.week1_start:%d %b})", 0.6)

        if re.search(r"\b(tomorrow)\b", text):
            d = (self.base + timedelta(days=1)).date()
            return Resolution(d, t, f"'{raw}' from lecture date {base_label} -> {d:%a %d %b %Y}", 0.9)
        if re.search(r"\b(today|tonight)\b", text):
            return Resolution(self.base.date(), t, f"'{raw}' = lecture date {base_label}", 0.9)

        m = re.search(r"\bin (\d{1,2}) (day|days|week|weeks)\b", text)
        if m:
            n = int(m.group(1))
            delta = timedelta(days=n) if m.group(2).startswith("day") else timedelta(weeks=n)
            d = (self.base + delta).date()
            return Resolution(d, t, f"'{raw}' from lecture date {base_label} -> {d:%a %d %b %Y}", 0.8)

        m = re.search(r"\b(next|this|on|by|before|until|coming)?\s*(" + "|".join(_WEEKDAYS) + r")\b", text)
        if m:
            target = _WEEKDAYS.index(m.group(2))
            d = self._next_weekday(target, m.group(1) or "")
            return Resolution(d, t, f"'{raw}' from lecture date {base_label} -> {d:%a %d %b %Y}", 0.8)

        if re.search(r"\bnext week\b", text):
            d = self._next_weekday(0, "next")  # Monday of next week
            return Resolution(d, t, f"'{raw}' = Monday of next week from {base_label} -> {d:%a %d %b %Y}", 0.6)
        if re.search(r"\bend of (the )?week\b", text):
            d = self._next_weekday(4, "this")
            return Resolution(d, t, f"'{raw}' = Friday {d:%d %b %Y}", 0.6)
        if re.search(r"\bend of (the )?(month)\b", text):
            first_next = (self.base.replace(day=1) + timedelta(days=32)).replace(day=1)
            d = (first_next - timedelta(days=1)).date()
            return Resolution(d, t, f"'{raw}' = {d:%a %d %b %Y}", 0.6)

        parsed = self._dateparser(text)
        if parsed is not None:
            d = parsed.date()
            return Resolution(d, t, f"'{raw}' -> {d:%a %d %b %Y}", 1.0 if re.search(r"\d", text) else 0.7)
        return Resolution(None, t, f"could not resolve '{raw}'", 0.0)

    # -- helpers -----------------------------------------------------------
    def _next_weekday(self, target: int, qualifier: str) -> date:
        today = self.base.date()
        ahead = (target - today.weekday()) % 7
        if ahead == 0:
            ahead = 7  # "Thursday" said on a Thursday means next week's
        d = today + timedelta(days=ahead)
        if qualifier == "next" and ahead < 7 and today.weekday() >= 4:
            # Said on Fri/Sat/Sun, "next Thursday" usually means the one after the coming week.
            d += timedelta(days=7)
        return d

    def _time(self, text: str) -> time | None:
        m = re.search(r"\b(\d{1,2})(?::(\d{2}))?\s*(am|pm)\b", text)
        if m:
            h, mins = int(m.group(1)), int(m.group(2) or 0)
            if m.group(3) == "pm" and h < 12:
                h += 12
            if m.group(3) == "am" and h == 12:
                h = 0
            if 0 <= h < 24 and 0 <= mins < 60:
                return time(h, mins)
        if re.search(r"\bnoon\b", text):
            return time(12, 0)
        if re.search(r"\bmidnight\b", text):
            return time(23, 59)
        m = re.search(r"\b(\d{1,2}):(\d{2})\b", text)
        if m and int(m.group(1)) < 24:
            return time(int(m.group(1)), int(m.group(2)))
        return None

    def _dateparser(self, text: str) -> datetime | None:
        import dateparser

        # Strip words dateparser misreads as dates ("due", "by", "before").
        cleaned = re.sub(r"\b(due|by|before|until|on|the|of|at)\b", " ", text)
        cleaned = re.sub(r"\b\d{1,2}(:\d{2})?\s*(am|pm)\b", " ", cleaned).strip()
        if not cleaned:
            return None
        return dateparser.parse(
            cleaned,
            settings={
                "RELATIVE_BASE": self.base.replace(tzinfo=None),
                "PREFER_DATES_FROM": "future",
                "PREFER_DAY_OF_MONTH": "first",
                "RETURN_AS_TIMEZONE_AWARE": False,
            },
        )
