"""ActionTargets (D1, D5, D19): the only code that writes to external services,
and it runs only from a user's confirm tap.

Every target implements the same small interface. Writes are idempotent per
suggestion (the idempotency key becomes the external record's identity where
the service allows it, or is stored in a hidden property and queried first).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass
class TargetHealth:
    name: str
    configured: bool
    connected: bool
    detail: str


@dataclass
class WriteResult:
    external_id: str
    url: str | None = None
    created: bool = True  # False when an existing record was found (idempotent replay)


class ActionTarget(Protocol):
    name: str

    def healthcheck(self) -> TargetHealth: ...

    def write(self, payload: dict, idempotency_key: str) -> WriteResult: ...

    def update(self, external_id: str, payload: dict) -> WriteResult: ...

    def delete(self, external_id: str) -> None: ...


class TargetError(RuntimeError):
    """A write failed. `retryable` tells the suggestion state machine which
    failed_* state to use."""

    def __init__(self, message: str, retryable: bool = True) -> None:
        super().__init__(message)
        self.retryable = retryable


def event_datetimes(payload: dict) -> tuple[str, str, bool]:
    """(start, end, all_day) in the course timezone. A dated deadline is an
    all-day event; a timed one is a 1-hour block ending at the deadline."""
    from datetime import date, datetime, timedelta
    from zoneinfo import ZoneInfo

    d = date.fromisoformat(payload["date"])
    if payload.get("time"):
        tz = ZoneInfo(payload.get("timezone") or "UTC")
        h, m = (int(x) for x in payload["time"].split(":")[:2])
        end = datetime(d.year, d.month, d.day, h, m, tzinfo=tz)
        start = end - timedelta(hours=1)
        return start.isoformat(), end.isoformat(), False
    return d.isoformat(), (d + timedelta(days=1)).isoformat(), True


def describe(payload: dict) -> str:
    quote = payload.get("evidence_quote", "")
    note = payload.get("resolution_note", "")
    t0 = payload.get("t0")
    where = f" at {int(t0 // 60):02d}:{int(t0 % 60):02d}" if t0 is not None else ""
    return f'Detected by Lecture Co-Pilot in {payload.get("course_name", "a lecture")}{where}.\nQuote: "{quote}"\n{note}'
