"""SuggestionService: the trust-critical policy between extracted candidates
and anything the user is asked to confirm (D5, D9, D15).

Tiers:
- one_tap : a real commitment with a resolved, future date, from clear audio
- maybe   : anything uncertain (tentative, unresolved date, low ASR confidence,
            another course); the card shows an editable date and no one-tap
- log     : hypotheticals, jokes, past references; visible in debug/eval only

States: proposed -> confirmed -> written -> (undone); proposed -> dismissed;
proposed | written -> superseded. Writes to external targets happen in M5;
until then a confirmed suggestion is exportable as .ics.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date, datetime

from lecture_copilot.extract import Candidate
from lecture_copilot.store import Store

ONE_TAP_MIN_CONF = 0.7
ONE_TAP_MIN_ASR_CONF = 0.5
ONE_TAP_MIN_DATE_CONF = 0.6


def tier_for(c: Candidate, lecture_date: date) -> str:
    if c.status == "log" or c.intent in ("hypothetical", "joke", "past_reference"):
        return "log"
    if c.status in ("duplicate", "superseded"):
        return "log"
    if c.course_hint.strip():
        return "maybe"
    if c.intent not in ("commitment", "correction"):
        return "maybe"
    r = c.resolution
    if not r.resolved or r.date < lecture_date:
        return "maybe"
    if c.confidence < ONE_TAP_MIN_CONF or c.asr_conf < ONE_TAP_MIN_ASR_CONF or r.confidence < ONE_TAP_MIN_DATE_CONF:
        return "maybe"
    return "one_tap"


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def idempotency_key(course_id: str, c: Candidate) -> str:
    raw = f"{course_id}|{c.type}|{_norm_title(c.title)}|{c.resolution.iso() or c.date_expression.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def payload_for(c: Candidate, lecture: dict, course: dict, tier: str) -> dict:
    return {
        "title": c.title,
        "type": c.type,
        "intent": c.intent,
        "date": c.resolution.date.isoformat() if c.resolution.date else None,
        "time": c.resolution.time.isoformat(timespec="minutes") if c.resolution.time else None,
        "date_expression": c.date_expression,
        "resolution_note": c.resolution.note,
        "evidence_quote": c.evidence_quote,
        "t0": c.t0,
        "confidence": c.confidence,
        "asr_conf": c.asr_conf,
        "course_hint": c.course_hint,
        "tier": tier,
        "lecture_id": lecture["id"],
        "course_id": course["id"],
        "course_name": course["name"],
        "timezone": course["timezone"],
    }


class SuggestionService:
    def __init__(self, store: Store) -> None:
        self.store = store

    def propose(self, candidates: list[Candidate], lecture: dict, course: dict) -> dict[str, int]:
        """Create suggestion rows for surfaced candidates; idempotent per key."""
        lecture_date = datetime.fromisoformat(lecture["started_at"]).date()
        counts = {"one_tap": 0, "maybe": 0, "log": 0, "existing": 0}
        for c in candidates:
            tier = tier_for(c, lecture_date)
            if tier == "log":
                counts["log"] += 1
                continue
            key = idempotency_key(course["id"], c)
            if self.store.suggestion_by_key(key):
                counts["existing"] += 1
                continue
            self.store.add_suggestion(
                candidate_id=c.id or "",
                lecture_id=lecture["id"],
                tier=tier,
                payload=payload_for(c, lecture, course, tier),
                idempotency_key=key,
            )
            counts[tier] += 1
        return counts

    # -- state machine -----------------------------------------------------
    def confirm(self, suggestion_id: str) -> dict:
        s = self._get(suggestion_id)
        if s["state"] not in ("proposed",):
            raise ValueError(f"cannot confirm a suggestion in state {s['state']}")
        if not s["payload"].get("date"):
            raise ValueError("set a date before confirming")
        self.store.set_suggestion_state(suggestion_id, "confirmed")
        return self._get(suggestion_id)

    def dismiss(self, suggestion_id: str) -> dict:
        s = self._get(suggestion_id)
        if s["state"] not in ("proposed", "confirmed"):
            raise ValueError(f"cannot dismiss a suggestion in state {s['state']}")
        self.store.set_suggestion_state(suggestion_id, "dismissed")
        return self._get(suggestion_id)

    def undo(self, suggestion_id: str) -> dict:
        s = self._get(suggestion_id)
        if s["state"] not in ("confirmed", "dismissed"):
            raise ValueError(f"cannot undo a suggestion in state {s['state']}")
        self.store.set_suggestion_state(suggestion_id, "proposed")
        return self._get(suggestion_id)

    def set_date(self, suggestion_id: str, iso_date: str, iso_time: str | None = None) -> dict:
        s = self._get(suggestion_id)
        if s["state"] != "proposed":
            raise ValueError("only a proposed suggestion can be re-dated")
        date.fromisoformat(iso_date)  # validates
        payload = s["payload"] | {"date": iso_date, "time": iso_time, "resolution_note": "date set by you", "tier": "one_tap"}
        self.store.update_suggestion_payload(suggestion_id, payload, tier="one_tap")
        return self._get(suggestion_id)

    def _get(self, suggestion_id: str) -> dict:
        s = self.store.get_suggestion(suggestion_id)
        if s is None:
            raise KeyError("unknown suggestion")
        return s

    # -- export ------------------------------------------------------------
    def ics(self, suggestions: list[dict]) -> str:
        lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Lecture Co-Pilot//EN", "CALSCALE:GREGORIAN"]
        for s in suggestions:
            p = s["payload"]
            if not p.get("date"):
                continue
            d = p["date"].replace("-", "")
            lines.append("BEGIN:VEVENT")
            lines.append(f"UID:{s['idempotency_key']}@lecture-copilot")
            lines.append(f"DTSTAMP:{datetime.now().strftime('%Y%m%dT%H%M%S')}")
            if p.get("time"):
                lines.append(f"DTSTART;TZID={p['timezone']}:{d}T{p['time'].replace(':', '')}00")
            else:
                lines.append(f"DTSTART;VALUE=DATE:{d}")
            lines.append(f"SUMMARY:{_esc(p['course_name'])}: {_esc(p['title'])}")
            desc = f"Detected in the lecture on {p['lecture_id']} at {p.get('t0')}s. Quote: {p['evidence_quote']} ({p['resolution_note']})"
            lines.append(f"DESCRIPTION:{_esc(desc)}")
            lines.append("END:VEVENT")
        lines.append("END:VCALENDAR")
        return "\r\n".join(lines) + "\r\n"


def _esc(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def dumps(payload: dict) -> str:
    return json.dumps(payload, ensure_ascii=False)
