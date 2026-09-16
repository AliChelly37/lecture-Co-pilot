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

from lecture_copilot.dates import normalise_numbers
from lecture_copilot.extract import _DATE_WORDS, Candidate, date_grounded
from lecture_copilot.store import Store
from lecture_copilot.targets import ActionTarget, TargetError

ONE_TAP_MIN_CONF = 0.7
ONE_TAP_MIN_ASR_CONF = 0.5
ONE_TAP_MIN_DATE_CONF = 0.6


JOKE_MARKERS = ("joking", "just kidding", "kidding", "i'm joking", "im joking", "haha", "lol", "only joking")
HYPOTHETICAL_MARKERS = ("if this were", "if it were", "if that were", "if these were", "imagine", "hypothetically", "suppose ", "were due")
TENTATIVE_MARKERS = (
    "might",
    "maybe",
    "haven't decided",
    "havent decided",
    "not sure",
    "probably",
    "possibly",
    "we may ",
    "perhaps",
    "tentatively",
)
COMMITMENT_MARKERS = (
    "is due",
    "are due",
    "due on",
    "due by",
    "due next",
    "hand in",
    "there will be a",
    "will be a quiz",
    "must be submitted",
    "submit by",
    "submitted by",
)
# "your Statistics midterm": another course named in the spoken words, whatever the model's hint says.
_OTHER_COURSE_RE = re.compile(r"\byour ([A-Z][A-Za-z]+(?: [A-Z][A-Za-z]+)?) (midterm|exam|quiz|assignment|deadline|test|final)\b")


def tier_and_reason(c: Candidate, lecture_date: date) -> tuple[str, str]:
    """The tier and a one-line reason (stored on the card, shown in the eval).
    Lexical guards read the quote *and* the located transcript line: the model's
    quote can omit the very words ("I'm joking") that change the meaning."""
    if c.status in ("duplicate", "superseded"):
        return "log", f"{c.status} by another mention"
    if c.status == "log" or c.intent in ("hypothetical", "joke", "past_reference"):
        return "log", f"intent {c.intent}"
    raw = f"{c.evidence_quote or ''} {c.extras.get('segment_text') or ''}"
    text = raw.lower()
    if any(m in text for m in JOKE_MARKERS):
        return "log", "the line says it was a joke"
    if any(m in text for m in HYPOTHETICAL_MARKERS):
        return "maybe", "the line sounds hypothetical"
    hint = c.course_hint.strip()
    if hint and hint.lower() in text:
        return "maybe", f"mentioned for another course: {hint}"
    # A hint that isn't in the spoken words is the model guessing a course code; ignore it.
    m = _OTHER_COURSE_RE.search(raw)
    if m and m.group(1).lower() not in (c.extras.get("course_name") or "").lower():
        return "maybe", f"mentioned for another course: {m.group(1)}"
    intent = c.intent
    hedged = any(mk in text for mk in TENTATIVE_MARKERS)
    override = ""
    if intent == "tentative" and not hedged and any(mk in text for mk in COMMITMENT_MARKERS):
        intent, override = "commitment", " (the model said tentative; the line states it plainly)"
    if intent not in ("commitment", "correction"):
        return "maybe", f"intent {intent}"
    if hedged:
        return "maybe", "the wording sounds tentative"
    expr_norm = normalise_numbers(c.date_expression or "")
    if not _DATE_WORDS.search(expr_norm) and not _DATE_WORDS.search(normalise_numbers(text)):
        return "log", "no date mentioned"
    r = c.resolution
    if not r.resolved:
        return "maybe", r.note
    if r.date < lecture_date:
        return "maybe", "date is in the past"
    if c.t0 is None:
        return "maybe", "quote not found in the transcript"
    if not date_grounded(c.date_expression, c.evidence_quote, c.extras.get("segment_text")):
        return "maybe", "date words not in the quoted sentence"
    if c.extras.get("source") == "trigger":
        return "maybe", "found by the keyword filter; the model did not report it"
    if c.confidence < ONE_TAP_MIN_CONF:
        return "maybe", f"model confidence {c.confidence:.2f} < {ONE_TAP_MIN_CONF}"
    if c.asr_conf < ONE_TAP_MIN_ASR_CONF:
        return "maybe", f"audio confidence {c.asr_conf:.2f} < {ONE_TAP_MIN_ASR_CONF}"
    if r.confidence < ONE_TAP_MIN_DATE_CONF:
        return "maybe", f"date resolution confidence {r.confidence:.2f} < {ONE_TAP_MIN_DATE_CONF}"
    return "one_tap", "commitment with a grounded, resolved future date" + override


def tier_for(c: Candidate, lecture_date: date) -> str:
    return tier_and_reason(c, lecture_date)[0]


def _norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", title.lower()).strip()


def idempotency_key(course_id: str, c: Candidate) -> str:
    raw = f"{course_id}|{c.type}|{_norm_title(c.title)}|{c.resolution.iso() or c.date_expression.lower()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def payload_for(c: Candidate, lecture: dict, course: dict, tier: str, reason: str = "") -> dict:
    return {
        "tier_reason": reason,
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
    def __init__(self, store: Store, targets: dict[str, ActionTarget] | None = None) -> None:
        self.store = store
        self.targets = targets or {}

    def propose(self, candidates: list[Candidate], lecture: dict, course: dict) -> dict[str, int]:
        """Create suggestion rows for surfaced candidates; idempotent per key.
        A re-run refreshes still-proposed cards and retires proposed cards it
        no longer produces; confirmed, written and dismissed rows are untouched."""
        lecture_date = datetime.fromisoformat(lecture["started_at"]).date()
        counts = {"one_tap": 0, "maybe": 0, "log": 0, "existing": 0, "retired": 0}
        produced: set[str] = set()
        for c in candidates:
            tier, reason = tier_and_reason(c, lecture_date)
            if tier == "log":
                counts["log"] += 1
                continue
            key = idempotency_key(course["id"], c)
            produced.add(key)
            existing = self.store.suggestion_by_key(key)
            if existing:
                counts["existing"] += 1
                if existing["state"] in ("proposed", "superseded"):
                    # Not acted on (or retired by an earlier re-run): refresh the card.
                    self.store.update_suggestion_payload(existing["id"], payload_for(c, lecture, course, tier, reason), tier=tier)
                    if existing["state"] == "superseded":
                        self.store.set_suggestion_state(existing["id"], "proposed")
                continue
            self.store.add_suggestion(
                candidate_id=c.id or "",
                lecture_id=lecture["id"],
                tier=tier,
                payload=payload_for(c, lecture, course, tier, reason),
                idempotency_key=key,
            )
            counts[tier] += 1
        for stale in self.store.suggestions(state="proposed", lecture_id=lecture["id"]):
            if stale["idempotency_key"] not in produced:
                self.store.set_suggestion_state(stale["id"], "superseded")
                counts["retired"] += 1
        return counts

    # -- state machine -----------------------------------------------------
    def confirm(self, suggestion_id: str) -> dict:
        """The user's tap. Moves to `confirmed`, then writes to every configured
        target. With no targets configured it stays `confirmed` (.ics export)."""
        s = self._get(suggestion_id)
        if s["state"] not in ("proposed",):
            raise ValueError(f"cannot confirm a suggestion in state {s['state']}")
        if not s["payload"].get("date"):
            raise ValueError("set a date before confirming")
        self.store.set_suggestion_state(suggestion_id, "confirmed")
        if self.targets:
            return self.write(suggestion_id)
        return self._get(suggestion_id)

    def write(self, suggestion_id: str) -> dict:
        """Write (or retry writing) a confirmed suggestion to each target that
        doesn't have it yet. Idempotent per target via the suggestion key."""
        s = self._get(suggestion_id)
        if s["state"] not in ("confirmed", "failed_retryable"):
            raise ValueError(f"cannot write a suggestion in state {s['state']}")
        payload = s["payload"]
        external: dict[str, dict] = dict(payload.get("external") or {})
        errors: list[tuple[str, str, bool]] = []
        for name, target in self.targets.items():
            if name in external:
                continue
            try:
                res = target.write(payload, s["idempotency_key"])
                external[name] = {"id": res.external_id, "url": res.url, "created": res.created}
            except TargetError as exc:
                errors.append((name, str(exc), exc.retryable))
            except Exception as exc:  # network layer, unexpected shapes: retryable
                errors.append((name, f"{type(exc).__name__}: {exc}", True))
        self.store.update_suggestion_payload(suggestion_id, payload | {"external": external})
        if not errors:
            self.store.set_suggestion_state(suggestion_id, "written", external_id=",".join(f"{k}:{v['id']}" for k, v in external.items()))
        else:
            state = "failed_retryable" if any(r for _, _, r in errors) else "failed_permanent"
            self.store.set_suggestion_state(suggestion_id, state, error="; ".join(f"{n}: {m}" for n, m, _ in errors))
        return self._get(suggestion_id)

    def dismiss(self, suggestion_id: str) -> dict:
        s = self._get(suggestion_id)
        if s["state"] not in ("proposed", "confirmed"):
            raise ValueError(f"cannot dismiss a suggestion in state {s['state']} (undo it first)")
        self.store.set_suggestion_state(suggestion_id, "dismissed")
        return self._get(suggestion_id)

    def undo(self, suggestion_id: str) -> dict:
        """Back to `proposed`. For a written suggestion this deletes what was
        written; a delete that fails leaves the record where it is."""
        s = self._get(suggestion_id)
        if s["state"] not in ("confirmed", "dismissed", "written", "failed_retryable", "failed_permanent"):
            raise ValueError(f"cannot undo a suggestion in state {s['state']}")
        external: dict[str, dict] = dict(s["payload"].get("external") or {})
        for name in list(external):
            target = self.targets.get(name)
            if target is None:
                continue  # target no longer configured; the external record stays, the link is dropped
            target.delete(external[name]["id"])  # TargetError propagates: nothing is silently orphaned
            external.pop(name)
        self.store.update_suggestion_payload(suggestion_id, s["payload"] | {"external": external})
        self.store.set_suggestion_state(suggestion_id, "proposed", error="")
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
