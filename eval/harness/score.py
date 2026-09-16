"""Scoring for the extraction eval. The headline is the trust metric: how many
non-events (hypotheticals, jokes, past references, superseded dates) reached
the one-tap tier. Then precision/recall of one-tap against gold commitments,
date exactness, and maybe-tier recall."""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_tok = re.compile(r"[a-z0-9]+")
_STOP = {"the", "a", "an", "of", "for", "to", "on", "in", "and", "is", "due", "your", "s"}


def _tokens(s: str) -> set[str]:
    return {t for t in _tok.findall((s or "").lower()) if t not in _STOP}


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    return len(ta & tb) / max(len(ta | tb), 1)


def _matches(pred: dict, gold: dict) -> bool:
    p = pred["payload"]
    same_date = bool(gold.get("date")) and p.get("date") == gold["date"]
    sim = _jaccard(p["title"], gold["title"])
    quote_sim = _jaccard(p.get("evidence_quote", ""), gold["text"])
    return sim >= 0.3 or quote_sim >= 0.5 or (same_date and sim >= 0.15)


@dataclass
class CaseScore:
    name: str
    gold_one_tap: int = 0
    one_tap_hits: int = 0  # gold one_tap events found in one_tap
    one_tap_in_maybe: int = 0  # gold one_tap events that only reached maybe
    one_tap_missed: int = 0
    one_tap_preds: int = 0
    false_one_taps: int = 0  # one_tap predictions matching no gold one_tap (includes traps)
    trap_one_taps: int = 0  # traps (expect=absent) that reached one_tap: the trust metric
    date_exact: int = 0
    date_checked: int = 0
    maybe_expected: int = 0
    maybe_hits: int = 0
    maybe_at_one_tap: int = 0  # tentative / other-course items that reached one_tap: soft trust failures
    details: list[str] = field(default_factory=list)

    @property
    def precision(self) -> float:
        return self.one_tap_hits / self.one_tap_preds if self.one_tap_preds else 1.0

    @property
    def recall(self) -> float:
        return self.one_tap_hits / self.gold_one_tap if self.gold_one_tap else 1.0

    @property
    def surfaced_recall(self) -> float:
        """Found anywhere the student will see it (one-tap or needs-a-date)."""
        return (self.one_tap_hits + self.one_tap_in_maybe) / self.gold_one_tap if self.gold_one_tap else 1.0

    def as_dict(self) -> dict:
        d = self.__dict__ | {
            "precision": round(self.precision, 3),
            "recall": round(self.recall, 3),
            "surfaced_recall": round(self.surfaced_recall, 3),
        }
        d["date_exact_rate"] = round(self.date_exact / self.date_checked, 3) if self.date_checked else None
        return d


def score_case(name: str, gold: list[dict], suggestions: list[dict]) -> CaseScore:
    s = CaseScore(name)
    one_tap = [x for x in suggestions if x["tier"] == "one_tap" and x["state"] == "proposed"]
    maybe = [x for x in suggestions if x["tier"] == "maybe" and x["state"] == "proposed"]
    used: set[str] = set()

    # Two gold events can share a title (two problem sets, two quizzes): assign
    # exact-date matches for every gold first, then fall back to title matches.
    assigned: dict[int, dict] = {}
    for i, g in enumerate(gold):
        if g["expect"] != "one_tap":
            continue
        hit = next((p for p in one_tap if p["id"] not in used and _matches(p, g) and p["payload"].get("date") == g["date"]), None)
        if hit:
            assigned[i] = hit
            used.add(hit["id"])
    for i, g in enumerate(gold):
        if g["expect"] != "one_tap" or i in assigned:
            continue
        hit = next((p for p in one_tap if p["id"] not in used and _matches(p, g)), None)
        if hit:
            assigned[i] = hit
            used.add(hit["id"])

    for i, g in enumerate(gold):
        if g["expect"] == "one_tap":
            s.gold_one_tap += 1
            hit = assigned.get(i)
            if hit:
                s.one_tap_hits += 1
                s.date_checked += 1
                if hit["payload"].get("date") == g["date"]:
                    s.date_exact += 1
                else:
                    s.details.append(f"date mismatch for '{g['title']}': got {hit['payload'].get('date')} expected {g['date']}")
            elif any(_matches(p, g) for p in maybe):
                s.one_tap_in_maybe += 1
                s.details.append(f"'{g['title']}' only reached the maybe tray")
            else:
                s.one_tap_missed += 1
                s.details.append(f"missed: '{g['title']}' ({g['intent']})")
        elif g["expect"] == "maybe":
            s.maybe_expected += 1
            if any(_matches(p, g) for p in maybe):
                s.maybe_hits += 1
            else:
                promoted = next((p for p in one_tap if p["id"] not in used and _matches(p, g)), None)
                if promoted:
                    s.maybe_hits += 1
                    s.maybe_at_one_tap += 1
                    used.add(promoted["id"])
                    s.details.append(
                        f"needs-a-date item reached one-tap: '{g['title']}' ({g['intent']}) as '{promoted['payload']['title']}' {promoted['payload'].get('date')}"
                    )
                else:
                    s.details.append(f"maybe-tier item not surfaced: '{g['title']}'")
        else:  # absent: traps and superseded dates
            if g["intent"] in ("hypothetical", "joke", "past_reference"):
                trap_hit = next((p for p in one_tap if p["id"] not in used and _matches(p, g)), None)
            else:  # superseded: only the specific old date counts as a failure
                trap_hit = next(
                    (p for p in one_tap if p["id"] not in used and _matches(p, g) and p["payload"].get("date") == g.get("date")), None
                )
            if trap_hit:
                s.trap_one_taps += 1
                used.add(trap_hit["id"])
                s.details.append(
                    f"TRAP reached one-tap: '{g['title']}' ({g['intent']}) as '{trap_hit['payload']['title']}' {trap_hit['payload'].get('date')}"
                )

    s.one_tap_preds = len(one_tap)
    s.false_one_taps = len([p for p in one_tap if p["id"] not in used])
    for p in one_tap:
        if p["id"] not in used:
            s.details.append(
                f"unmatched one-tap: '{p['payload']['title']}' {p['payload'].get('date')} — \"{p['payload'].get('evidence_quote', '')[:80]}\""
            )
    return s


def aggregate(scores: list[CaseScore]) -> dict:
    tot = CaseScore("all")
    for c in scores:
        for k in (
            "gold_one_tap",
            "one_tap_hits",
            "one_tap_in_maybe",
            "one_tap_missed",
            "one_tap_preds",
            "false_one_taps",
            "trap_one_taps",
            "date_exact",
            "date_checked",
            "maybe_expected",
            "maybe_hits",
            "maybe_at_one_tap",
        ):
            setattr(tot, k, getattr(tot, k) + getattr(c, k))
    return tot.as_dict()
