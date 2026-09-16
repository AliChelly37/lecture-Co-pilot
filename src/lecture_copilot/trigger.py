"""Deterministic trigger filter (D6 Stage A, kept after D15 as the free live
signal). Scores a transcript segment for "this might mention a deadline,
exam, assignment or reading" without any model call.

A hit is only a badge tick and a hint to the post-lecture extractor; it never
creates a suggestion by itself.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

_LEXICON = {
    # strong: almost always administrative
    "due": 0.6,
    "deadline": 0.6,
    "submit": 0.5,
    "submission": 0.5,
    "hand in": 0.5,
    "exam": 0.5,
    "midterm": 0.6,
    "final exam": 0.6,
    "quiz": 0.5,
    "test on": 0.4,
    "problem set": 0.5,
    "pset": 0.5,
    "homework": 0.5,
    "assignment": 0.5,
    "coursework": 0.5,
    "project proposal": 0.4,
    "lab report": 0.4,
    "essay": 0.3,
    # medium: often frames an instruction
    "reading for": 0.4,
    "read chapter": 0.4,
    "before next": 0.3,
    "by next": 0.4,
    "make sure you": 0.2,
    "don't forget": 0.3,
    "remember to": 0.3,
    "office hours": 0.2,
    "on the exam": 0.4,
    "on the test": 0.4,
    "will be on the": 0.3,
    "moved to": 0.3,
    "rescheduled": 0.4,
    "extension": 0.3,
    "late penalty": 0.4,
}

_WEEKDAY = r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)"
_MONTH = r"(january|february|march|april|may|june|july|august|september|october|november|december)"
_DATE_PATTERNS = [
    re.compile(rf"\b(next|this|on|by|before|until)\s+{_WEEKDAY}\b"),
    re.compile(rf"\b{_MONTH}\s+(the\s+)?\d{{1,2}}(st|nd|rd|th)?\b"),
    re.compile(rf"\b\d{{1,2}}(st|nd|rd|th)?\s+(of\s+)?{_MONTH}\b"),
    re.compile(
        rf"\b{_MONTH}\s+(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|"
        r"eleventh|twelfth|thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|"
        r"nineteenth|twentieth|twenty[- ]\w+|thirtieth|thirty[- ]first)\b"
    ),
    re.compile(r"\bweek\s+(\d{1,2}|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\b"),
    re.compile(r"\b(tomorrow|tonight|end of (the )?(week|month|term|semester)|reading week|in two weeks)\b"),
    re.compile(r"\bnext (time|week|lecture|class|session|tutorial|lab)\b"),
    re.compile(r"\b\d{1,2}(:\d{2})?\s*(am|pm|a\.m\.|p\.m\.)\b"),
    re.compile(r"\b(noon|midnight)\b"),
]


@dataclass
class TriggerResult:
    score: float
    terms: list[str] = field(default_factory=list)
    has_date: bool = False

    @property
    def hit(self) -> bool:
        # A badge tick needs either a strong administrative word, or a
        # lexicon word together with a date expression.
        return self.score >= 0.6 or (self.has_date and self.score >= 0.3)


def score_segment(text: str) -> TriggerResult:
    lowered = f" {text.lower()} "
    terms: list[str] = []
    score = 0.0
    for term, weight in _LEXICON.items():
        if f" {term}" in lowered or f"{term} " in lowered:
            terms.append(term)
            score += weight
    has_date = False
    for pat in _DATE_PATTERNS:
        m = pat.search(lowered)
        if m:
            has_date = True
            terms.append(m.group(0).strip())
    if has_date:
        score += 0.3
    return TriggerResult(score=round(min(score, 1.5), 2), terms=terms, has_date=has_date)
