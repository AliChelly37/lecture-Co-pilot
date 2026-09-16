"""Deterministic slide alignment: which slide (if any) was the professor on
during each transcript window? Lexical overlap between the window and each
slide's text, weighted by how rare a term is across the deck (IDF), so common
words like "the" and "equation" don't glue every window to slide 1.

Windows scoring below `min_score` are `off_slide`; long off-slide stretches
are what the recap reports as "off-slide material".
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

_token_re = re.compile(r"[a-z][a-z0-9'-]{2,}")
_STOP = frozenset(
    (
        "the and for that with this from are was were will have has had not but you your our its their they them which "
        "what when where how why can could should would may might into onto over under about after before also then than "
        "there here these those such very just more most some any all each other one two three four five six seven eight "
        "nine ten first second next last because while does did done being been get got let lets like see look going make "
        "made say said think know want need use used using well okay right now today lecture slide slides"
    ).split()
)


def tokens(text: str) -> list[str]:
    return [t for t in _token_re.findall(text.lower()) if t not in _STOP]


@dataclass
class Window:
    t0: float
    t1: float
    slide: int | None  # 1-based; None = off_slide
    score: float


class SlideAligner:
    def __init__(self, slides: list[str], min_score: float = 0.12) -> None:
        self.min_score = min_score
        self.slide_tokens = [Counter(tokens(s)) for s in slides]
        df: Counter[str] = Counter()
        for c in self.slide_tokens:
            df.update(c.keys())
        n = max(len(slides), 1)
        self.idf = {t: math.log(1 + n / d) for t, d in df.items()}
        self.slide_norm = [self._norm(c) for c in self.slide_tokens]

    def _norm(self, c: Counter[str]) -> float:
        return math.sqrt(sum((self.idf.get(t, 0.0) * v) ** 2 for t, v in c.items())) or 1.0

    def score_window(self, text: str) -> tuple[int | None, float]:
        wt = Counter(tokens(text))
        if not wt or not self.slide_tokens:
            return None, 0.0
        wnorm = math.sqrt(sum((self.idf.get(t, 0.0) * v) ** 2 for t, v in wt.items())) or 1.0
        best, best_score = None, 0.0
        for i, sc in enumerate(self.slide_tokens):
            dot = sum(self.idf.get(t, 0.0) ** 2 * v * sc[t] for t, v in wt.items() if t in sc)
            score = dot / (wnorm * self.slide_norm[i])
            if score > best_score:
                best, best_score = i + 1, score
        if best_score < self.min_score:
            return None, round(best_score, 3)
        return best, round(best_score, 3)

    def align(self, segments: list[dict], window_s: float = 30.0) -> list[Window]:
        """segments: dicts with t0, t1, text (as stored). Returns fixed-size windows."""
        if not segments:
            return []
        end = max(s["t1"] for s in segments)
        out: list[Window] = []
        t = 0.0
        while t < end:
            t1 = t + window_s
            text = " ".join(s["text"] for s in segments if s["t1"] > t and s["t0"] < t1)
            slide, score = self.score_window(text)
            out.append(Window(t0=t, t1=min(t1, end), slide=slide, score=score))
            t = t1
        return _smooth(out)


def _smooth(windows: list[Window]) -> list[Window]:
    """A single off-slide window between two windows on the same slide is
    almost always the same slide (a quiet half-minute), so fill it in."""
    for i in range(1, len(windows) - 1):
        prev, cur, nxt = windows[i - 1], windows[i], windows[i + 1]
        if cur.slide is None and prev.slide is not None and prev.slide == nxt.slide:
            cur.slide = prev.slide
    return windows


def coverage(windows: list[Window]) -> float:
    if not windows:
        return 0.0
    return round(sum(1 for w in windows if w.slide is not None) / len(windows), 3)


def off_slide_stretches(windows: list[Window], min_windows: int = 4) -> list[tuple[float, float]]:
    """Contiguous off-slide runs of at least `min_windows` windows (default 2 min)."""
    out: list[tuple[float, float]] = []
    run: list[Window] = []
    for w in [*windows, Window(0, 0, 0, 0)]:  # sentinel closes the last run
        if w.slide is None:
            run.append(w)
        else:
            if len(run) >= min_windows:
                out.append((run[0].t0, run[-1].t1))
            run = []
    return out
