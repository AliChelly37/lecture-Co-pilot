"""Deterministic recap faithfulness check (M6, L3 part 1): for every claim
with a `t=MM:SS` source, does that moment exist in the transcript, and does
the cited window share content words with the claim? No judge model needed;
this catches invented timestamps and citations that point at the wrong place.

  uv run python -m harness.recap_citations --db ../data/copilot.sqlite3 --lecture <id>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lecture_copilot.store import Store  # noqa: E402

_tok = re.compile(r"[a-z][a-z0-9-]{3,}")
_STOP = {
    "this",
    "that",
    "with",
    "from",
    "have",
    "which",
    "their",
    "there",
    "about",
    "into",
    "than",
    "then",
    "also",
    "very",
    "when",
    "what",
    "will",
    "were",
    "been",
    "being",
    "lecture",
    "student",
    "students",
}


def tokens(text: str) -> set[str]:
    return {t for t in _tok.findall(text.lower()) if t not in _STOP}


def check_recap(sections: dict, segments: list[dict], window_s: float = 45.0) -> dict:
    end = max((s["t1"] for s in segments), default=0.0)
    claims: list[tuple[str, list[str]]] = []
    for h in sections.get("highlights", []):
        claims.append((h, re.findall(r"t=(\d{1,3}:\d{2})", h)))
    for c in sections.get("concepts", []):
        claims.append((c["name"] + " " + c["explanation"], [s[2:] for s in c.get("sources", []) if s.startswith("t=")]))
    for q in sections.get("review_questions", []):
        claims.append((q["question"] + " " + q["answer"], [s[2:] for s in q.get("sources", []) if s.startswith("t=")]))
    for f in sections.get("flag_explanations", []):
        claims.append(
            (f.get("what_was_confusing", "") + " " + f.get("explanation", ""), [s[2:] for s in f.get("sources", []) if s.startswith("t=")])
        )

    n_claims = len(claims)
    with_source = 0
    valid_time = 0
    supported = 0
    problems: list[str] = []
    for text, times in claims:
        if not times:
            problems.append(f"no source: {text[:70]}")
            continue
        with_source += 1
        ok_any, sup_any = False, False
        for mmss in times:
            m, s = mmss.split(":")
            t = int(m) * 60 + int(s)
            if t > end + 5:
                problems.append(f"timestamp {mmss} beyond the lecture ({end:.0f}s): {text[:60]}")
                continue
            ok_any = True
            window = " ".join(x["text"] for x in segments if x["t1"] >= t - window_s and x["t0"] <= t + window_s)
            overlap = tokens(text) & tokens(window)
            if len(overlap) >= 2 or (tokens(text) and len(overlap) / len(tokens(text)) >= 0.2):
                sup_any = True
        valid_time += int(ok_any)
        supported += int(sup_any)
        if ok_any and not sup_any:
            problems.append(f"cited window doesn't mention the claim: {text[:70]} @ {times}")
    return {
        "claims": n_claims,
        "with_source": with_source,
        "valid_timestamp": valid_time,
        "supported_by_window": supported,
        "support_rate": round(supported / with_source, 3) if with_source else None,
        "problems": problems,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data/copilot.sqlite3"))
    ap.add_argument("--lecture", required=True)
    args = ap.parse_args()
    store = Store(Path(args.db))
    recap = store.latest_recap(args.lecture)
    if recap is None:
        raise SystemExit("no recap for this lecture")
    result = check_recap(recap["sections"], store.segments(args.lecture))
    store.close()
    import json

    print(json.dumps({k: v for k, v in result.items() if k != "problems"}, indent=1))
    for p in result["problems"]:
        print(" -", p)


if __name__ == "__main__":
    main()
