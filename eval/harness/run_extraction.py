"""Extraction eval (M6, L2): run the real pipeline on labelled cases with any
provider/model and score it. Writes eval/results/<run>.json and appends a row
to eval/results/summary.md.

  uv run python -m harness.run_extraction --provider ollama --model gemma3:4b
  uv run python -m harness.run_extraction --provider cloudflare --model @cf/meta/llama-3.3-70b-instruct-fp8-fast
  uv run python -m harness.run_extraction --cases eval/cases/synthetic-11.json   # subset
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from harness.inserts import load_case  # noqa: E402
from harness.score import aggregate, score_case  # noqa: E402
from lecture_copilot import pipeline  # noqa: E402
from lecture_copilot.config import Settings  # noqa: E402
from lecture_copilot.llm import LlmGateway  # noqa: E402
from lecture_copilot.store import Store  # noqa: E402
from lecture_copilot.trigger import score_segment  # noqa: E402


def run_case(case: dict, settings: Settings) -> tuple[dict, dict]:
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "eval.sqlite3")
        try:
            course = store.create_course("Eval course", case["timezone"], case["term_calendar"])
            lecture = store.start_lecture(course["id"], "eval", "cpu", "ac", None)
            for s in case["segments"]:
                trig = score_segment(s["text"])
                store.add_segment(
                    lecture["id"], s["t0"], s["t1"], s["text"], s["asr_conf"], "eval", trig.score, trig.terms if trig.hit else []
                )
            # The lecture clock must be the case's, not now: the resolver reads started_at.
            store.execute(
                "UPDATE lectures SET started_at=?, ended_at=?, status='ended' WHERE id=?",
                (case["lecture_start"], case["lecture_start"], lecture["id"]),
            )
            llm = LlmGateway(settings, store)
            window = settings.extract_window_s if settings.llm_provider == "ollama" else 0.0
            t0 = time.perf_counter()
            summary = pipeline.extract_lecture(store, llm, lecture["id"], window, settings.extract_overlap_s)
            elapsed = time.perf_counter() - t0
            suggestions = store.suggestions(None, lecture["id"])
            usage = store.usage_by_stage(lecture["id"])
            score = score_case(case["name"], case["gold"], suggestions)
            predictions = [
                {
                    "tier": s["tier"],
                    "title": s["payload"]["title"],
                    "intent": s["payload"]["intent"],
                    "date": s["payload"].get("date"),
                    "date_expression": s["payload"].get("date_expression"),
                    "confidence": s["payload"].get("confidence"),
                    "reason": s["payload"].get("tier_reason"),
                    "quote": (s["payload"].get("evidence_quote") or "")[:100],
                }
                for s in suggestions
                if s["state"] == "proposed"
            ]
            return score.as_dict(), {
                "predictions": predictions,
                "elapsed_s": round(elapsed, 1),
                "calls": sum(u["calls"] for u in usage),
                "input_tokens": sum(u["input_tokens"] for u in usage),
                "output_tokens": sum(u["output_tokens"] for u in usage),
                "cost_usd": round(sum(u["cost_usd"] for u in usage), 5),
                "summary": summary,
            }
        finally:
            store.close()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--cases", nargs="*", default=None)
    ap.add_argument("--label", default=None, help="run label (default provider:model)")
    args = ap.parse_args()

    settings = Settings()
    if args.provider:
        settings.llm_provider = args.provider
    if args.model:
        if settings.llm_provider == "ollama":
            settings.ollama_model_text = args.model
        elif settings.llm_provider == "cloudflare":
            settings.cloudflare_model_text = args.model
        else:
            settings.anthropic_model = args.model
    label = args.label or f"{settings.llm_provider}:{LlmGateway(settings, None).describe()['text_model']}"  # type: ignore[arg-type]

    paths = [Path(p) for p in args.cases] if args.cases else sorted((ROOT / "eval/cases").glob("*.json"))
    results, scores = [], []
    for path in paths:
        case = load_case(path)
        print(f"[{label}] {case['name']} ...", flush=True)
        score, meta = run_case(case, settings)
        scores.append(score)
        results.append({"case": case["name"], "score": score, "meta": meta})
        print(
            f"  one-tap P={score['precision']:.2f} R={score['recall']:.2f} surfaced R={score['surfaced_recall']:.2f} traps->one-tap={score['trap_one_taps']} "
            f"false one-taps={score['false_one_taps']} dates exact={score['date_exact']}/{score['date_checked']} "
            f"maybe={score['maybe_hits']}/{score['maybe_expected']} in {meta['elapsed_s']}s, {meta['calls']} calls, ${meta['cost_usd']}",
            flush=True,
        )
        for d in score["details"]:
            print("    -", d)
        for p in meta["predictions"]:
            if p["tier"] == "maybe":
                print(f"    ? maybe '{p['title']}' ({p['intent']}, {p['date'] or p['date_expression']!r}): {p['reason']}")

    from harness.score import CaseScore

    agg = aggregate([CaseScore(**{k: v for k, v in s.items() if k in CaseScore.__dataclass_fields__}) for s in scores])
    total_s = sum(r["meta"]["elapsed_s"] for r in results)
    total_cost = sum(r["meta"]["cost_usd"] for r in results)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    out_dir = ROOT / "eval/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    run = {
        "label": label,
        "at": stamp,
        "cases": results,
        "aggregate": agg,
        "total_elapsed_s": round(total_s, 1),
        "total_cost_usd": round(total_cost, 5),
    }
    (out_dir / f"extraction-{label.replace(':', '_').replace('/', '_').replace('@', '')}-{stamp}.json").write_text(
        json.dumps(run, indent=1), encoding="utf-8"
    )

    line = (
        f"| {stamp} | {label} | {len(results)} | {agg['precision']:.2f} | {agg['recall']:.2f} | {agg['surfaced_recall']:.2f} | "
        f"{agg['trap_one_taps']} | {agg['false_one_taps']} | {agg.get('maybe_at_one_tap', 0)} | "
        f"{agg['date_exact']}/{agg['date_checked']} | {agg['maybe_hits']}/{agg['maybe_expected']} | {total_s:.0f}s | ${total_cost:.4f} |\n"
    )
    header = (
        "# Extraction eval runs\n\n| run | provider:model | cases | one-tap precision | one-tap recall | surfaced recall | traps → one-tap | "
        "false one-taps | needs-a-date → one-tap | dates exact | maybe surfaced | time | cost |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    summary = out_dir / "summary.md"
    text = summary.read_text(encoding="utf-8") if summary.exists() else header
    # New rows go under the current table, above any legacy section kept at the bottom.
    head, sep, legacy = text.partition("\n## Earlier runs")
    summary.write_text(head.rstrip("\n") + "\n" + line + (("\n" + sep.lstrip("\n") + legacy) if sep else ""), encoding="utf-8")
    print("\nAGGREGATE", json.dumps(agg))
    print(f"total {total_s:.0f}s, ${total_cost:.4f}; appended to {summary}")


if __name__ == "__main__":
    main()
