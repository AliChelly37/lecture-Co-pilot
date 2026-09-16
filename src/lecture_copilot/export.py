"""Markdown export of a lecture bundle (D2/D19: view, export, delete)."""

from __future__ import annotations


def _mmss(t: float | None) -> str:
    if t is None:
        return "--:--"
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def lecture_markdown(bundle: dict) -> str:
    lecture, course = bundle["lecture"], bundle["course"] or {}
    out: list[str] = []
    out.append(f"# {course.get('name', 'Lecture')} — {lecture['started_at'][:16].replace('T', ' ')}")
    out.append("")
    out.append(
        f"Recorded with Lecture Co-Pilot. Speech-to-text: {lecture.get('asr_model')} on {lecture.get('asr_device')}"
        f" (RTF p95 {lecture.get('asr_rtf_p95')}). Power: {lecture.get('power_state')}."
    )
    if bundle.get("gaps"):
        out.append("Recording gaps: " + ", ".join(f"{_mmss(g['t0'])} to {_mmss(g['t1'])} ({g['cause']})" for g in bundle["gaps"]))
    out.append("")

    recaps = bundle.get("recaps") or []
    if recaps:
        r = recaps[-1]["sections"]
        out.append(f"## Recap: {r.get('title', '')}")
        if r.get("gaps_note"):
            out.append(f"> {r['gaps_note']}")
        out.append("")
        out.append("### Highlights")
        out += [f"- {h}" for h in r.get("highlights", [])]
        if r.get("flag_explanations"):
            out.append("")
            out.append("### What you flagged")
            for f in r["flag_explanations"]:
                out.append(f"- **{_mmss(f.get('t'))}** {f.get('what_was_confusing', f.get('error', ''))}")
                if f.get("explanation"):
                    out.append(f"  {f['explanation']}")
                if f.get("prerequisite"):
                    out.append(f"  _Check first: {f['prerequisite']}_")
        out.append("")
        out.append("### Key concepts")
        for c in r.get("concepts", []):
            out.append(f"- **{c['name']}** ({c['importance']}): {c['explanation']} _{', '.join(c.get('sources', []))}_")
        if r.get("off_slide_notes"):
            out.append("")
            out.append("### Off the slides")
            out += [f"- {n}" for n in r["off_slide_notes"]]
        out.append("")
        out.append("### Review questions")
        for q in r.get("review_questions", []):
            out.append(f"- **Q:** {q['question']}")
            out.append(f"  **A:** {q['answer']}")
        out.append("")

    suggestions = [s for s in bundle.get("suggestions") or [] if s["state"] != "dismissed"]
    if suggestions:
        out.append("## Deadlines detected")
        for s in suggestions:
            p = s["payload"]
            when = f"{p.get('date') or p.get('date_expression') or '?'}{' ' + p['time'] if p.get('time') else ''}"
            out.append(f'- [{s["state"]}] **{p["title"]}** ({p["type"]}) — {when} — "{p["evidence_quote"]}"')
        out.append("")

    captures = [c for c in bundle.get("board_captures") or [] if c.get("status") == "derived"]
    if captures:
        out.append("## Whiteboard")
        for c in captures:
            out.append(f"### Board at {_mmss(c.get('t_shutter'))}")
            out.append(c.get("text") or "")
            out.append("")

    segments = bundle.get("segments") or []
    if segments:
        flags = bundle.get("flags") or []
        out.append("## Transcript")
        for s in segments:
            mark = " 🚩" if any(s["t1"] > f["window_t0"] and s["t0"] < f["window_t1"] for f in flags) else ""
            out.append(f"- `{_mmss(s['t0'])}` {s['text']}{mark}")
    elif lecture.get("notes") == "transcript purged":
        out.append("## Transcript")
        out.append("_Purged by the retention policy._")
    return "\n".join(out) + "\n"
