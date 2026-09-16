"""Read-only MCP server over the local lecture store (D4, D19).

The first defensible MCP use in this project: a *second client* (Claude
Desktop, Claude Code, any MCP host) can ask "what did I flag in last week's
thermodynamics lecture?" against the same SQLite file. Nothing here writes.

Run:  uv run lecture-copilot-mcp
Claude Desktop config: {"mcpServers": {"lecture-copilot": {"command": "uv",
  "args": ["run", "--directory", "<project dir>", "lecture-copilot-mcp"]}}}
"""

from __future__ import annotations

import json

from mcp.server.mcpserver import MCPServer  # mcp 2.x (FastMCP was renamed)

from lecture_copilot.config import settings
from lecture_copilot.store import Store

mcp = MCPServer("lecture-copilot", instructions="Read-only access to the student's recorded lectures, recaps, flags and deadlines.")
_store: Store | None = None


def store() -> Store:
    global _store
    if _store is None:
        _store = Store(settings.db_path)
    return _store


@mcp.tool()
def list_lectures(limit: int = 20) -> str:
    """List recent lectures with course, date, status and counts."""
    rows = store().list_lectures()[:limit]
    out = []
    for r in rows:
        out.append(
            {
                "lecture_id": r["id"],
                "course": r["course_name"],
                "started_at": r["started_at"],
                "status": r["status"],
                "flags": len(store().flags(r["id"])),
                "has_recap": store().latest_recap(r["id"]) is not None,
            }
        )
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def search_transcripts(query: str, limit: int = 10) -> str:
    """Find transcript segments containing the query words (case-insensitive), with lecture, course and timestamp."""
    words = [w for w in query.lower().split() if len(w) > 2][:6]
    if not words:
        return "[]"
    where = " AND ".join("LOWER(s.text) LIKE ?" for _ in words)
    rows = store().query(
        f"SELECT s.lecture_id, s.t0, s.text, c.name AS course, l.started_at FROM segments s"
        f" JOIN lectures l ON l.id=s.lecture_id JOIN courses c ON c.id=l.course_id WHERE {where} ORDER BY l.started_at DESC, s.t0 LIMIT ?",
        [f"%{w}%" for w in words] + [limit],
    )
    return json.dumps(rows, ensure_ascii=False)


@mcp.tool()
def get_recap(lecture_id: str) -> str:
    """Return the latest recap for a lecture (highlights, concepts, flag explanations, review questions)."""
    r = store().latest_recap(lecture_id)
    if r is None:
        return json.dumps({"error": "no recap for this lecture"})
    s = r["sections"]
    return json.dumps(
        {
            "title": s.get("title"),
            "version": r["version"],
            "highlights": s.get("highlights", []),
            "concepts": s.get("concepts", []),
            "flag_explanations": s.get("flag_explanations", []),
            "review_questions": s.get("review_questions", []),
            "off_slide_notes": s.get("off_slide_notes", []),
        },
        ensure_ascii=False,
    )


@mcp.tool()
def list_open_flags(course: str | None = None) -> str:
    """Confusion flags not yet marked resolved, with the transcript around each and any recap explanation."""
    rows = store().query(
        "SELECT f.id, f.lecture_id, f.t, f.window_t0, f.window_t1, c.name AS course, l.started_at FROM flags f"
        " JOIN lectures l ON l.id=f.lecture_id JOIN courses c ON c.id=l.course_id WHERE f.resolved=0 ORDER BY l.started_at DESC, f.t"
    )
    out = []
    for f in rows:
        if course and course.lower() not in f["course"].lower():
            continue
        segs = store().query(
            "SELECT t0, text FROM segments WHERE lecture_id=? AND t1>? AND t0<? ORDER BY t0",
            (f["lecture_id"], f["window_t0"], f["window_t1"]),
        )
        recap = store().latest_recap(f["lecture_id"])
        expl = None
        if recap:
            expl = next((e for e in recap["sections"].get("flag_explanations", []) if e.get("flag_id") == f["id"]), None)
        out.append({**f, "context": " ".join(s["text"] for s in segs)[:1500], "explanation": expl})
    return json.dumps(out, ensure_ascii=False)


@mcp.tool()
def upcoming_deadlines() -> str:
    """Confirmed or written deadlines from today on, soonest first."""
    d = store().dashboard()
    return json.dumps(
        [
            {
                "title": s["payload"]["title"],
                "date": s["payload"]["date"],
                "time": s["payload"].get("time"),
                "course": s["payload"].get("course_name"),
                "state": s["state"],
            }
            for s in d["upcoming"]
        ],
        ensure_ascii=False,
    )


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
