"""SQLite persistence. One file, WAL mode, a process-wide lock so the ASR worker
thread and the API handlers can share a connection safely.

Only derived text is ever stored here (transcript, OCR text, recaps, events).
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

SCHEMA = """
CREATE TABLE IF NOT EXISTS courses (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  timezone TEXT NOT NULL,
  term_calendar TEXT NOT NULL DEFAULT '{}',   -- JSON: {"week1_start": "2026-09-07", "breaks": [...]}
  capture_enabled INTEGER NOT NULL DEFAULT 1,
  policy_ack_at TEXT,
  transcript_retention_days INTEGER,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS lectures (
  id TEXT PRIMARY KEY,
  course_id TEXT NOT NULL REFERENCES courses(id),
  started_at TEXT NOT NULL,                   -- wall clock, ISO-8601 UTC; t=0 of the timeline
  ended_at TEXT,
  status TEXT NOT NULL,                       -- recording | ended | processed
  deck_id TEXT,
  asr_model TEXT,
  asr_device TEXT,
  asr_rtf_p95 REAL,
  power_state TEXT,                           -- ac | battery
  battery_start INTEGER,
  battery_end INTEGER,
  source TEXT NOT NULL DEFAULT 'mic',         -- mic | replay (D26) | youtube (D27)
  source_url TEXT,                            -- the YouTube link, when source = 'youtube'
  notes TEXT
);

CREATE TABLE IF NOT EXISTS segments (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lecture_id TEXT NOT NULL REFERENCES lectures(id),
  t0 REAL NOT NULL, t1 REAL NOT NULL,
  text TEXT NOT NULL,
  asr_conf REAL,
  asr_model TEXT,
  trigger_score REAL NOT NULL DEFAULT 0,
  trigger_terms TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS segments_lecture ON segments(lecture_id, t0);

CREATE TABLE IF NOT EXISTS gaps (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lecture_id TEXT NOT NULL REFERENCES lectures(id),
  t0 REAL NOT NULL, t1 REAL,
  cause TEXT NOT NULL                          -- mic_lost | asr_backlog | paused | crash
);

CREATE TABLE IF NOT EXISTS flags (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lecture_id TEXT NOT NULL REFERENCES lectures(id),
  t REAL NOT NULL,
  window_t0 REAL NOT NULL, window_t1 REAL NOT NULL,
  reason TEXT,
  resolved INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS decks (
  id TEXT PRIMARY KEY,                         -- sha256 of the file
  course_id TEXT REFERENCES courses(id),
  filename TEXT NOT NULL,
  slide_count INTEGER NOT NULL,
  slides_text TEXT NOT NULL,                   -- JSON list of per-slide text
  slide_index TEXT,                            -- JSON produced by the SlideIndex call
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS slide_alignment (
  lecture_id TEXT NOT NULL REFERENCES lectures(id),
  t0 REAL NOT NULL, t1 REAL NOT NULL,
  slide INTEGER,                               -- NULL = off_slide
  score REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS board_captures (
  id TEXT PRIMARY KEY,
  lecture_id TEXT REFERENCES lectures(id),
  t_shutter REAL,                              -- seconds into the lecture; NULL = unassigned
  shot_at TEXT,                                -- EXIF wall time if present
  content_kind TEXT,
  text TEXT, latex TEXT, diagram_notes TEXT,
  legibility REAL,
  status TEXT NOT NULL                         -- pending | derived | failed | discarded
);

CREATE TABLE IF NOT EXISTS candidate_events (
  id TEXT PRIMARY KEY,
  lecture_id TEXT NOT NULL REFERENCES lectures(id),
  type TEXT NOT NULL, title TEXT NOT NULL,
  date_expression TEXT, intent TEXT NOT NULL,
  evidence_quote TEXT NOT NULL, t0 REAL,
  confidence REAL NOT NULL,
  resolved_date TEXT, resolution_note TEXT,
  course_hint TEXT,
  status TEXT NOT NULL DEFAULT 'surfaced',     -- surfaced | duplicate | superseded | log
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS suggestions (
  id TEXT PRIMARY KEY,
  candidate_id TEXT NOT NULL,
  lecture_id TEXT,
  target TEXT NOT NULL,                        -- local (M3) | gcal | notion (M5)
  tier TEXT NOT NULL DEFAULT 'maybe',          -- one_tap | maybe
  payload TEXT NOT NULL,                       -- JSON card content
  state TEXT NOT NULL,                         -- proposed | confirmed | written | dismissed | superseded | failed_*
  idempotency_key TEXT NOT NULL UNIQUE,
  external_id TEXT,
  supersedes_id TEXT,
  error TEXT,
  updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS recaps (
  id TEXT PRIMARY KEY,
  lecture_id TEXT NOT NULL REFERENCES lectures(id),
  version INTEGER NOT NULL,
  model TEXT NOT NULL, effort TEXT NOT NULL,
  sections TEXT NOT NULL,                      -- JSON
  rating INTEGER,
  flag_helpful TEXT,                           -- JSON {flag_id: bool}
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS usage_ledger (
  call_id TEXT PRIMARY KEY,
  lecture_id TEXT,
  stage TEXT NOT NULL,
  model TEXT NOT NULL, effort TEXT,
  input_tokens INTEGER NOT NULL,
  cache_read_tokens INTEGER NOT NULL,
  cache_write_tokens INTEGER NOT NULL,
  output_tokens INTEGER NOT NULL,
  cost_usd REAL NOT NULL,
  latency_ms INTEGER NOT NULL,
  stop_reason TEXT,
  created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS known_concepts (
  concept TEXT NOT NULL,
  course_id TEXT NOT NULL REFERENCES courses(id),
  marked_at TEXT NOT NULL,
  PRIMARY KEY (concept, course_id)
);
"""


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def new_id() -> str:
    return uuid.uuid4().hex[:12]


class Store:
    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        with self._lock:
            self._conn.executescript(SCHEMA)
            self._migrate()

    def _migrate(self) -> None:
        # Columns added after the first release; CREATE TABLE IF NOT EXISTS leaves old files untouched.
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(lectures)")}
        if "source" not in cols:
            self._conn.execute("ALTER TABLE lectures ADD COLUMN source TEXT NOT NULL DEFAULT 'mic'")  # mic | replay | youtube
        if "source_url" not in cols:
            self._conn.execute("ALTER TABLE lectures ADD COLUMN source_url TEXT")  # D27: the YouTube link, for provenance
        self._conn.commit()

    # -- generic helpers -------------------------------------------------
    def execute(self, sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return cur

    def query(self, sql: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(r) for r in self._conn.execute(sql, tuple(params)).fetchall()]

    def one(self, sql: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    # -- courses ---------------------------------------------------------
    def create_course(self, name: str, timezone_name: str, term_calendar: dict | None = None) -> dict:
        cid = new_id()
        self.execute(
            "INSERT INTO courses (id, name, timezone, term_calendar, created_at) VALUES (?,?,?,?,?)",
            (cid, name, timezone_name, json.dumps(term_calendar or {}), now_iso()),
        )
        return self.one("SELECT * FROM courses WHERE id=?", (cid,))  # type: ignore[return-value]

    def list_courses(self) -> list[dict]:
        return self.query("SELECT * FROM courses ORDER BY created_at")

    # -- lectures --------------------------------------------------------
    def start_lecture(
        self,
        course_id: str,
        asr_model: str,
        asr_device: str,
        power_state: str,
        battery: int | None,
        source: str = "mic",
        notes: str | None = None,
        source_url: str | None = None,
    ) -> dict:
        lid = new_id()
        self.execute(
            "INSERT INTO lectures (id, course_id, started_at, status, asr_model, asr_device, power_state, battery_start, source, notes, source_url)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (lid, course_id, now_iso(), "recording", asr_model, asr_device, power_state, battery, source, notes, source_url),
        )
        return self.one("SELECT * FROM lectures WHERE id=?", (lid,))  # type: ignore[return-value]

    def end_lecture(self, lecture_id: str, rtf_p95: float | None, battery: int | None) -> None:
        self.execute(
            "UPDATE lectures SET ended_at=?, status='ended', asr_rtf_p95=?, battery_end=? WHERE id=?",
            (now_iso(), rtf_p95, battery, lecture_id),
        )

    def get_lecture(self, lecture_id: str) -> dict | None:
        return self.one("SELECT * FROM lectures WHERE id=?", (lecture_id,))

    def list_lectures(self) -> list[dict]:
        return self.query("SELECT l.*, c.name AS course_name FROM lectures l JOIN courses c ON c.id=l.course_id ORDER BY l.started_at DESC")

    # -- timeline rows ---------------------------------------------------
    def add_segment(
        self,
        lecture_id: str,
        t0: float,
        t1: float,
        text: str,
        conf: float | None,
        model: str,
        trigger_score: float,
        trigger_terms: list[str],
    ) -> int:
        cur = self.execute(
            "INSERT INTO segments (lecture_id, t0, t1, text, asr_conf, asr_model, trigger_score, trigger_terms) VALUES (?,?,?,?,?,?,?,?)",
            (lecture_id, t0, t1, text, conf, model, trigger_score, json.dumps(trigger_terms)),
        )
        return int(cur.lastrowid)

    def segments(self, lecture_id: str) -> list[dict]:
        rows = self.query("SELECT * FROM segments WHERE lecture_id=? ORDER BY t0", (lecture_id,))
        for r in rows:
            r["trigger_terms"] = json.loads(r["trigger_terms"])
        return rows

    def add_gap(self, lecture_id: str, t0: float, cause: str) -> int:
        cur = self.execute("INSERT INTO gaps (lecture_id, t0, cause) VALUES (?,?,?)", (lecture_id, t0, cause))
        return int(cur.lastrowid)

    def close_gap(self, gap_id: int, t1: float) -> None:
        self.execute("UPDATE gaps SET t1=? WHERE id=?", (t1, gap_id))

    def gaps(self, lecture_id: str) -> list[dict]:
        return self.query("SELECT * FROM gaps WHERE lecture_id=? ORDER BY t0", (lecture_id,))

    def add_flag(self, lecture_id: str, t: float, w0: float, w1: float) -> int:
        cur = self.execute(
            "INSERT INTO flags (lecture_id, t, window_t0, window_t1) VALUES (?,?,?,?)",
            (lecture_id, t, w0, w1),
        )
        return int(cur.lastrowid)

    def flags(self, lecture_id: str) -> list[dict]:
        return self.query("SELECT * FROM flags WHERE lecture_id=? ORDER BY t", (lecture_id,))

    # -- decks -----------------------------------------------------------
    def upsert_deck(self, sha256: str, course_id: str | None, filename: str, slides: list[str]) -> dict:
        if self.one("SELECT id FROM decks WHERE id=?", (sha256,)) is None:
            self.execute(
                "INSERT INTO decks (id, course_id, filename, slide_count, slides_text, created_at) VALUES (?,?,?,?,?,?)",
                (sha256, course_id, filename, len(slides), json.dumps(slides, ensure_ascii=False), now_iso()),
            )
        return self.get_deck(sha256)  # type: ignore[return-value]

    def set_slide_index(self, sha256: str, index_json: str) -> None:
        self.execute("UPDATE decks SET slide_index=? WHERE id=?", (index_json, sha256))

    def get_deck(self, sha256: str) -> dict | None:
        row = self.one("SELECT * FROM decks WHERE id=?", (sha256,))
        if row:
            row["slides_text"] = json.loads(row["slides_text"])
            row["slide_index"] = json.loads(row["slide_index"]) if row["slide_index"] else None
        return row

    def list_decks(self, course_id: str | None = None) -> list[dict]:
        sql = "SELECT id, course_id, filename, slide_count, slide_index IS NOT NULL AS indexed, created_at FROM decks"
        rows = self.query(
            sql + (" WHERE course_id=?" if course_id else "") + " ORDER BY created_at DESC", (course_id,) if course_id else ()
        )
        return rows

    def attach_deck(self, lecture_id: str, deck_id: str | None) -> None:
        self.execute("UPDATE lectures SET deck_id=? WHERE id=?", (deck_id, lecture_id))

    # -- board captures --------------------------------------------------
    def add_board_capture(self, lecture_id: str | None, t_shutter: float | None, shot_at: str | None, status: str) -> str:
        cid = new_id()
        self.execute(
            "INSERT INTO board_captures (id, lecture_id, t_shutter, shot_at, status) VALUES (?,?,?,?,?)",
            (cid, lecture_id, t_shutter, shot_at, status),
        )
        return cid

    def set_board_reading(
        self, capture_id: str, content_kind: str, text: str, latex: list[str], notes: list[str], legibility: float, status: str
    ) -> None:
        self.execute(
            "UPDATE board_captures SET content_kind=?, text=?, latex=?, diagram_notes=?, legibility=?, status=? WHERE id=?",
            (content_kind, text, json.dumps(latex), json.dumps(notes), legibility, status, capture_id),
        )

    def set_board_status(self, capture_id: str, status: str) -> None:
        self.execute("UPDATE board_captures SET status=? WHERE id=?", (status, capture_id))

    def board_captures(self, lecture_id: str) -> list[dict]:
        rows = self.query("SELECT * FROM board_captures WHERE lecture_id=? ORDER BY t_shutter", (lecture_id,))
        for r in rows:
            r["latex"] = json.loads(r["latex"]) if r["latex"] else []
            r["diagram_notes"] = json.loads(r["diagram_notes"]) if r["diagram_notes"] else []
        return rows

    # -- alignment -------------------------------------------------------
    def save_alignment(self, lecture_id: str, windows: list[tuple[float, float, int | None, float]]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM slide_alignment WHERE lecture_id=?", (lecture_id,))
            self._conn.executemany(
                "INSERT INTO slide_alignment (lecture_id, t0, t1, slide, score) VALUES (?,?,?,?,?)",
                [(lecture_id, t0, t1, slide, score) for t0, t1, slide, score in windows],
            )
            self._conn.commit()

    def alignment(self, lecture_id: str) -> list[dict]:
        return self.query("SELECT t0, t1, slide, score FROM slide_alignment WHERE lecture_id=? ORDER BY t0", (lecture_id,))

    # -- usage ledger ----------------------------------------------------
    def add_usage(
        self,
        *,
        call_id: str,
        lecture_id: str | None,
        stage: str,
        model: str,
        effort: str | None,
        input_tokens: int,
        cache_read: int,
        cache_write: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: int,
        stop_reason: str | None,
    ) -> None:
        self.execute(
            "INSERT INTO usage_ledger (call_id, lecture_id, stage, model, effort, input_tokens, cache_read_tokens, cache_write_tokens,"
            " output_tokens, cost_usd, latency_ms, stop_reason, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                call_id,
                lecture_id,
                stage,
                model,
                effort,
                input_tokens,
                cache_read,
                cache_write,
                output_tokens,
                cost_usd,
                latency_ms,
                stop_reason,
                now_iso(),
            ),
        )

    # -- candidate events & suggestions (M3) -----------------------------
    def add_candidate(self, lecture_id: str, c: dict) -> str:
        cid = new_id()
        self.execute(
            "INSERT INTO candidate_events (id, lecture_id, type, title, date_expression, intent, evidence_quote, t0, confidence,"
            " resolved_date, resolution_note, course_hint, status, created_at) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                cid,
                lecture_id,
                c["type"],
                c["title"],
                c.get("date_expression"),
                c["intent"],
                c["evidence_quote"],
                c.get("t0"),
                c["confidence"],
                c.get("resolved_date"),
                c.get("resolution_note"),
                c.get("course_hint"),
                c.get("status", "surfaced"),
                now_iso(),
            ),
        )
        return cid

    def candidates(self, lecture_id: str) -> list[dict]:
        return self.query("SELECT * FROM candidate_events WHERE lecture_id=? ORDER BY t0", (lecture_id,))

    def add_suggestion(self, *, candidate_id: str, lecture_id: str, tier: str, payload: dict, idempotency_key: str) -> str:
        sid = new_id()
        self.execute(
            "INSERT INTO suggestions (id, candidate_id, lecture_id, target, tier, payload, state, idempotency_key, updated_at)"
            " VALUES (?,?,?,?,?,?,?,?,?)",
            (sid, candidate_id, lecture_id, "local", tier, json.dumps(payload, ensure_ascii=False), "proposed", idempotency_key, now_iso()),
        )
        return sid

    def _suggestion_row(self, row: dict | None) -> dict | None:
        if row:
            row["payload"] = json.loads(row["payload"])
        return row

    def get_suggestion(self, sid: str) -> dict | None:
        return self._suggestion_row(self.one("SELECT * FROM suggestions WHERE id=?", (sid,)))

    def suggestion_by_key(self, key: str) -> dict | None:
        return self._suggestion_row(self.one("SELECT * FROM suggestions WHERE idempotency_key=?", (key,)))

    def suggestions(self, state: str | None = None, lecture_id: str | None = None) -> list[dict]:
        clauses, params = [], []
        if state:
            clauses.append("state=?")
            params.append(state)
        if lecture_id:
            clauses.append("lecture_id=?")
            params.append(lecture_id)
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        rows = self.query(f"SELECT * FROM suggestions{where} ORDER BY updated_at DESC", params)
        return [self._suggestion_row(r) for r in rows]  # type: ignore[misc]

    def set_suggestion_state(self, sid: str, state: str, external_id: str | None = None, error: str | None = None) -> None:
        self.execute(
            "UPDATE suggestions SET state=?, external_id=COALESCE(?, external_id), error=?, updated_at=? WHERE id=?",
            (state, external_id, error, now_iso(), sid),
        )

    def update_suggestion_payload(self, sid: str, payload: dict, tier: str | None = None) -> None:
        self.execute(
            "UPDATE suggestions SET payload=?, tier=COALESCE(?, tier), updated_at=? WHERE id=?",
            (json.dumps(payload, ensure_ascii=False), tier, now_iso(), sid),
        )

    # -- recaps (M4) -----------------------------------------------------
    def next_recap_version(self, lecture_id: str) -> int:
        row = self.one("SELECT MAX(version) AS v FROM recaps WHERE lecture_id=?", (lecture_id,))
        return int(row["v"] or 0) + 1 if row else 1

    def add_recap(self, lecture_id: str, version: int, model: str, effort: str, sections: dict) -> str:
        rid = new_id()
        self.execute(
            "INSERT INTO recaps (id, lecture_id, version, model, effort, sections, created_at) VALUES (?,?,?,?,?,?,?)",
            (rid, lecture_id, version, model, effort, json.dumps(sections, ensure_ascii=False), now_iso()),
        )
        return rid

    def _recap_row(self, row: dict | None) -> dict | None:
        if row:
            row["sections"] = json.loads(row["sections"])
            row["flag_helpful"] = json.loads(row["flag_helpful"]) if row.get("flag_helpful") else {}
        return row

    def get_recap(self, rid: str) -> dict | None:
        return self._recap_row(self.one("SELECT * FROM recaps WHERE id=?", (rid,)))

    def latest_recap(self, lecture_id: str) -> dict | None:
        return self._recap_row(self.one("SELECT * FROM recaps WHERE lecture_id=? ORDER BY version DESC LIMIT 1", (lecture_id,)))

    def rate_recap(self, rid: str, rating: int | None, flag_helpful: dict | None) -> None:
        self.execute(
            "UPDATE recaps SET rating=COALESCE(?, rating), flag_helpful=COALESCE(?, flag_helpful) WHERE id=?",
            (rating, json.dumps(flag_helpful) if flag_helpful is not None else None, rid),
        )

    # -- export / delete / retention (M7, D2, D19) ------------------------
    LECTURE_TABLES = (
        "segments",
        "gaps",
        "flags",
        "board_captures",
        "slide_alignment",
        "candidate_events",
        "suggestions",
        "recaps",
        "usage_ledger",
    )

    def export_lecture(self, lecture_id: str) -> dict | None:
        lecture = self.get_lecture(lecture_id)
        if lecture is None:
            return None
        course = self.one("SELECT * FROM courses WHERE id=?", (lecture["course_id"],))
        deck = self.get_deck(lecture["deck_id"]) if lecture.get("deck_id") else None
        return {
            "exported_at": now_iso(),
            "course": course,
            "lecture": lecture,
            "deck": {k: v for k, v in deck.items() if k != "slides_text"} if deck else None,
            "segments": self.segments(lecture_id),
            "gaps": self.gaps(lecture_id),
            "flags": self.flags(lecture_id),
            "board_captures": self.board_captures(lecture_id),
            "alignment": self.alignment(lecture_id),
            "candidates": self.candidates(lecture_id),
            "suggestions": self.suggestions(None, lecture_id),
            "recaps": [self._recap_row(r) for r in self.query("SELECT * FROM recaps WHERE lecture_id=? ORDER BY version", (lecture_id,))],
            "usage": self.usage_by_stage(lecture_id),
        }

    def delete_lecture(self, lecture_id: str) -> bool:
        if self.get_lecture(lecture_id) is None:
            return False
        with self._lock:
            for table in self.LECTURE_TABLES:
                self._conn.execute(f"DELETE FROM {table} WHERE lecture_id=?", (lecture_id,))
            self._conn.execute("DELETE FROM lectures WHERE id=?", (lecture_id,))
            self._conn.commit()
        return True

    def delete_course(self, course_id: str) -> int:
        lectures = self.query("SELECT id FROM lectures WHERE course_id=?", (course_id,))
        for row in lectures:
            self.delete_lecture(row["id"])
        with self._lock:
            self._conn.execute("DELETE FROM known_concepts WHERE course_id=?", (course_id,))
            self._conn.execute("DELETE FROM decks WHERE course_id=?", (course_id,))
            self._conn.execute("DELETE FROM courses WHERE id=?", (course_id,))
            self._conn.commit()
        return len(lectures)

    def purge_transcripts(self, default_days: int) -> int:
        """Delete transcript segments older than the retention window (course
        override, else default). Recaps, flags and suggestions stay. Returns
        the number of lectures purged."""
        rows = self.query(
            "SELECT l.id, l.ended_at, COALESCE(c.transcript_retention_days, ?) AS days FROM lectures l JOIN courses c ON c.id=l.course_id"
            " WHERE l.ended_at IS NOT NULL AND l.notes IS NOT 'transcript purged'",
            (default_days,),
        )
        purged = 0
        now = datetime.now(UTC)
        for r in rows:
            if not r["days"] or r["days"] <= 0:
                continue
            ended = datetime.fromisoformat(r["ended_at"])
            if (now - ended).days >= int(r["days"]):
                with self._lock:
                    self._conn.execute("DELETE FROM segments WHERE lecture_id=?", (r["id"],))
                    self._conn.execute("UPDATE lectures SET notes='transcript purged' WHERE id=?", (r["id"],))
                    self._conn.commit()
                purged += 1
        return purged

    def db_size_bytes(self) -> int:
        row = self.one("SELECT page_count * page_size AS b FROM pragma_page_count(), pragma_page_size()")
        return int(row["b"]) if row else 0

    # -- dashboard (M7) --------------------------------------------------
    def dashboard(self) -> dict:
        courses = self.query(
            "SELECT c.id, c.name, COUNT(l.id) AS lectures, MAX(l.started_at) AS last_lecture,"
            " (SELECT COUNT(*) FROM flags f JOIN lectures l2 ON l2.id=f.lecture_id WHERE l2.course_id=c.id AND f.resolved=0) AS open_flags,"
            " (SELECT COUNT(*) FROM suggestions s WHERE s.state='proposed' AND s.lecture_id IN (SELECT id FROM lectures WHERE course_id=c.id)) AS pending_suggestions,"
            " (SELECT COUNT(DISTINCT r.lecture_id) FROM recaps r JOIN lectures l3 ON l3.id=r.lecture_id WHERE l3.course_id=c.id) AS recaps"
            " FROM courses c LEFT JOIN lectures l ON l.course_id=c.id GROUP BY c.id ORDER BY c.name"
        )
        inbox = {r["tier"]: r["n"] for r in self.query("SELECT tier, COUNT(*) AS n FROM suggestions WHERE state='proposed' GROUP BY tier")}
        upcoming = [
            s
            for s in self.suggestions(None)
            if s["state"] in ("confirmed", "written")
            and s["payload"].get("date")
            and s["payload"]["date"] >= datetime.now(UTC).date().isoformat()
        ]
        upcoming.sort(key=lambda s: (s["payload"]["date"], s["payload"].get("time") or ""))
        usage = self.usage_by_stage()
        asr = self.one(
            "SELECT AVG(asr_rtf_p95) AS rtf, AVG(CASE WHEN power_state='battery' AND battery_start IS NOT NULL AND battery_end IS NOT NULL"
            " AND (julianday(ended_at) - julianday(started_at)) * 86400 >= 600"  # a rate needs at least 10 minutes to mean anything
            " THEN (battery_start - battery_end) * 3600.0 / ((julianday(ended_at) - julianday(started_at)) * 86400) END) AS drain_per_hour,"
            " SUM((julianday(ended_at) - julianday(started_at)) * 24) AS hours FROM lectures WHERE ended_at IS NOT NULL"
            # Live capture only: a replay transcribes a file at full speed, so its battery use and hours would mislead.
            " AND source = 'mic'"
        )
        return {
            "courses": courses,
            "inbox": {"one_tap": inbox.get("one_tap", 0), "maybe": inbox.get("maybe", 0)},
            "upcoming": upcoming[:20],
            "usage": {
                "by_stage": usage,
                "total_cost_usd": round(sum(u["cost_usd"] for u in usage), 4),
                "calls": sum(u["calls"] for u in usage),
            },
            "asr": {
                "rtf_p95_avg": round(asr["rtf"], 3) if asr and asr["rtf"] else None,
                "battery_drain_pct_per_hour": round(asr["drain_per_hour"], 1) if asr and asr["drain_per_hour"] else None,
                "recorded_hours": round(asr["hours"], 2) if asr and asr["hours"] else 0.0,
            },
            "storage": {"db_bytes": self.db_size_bytes()},
        }

    def usage_by_stage(self, lecture_id: str | None = None) -> list[dict]:
        where = " WHERE lecture_id=?" if lecture_id else ""
        return self.query(
            "SELECT stage, model, COUNT(*) AS calls, SUM(input_tokens) AS input_tokens, SUM(cache_read_tokens) AS cache_read,"
            " SUM(cache_write_tokens) AS cache_write, SUM(output_tokens) AS output_tokens, ROUND(SUM(cost_usd), 4) AS cost_usd"
            f" FROM usage_ledger{where} GROUP BY stage, model ORDER BY stage",
            (lecture_id,) if lecture_id else (),
        )
