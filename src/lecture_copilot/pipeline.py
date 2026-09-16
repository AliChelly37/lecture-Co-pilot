"""Post-lecture pipeline steps that combine the store, the local processors and
the LlmGateway. Each step is idempotent and resumable on its own."""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

from lecture_copilot.align import SlideAligner, coverage, off_slide_stretches
from lecture_copilot.dates import DateResolver
from lecture_copilot.deck import SLIDE_INDEX_SYSTEM, SlideIndex, extract_deck, index_to_json, slide_index_user_message
from lecture_copilot.extract import (
    EXTRACT_SYSTEM,
    Candidate,
    Chunk,
    ChunkExtraction,
    chunk_prompt,
    chunk_segments,
    clean_hint,
    locate,
    merge,
)
from lecture_copilot.llm import LlmGateway
from lecture_copilot.photos import BOARD_SYSTEM, BoardReading, prepare_photo, seconds_into_lecture
from lecture_copilot.recap import (
    ASK_SYSTEM,
    FLAG_SYSTEM,
    NOTES_SYSTEM,
    RECAP_SYSTEM,
    Answer,
    ChunkNotes,
    FlagExplanation,
    Recap,
    ask_prompt,
    boards_in,
    flag_prompt,
    normalise_recap,
    notes_prompt,
    pick_excerpts,
    recap_prompt,
    slides_in,
    window_text,
)
from lecture_copilot.store import Store
from lecture_copilot.suggest import SuggestionService

log = logging.getLogger(__name__)

KEEP_KINDS = {"board", "slide_projection", "paper"}


def ingest_deck(store: Store, path: Path, course_id: str | None) -> dict:
    """Local only: extract text and store the deck keyed by its hash."""
    extracted = extract_deck(path)
    return store.upsert_deck(extracted.sha256, course_id, extracted.filename, extracted.slides)


def index_deck(store: Store, llm: LlmGateway, deck_id: str) -> dict:
    """One Sonnet 5 `low` call per deck; cached by deck hash in the store."""
    deck = store.get_deck(deck_id)
    if deck is None:
        raise KeyError("unknown deck")
    if deck["slide_index"]:
        return deck
    index = llm.structured(
        stage="slide_index",
        schema=SlideIndex,
        system=[SLIDE_INDEX_SYSTEM],
        user=slide_index_user_message(deck["slides_text"]),
        effort="low",
        cache=False,
        max_tokens=8000,
    )
    store.set_slide_index(deck_id, index_to_json(index))
    return store.get_deck(deck_id)  # type: ignore[return-value]


def import_photos(store: Store, llm: LlmGateway, lecture: dict, course: dict, files: list[tuple[str, bytes]]) -> list[dict]:
    """Prepare each photo in memory, place it on the timeline by EXIF time,
    read it with Sonnet 5 vision, store only the derived text."""
    for filename, data in files:
        try:
            photo = prepare_photo(data, filename, course["timezone"])
        except Exception as exc:
            log.warning("skipping %s: %s", filename, exc)
            continue
        t = seconds_into_lecture(photo.shot_at, lecture["started_at"], lecture["ended_at"])
        shot_iso = photo.shot_at.isoformat() if photo.shot_at else None
        cid = store.add_board_capture(lecture["id"], t, shot_iso, "pending")
        try:
            reading = llm.structured(
                stage="board_ocr",
                schema=BoardReading,
                system=[BOARD_SYSTEM],
                user="Read this photo.",
                images=[photo.jpeg],
                effort="low",
                cache=False,
                lecture_id=lecture["id"],
                max_tokens=4000,
            )
        except Exception:
            log.exception("board reading failed for %s", filename)
            store.set_board_status(cid, "failed")
            continue
        if reading.content_kind not in KEEP_KINDS:
            store.set_board_reading(cid, reading.content_kind, "", [], [], 0.0, "discarded")
            continue
        store.set_board_reading(
            cid, reading.content_kind, reading.text, reading.latex, reading.diagram_notes, reading.legibility, "derived"
        )
        # `photo.jpeg` goes out of scope here; nothing was written to disk.
    return store.board_captures(lecture["id"])


def extract_lecture(store: Store, llm: LlmGateway, lecture_id: str, window_s: float = 600.0, overlap_s: float = 60.0) -> dict:
    """Deadline extraction after the lecture (D15): windows sized for the
    provider, one schema-constrained call per window, then code resolves
    dates, merges duplicates, applies corrections and files suggestions."""
    lecture = store.get_lecture(lecture_id)
    if lecture is None:
        raise KeyError("unknown lecture")
    course = store.one("SELECT * FROM courses WHERE id=?", (lecture["course_id"],))
    segments = store.segments(lecture_id)
    empty = {"chunks": 0, "calls": 0, "candidates": 0, "surfaced": 0, "suggestions": {"one_tap": 0, "maybe": 0, "log": 0, "existing": 0}}
    if not segments or course is None:
        return empty | {"note": "no transcript"}

    deck = store.get_deck(lecture["deck_id"]) if lecture.get("deck_id") else None
    slide_dates: list[str] = []
    if deck and deck.get("slide_index"):
        for entry in deck["slide_index"].get("slides", []):
            slide_dates += [f"slide {entry['slide']}: {d}" for d in entry.get("stated_dates", [])]

    resolver = DateResolver(datetime.fromisoformat(lecture["started_at"]), course["timezone"], json.loads(course["term_calendar"] or "{}"))
    chunks = chunk_segments(segments, window_s=window_s, overlap_s=overlap_s)
    cands: list[Candidate] = []
    for ch in chunks:
        hints = [f"[{int(s['t0'] // 60):02d}:{int(s['t0'] % 60):02d}] {s['text'][:90]}" for s in ch.segments if s.get("trigger_terms")]
        out = llm.structured(
            stage="extract",
            schema=ChunkExtraction,
            system=[EXTRACT_SYSTEM],
            user=chunk_prompt(ch, slide_dates, hints),
            effort="low",
            cache=False,
            lecture_id=lecture_id,
            max_tokens=2000,
        )
        for e in out.events:
            t0, asr_conf = locate(e.evidence_quote, ch.segments)
            res = resolver.resolve(e.date_expression, e.time_expression or None)
            cands.append(
                Candidate(
                    e.type,
                    e.title,
                    e.date_expression,
                    e.time_expression,
                    e.intent,
                    e.evidence_quote,
                    e.confidence,
                    clean_hint(e.course_hint, course["name"]),
                    t0,
                    asr_conf,
                    res,
                    chunk=ch.index,
                )
            )
    merge(cands)
    for c in cands:
        c.id = store.add_candidate(
            lecture_id,
            {
                "type": c.type,
                "title": c.title,
                "date_expression": c.date_expression,
                "intent": c.intent,
                "evidence_quote": c.evidence_quote,
                "t0": c.t0,
                "confidence": c.confidence,
                "resolved_date": c.resolution.iso(),
                "resolution_note": c.resolution.note,
                "course_hint": c.course_hint,
                "status": c.status,
            },
        )
    counts = SuggestionService(store).propose(cands, lecture, course)
    return {
        "chunks": len(chunks),
        "calls": len(chunks),
        "candidates": len(cands),
        "surfaced": sum(1 for c in cands if c.status == "surfaced"),
        "suggestions": counts,
    }


def recap_lecture(
    store: Store,
    llm: LlmGateway,
    lecture_id: str,
    window_s: float = 600.0,
    progress: Callable[[dict], None] | None = None,
) -> dict:
    """Map-reduce recap (M4). Emits progress events so a 2-3 minute local
    run is visible in the UI."""
    lecture = store.get_lecture(lecture_id)
    if lecture is None:
        raise KeyError("unknown lecture")
    course = store.one("SELECT * FROM courses WHERE id=?", (lecture["course_id"],))
    segments = store.segments(lecture_id)
    if not segments or course is None:
        raise ValueError("no transcript to recap")
    deck = store.get_deck(lecture["deck_id"]) if lecture.get("deck_id") else None
    alignment = store.alignment(lecture_id)
    captures = store.board_captures(lecture_id)
    flags = store.flags(lecture_id)
    gaps = store.gaps(lecture_id)
    chunks = chunk_segments(segments, window_s=window_s, overlap_s=0.0)
    total = len(chunks) + len(flags) + 1
    done = 0

    def tick(stage: str) -> None:
        nonlocal done
        done += 1
        if progress:
            progress({"type": "progress", "job": "recap", "lecture_id": lecture_id, "stage": stage, "done": done, "total": total})

    notes: list[tuple[Chunk, ChunkNotes]] = []
    for ch in chunks:
        n = llm.structured(
            stage="recap_notes",
            schema=ChunkNotes,
            system=[NOTES_SYSTEM],
            user=notes_prompt(ch, slides_in(alignment, deck, ch.t0, ch.t1), boards_in(captures, ch.t0, ch.t1)),
            effort="low",
            cache=False,
            lecture_id=lecture_id,
            max_tokens=1200,
        )
        notes.append((ch, n))
        tick("notes")

    flag_out: list[dict] = []
    for f in flags:
        text = window_text(segments, f["window_t0"], f["window_t1"])
        if not text.strip():
            tick("flag")
            continue
        try:
            fe = llm.structured(
                stage="recap_flag",
                schema=FlagExplanation,
                system=[FLAG_SYSTEM],
                user=flag_prompt(
                    f["t"],
                    text,
                    slides_in(alignment, deck, f["window_t0"], f["window_t1"]),
                    boards_in(captures, f["window_t0"], f["window_t1"]),
                ),
                effort="medium",
                cache=False,
                lecture_id=lecture_id,
                max_tokens=900,
            )
            flag_out.append({"flag_id": f["id"], "t": f["t"], **fe.model_dump()})
        except Exception:
            log.exception("flag explanation failed for flag %s", f["id"])
            flag_out.append({"flag_id": f["id"], "t": f["t"], "error": "explanation failed"})
        tick("flag")

    events = [
        f"{s['payload']['title']} ({s['payload']['type']}) {s['payload']['date'] or s['payload']['date_expression']}"
        for s in store.suggestions(None, lecture_id)
        if s["state"] != "dismissed"
    ]
    windows = SlideAligner(deck["slides_text"]).align(segments) if deck else []
    off = off_slide_stretches(windows) if windows else []
    recap = llm.structured(
        stage="recap",
        schema=Recap,
        system=[RECAP_SYSTEM],
        user=recap_prompt(course["name"], notes, off, gaps, events),
        effort="high",
        cache=False,
        lecture_id=lecture_id,
        max_tokens=3000,
    )
    tick("recap")

    sections = recap.model_dump() | {
        "flag_explanations": flag_out,
        "chunk_notes": [{"t0": ch.t0, "t1": ch.t1, **n.model_dump()} for ch, n in notes],
        "detected_events": events,
    }
    sections = normalise_recap(
        sections, has_deck=deck is not None, n_boards=len([c for c in captures if c.get("status") == "derived"]), has_gaps=bool(gaps)
    )
    version = store.next_recap_version(lecture_id)
    rid = store.add_recap(lecture_id, version, f"{llm.provider}:{llm.describe()['text_model']}", "high", sections)
    store.execute("UPDATE lectures SET status='processed' WHERE id=?", (lecture_id,))
    return store.get_recap(rid)  # type: ignore[return-value]


def ask_lecture(store: Store, llm: LlmGateway, lecture_id: str, question: str, window_s: float = 600.0) -> dict:
    lecture = store.get_lecture(lecture_id)
    if lecture is None:
        raise KeyError("unknown lecture")
    segments = store.segments(lecture_id)
    if not segments:
        raise ValueError("no transcript")
    chunks = chunk_segments(segments, window_s=window_s, overlap_s=0.0)
    excerpts = pick_excerpts(chunks, question)
    latest = store.latest_recap(lecture_id)
    summary = "\n".join(f"- {h}" for h in latest["sections"].get("highlights", [])) if latest else ""
    answer = llm.structured(
        stage="ask",
        schema=Answer,
        system=[ASK_SYSTEM],
        user=ask_prompt(question, excerpts, summary),
        effort="medium",
        cache=False,
        lecture_id=lecture_id,
        max_tokens=800,
    )
    return answer.model_dump() | {"excerpts": [{"t0": c.t0, "t1": c.t1} for c in excerpts]}


def align_lecture(store: Store, lecture_id: str) -> dict:
    lecture = store.get_lecture(lecture_id)
    if lecture is None:
        raise KeyError("unknown lecture")
    deck = store.get_deck(lecture["deck_id"]) if lecture.get("deck_id") else None
    if deck is None:
        store.save_alignment(lecture_id, [])
        return {"windows": [], "coverage": 0.0, "off_slide": [], "note": "no deck attached"}
    windows = SlideAligner(deck["slides_text"]).align(store.segments(lecture_id))
    store.save_alignment(lecture_id, [(w.t0, w.t1, w.slide, w.score) for w in windows])
    cov = coverage(windows)
    return {
        "windows": [{"t0": w.t0, "t1": w.t1, "slide": w.slide, "score": w.score} for w in windows],
        "coverage": cov,
        "off_slide": [{"t0": a, "t1": b} for a, b in off_slide_stretches(windows)],
        "deck_mismatch": len(windows) >= 10 and cov < 0.2,
    }
