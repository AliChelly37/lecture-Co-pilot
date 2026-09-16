"""Post-lecture pipeline steps that combine the store, the local processors and
the LlmGateway. Each step is idempotent and resumable on its own."""

from __future__ import annotations

import logging
from pathlib import Path

from lecture_copilot.align import SlideAligner, coverage, off_slide_stretches
from lecture_copilot.deck import SLIDE_INDEX_SYSTEM, SlideIndex, extract_deck, index_to_json, slide_index_user_message
from lecture_copilot.llm import LlmGateway
from lecture_copilot.photos import BOARD_SYSTEM, BoardReading, prepare_photo, seconds_into_lecture
from lecture_copilot.store import Store

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
