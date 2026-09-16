"""HTTP + WebSocket API, localhost only. Serves the built frontend when
frontend/dist exists."""

from __future__ import annotations

import asyncio
import logging
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, Response, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lecture_copilot import pipeline
from lecture_copilot.bus import EventBus
from lecture_copilot.capture import MicCapture
from lecture_copilot.config import settings
from lecture_copilot.llm import LlmGateway, LlmUnavailable
from lecture_copilot.power import SleepGuard
from lecture_copilot.session import LectureSession
from lecture_copilot.store import Store
from lecture_copilot.suggest import SuggestionService

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT / "frontend" / "dist"


class AppState:
    store: Store
    bus: EventBus
    session: LectureSession
    sleep_guard: SleepGuard
    llm: LlmGateway


state = AppState()


@asynccontextmanager
async def lifespan(app: FastAPI):
    state.store = Store(settings.db_path)
    state.bus = EventBus()
    state.bus.bind(asyncio.get_running_loop())
    state.session = LectureSession(settings, state.store, state.bus)
    state.sleep_guard = SleepGuard()
    state.llm = LlmGateway(settings, state.store)
    log.info("data dir: %s; Claude API %s", settings.data_dir.resolve(), "configured" if state.llm.available else "NOT configured")
    try:
        yield
    finally:
        if state.session.lecture is not None:
            await asyncio.to_thread(state.session.stop)
        state.sleep_guard.release()
        state.store.close()


app = FastAPI(title="Lecture Co-Pilot", lifespan=lifespan)


# -- models --------------------------------------------------------------
class CourseIn(BaseModel):
    name: str
    timezone: str | None = None
    term_calendar: dict | None = None


class StartIn(BaseModel):
    course_id: str
    language: str = "en"


class AttachDeckIn(BaseModel):
    deck_id: str | None


def _lecture_or_404(lecture_id: str) -> dict:
    lecture = state.store.get_lecture(lecture_id)
    if lecture is None:
        raise HTTPException(404, "unknown lecture")
    return lecture


def _llm_or_503() -> LlmGateway:
    if not state.llm.available:
        raise HTTPException(503, "Claude API not configured: set ANTHROPIC_API_KEY in .env (capture works without it)")
    return state.llm


# -- status & devices ----------------------------------------------------
@app.get("/api/status")
async def get_status() -> dict:
    return {**state.session.status(), "llm_available": state.llm.available}


@app.get("/api/devices")
async def get_devices() -> list[dict]:
    return await asyncio.to_thread(MicCapture.list_devices)


# -- courses -------------------------------------------------------------
@app.get("/api/courses")
async def list_courses() -> list[dict]:
    return state.store.list_courses()


@app.post("/api/courses")
async def create_course(body: CourseIn) -> dict:
    tz = body.timezone
    if not tz:
        import tzlocal

        tz = tzlocal.get_localzone_name()
    return state.store.create_course(body.name, tz, body.term_calendar)


# -- lectures: live ------------------------------------------------------
@app.post("/api/lectures/start")
async def start_lecture(body: StartIn) -> dict:
    try:
        lecture = await asyncio.to_thread(state.session.start, body.course_id, body.language)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except PermissionError as exc:
        raise HTTPException(403, str(exc)) from exc
    except Exception as exc:  # mic unavailable, etc.
        log.exception("start failed")
        raise HTTPException(500, f"could not start capture: {exc}") from exc
    state.sleep_guard.acquire()  # loop thread; released on the same thread in stop()
    return lecture


@app.post("/api/lectures/stop")
async def stop_lecture() -> dict:
    try:
        lecture = await asyncio.to_thread(state.session.stop)
    except RuntimeError as exc:
        raise HTTPException(409, str(exc)) from exc
    finally:
        state.sleep_guard.release()
    return lecture


@app.post("/api/lectures/flag")
async def flag() -> dict:
    result = state.session.flag()
    if result is None:
        raise HTTPException(409, "no lecture is recording")
    return result


@app.post("/api/lectures/pause")
async def pause() -> dict:
    if state.session.lecture is None:
        raise HTTPException(409, "no lecture is recording")
    return {"paused": state.session.toggle_pause()}


# -- lectures: after class -----------------------------------------------
@app.get("/api/lectures")
async def list_lectures() -> list[dict]:
    return state.store.list_lectures()


@app.get("/api/lectures/{lecture_id}")
async def get_lecture(lecture_id: str) -> dict:
    lecture = _lecture_or_404(lecture_id)
    deck = state.store.get_deck(lecture["deck_id"]) if lecture.get("deck_id") else None
    return {
        "lecture": lecture,
        "segments": state.store.segments(lecture_id),
        "flags": state.store.flags(lecture_id),
        "gaps": state.store.gaps(lecture_id),
        "deck": {k: v for k, v in deck.items() if k != "slides_text"} | {"indexed": deck["slide_index"] is not None} if deck else None,
        "captures": state.store.board_captures(lecture_id),
        "alignment": state.store.alignment(lecture_id),
        "candidates": state.store.candidates(lecture_id),
        "suggestions": state.store.suggestions(None, lecture_id),
        "usage": state.store.usage_by_stage(lecture_id),
    }


@app.post("/api/lectures/{lecture_id}/deck")
async def attach_deck(lecture_id: str, body: AttachDeckIn) -> dict:
    _lecture_or_404(lecture_id)
    if body.deck_id and state.store.get_deck(body.deck_id) is None:
        raise HTTPException(404, "unknown deck")
    state.store.attach_deck(lecture_id, body.deck_id)
    return {"ok": True}


@app.post("/api/lectures/{lecture_id}/photos")
async def import_photos(lecture_id: str, files: Annotated[list[UploadFile], File()]) -> list[dict]:
    lecture = _lecture_or_404(lecture_id)
    if lecture["status"] == "recording":
        raise HTTPException(409, "stop the lecture before importing photos")
    llm = _llm_or_503()  # refuse before reading any image: images are never held for later
    course = state.store.one("SELECT * FROM courses WHERE id=?", (lecture["course_id"],))
    payload = [(f.filename or "photo", await f.read()) for f in files]
    return await asyncio.to_thread(pipeline.import_photos, state.store, llm, lecture, course, payload)


@app.post("/api/lectures/{lecture_id}/align")
async def align(lecture_id: str) -> dict:
    _lecture_or_404(lecture_id)
    return await asyncio.to_thread(pipeline.align_lecture, state.store, lecture_id)


@app.post("/api/lectures/{lecture_id}/extract")
async def extract(lecture_id: str) -> dict:
    lecture = _lecture_or_404(lecture_id)
    if lecture["status"] == "recording":
        raise HTTPException(409, "stop the lecture before extracting deadlines")
    llm = _llm_or_503()
    window = settings.extract_window_s if llm.provider == "ollama" else 0.0
    try:
        return await asyncio.to_thread(pipeline.extract_lecture, state.store, llm, lecture_id, window, settings.extract_overlap_s)
    except LlmUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc


# -- suggestions (the inbox) --------------------------------------------
class DateIn(BaseModel):
    date: str
    time: str | None = None


def _svc() -> SuggestionService:
    return SuggestionService(state.store)


def _suggestion_action(fn, sid: str) -> dict:
    try:
        return fn(sid)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/suggestions")
async def list_suggestions(state_filter: str | None = "proposed", lecture_id: str | None = None) -> list[dict]:
    return state.store.suggestions(None if state_filter in (None, "", "all") else state_filter, lecture_id)


@app.post("/api/suggestions/{sid}/confirm")
async def confirm_suggestion(sid: str) -> dict:
    return _suggestion_action(_svc().confirm, sid)


@app.post("/api/suggestions/{sid}/dismiss")
async def dismiss_suggestion(sid: str) -> dict:
    return _suggestion_action(_svc().dismiss, sid)


@app.post("/api/suggestions/{sid}/undo")
async def undo_suggestion(sid: str) -> dict:
    return _suggestion_action(_svc().undo, sid)


@app.post("/api/suggestions/{sid}/date")
async def set_suggestion_date(sid: str, body: DateIn) -> dict:
    try:
        return _svc().set_date(sid, body.date, body.time)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(409, str(exc)) from exc


@app.get("/api/suggestions/export.ics")
async def export_ics() -> Response:
    ics = _svc().ics(state.store.suggestions(state="confirmed"))
    return Response(content=ics, media_type="text/calendar", headers={"Content-Disposition": "attachment; filename=lecture-copilot.ics"})


# -- decks ---------------------------------------------------------------
@app.get("/api/decks")
async def list_decks(course_id: str | None = None) -> list[dict]:
    return state.store.list_decks(course_id)


@app.post("/api/decks")
async def upload_deck(file: Annotated[UploadFile, File()], course_id: Annotated[str | None, Form()] = None) -> dict:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in {".pdf", ".pptx"}:
        raise HTTPException(400, "upload a .pdf or .pptx")
    data = await file.read()
    # Extraction libraries want a path; the temp file is deleted immediately after.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / (file.filename or f"deck{suffix}")
        path.write_bytes(data)
        try:
            deck = await asyncio.to_thread(pipeline.ingest_deck, state.store, path, course_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
    return {k: v for k, v in deck.items() if k != "slides_text"} | {"indexed": deck["slide_index"] is not None}


@app.post("/api/decks/{deck_id}/index")
async def build_index(deck_id: str) -> dict:
    llm = _llm_or_503()
    try:
        deck = await asyncio.to_thread(pipeline.index_deck, state.store, llm, deck_id)
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except LlmUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    return {k: v for k, v in deck.items() if k != "slides_text"} | {"indexed": True}


# -- usage ---------------------------------------------------------------
@app.get("/api/usage")
async def usage(lecture_id: str | None = None) -> list[dict]:
    return state.store.usage_by_stage(lecture_id)


# -- live events ---------------------------------------------------------
@app.websocket("/ws")
async def ws(websocket: WebSocket) -> None:
    await websocket.accept()
    q = state.bus.subscribe()
    try:
        await websocket.send_json({"type": "hello", **state.session.status()})
        while True:
            event = await q.get()
            await websocket.send_json(event)
    except WebSocketDisconnect:
        pass
    except Exception:
        log.debug("websocket closed", exc_info=True)
    finally:
        state.bus.unsubscribe(q)


# -- frontend ------------------------------------------------------------
if FRONTEND_DIST.is_dir():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
else:

    @app.get("/")
    async def no_frontend() -> dict:
        return {"message": "frontend not built: run `npm run build` in frontend/, or use the Vite dev server"}
