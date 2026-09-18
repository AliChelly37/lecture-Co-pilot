"""API wiring without a microphone or an API key: courses, deck upload and
extraction, deck attach, alignment, and the honest 503s where Claude is
required."""

import io
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lecture_copilot import api
    from lecture_copilot.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    # An unconfigured paid provider: the API must answer 503 where a model is required.
    monkeypatch.setattr(settings, "llm_provider", "anthropic")
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    with TestClient(api.app) as c:
        yield c


def make_pdf_bytes(pages: list[str]) -> bytes:
    import pymupdf

    doc = pymupdf.open()
    for text in pages:
        doc.new_page().insert_text((72, 72), text, fontsize=14)
    data = doc.tobytes()
    doc.close()
    return data


def test_deck_upload_attach_and_align(client: TestClient) -> None:
    assert client.get("/api/status").json()["llm_available"] is False
    course = client.post("/api/courses", json={"name": "Thermo", "timezone": "Europe/Paris"}).json()

    pdf = make_pdf_bytes(["Fugacity: effective pressure of a real gas", "Peng-Robinson equation of state, compressibility factor"])
    r = client.post("/api/decks", files={"file": ("deck.pdf", pdf, "application/pdf")}, data={"course_id": course["id"]})
    assert r.status_code == 200, r.text
    deck = r.json()
    assert deck["slide_count"] == 2 and deck["indexed"] is False
    assert client.get("/api/decks", params={"course_id": course["id"]}).json()[0]["id"] == deck["id"]

    # Indexing needs Claude; without a key it is an honest 503, not a crash.
    assert client.post(f"/api/decks/{deck['id']}/index").status_code == 503

    # Fake an ended lecture with a transcript (no mic in tests).
    from lecture_copilot.api import state

    lec = state.store.start_lecture(course["id"], "small", "cpu", "ac", None)
    state.store.add_segment(lec["id"], 0, 25, "fugacity is the effective pressure of a real gas", 0.9, "small", 0, [])
    state.store.add_segment(lec["id"], 30, 55, "peng robinson gives the compressibility factor", 0.9, "small", 0, [])
    state.store.end_lecture(lec["id"], 0.1, None)

    assert client.post(f"/api/lectures/{lec['id']}/deck", json={"deck_id": deck["id"]}).status_code == 200
    a = client.post(f"/api/lectures/{lec['id']}/align").json()
    assert [w["slide"] for w in a["windows"]] == [1, 2]
    assert a["coverage"] == 1.0 and a["deck_mismatch"] is False

    detail = client.get(f"/api/lectures/{lec['id']}").json()
    assert detail["deck"]["id"] == deck["id"] and len(detail["alignment"]) == 2
    assert detail["usage"] == []

    # Photo import refuses before touching images when Claude is unavailable.
    r = client.post(f"/api/lectures/{lec['id']}/photos", files={"files": ("b.jpg", io.BytesIO(b"x"), "image/jpeg")})
    assert r.status_code == 503


@pytest.mark.usefixtures("frozen_now")
def test_extract_and_inbox_with_fake_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    """The whole M3 path with the model replaced by canned output: chunking,
    evidence location, date resolution, merge, tiers, confirm and .ics."""
    from lecture_copilot.api import state
    from lecture_copilot.config import settings
    from lecture_copilot.extract import ChunkExtraction, EventOut

    course = client.post(
        "/api/courses", json={"name": "Thermo", "timezone": "Europe/Paris", "term_calendar": {"week1_start": "2026-09-07"}}
    ).json()
    lec = state.store.start_lecture(course["id"], "gemma3:4b", "cuda", "ac", None)
    state.store.add_segment(
        lec["id"], 47.2, 53.5, "Problem Set 4 is due next Thursday at 5 p.m. Submit it through the course website.", 0.84, "g", 1.2, ["due"]
    )
    state.store.add_segment(lec["id"], 64.5, 68.5, "The midterm exam is on October 14th.", 0.7, "g", 0.9, ["midterm"])
    state.store.add_segment(
        lec["id"], 68.5, 75.0, "Actually, let me correct that, the midterm has moved to October 21st.", 0.7, "g", 0.9, ["moved to"]
    )
    state.store.add_segment(lec["id"], 80.0, 85.0, "If this were due tomorrow you would all be panicking.", 0.8, "g", 0.5, [])
    state.store.add_segment(
        lec["id"],
        90.0,
        96.0,
        "The reading for week 6 is chapter 11; make sure it is done before that week's lecture.",
        0.8,
        "g",
        0.6,
        ["week 6"],
    )
    state.store.add_segment(
        lec["id"],
        100.0,
        105.0,
        "Please hand in the lab report by October 20th, no extensions this time.",
        0.8,
        "g",
        0.9,
        ["hand in", "lab report", "october 20"],
    )
    state.store.end_lecture(lec["id"], 0.1, None)

    canned = ChunkExtraction(
        events=[
            EventOut(
                type="assignment",
                title="Problem set 4",
                date_expression="next Thursday",
                time_expression="5 p.m.",
                intent="commitment",
                evidence_quote="Problem Set 4 is due next Thursday at 5 p.m.",
                confidence=0.95,
                course_hint="",
            ),
            EventOut(
                type="exam",
                title="Midterm exam",
                date_expression="October 14th",
                time_expression="",
                intent="commitment",
                evidence_quote="The midterm exam is on October 14th.",
                confidence=0.9,
                course_hint="",
            ),
            EventOut(
                type="exam",
                title="Midterm exam",
                date_expression="October 21st",
                time_expression="",
                intent="correction",
                evidence_quote="the midterm has moved to October 21st",
                confidence=0.95,
                course_hint="",
            ),
            EventOut(
                type="assignment",
                title="Problem set 4",
                date_expression="tomorrow",
                time_expression="",
                intent="hypothetical",
                evidence_quote="If this were due tomorrow you would all be panicking.",
                confidence=0.9,
                course_hint="",
            ),
            EventOut(
                type="reading",
                title="Chapter 11",
                date_expression="next time",
                time_expression="",
                intent="commitment",
                evidence_quote="reading for next time is chapter eleven",
                confidence=0.9,
                course_hint="",
            ),
            # A junk card on the lab-report line (logged as a joke) must not block the keyword fallback for that line.
            EventOut(
                type="other",
                title="junk on the lab report line",
                date_expression="",
                time_expression="",
                intent="joke",
                evidence_quote="Please hand in the lab report by October 20th, no extensions this time.",
                confidence=0.5,
                course_hint="",
            ),
            # The model put the wrong words in the date field; the line says "week 6" -> rescued.
            EventOut(
                type="reading",
                title="Reading week 6",
                date_expression="chapter 11",
                time_expression="",
                intent="commitment",
                evidence_quote="The reading for week 6 is chapter 11",
                confidence=0.9,
                course_hint="CHEM 300",
            ),
        ]
    )
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(state.llm, "structured", lambda **kw: canned)

    r = client.post(f"/api/lectures/{lec['id']}/extract")
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["chunks"] == 1 and summary["candidates"] == 8
    assert summary["suggestions"] == {"one_tap": 3, "maybe": 2, "log": 3, "existing": 0, "retired": 0}

    inbox = client.get("/api/suggestions").json()
    by_title = {s["payload"]["title"]: s for s in inbox}
    assert by_title["Problem set 4"]["tier"] == "one_tap" and by_title["Problem set 4"]["payload"]["date"] == "2026-09-17"
    assert by_title["Problem set 4"]["payload"]["time"] == "17:00" and by_title["Problem set 4"]["payload"]["t0"] == 47.2
    assert by_title["Midterm exam"]["payload"]["date"] == "2026-10-21"  # the correction, not the superseded date
    assert by_title["Chapter 11"]["tier"] == "maybe" and by_title["Chapter 11"]["payload"]["date"] is None
    fallback = next(x for x in inbox if x["payload"]["evidence_quote"].startswith("Please hand in the lab report"))
    assert fallback["tier"] == "maybe" and fallback["payload"]["date"] == "2026-10-20"
    assert fallback["payload"]["tier_reason"].startswith("found by the keyword filter")
    rescued = by_title["Reading week 6"]
    assert rescued["tier"] == "one_tap" and rescued["payload"]["date"] == "2026-10-12"  # week 1 starts 7 Sep
    assert rescued["payload"]["resolution_note"].startswith("date taken from the transcript line")

    # Re-running is idempotent for suggestions.
    assert client.post(f"/api/lectures/{lec['id']}/extract").json()["suggestions"]["existing"] == 5

    # Maybe tray: needs a date before it can be confirmed.
    sid = by_title["Chapter 11"]["id"]
    assert client.post(f"/api/suggestions/{sid}/confirm").status_code == 409
    assert client.post(f"/api/suggestions/{sid}/date", json={"date": "2026-09-18"}).json()["tier"] == "one_tap"
    assert client.post(f"/api/suggestions/{sid}/confirm").json()["state"] == "confirmed"
    assert client.post(f"/api/suggestions/{by_title['Problem set 4']['id']}/dismiss").json()["state"] == "dismissed"

    ics = client.get("/api/suggestions/export.ics")
    assert ics.status_code == 200 and "Thermo: Chapter 11" in ics.text and "DTSTART;VALUE=DATE:20260918" in ics.text
    detail = client.get(f"/api/lectures/{lec['id']}").json()
    assert {c["status"] for c in detail["candidates"]} >= {"surfaced", "superseded", "log"}


def test_bad_deck_type(client: TestClient) -> None:
    r = client.post("/api/decks", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_html_shell_is_revalidated_but_bundles_may_be_cached(client: TestClient) -> None:
    from lecture_copilot.api import FRONTEND_DIST

    bundles = sorted((FRONTEND_DIST / "assets").glob("*.js")) if FRONTEND_DIST.is_dir() else []
    if not bundles:
        pytest.skip("frontend not built")
    assert client.get("/").headers["cache-control"] == "no-cache"
    assert "cache-control" not in client.get(f"/assets/{bundles[0].name}").headers
