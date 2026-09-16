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


def test_bad_deck_type(client: TestClient) -> None:
    r = client.post("/api/decks", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 400
