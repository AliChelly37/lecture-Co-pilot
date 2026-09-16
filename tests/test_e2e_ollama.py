"""End-to-end through the real API with the local provider (Ollama).

Opt-in: runs only with LC_E2E=1 and a reachable Ollama, because it loads the
model and takes ~1 minute on the GPU:

    $env:LC_E2E='1'; uv run pytest -q tests/test_e2e_ollama.py
"""

import io
import os
from pathlib import Path

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

pytestmark = pytest.mark.skipif(os.environ.get("LC_E2E") != "1", reason="set LC_E2E=1 to run the Ollama end-to-end test")


def ollama_up(url: str) -> bool:
    try:
        return httpx.get(f"{url}/api/tags", timeout=3).status_code == 200
    except httpx.HTTPError:
        return False


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lecture_copilot import api
    from lecture_copilot.config import settings

    if not ollama_up(settings.ollama_url):
        pytest.skip("Ollama not reachable")
    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "llm_provider", "ollama")
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


def make_board_jpeg() -> bytes:
    from PIL import ImageFont

    img = Image.new("RGB", (1400, 900), (236, 238, 232))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)  # PIL's default bitmap font is too small to read after downscaling
    except OSError:
        font = ImageFont.load_default(size=48)
    for i, line in enumerate(["Residual Gibbs energy", "G_R / RT = integral (Z - 1)/P dP", "PS4 due Thursday 5pm"]):
        d.text((60, 80 + i * 140), line, fill=(25, 35, 90), font=font)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def test_deck_index_and_photo_reading_locally(client: TestClient) -> None:
    assert client.get("/api/status").json()["llm_available"] is True
    course = client.post("/api/courses", json={"name": "Thermo", "timezone": "Europe/Paris"}).json()

    pdf = make_pdf_bytes(
        [
            "Fugacity and the fugacity coefficient. Effective pressure of a real gas. Chemical potential.",
            "Peng-Robinson equation of state. Compressibility factor Z. Midterm: October 21.",
        ]
    )
    deck = client.post("/api/decks", files={"file": ("deck.pdf", pdf, "application/pdf")}, data={"course_id": course["id"]}).json()
    r = client.post(f"/api/decks/{deck['id']}/index")
    assert r.status_code == 200, r.text
    indexed = r.json()
    assert indexed["indexed"] is True

    from lecture_copilot.api import state

    full = state.store.get_deck(deck["id"])
    idx = full["slide_index"]
    assert len(idx["slides"]) == 2
    assert any("fugacity" in t.lower() for t in idx["course_terms"] + idx["slides"][0]["key_terms"])
    assert any("october" in d.lower() for d in idx["slides"][1]["stated_dates"])

    lec = state.store.start_lecture(course["id"], "small", "cpu", "ac", None)
    state.store.end_lecture(lec["id"], 0.1, None)
    r = client.post(f"/api/lectures/{lec['id']}/photos", files={"files": ("board.jpg", make_board_jpeg(), "image/jpeg")})
    assert r.status_code == 200, r.text
    captures = r.json()
    assert len(captures) == 1
    cap = captures[0]
    assert cap["status"] == "derived", cap
    assert cap["content_kind"] in {"board", "paper", "slide_projection"}
    text = cap["text"].lower()
    assert "ps4" in text and any(k in text for k in ("gibbs", "residual", "energy")), text

    usage = client.get("/api/usage").json()
    stages = {u["stage"] for u in usage}
    assert {"slide_index", "board_ocr"} <= stages
    assert all(u["cost_usd"] == 0 for u in usage)


def test_extraction_locally(client: TestClient) -> None:
    """The fixture transcript through the real local model: the two real
    deadlines must reach the inbox with plausible dates; the joke must not."""
    from lecture_copilot.api import state

    course = client.post(
        "/api/courses", json={"name": "Thermo", "timezone": "Europe/Paris", "term_calendar": {"week1_start": "2026-09-07"}}
    ).json()
    lec = state.store.start_lecture(course["id"], "gemma3:4b", "cuda", "ac", None)
    text = (Path(__file__).resolve().parents[1] / "eval/fixtures/tts_lecture.txt").read_text(encoding="utf-8")
    t = 0.0
    for sentence in [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]:
        dur = max(2.0, len(sentence.split()) / 2.5)
        state.store.add_segment(lec["id"], t, t + dur, sentence, 0.85, "gemma3:4b", 0, [])
        t += dur + 0.5
    state.store.end_lecture(lec["id"], 0.1, None)

    r = client.post(f"/api/lectures/{lec['id']}/extract")
    assert r.status_code == 200, r.text
    summary = r.json()
    assert summary["candidates"] >= 3, summary

    inbox = client.get("/api/suggestions").json()
    titles = " | ".join(s["payload"]["title"].lower() for s in inbox)
    assert "problem set" in titles or "ps" in titles, titles
    assert "midterm" in titles, titles
    midterm = next(s for s in inbox if "midterm" in s["payload"]["title"].lower())
    assert midterm["payload"]["date"] == "2026-10-21", midterm["payload"]  # the corrected date, not the 14th
    assert not any("navier" in s["payload"]["title"].lower() or "joke" in s["payload"]["title"].lower() for s in inbox)
    for s in inbox:
        assert s["payload"]["t0"] is not None, s["payload"]  # evidence located on the timeline
