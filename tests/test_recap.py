"""M4 wiring with a fake model: map-reduce order, per-flag calls, source
pointers preserved, versioning, ratings and ask-the-lecture retrieval."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lecture_copilot.extract import Chunk
from lecture_copilot.recap import Answer, ChunkNotes, Concept, FlagExplanation, Recap, ReviewQuestion, boards_in, pick_excerpts, slides_in


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lecture_copilot import api
    from lecture_copilot.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    with TestClient(api.app) as c:
        yield c


def test_pick_excerpts_by_overlap() -> None:
    chunks = [
        Chunk(0, 0, 600, [{"t0": 0, "t1": 5, "text": "fugacity is an effective pressure"}]),
        Chunk(1, 600, 1200, [{"t0": 600, "t1": 605, "text": "the residual gibbs energy integral"}]),
        Chunk(2, 1200, 1800, [{"t0": 1200, "t1": 1205, "text": "nitrogen example at fifty bar"}]),
    ]
    assert [c.index for c in pick_excerpts(chunks, "what is the residual Gibbs energy?", k=2)] == [1]
    assert pick_excerpts(chunks, "???", k=2) == chunks[:2]  # no usable tokens: fall back to the start


def test_slides_and_boards_in_window() -> None:
    deck = {"slides_text": ["slide one text", "slide two text", "slide three text"]}
    alignment = [
        {"t0": 0, "t1": 30, "slide": 1, "score": 0.5},
        {"t0": 30, "t1": 60, "slide": None, "score": 0},
        {"t0": 60, "t1": 90, "slide": 3, "score": 0.4},
    ]
    assert slides_in(alignment, deck, 0, 45) == [(1, "slide one text")]
    assert slides_in(alignment, deck, 0, 90) == [(1, "slide one text"), (3, "slide three text")]
    caps = [{"status": "derived", "text": "board A", "t_shutter": 100.0}, {"status": "failed", "text": None, "t_shutter": 10.0}]
    assert boards_in(caps, 0, 30) == [(1, "board A")]  # within the 120 s slack
    assert boards_in(caps, 400, 500) == []


def test_recap_and_ask_with_fake_model(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from lecture_copilot.api import state

    course = client.post("/api/courses", json={"name": "Thermo", "timezone": "Europe/Paris"}).json()
    lec = state.store.start_lecture(course["id"], "gemma3:4b", "cuda", "ac", None)
    for i in range(70):  # 35 minutes -> 4 windows of 10 min
        state.store.add_segment(lec["id"], i * 30, i * 30 + 25, f"segment {i} about fugacity and gibbs energy", 0.9, "g", 0, [])
    state.store.add_flag(lec["id"], 700.0, 610.0, 715.0)
    state.store.add_gap(lec["id"], 1500.0, "mic_lost")
    state.store.end_lecture(lec["id"], 0.1, None)

    calls: list[str] = []

    def fake(**kw):
        calls.append(kw["stage"])
        schema = kw["schema"]
        if schema is ChunkNotes:
            return ChunkNotes(
                summary="covered fugacity", key_points=[f"fugacity defined ({kw['user'][18:29]})"], terms=["fugacity"], announcements=[]
            )
        if schema is FlagExplanation:
            assert "flagged confusion at 11:40" in kw["user"]
            return FlagExplanation(
                what_was_confusing="the integral", explanation="think of it as area", prerequisite="integration", sources=["t=11:30"]
            )
        if schema is Recap:
            assert "Recording gaps" in kw["user"] and "mic_lost" in kw["user"]
            return Recap(
                title="Fugacity",
                highlights=["fugacity is effective pressure (t=00:00)"],
                concepts=[Concept(name="Fugacity", importance="high", explanation="…", sources=["t=00:00", "slide 1"])],
                review_questions=[ReviewQuestion(question="Why does phi -> 1 at low P?", answer="Z -> 1", sources=["t=05:00"])],
                off_slide_notes=[],
                gaps_note="25:00 onwards mic lost",
            )
        if schema is Answer:
            assert "Excerpt" in kw["user"]
            return Answer(answer="It is the effective pressure.", sources=["t=00:00"], coverage="answered_from_lecture")
        raise AssertionError(schema)

    monkeypatch.setattr(state.llm, "structured", fake)

    r = client.post(f"/api/lectures/{lec['id']}/recap")
    assert r.status_code == 200, r.text
    recap = r.json()
    assert calls == ["recap_notes"] * 4 + ["recap_flag", "recap"]
    assert recap["version"] == 1 and recap["sections"]["title"] == "Fugacity"
    assert len(recap["sections"]["chunk_notes"]) == 4
    assert recap["sections"]["flag_explanations"][0]["flag_id"] == 1
    assert recap["sections"]["concepts"][0]["sources"] == ["t=00:00", "slide 1"]
    assert state.store.get_lecture(lec["id"])["status"] == "processed"

    # Regenerate -> version 2, never overwrites.
    assert client.post(f"/api/lectures/{lec['id']}/recap").json()["version"] == 2
    assert client.get(f"/api/lectures/{lec['id']}/recap").json()["version"] == 2

    rated = client.post(f"/api/recaps/{recap['id']}/rating", json={"rating": 4, "flag_helpful": {"1": True}}).json()
    assert rated["rating"] == 4 and rated["flag_helpful"] == {"1": True}

    a = client.post(f"/api/lectures/{lec['id']}/ask", json={"question": "what is fugacity?"}).json()
    assert a["answer"].startswith("It is") and a["excerpts"]
    assert client.post(f"/api/lectures/{lec['id']}/ask", json={"question": "  "}).status_code == 400
