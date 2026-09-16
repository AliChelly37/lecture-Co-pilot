from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lecture_copilot.export import lecture_markdown
from lecture_copilot.store import Store


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lecture_copilot import api
    from lecture_copilot.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    with TestClient(api.app) as c:
        yield c


def test_export_delete_and_dashboard(client: TestClient) -> None:
    from lecture_copilot.api import state

    course = client.post("/api/courses", json={"name": "Thermo", "timezone": "Europe/Paris"}).json()
    lec = state.store.start_lecture(course["id"], "gemma3:4b", "cuda", "battery", 60)
    state.store.add_segment(lec["id"], 0, 5, "fugacity is an effective pressure", 0.9, "g", 0, [])
    state.store.add_flag(lec["id"], 4.0, 0.0, 19.0)
    state.store.add_recap(
        lec["id"],
        1,
        "ollama:gemma3:4b",
        "high",
        {
            "title": "Fugacity",
            "highlights": ["h1 (t=00:00)"],
            "concepts": [],
            "review_questions": [],
            "off_slide_notes": [],
            "gaps_note": "",
            "flag_explanations": [],
            "detected_events": [],
        },
    )
    state.store.add_usage(
        call_id="c1",
        lecture_id=lec["id"],
        stage="recap",
        model="ollama:gemma3:4b",
        effort="high",
        input_tokens=100,
        cache_read=0,
        cache_write=0,
        output_tokens=50,
        cost_usd=0.0,
        latency_ms=10,
        stop_reason="stop",
    )
    state.store.end_lecture(lec["id"], 0.2, 50)

    bundle = client.get(f"/api/lectures/{lec['id']}/export.json").json()
    assert bundle["course"]["name"] == "Thermo" and len(bundle["segments"]) == 1 and bundle["recaps"][0]["sections"]["title"] == "Fugacity"
    md = client.get(f"/api/lectures/{lec['id']}/export.md")
    assert md.status_code == 200 and "## Recap: Fugacity" in md.text and "🚩" in md.text and "`00:00` fugacity" in md.text

    d = client.get("/api/dashboard").json()
    assert d["courses"][0]["lectures"] == 1 and d["courses"][0]["open_flags"] == 1 and d["courses"][0]["recaps"] == 1
    assert d["usage"]["calls"] == 1 and d["usage"]["total_cost_usd"] == 0
    assert d["asr"]["rtf_p95_avg"] == 0.2 and d["storage"]["db_bytes"] > 0

    assert client.delete(f"/api/lectures/{lec['id']}").json() == {"deleted": lec["id"]}
    assert client.get(f"/api/lectures/{lec['id']}").status_code == 404
    assert state.store.query("SELECT COUNT(*) AS n FROM segments")[0]["n"] == 0
    assert state.store.query("SELECT COUNT(*) AS n FROM usage_ledger")[0]["n"] == 0
    assert client.delete(f"/api/courses/{course['id']}").json()["deleted"] == course["id"]
    assert client.get("/api/courses").json() == []


def test_retention_purges_only_old_transcripts(tmp_path: Path) -> None:
    store = Store(tmp_path / "t.sqlite3")
    course = store.create_course("Thermo", "UTC")
    old = store.start_lecture(course["id"], "g", "cpu", "ac", None)
    store.add_segment(old["id"], 0, 5, "old words", 0.9, "g", 0, [])
    store.add_flag(old["id"], 1.0, 0.0, 10.0)
    store.end_lecture(old["id"], 0.1, None)
    store.execute(
        "UPDATE lectures SET ended_at=? WHERE id=?", ((datetime.now(UTC) - timedelta(days=40)).isoformat(timespec="seconds"), old["id"])
    )
    new = store.start_lecture(course["id"], "g", "cpu", "ac", None)
    store.add_segment(new["id"], 0, 5, "new words", 0.9, "g", 0, [])
    store.end_lecture(new["id"], 0.1, None)

    assert store.purge_transcripts(30) == 1
    assert store.segments(old["id"]) == [] and store.flags(old["id"])  # flags survive
    assert store.get_lecture(old["id"])["notes"] == "transcript purged"
    assert len(store.segments(new["id"])) == 1
    assert store.purge_transcripts(30) == 0  # idempotent
    bundle = store.export_lecture(old["id"])
    assert "Purged by the retention policy" in lecture_markdown(bundle)
    store.close()


def test_mcp_tools_read_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import json

    from lecture_copilot import mcp_server
    from lecture_copilot.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(mcp_server, "_store", None)
    store = mcp_server.store()
    course = store.create_course("Thermo", "UTC")
    lec = store.start_lecture(course["id"], "g", "cpu", "ac", None)
    store.add_segment(lec["id"], 30, 35, "the residual Gibbs energy integral", 0.9, "g", 0, [])
    store.add_flag(lec["id"], 34.0, 0.0, 49.0)
    store.end_lecture(lec["id"], 0.1, None)

    assert json.loads(mcp_server.list_lectures())[0]["flags"] == 1
    hits = json.loads(mcp_server.search_transcripts("gibbs energy"))
    assert hits and hits[0]["t0"] == 30 and hits[0]["course"] == "Thermo"
    assert json.loads(mcp_server.search_transcripts("x")) == []
    flags = json.loads(mcp_server.list_open_flags("thermo"))
    assert flags[0]["context"].startswith("the residual") and flags[0]["explanation"] is None
    assert "error" in json.loads(mcp_server.get_recap(lec["id"]))
    assert json.loads(mcp_server.upcoming_deadlines()) == []
