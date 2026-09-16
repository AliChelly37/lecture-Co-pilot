"""confirm -> write -> undo/retry through the API with fake targets."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lecture_copilot.targets import TargetError, TargetHealth, WriteResult


class FakeTarget:
    def __init__(self, name: str, fail_times: int = 0) -> None:
        self.name = name
        self.fail_times = fail_times
        self.written: dict[str, dict] = {}
        self.deleted: list[str] = []

    def healthcheck(self) -> TargetHealth:
        return TargetHealth(self.name, True, True, "fake")

    def write(self, payload: dict, idempotency_key: str) -> WriteResult:
        if self.fail_times > 0:
            self.fail_times -= 1
            raise TargetError("temporarily down", retryable=True)
        if idempotency_key in self.written:
            return WriteResult(external_id=f"{self.name}-{idempotency_key}", created=False)
        self.written[idempotency_key] = payload
        return WriteResult(external_id=f"{self.name}-{idempotency_key}", url=f"https://{self.name}/x")

    def update(self, external_id: str, payload: dict) -> WriteResult:
        return WriteResult(external_id=external_id, created=False)

    def delete(self, external_id: str) -> None:
        self.deleted.append(external_id)


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lecture_copilot import api
    from lecture_copilot.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    with TestClient(api.app) as c:
        yield c


def seed(client: TestClient) -> str:
    from datetime import date

    from lecture_copilot.api import state
    from lecture_copilot.dates import Resolution
    from lecture_copilot.extract import Candidate
    from lecture_copilot.suggest import SuggestionService

    course = client.post("/api/courses", json={"name": "Thermo", "timezone": "Europe/Paris"}).json()
    lec = state.store.start_lecture(course["id"], "g", "cuda", "ac", None)
    state.store.end_lecture(lec["id"], 0.1, None)
    c = Candidate(
        "assignment",
        "Problem set 4",
        "next Thursday",
        "5 pm",
        "commitment",
        "q",
        0.95,
        "",
        47.2,
        0.9,
        Resolution(date(2099, 9, 17), None, "note", 0.8),
    )
    SuggestionService(state.store).propose([c], lec, course)
    return client.get("/api/suggestions").json()[0]["id"]


def test_confirm_writes_to_all_targets_then_undo_deletes(client: TestClient) -> None:
    from lecture_copilot.api import state

    gcal, notion = FakeTarget("gcal"), FakeTarget("notion")
    state.targets = {"gcal": gcal, "notion": notion}
    sid = seed(client)

    s = client.post(f"/api/suggestions/{sid}/confirm").json()
    assert s["state"] == "written", s
    assert set(s["payload"]["external"]) == {"gcal", "notion"}
    assert s["payload"]["external"]["gcal"]["url"] == "https://gcal/x"
    assert len(gcal.written) == 1 and len(notion.written) == 1
    assert client.get("/api/suggestions?state_filter=done").json()[0]["id"] == sid

    # Dismissing a written item is refused; undo deletes both external records.
    assert client.post(f"/api/suggestions/{sid}/dismiss").status_code == 409
    s = client.post(f"/api/suggestions/{sid}/undo").json()
    assert s["state"] == "proposed" and s["payload"]["external"] == {}
    assert gcal.deleted == [f"gcal-{s['idempotency_key']}"] and notion.deleted == [f"notion-{s['idempotency_key']}"]


def test_partial_failure_is_retryable_and_idempotent(client: TestClient) -> None:
    from lecture_copilot.api import state

    gcal, notion = FakeTarget("gcal"), FakeTarget("notion", fail_times=1)
    state.targets = {"gcal": gcal, "notion": notion}
    sid = seed(client)

    s = client.post(f"/api/suggestions/{sid}/confirm").json()
    assert s["state"] == "failed_retryable" and "notion: temporarily down" in s["error"]
    assert set(s["payload"]["external"]) == {"gcal"}  # the successful write is kept

    s = client.post(f"/api/suggestions/{sid}/retry").json()
    assert s["state"] == "written" and set(s["payload"]["external"]) == {"gcal", "notion"}
    assert len(gcal.written) == 1  # not written twice


def test_no_targets_keeps_confirmed_for_ics(client: TestClient) -> None:
    from lecture_copilot.api import state

    state.targets = {}
    sid = seed(client)
    assert client.post(f"/api/suggestions/{sid}/confirm").json()["state"] == "confirmed"
    assert client.get("/api/targets").json() == []
    assert client.post("/api/targets/gcal/connect").status_code == 404
    assert "BEGIN:VEVENT" in client.get("/api/suggestions/export.ics").text
