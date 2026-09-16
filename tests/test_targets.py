"""Target mappings and the confirm -> write -> undo path with fake services.
Real Google/Notion calls need credentials and are exercised manually."""

from pathlib import Path

import httpx

from lecture_copilot.targets import TargetError, event_datetimes
from lecture_copilot.targets.gcal import GoogleCalendarTarget
from lecture_copilot.targets.notion import NotionTarget

PAYLOAD = {
    "title": "Problem set 4",
    "type": "assignment",
    "date": "2026-09-17",
    "time": "17:00",
    "timezone": "Europe/Paris",
    "course_name": "Thermo",
    "evidence_quote": "Problem Set 4 is due next Thursday at 5 p.m.",
    "resolution_note": "'next Thursday' from lecture date Wed 16 Sep 2026 -> Thu 17 Sep 2026",
    "t0": 47.2,
}


def test_event_datetimes_timed_and_all_day() -> None:
    start, end, all_day = event_datetimes(PAYLOAD)
    assert not all_day and start == "2026-09-17T16:00:00+02:00" and end == "2026-09-17T17:00:00+02:00"
    start, end, all_day = event_datetimes(PAYLOAD | {"time": None})
    assert all_day and start == "2026-09-17" and end == "2026-09-18"


def test_gcal_event_body_and_idempotent_409(tmp_path: Path, monkeypatch) -> None:
    body = GoogleCalendarTarget.event_body(PAYLOAD)
    assert body["summary"] == "Thermo: Problem set 4"
    assert body["start"] == {"dateTime": "2026-09-17T16:00:00+02:00", "timeZone": "Europe/Paris"}
    assert "Quote:" in body["description"] and "00:47" in body["description"]

    target = GoogleCalendarTarget(tmp_path / "secret.json", tmp_path / "token.json", tmp_path / "state.json")
    assert target.healthcheck().configured is False
    monkeypatch.setattr(target, "_headers", lambda: {"Authorization": "Bearer x"})
    (tmp_path / "state.json").write_text('{"calendar_id": "cal1"}')

    def fake_post(url, headers=None, json=None):
        assert url.endswith("/calendars/cal1/events") and json["id"] == "abc123"
        return httpx.Response(409, request=httpx.Request("POST", url))

    monkeypatch.setattr(target._http, "post", fake_post)
    res = target.write(PAYLOAD, "abc123")
    assert res.created is False and res.external_id == "abc123"


def test_notion_properties_and_find_before_create(monkeypatch) -> None:
    t = NotionTarget("tok", "db", prop_title="Name", prop_date="Due", prop_status="Status", status_value="To do", prop_course="Course")
    schema = {
        "Name": {"type": "title"},
        "Due": {"type": "date"},
        "copilot_id": {"type": "rich_text"},
        "Status": {"type": "status"},
        "Course": {"type": "select"},
    }
    props = t.page_properties(PAYLOAD, "key1", schema)
    assert props["Name"]["title"][0]["text"]["content"] == "Problem set 4 (Thermo)"
    assert props["Due"]["date"]["start"] == "2026-09-17T17:00:00+02:00"  # explicit offset, no time_zone field
    assert props["Status"] == {"status": {"name": "To do"}} and props["Course"] == {"select": {"name": "Thermo"}}
    assert props["copilot_id"]["rich_text"][0]["text"]["content"] == "key1"

    monkeypatch.setattr(t, "_data_source", lambda: ("ds1", schema))
    calls: list[str] = []

    def fake_post(url, headers=None, json=None):
        calls.append(url)
        if url.endswith("/query"):
            return httpx.Response(
                200, json={"results": [{"id": "page-existing", "url": "https://n/x"}]}, request=httpx.Request("POST", url)
            )
        raise AssertionError("must not create when the key already exists")

    monkeypatch.setattr(t._http, "post", fake_post)
    res = t.write(PAYLOAD, "key1")
    assert res.created is False and res.external_id == "page-existing" and calls == ["https://api.notion.com/v1/data_sources/ds1/query"]


def test_notion_healthcheck_reports_missing_mapping(monkeypatch) -> None:
    t = NotionTarget("tok", "db", prop_title="Name", prop_date="Deadline")
    monkeypatch.setattr(t, "_data_source", lambda: ("ds1", {"Name": {"type": "title"}, "copilot_id": {"type": "rich_text"}}))
    h = t.healthcheck()
    assert h.configured and not h.connected and "Deadline" in h.detail
    assert NotionTarget(None, None).healthcheck().configured is False


def test_target_error_retryable_flag() -> None:
    assert TargetError("x").retryable is True and TargetError("y", retryable=False).retryable is False
