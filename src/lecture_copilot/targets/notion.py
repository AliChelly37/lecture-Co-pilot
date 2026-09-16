"""Notion target (D1, D19): writes a page into the user's existing task
database via an internal integration shared with that one database.

- API version 2025-09-03: pages attach to a *data source*, not a database;
  the data source id is discovered once from the database id.
- No native idempotency: a hidden `copilot_id` rich-text property is queried
  before every create.
- Property mapping comes from config (title / date / optional status and
  course properties); `healthcheck()` verifies the mapped properties exist.
- Dates are sent with an explicit offset and no `time_zone` field.
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from zoneinfo import ZoneInfo

import httpx

from lecture_copilot.targets import ActionTarget, TargetError, TargetHealth, WriteResult, describe

log = logging.getLogger(__name__)

API = "https://api.notion.com/v1"
VERSION = "2025-09-03"


class NotionTarget(ActionTarget):
    name = "notion"

    def __init__(
        self,
        token: str | None,
        database_id: str | None,
        prop_title: str = "Name",
        prop_date: str = "Due",
        prop_status: str | None = None,
        status_value: str | None = None,
        prop_course: str | None = None,
        prop_key: str = "copilot_id",
    ) -> None:
        self.token = token
        self.database_id = database_id
        self.prop_title, self.prop_date, self.prop_status, self.status_value = prop_title, prop_date, prop_status, status_value
        self.prop_course, self.prop_key = prop_course, prop_key
        self._http = httpx.Client(timeout=30)
        self._data_source_id: str | None = None
        self._schema: dict | None = None

    @property
    def configured(self) -> bool:
        return bool(self.token and self.database_id)

    def _headers(self) -> dict:
        if not self.configured:
            raise TargetError("Notion is not configured (NOTION_TOKEN and NOTION_DATABASE_ID)", retryable=False)
        return {"Authorization": f"Bearer {self.token}", "Notion-Version": VERSION, "Content-Type": "application/json"}

    # -- discovery -------------------------------------------------------------
    def _data_source(self) -> tuple[str, dict]:
        if self._data_source_id and self._schema is not None:
            return self._data_source_id, self._schema
        r = self._http.get(f"{API}/databases/{self.database_id}", headers=self._headers())
        self._check(r)
        sources = r.json().get("data_sources") or []
        if not sources:
            raise TargetError("Notion database has no data source", retryable=False)
        ds_id = sources[0]["id"]
        r = self._http.get(f"{API}/data_sources/{ds_id}", headers=self._headers())
        self._check(r)
        self._data_source_id, self._schema = ds_id, r.json().get("properties", {})
        return ds_id, self._schema

    def healthcheck(self) -> TargetHealth:
        if not self.configured:
            return TargetHealth(self.name, False, False, "NOTION_TOKEN / NOTION_DATABASE_ID not set")
        try:
            _, schema = self._data_source()
        except TargetError as exc:
            return TargetHealth(self.name, True, False, str(exc))
        missing = [p for p in (self.prop_title, self.prop_date, self.prop_key) if p not in schema]
        if self.prop_status and self.prop_status not in schema:
            missing.append(self.prop_status)
        if self.prop_course and self.prop_course not in schema:
            missing.append(self.prop_course)
        if missing:
            return TargetHealth(
                self.name, True, False, f"database is missing properties: {', '.join(missing)} (fix the mapping or add them)"
            )
        if schema[self.prop_title].get("type") != "title":
            return TargetHealth(self.name, True, False, f"'{self.prop_title}' is not the title property")
        if schema[self.prop_date].get("type") != "date":
            return TargetHealth(self.name, True, False, f"'{self.prop_date}' is not a date property")
        return TargetHealth(self.name, True, True, "mapping verified")

    # -- mapping ---------------------------------------------------------------
    @staticmethod
    def date_value(payload: dict) -> dict:
        d = date.fromisoformat(payload["date"])
        if payload.get("time"):
            tz = ZoneInfo(payload.get("timezone") or "UTC")
            h, m = (int(x) for x in payload["time"].split(":")[:2])
            return {"start": datetime(d.year, d.month, d.day, h, m, tzinfo=tz).isoformat(), "end": None}
        return {"start": d.isoformat(), "end": None}

    def page_properties(self, payload: dict, idempotency_key: str, schema: dict | None = None) -> dict:
        props: dict = {
            self.prop_title: {"title": [{"text": {"content": f"{payload['title']} ({payload.get('course_name', '')})".strip()}}]},
            self.prop_date: {"date": self.date_value(payload)},
            self.prop_key: {"rich_text": [{"text": {"content": idempotency_key}}]},
        }
        if self.prop_status and self.status_value:
            kind = (schema or {}).get(self.prop_status, {}).get("type", "status")
            props[self.prop_status] = {kind: {"name": self.status_value}}
        if self.prop_course and payload.get("course_name"):
            kind = (schema or {}).get(self.prop_course, {}).get("type", "select")
            if kind == "rich_text":
                props[self.prop_course] = {"rich_text": [{"text": {"content": payload["course_name"]}}]}
            else:
                props[self.prop_course] = {kind: {"name": payload["course_name"]}}
        return props

    # -- writes ----------------------------------------------------------------
    def _find(self, ds_id: str, idempotency_key: str) -> dict | None:
        body = {"filter": {"property": self.prop_key, "rich_text": {"equals": idempotency_key}}, "page_size": 1}
        r = self._http.post(f"{API}/data_sources/{ds_id}/query", headers=self._headers(), json=body)
        self._check(r)
        results = r.json().get("results") or []
        return results[0] if results else None

    def write(self, payload: dict, idempotency_key: str) -> WriteResult:
        ds_id, schema = self._data_source()
        existing = self._find(ds_id, idempotency_key)
        if existing:
            return WriteResult(external_id=existing["id"], url=existing.get("url"), created=False)
        body = {
            "parent": {"type": "data_source_id", "data_source_id": ds_id},
            "properties": self.page_properties(payload, idempotency_key, schema),
            "children": [
                {"object": "block", "type": "paragraph", "paragraph": {"rich_text": [{"text": {"content": describe(payload)[:1900]}}]}}
            ],
        }
        r = self._http.post(f"{API}/pages", headers=self._headers(), json=body)
        self._check(r)
        data = r.json()
        return WriteResult(external_id=data["id"], url=data.get("url"))

    def update(self, external_id: str, payload: dict) -> WriteResult:
        _, schema = self._data_source()
        props = self.page_properties(payload, "", schema)
        props.pop(self.prop_key, None)
        r = self._http.patch(f"{API}/pages/{external_id}", headers=self._headers(), json={"properties": props})
        self._check(r)
        return WriteResult(external_id=external_id, url=r.json().get("url"), created=False)

    def delete(self, external_id: str) -> None:
        r = self._http.patch(f"{API}/pages/{external_id}", headers=self._headers(), json={"in_trash": True})
        if r.status_code == 404:
            return
        self._check(r)

    @staticmethod
    def _check(r: httpx.Response) -> None:
        if r.status_code == 429:
            raise TargetError(f"Notion rate limited; retry after {r.headers.get('Retry-After', '?')}s", retryable=True)
        if r.status_code in (401, 403):
            raise TargetError("Notion rejected the token, or the database is not shared with the integration", retryable=False)
        if r.status_code >= 500:
            raise TargetError(f"Notion temporary error {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise TargetError(f"Notion error {r.status_code}: {r.text[:200]}", retryable=False)
