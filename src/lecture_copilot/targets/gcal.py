"""Google Calendar target (D19).

- Scope `calendar.app.created`: the app can only touch calendars it created,
  so it makes one "Lecture Co-Pilot" calendar and writes there. Your other
  calendars are unreachable by construction.
- Event ids are client-generated from the idempotency key (24 hex chars are
  valid base32hex), so a repeated confirm can never create a duplicate: Google
  answers 409 for an existing id and we treat that as success.
- OAuth is the installed-app flow with the user's own client secret; the
  refresh token lives in the data dir. With the consent screen "In production"
  (unverified, personal use) tokens don't expire weekly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import httpx

from lecture_copilot.targets import ActionTarget, TargetError, TargetHealth, WriteResult, describe, event_datetimes

log = logging.getLogger(__name__)

SCOPES = ["https://www.googleapis.com/auth/calendar.app.created"]
API = "https://www.googleapis.com/calendar/v3"
CALENDAR_SUMMARY = "Lecture Co-Pilot"


class GoogleCalendarTarget(ActionTarget):
    name = "gcal"

    def __init__(self, client_secret_path: Path, token_path: Path, state_path: Path) -> None:
        self.client_secret_path = client_secret_path
        self.token_path = token_path
        self.state_path = state_path  # stores the calendar id we created
        self._http = httpx.Client(timeout=30)

    # -- auth ------------------------------------------------------------------
    @property
    def configured(self) -> bool:
        return self.client_secret_path.is_file()

    def connect(self) -> TargetHealth:
        """Run the OAuth flow in the browser (blocking). Called from a user action."""
        if not self.configured:
            raise TargetError(f"Google client secret not found at {self.client_secret_path}", retryable=False)
        from google_auth_oauthlib.flow import InstalledAppFlow

        flow = InstalledAppFlow.from_client_secrets_file(str(self.client_secret_path), SCOPES)
        creds = flow.run_local_server(port=0, open_browser=True)
        self.token_path.parent.mkdir(parents=True, exist_ok=True)
        self.token_path.write_text(creds.to_json(), encoding="utf-8")
        return self.healthcheck()

    def _credentials(self):
        from google.auth.transport.requests import Request
        from google.oauth2.credentials import Credentials

        if not self.token_path.is_file():
            return None
        creds = Credentials.from_authorized_user_file(str(self.token_path), SCOPES)
        if creds.expired and creds.refresh_token:
            try:
                creds.refresh(Request())
                self.token_path.write_text(creds.to_json(), encoding="utf-8")
            except Exception as exc:  # revoked, or 7-day expiry on a Testing consent screen
                log.warning("google token refresh failed: %s", exc)
                return None
        return creds if creds.valid else None

    def _headers(self) -> dict:
        creds = self._credentials()
        if creds is None:
            raise TargetError("Google Calendar is not connected (connect it in Settings)", retryable=True)
        return {"Authorization": f"Bearer {creds.token}"}

    # -- calendar --------------------------------------------------------------
    def _calendar_id(self) -> str:
        if self.state_path.is_file():
            cid = json.loads(self.state_path.read_text(encoding="utf-8")).get("calendar_id")
            if cid:
                return cid
        r = self._http.post(f"{API}/calendars", headers=self._headers(), json={"summary": CALENDAR_SUMMARY})
        self._check(r)
        cid = r.json()["id"]
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({"calendar_id": cid}), encoding="utf-8")
        return cid

    def healthcheck(self) -> TargetHealth:
        if not self.configured:
            return TargetHealth(self.name, False, False, f"no client secret at {self.client_secret_path}")
        creds = self._credentials()
        if creds is None:
            return TargetHealth(self.name, True, False, "not connected: run the Google sign-in")
        try:
            cid = self._calendar_id()
            r = self._http.get(f"{API}/calendars/{cid}", headers=self._headers())
            self._check(r)
            return TargetHealth(self.name, True, True, f"writes to calendar '{r.json().get('summary', CALENDAR_SUMMARY)}'")
        except TargetError as exc:
            return TargetHealth(self.name, True, False, str(exc))

    # -- writes ----------------------------------------------------------------
    @staticmethod
    def event_body(payload: dict) -> dict:
        start, end, all_day = event_datetimes(payload)
        when = {"date": start} if all_day else {"dateTime": start, "timeZone": payload.get("timezone", "UTC")}
        when_end = {"date": end} if all_day else {"dateTime": end, "timeZone": payload.get("timezone", "UTC")}
        return {
            "summary": f"{payload.get('course_name', '')}: {payload['title']}".strip(": "),
            "description": describe(payload),
            "start": when,
            "end": when_end,
            "reminders": {"useDefault": False, "overrides": [{"method": "popup", "minutes": 24 * 60}]},
        }

    def write(self, payload: dict, idempotency_key: str) -> WriteResult:
        cid = self._calendar_id()
        body = self.event_body(payload) | {"id": idempotency_key}
        r = self._http.post(f"{API}/calendars/{cid}/events", headers=self._headers(), json=body)
        if r.status_code == 409:  # our id already exists: idempotent replay
            return WriteResult(external_id=idempotency_key, url=None, created=False)
        self._check(r)
        data = r.json()
        return WriteResult(external_id=data["id"], url=data.get("htmlLink"))

    def update(self, external_id: str, payload: dict) -> WriteResult:
        cid = self._calendar_id()
        r = self._http.patch(f"{API}/calendars/{cid}/events/{external_id}", headers=self._headers(), json=self.event_body(payload))
        self._check(r)
        return WriteResult(external_id=external_id, url=r.json().get("htmlLink"), created=False)

    def delete(self, external_id: str) -> None:
        cid = self._calendar_id()
        r = self._http.delete(f"{API}/calendars/{cid}/events/{external_id}", headers=self._headers())
        if r.status_code in (404, 410):
            return
        self._check(r)

    @staticmethod
    def _check(r: httpx.Response) -> None:
        if r.status_code in (401, 403):
            raise TargetError(f"Google Calendar rejected the request ({r.status_code}): reconnect the account", retryable=True)
        if r.status_code == 429 or r.status_code >= 500:
            raise TargetError(f"Google Calendar temporary error {r.status_code}", retryable=True)
        if r.status_code >= 400:
            raise TargetError(f"Google Calendar error {r.status_code}: {r.text[:200]}", retryable=False)
