from __future__ import annotations

import logging

from lecture_copilot.config import Settings
from lecture_copilot.targets import ActionTarget
from lecture_copilot.targets.gcal import GoogleCalendarTarget
from lecture_copilot.targets.notion import NotionTarget

log = logging.getLogger(__name__)


def build_targets(s: Settings) -> dict[str, ActionTarget]:
    out: dict[str, ActionTarget] = {}
    for name in [n.strip() for n in s.targets.split(",") if n.strip()]:
        if name == "gcal":
            out[name] = GoogleCalendarTarget(s.google_client_secret, s.data_dir / "google_token.json", s.data_dir / "google_calendar.json")
        elif name == "notion":
            out[name] = NotionTarget(
                s.notion_token,
                s.notion_database_id,
                s.notion_prop_title,
                s.notion_prop_date,
                s.notion_prop_status or None,
                s.notion_status_value or None,
                s.notion_prop_course or None,
            )
        else:
            log.warning("unknown target '%s' in LC_TARGETS (use gcal, notion)", name)
    return out
