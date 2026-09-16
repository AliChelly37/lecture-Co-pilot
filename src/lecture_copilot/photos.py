"""Whiteboard photo import (D14, D19): photos taken with the phone's normal
camera are imported after class. EXIF `DateTimeOriginal` places each photo on
the lecture timeline; the image is downscaled in memory for the vision call and
never written to disk by the app.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal
from zoneinfo import ZoneInfo

from PIL import Image, ImageOps
from pydantic import BaseModel, Field

EXIF_DATETIME_ORIGINAL = 36867
EXIF_OFFSET_TIME_ORIGINAL = 36881


@dataclass
class PreparedPhoto:
    filename: str
    shot_at: datetime | None  # timezone-aware if EXIF had an offset or a course tz was supplied
    jpeg: bytes  # downscaled, for the vision call only
    width: int
    height: int


def prepare_photo(data: bytes, filename: str, course_tz: str, max_edge: int = 1280) -> PreparedPhoto:
    with Image.open(io.BytesIO(data)) as img:
        shot_at = _exif_datetime(img, course_tz)
        img = ImageOps.exif_transpose(img)  # honour the phone's rotation flag
        img = img.convert("RGB")
        img.thumbnail((max_edge, max_edge))
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=True)
        return PreparedPhoto(filename=filename, shot_at=shot_at, jpeg=buf.getvalue(), width=img.width, height=img.height)


def _exif_datetime(img: Image.Image, course_tz: str) -> datetime | None:
    try:
        exif = img.getexif()
    except Exception:
        return None
    raw = exif.get(EXIF_DATETIME_ORIGINAL)
    if not raw:
        # DateTimeOriginal lives in the Exif IFD on most phones.
        try:
            raw = exif.get_ifd(0x8769).get(EXIF_DATETIME_ORIGINAL)
            offset = exif.get_ifd(0x8769).get(EXIF_OFFSET_TIME_ORIGINAL)
        except Exception:
            raw, offset = None, None
    else:
        offset = exif.get(EXIF_OFFSET_TIME_ORIGINAL)
    if not raw:
        return None
    try:
        naive = datetime.strptime(str(raw).strip(), "%Y:%m:%d %H:%M:%S")
    except ValueError:
        return None
    if offset:
        try:
            sign = 1 if str(offset)[0] == "+" else -1
            hh, mm = str(offset)[1:].split(":")
            from datetime import timezone

            return naive.replace(tzinfo=timezone(sign * timedelta(hours=int(hh), minutes=int(mm))))
        except Exception:
            pass
    return naive.replace(tzinfo=ZoneInfo(course_tz))


def seconds_into_lecture(shot_at: datetime | None, started_at_iso: str, ended_at_iso: str | None, slack_s: float = 300) -> float | None:
    """Map a wall-clock shot time to lecture seconds, or None if it falls
    outside the lecture window (plus slack for a photo taken as people leave)."""
    if shot_at is None:
        return None
    start = datetime.fromisoformat(started_at_iso)
    t = (shot_at - start).total_seconds()
    end = (datetime.fromisoformat(ended_at_iso) - start).total_seconds() if ended_at_iso else None
    if t < -slack_s or (end is not None and t > end + slack_s):
        return None
    return round(max(t, 0.0), 1)


# -- BoardReader schema (structured output of the Sonnet vision call) ------
class BoardReading(BaseModel):
    content_kind: Literal["board", "slide_projection", "paper", "other"] = Field(description="What the photo shows")
    text: str = Field(description="All legible text, in reading order; equations in plain words if not LaTeX")
    latex: list[str] = Field(description="Each equation as LaTeX, in reading order", max_length=30)
    diagram_notes: list[str] = Field(description="Short descriptions of diagrams, graphs or arrows", max_length=10)
    legibility: float = Field(description="0-1: how much of the writing could be read confidently")


BOARD_SYSTEM = (
    "You read photos of lecture whiteboards or blackboards for a study tool. Transcribe faithfully; "
    "never invent content that is not visible. If the photo is not a board (a person, a room, a paper), "
    "set content_kind accordingly and leave text empty."
)
