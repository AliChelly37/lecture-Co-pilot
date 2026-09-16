"""Probe the board-reading path through the real LlmGateway with a synthetic
whiteboard image (no real classroom photo needed).

  uv run python scripts/probe_vision.py            # provider from .env / defaults (ollama)
  uv run python scripts/probe_vision.py --provider cloudflare
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
import time
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from lecture_copilot.config import Settings  # noqa: E402
from lecture_copilot.llm import LlmGateway  # noqa: E402
from lecture_copilot.photos import BOARD_SYSTEM, BoardReading, prepare_photo  # noqa: E402
from lecture_copilot.store import Store  # noqa: E402

BOARD_LINES = [
    "Residual Gibbs energy",
    "G_R / RT = integral_0^P (Z - 1) / P dP   (const T)",
    "Ideal gas: Z = 1  =>  G_R = 0",
    "Peng-Robinson: Z^3 - (1-B) Z^2 + (A - 3B^2 - 2B) Z - (AB - B^2 - B^3) = 0",
    "PS4 due Thu 5pm",
]


def synthetic_board(width: int = 1600, height: int = 1000) -> bytes:
    img = Image.new("RGB", (width, height), (235, 238, 232))  # whiteboard-ish
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 44)
    except OSError:
        font = ImageFont.load_default()
    y = 80
    for line in BOARD_LINES:
        draw.text((80, y), line, fill=(30, 40, 90), font=font)
        y += 110
    draw.line((80, 620, 900, 620), fill=(160, 40, 40), width=4)  # an underline "stroke"
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    s = Settings()
    if args.provider:
        s.llm_provider = args.provider
    with tempfile.TemporaryDirectory() as tmp:
        store = Store(Path(tmp) / "probe.sqlite3")
        gw = LlmGateway(s, store)
        print(json.dumps(gw.describe()))
        photo = prepare_photo(synthetic_board(), "board.jpg", "UTC")
        t0 = time.perf_counter()
        reading = gw.structured(
            stage="board_ocr",
            schema=BoardReading,
            system=[BOARD_SYSTEM],
            user="Read this photo.",
            images=[photo.jpeg],
            model=args.model,
            cache=False,
            max_tokens=1500,
        )
        wall = time.perf_counter() - t0
        print(json.dumps({"wall_s": round(wall, 1), "usage": store.usage_by_stage()}, indent=1))
        print(json.dumps(reading.model_dump(), indent=1, ensure_ascii=False))
        expected = ["Gibbs", "Z", "Peng", "PS4"]
        found = [w for w in expected if w.lower() in reading.text.lower()]
        print(f"expected terms found: {found} / {expected}")
        store.close()  # Windows: the temp dir can't be removed while SQLite holds the file


if __name__ == "__main__":
    main()
