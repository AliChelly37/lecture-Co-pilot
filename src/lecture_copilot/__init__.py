"""Lecture Co-Pilot: local-first lecture capture, recap and deadline assistant."""

from __future__ import annotations

import logging
import threading
import webbrowser


def main() -> None:
    import uvicorn

    from lecture_copilot.config import settings

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    url = f"http://{settings.host}:{settings.port}"
    if settings.open_browser:
        threading.Timer(1.5, lambda: webbrowser.open(url)).start()
    uvicorn.run("lecture_copilot.api:app", host=settings.host, port=settings.port, log_level="info")
