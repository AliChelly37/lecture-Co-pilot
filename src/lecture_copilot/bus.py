"""In-process event bus: worker threads publish, WebSocket handlers subscribe.

Publishing is thread-safe; delivery happens on the asyncio loop the bus was
bound to. Events are plain dicts with a "type" key.
"""

from __future__ import annotations

import asyncio
import contextlib
import threading
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: set[asyncio.Queue] = set()
        self._lock = threading.Lock()

    def bind(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, maxsize: int = 1000) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=maxsize)
        with self._lock:
            self._subs.add(q)
        return q

    def unsubscribe(self, q: asyncio.Queue) -> None:
        with self._lock:
            self._subs.discard(q)

    def publish(self, event: dict[str, Any]) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        with self._lock:
            subs = list(self._subs)
        for q in subs:
            loop.call_soon_threadsafe(_put, q, event)


def _put(q: asyncio.Queue, event: dict[str, Any]) -> None:
    # A slow client loses events rather than stalling the recorder.
    with contextlib.suppress(asyncio.QueueFull):
        q.put_nowait(event)
