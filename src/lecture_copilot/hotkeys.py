"""Global hotkeys via pynput (D14): the flag and pause keys work while the
student is typing in any other application.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)


class Hotkeys:
    def __init__(self, bindings: dict[str, Callable[[], None]]) -> None:
        self.bindings = bindings
        self._listener = None

    def start(self) -> None:
        try:
            from pynput import keyboard

            self._listener = keyboard.GlobalHotKeys(self.bindings)
            self._listener.daemon = True
            self._listener.start()
        except Exception:
            log.exception("global hotkeys unavailable; UI buttons still work")
            self._listener = None

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
            self._listener = None

    @property
    def active(self) -> bool:
        return self._listener is not None
