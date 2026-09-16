"""Microphone capture owned by the backend process (D14).

Frames are handed to a callback with their absolute sample index; nothing is
written to disk. Device loss is reported, never silently recovered with a
different device.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable

import numpy as np

log = logging.getLogger(__name__)


class MicCapture:
    def __init__(
        self,
        sample_rate: int,
        on_frames: Callable[[np.ndarray, int], None],
        on_error: Callable[[str], None],
        device: int | str | None = None,
        block_ms: int = 100,
    ) -> None:
        self.sample_rate = sample_rate
        self.on_frames = on_frames
        self.on_error = on_error
        self.device = device
        self.blocksize = sample_rate * block_ms // 1000
        self._stream = None
        self._samples = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        import sounddevice as sd

        def callback(indata, frames, time_info, status) -> None:
            if status and status.input_overflow:
                log.warning("input overflow")
            mono = np.ascontiguousarray(indata[:, 0], dtype=np.float32)
            with self._lock:
                index = self._samples
                self._samples += len(mono)
            self.on_frames(mono, index)

        def finished() -> None:
            # Called when the stream stops on its own (device unplugged).
            if self._stream is not None:
                self.on_error("mic_lost")

        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=1,
            dtype="float32",
            blocksize=self.blocksize,
            device=self.device,
            callback=callback,
            finished_callback=finished,
        )
        self._stream.start()

    def stop(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            stream.stop()
            stream.close()

    @property
    def seconds(self) -> float:
        with self._lock:
            return self._samples / self.sample_rate

    @staticmethod
    def list_devices() -> list[dict]:
        import sounddevice as sd

        out = []
        for i, d in enumerate(sd.query_devices()):
            if d.get("max_input_channels", 0) > 0:
                out.append({"index": i, "name": d["name"], "default": i == sd.default.device[0]})
        return out
