"""Replay mode (D26): catching up on a recording of a lecture you missed.

The browser plays the file; the backend transcribes the same audio ahead of
playback and keeps one clock, the playback position the browser reports, so
flags and pauses land on the recording's own timeline. The audio arrives as
the raw request body, is decoded in memory and fed to the Whisper worker
exactly as the microphone would feed it; nothing is written to disk (D8).
"""

from __future__ import annotations

import io
import threading
import time
from typing import Protocol

import numpy as np


class Clock(Protocol):
    @property
    def seconds(self) -> float: ...


class PlaybackClock:
    """The lecture clock during replay. The browser reports (playing, t) on
    play, pause, seek and every couple of seconds; between reports the clock
    runs at real time while playing and stands still while paused."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._t = 0.0
        self._wall: float | None = None  # perf_counter at the last report; None while paused
        self._reach = 0.0  # furthest point played, kept across seeks back and page reloads

    def report(self, playing: bool, t: float) -> None:
        with self._lock:
            self._reach = max(self._reach, self._seconds_locked())  # where playback was before a jump
            self._t = max(0.0, float(t))
            self._wall = time.perf_counter() if playing else None
            self._reach = max(self._reach, self._t)

    def pause(self) -> None:
        with self._lock:
            self._t = self._seconds_locked()
            self._wall = None

    def play(self) -> None:
        with self._lock:
            if self._wall is None:
                self._wall = time.perf_counter()

    @property
    def playing(self) -> bool:
        return self._wall is not None

    @property
    def seconds(self) -> float:
        with self._lock:
            return self._seconds_locked()

    @property
    def reach(self) -> float:
        with self._lock:
            return max(self._reach, self._seconds_locked())

    def _seconds_locked(self) -> float:
        if self._wall is None:
            return self._t
        return self._t + (time.perf_counter() - self._wall)


class FileFeeder:
    """Feeds decoded audio to the worker in 100 ms frames, as the mic would.

    The feed is paced by the worker's backlog, not by wall time: the model
    runs at its own speed and the queue stays under `cap_s`. With the default
    8 s, even a full 20 s chunk on top stays below the 30 s backlog at which
    the live path downgrades the model, so a file that arrived all at once
    never costs transcript quality.
    """

    def __init__(self, audio: np.ndarray, sample_rate: int, worker, cap_s: float = 8.0, frame_ms: int = 100) -> None:
        self.audio = audio
        self.sr = sample_rate
        self.worker = worker
        self.cap_s = cap_s
        self.frame = max(1, sample_rate * frame_ms // 1000)
        self._samples = len(audio)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._fed = 0
        self.done = False

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="replay-feeder", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def fed_s(self) -> float:
        return self._fed / self.sr

    @property
    def duration_s(self) -> float:
        return self._samples / self.sr

    def _run(self) -> None:
        i = 0
        while i < self._samples and not self._stop.is_set():
            if self.worker.status()["backlog_s"] >= self.cap_s:
                self._stop.wait(0.05)
                continue
            self.worker.feed(self.audio[i : i + self.frame], i)
            i += self.frame
            self._fed = min(i, self._samples)
        if i >= self._samples:
            # A file can end mid-sentence, and no pause will ever close that chunk.
            self.worker.flush()
            self.done = True
        self.audio = np.empty(0, dtype=np.float32)  # free the buffer as soon as it is consumed


def decode_recording(data: bytes | bytearray, sample_rate: int) -> np.ndarray:
    """Any format ffmpeg reads (mp3, m4a, wav, ogg, webm, a phone video) to
    16 kHz float32 mono, without touching the disk."""
    from faster_whisper.audio import decode_audio

    try:
        audio = decode_audio(io.BytesIO(data), sampling_rate=sample_rate)
    except Exception as exc:  # PyAV raises a zoo of errors for unreadable input
        raise ValueError(f"could not read that recording ({type(exc).__name__}); try an mp3, m4a or wav") from exc
    if len(audio) < sample_rate:
        raise ValueError("the recording is shorter than a second")
    return np.asarray(audio, dtype=np.float32)
