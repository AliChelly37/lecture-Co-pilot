"""Whisper worker: turns a live audio stream into transcript segments.

Design (D12, D14):
- Audio arrives in 100 ms frames from the capture thread. `feed()` groups them
  into 0.5 s blocks, runs Silero VAD on each block, and closes a chunk when a
  pause of `min_silence_ms` follows speech (or when the chunk reaches
  `max_chunk_s`). Blocks of pure silence are dropped, so silence never reaches
  Whisper - this starves its hallucination-on-silence failure mode.
- Closed chunks go to a queue; one worker thread transcribes them
  (decode-once; no sliding window, no conditioning on previous text).
- Segments failing the no-speech / log-prob / compression-ratio filters are
  dropped, as are exact repeats (loop guard).
- Backlog (un-transcribed audio) and real-time factor are reported. When the
  backlog exceeds a threshold the worker reloads a smaller model once.
"""

from __future__ import annotations

import logging
import math
import queue
import statistics
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from lecture_copilot.asr.cuda import cuda_available, ensure_cuda_dlls
from lecture_copilot.config import Settings

log = logging.getLogger(__name__)


@dataclass
class Segment:
    t0: float
    t1: float
    text: str
    conf: float
    model: str


@dataclass
class _Chunk:
    t0: float
    audio: np.ndarray


class WhisperWorker:
    def __init__(
        self,
        settings: Settings,
        on_segment: Callable[[Segment], None],
        on_status: Callable[[dict], None],
        language: str = "en",
        hotwords: str | None = None,
    ) -> None:
        self.s = settings
        self.on_segment = on_segment
        self.on_status = on_status
        self.language = language
        self.hotwords = hotwords

        self.model = None
        self.model_name = ""
        self.device = "cpu"
        self.compute = ""
        self.loaded = threading.Event()
        self.downgraded = False

        self._chunks: queue.Queue[_Chunk | None] = queue.Queue()
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._paused = False

        # VAD / chunking state, touched only from the capture thread.
        self._block: list[np.ndarray] = []
        self._block_samples = 0
        self._pending: list[np.ndarray] = []
        self._pending_samples = 0
        self._pending_t0 = 0.0
        self._speech_seen = False
        self._silence_s = 0.0
        self._consumed_samples = 0
        self._vad = None

        # Telemetry.
        self._rtfs: deque[float] = deque(maxlen=50)
        self._all_rtfs: list[float] = []
        self._backlog_s = 0.0
        self._last_text = ""

    # -- lifecycle -------------------------------------------------------
    def pick_model(self, on_ac: bool) -> tuple[str, str, str]:
        ensure_cuda_dlls()
        use_cuda = self.s.asr_device == "cuda" or (self.s.asr_device == "auto" and cuda_available())
        device = "cuda" if use_cuda else "cpu"
        compute = self.s.asr_compute_cuda if use_cuda else self.s.asr_compute_cpu
        name = self.s.asr_model_ac if on_ac else self.s.asr_model_battery
        return name, device, compute

    def start(self, model_name: str, device: str, compute: str) -> None:
        self.model_name, self.device, self.compute = model_name, device, compute
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="whisper-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._flush_pending(force=True)
        self._chunks.put(None)
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=120)

    def set_paused(self, paused: bool) -> None:
        self._paused = paused
        if paused:
            self._flush_pending(force=True)

    # -- capture side ----------------------------------------------------
    def feed(self, frames: np.ndarray, sample_index: int) -> None:
        """Called from the capture thread with float32 mono frames.

        `sample_index` is the absolute index of the first sample, which keeps
        the lecture clock tied to audio, not wall time.
        """
        if self._paused:
            self._consumed_samples = sample_index + len(frames)
            return
        if not self._pending and not self._block:
            self._pending_t0 = sample_index / self.s.sample_rate
        self._block.append(frames)
        self._block_samples += len(frames)
        if self._block_samples >= self.s.sample_rate // 2:  # 0.5 s block
            block = np.concatenate(self._block)
            self._block, self._block_samples = [], 0
            self._on_block(block)

    def _on_block(self, block: np.ndarray) -> None:
        has_speech = self._block_has_speech(block)
        self._pending.append(block)
        self._pending_samples += len(block)
        block_s = len(block) / self.s.sample_rate

        if has_speech:
            self._speech_seen = True
            self._silence_s = 0.0
        else:
            self._silence_s += block_s

        pending_s = self._pending_samples / self.s.sample_rate
        pause_closed = self._silence_s * 1000 >= self.s.asr_min_silence_ms
        if self._speech_seen and (pause_closed or pending_s >= self.s.asr_max_chunk_s):
            self._flush_pending()
        elif not self._speech_seen and pending_s >= 3.0:
            # Pure silence: drop it, keep the clock moving.
            self._drop_pending()

    def _block_has_speech(self, block: np.ndarray) -> bool:
        if self._vad is None:
            from faster_whisper.vad import VadOptions, get_speech_timestamps

            self._vad = (get_speech_timestamps, VadOptions(min_speech_duration_ms=100, min_silence_duration_ms=100, speech_pad_ms=0))
        fn, opts = self._vad
        try:
            return bool(fn(block, opts, sampling_rate=self.s.sample_rate))
        except Exception:  # VAD failure must never stop capture
            return True

    def _flush_pending(self, force: bool = False) -> None:
        if not self._pending:
            return
        if not self._speech_seen and not force:
            self._drop_pending()
            return
        audio = np.concatenate(self._pending)
        if self._speech_seen or (force and len(audio) >= self.s.sample_rate):
            self._chunks.put(_Chunk(t0=self._pending_t0, audio=audio))
            self._backlog_s += len(audio) / self.s.sample_rate
        self._drop_pending()

    def _drop_pending(self) -> None:
        self._consumed_samples += self._pending_samples
        self._pending, self._pending_samples = [], 0
        self._pending_t0 = self._consumed_samples / self.s.sample_rate
        self._speech_seen = False
        self._silence_s = 0.0

    # -- worker thread ---------------------------------------------------
    def _load(self, name: str, device: str, compute: str) -> None:
        from faster_whisper import WhisperModel

        t0 = time.perf_counter()
        self.model = WhisperModel(name, device=device, compute_type=compute)
        self.model_name, self.device, self.compute = name, device, compute
        log.info("loaded %s on %s (%s) in %.1fs", name, device, compute, time.perf_counter() - t0)

    def _run(self) -> None:
        try:
            self._load(self.model_name, self.device, self.compute)
        except Exception as exc:
            log.exception("model load failed, falling back to CPU %s", self.s.asr_model_fallback)
            self.on_status({"type": "status", "asr": {"error": str(exc)[:200]}})
            try:
                self._load(self.s.asr_model_fallback, "cpu", self.s.asr_compute_cpu)
                self.downgraded = True
            except Exception as exc2:
                self.on_status({"type": "status", "asr": {"fatal": str(exc2)[:200]}})
                return
        self.loaded.set()
        self._emit_status()

        while True:
            chunk = self._chunks.get()
            if chunk is None:
                break
            self._transcribe(chunk)
            self._maybe_downgrade()

    def _transcribe(self, chunk: _Chunk) -> None:
        assert self.model is not None
        audio_s = len(chunk.audio) / self.s.sample_rate
        t_start = time.perf_counter()
        try:
            segments, _info = self.model.transcribe(
                chunk.audio,
                language=self.language,
                beam_size=self.s.asr_beam_size,
                vad_filter=True,
                condition_on_previous_text=False,
                without_timestamps=False,
                hotwords=self.hotwords,
                no_speech_threshold=self.s.asr_no_speech_threshold,
                log_prob_threshold=self.s.asr_log_prob_threshold,
                compression_ratio_threshold=self.s.asr_compression_ratio_threshold,
            )
            for seg in segments:
                text = seg.text.strip()
                if not text:
                    continue
                if seg.no_speech_prob > self.s.asr_no_speech_threshold and seg.avg_logprob < self.s.asr_log_prob_threshold:
                    continue
                if text == self._last_text:  # loop guard
                    continue
                self._last_text = text
                conf = max(0.0, min(1.0, math.exp(seg.avg_logprob)))
                # Whisper timestamps can overshoot the chunk; keep them inside it.
                t0 = chunk.t0 + min(seg.start, audio_s)
                t1 = chunk.t0 + min(seg.end, audio_s)
                self.on_segment(Segment(t0=t0, t1=t1, text=text, conf=round(conf, 3), model=self.model_name))
        except Exception:
            log.exception("transcription failed for chunk at %.1fs", chunk.t0)
        finally:
            elapsed = time.perf_counter() - t_start
            rtf = elapsed / max(audio_s, 0.1)
            self._rtfs.append(rtf)
            self._all_rtfs.append(rtf)
            self._backlog_s = max(0.0, self._backlog_s - audio_s)
            self._emit_status()

    def _maybe_downgrade(self) -> None:
        """Downgrade only when the model itself is too slow: a large backlog
        *and* a recent real-time factor near 1. A queue that built up during
        the cold start drains on its own with a fast model and must not
        trigger this (found by the replay harness)."""
        if self.downgraded or self._backlog_s < self.s.asr_backlog_downgrade_s:
            return
        if self.model_name == self.s.asr_model_fallback:
            return
        if len(self._rtfs) < 3 or statistics.median(self._rtfs) < self.s.asr_downgrade_rtf:
            return
        log.warning("backlog %.0fs; downgrading to %s", self._backlog_s, self.s.asr_model_fallback)
        try:
            self._load(self.s.asr_model_fallback, self.device, self.compute)
            self.downgraded = True
            self._emit_status(event="downgraded")
        except Exception:
            log.exception("downgrade failed; keeping current model")

    # -- telemetry -------------------------------------------------------
    def status(self) -> dict:
        return {
            "model": self.model_name,
            "device": self.device,
            "compute": self.compute,
            "loaded": self.loaded.is_set(),
            "downgraded": self.downgraded,
            "backlog_s": round(self._backlog_s, 1),
            "rtf_recent": round(statistics.median(self._rtfs), 3) if self._rtfs else None,
        }

    def rtf_p95(self) -> float | None:
        if not self._all_rtfs:
            return None
        ordered = sorted(self._all_rtfs)
        return round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 3)

    def _emit_status(self, event: str = "asr") -> None:
        self.on_status({"type": "status", "event": event, "asr": self.status()})
