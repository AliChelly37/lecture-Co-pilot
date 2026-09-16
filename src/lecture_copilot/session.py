"""The single active lecture session: wires mic capture, the Whisper worker,
hotkeys, the trigger filter and the timeline together (M1).

Everything here runs on the laptop; no network access is needed during class.
"""

from __future__ import annotations

import logging
import threading
from typing import Any

import numpy as np

from lecture_copilot.asr.worker import Segment, WhisperWorker
from lecture_copilot.bus import EventBus
from lecture_copilot.capture import MicCapture
from lecture_copilot.config import Settings
from lecture_copilot.hotkeys import Hotkeys
from lecture_copilot.power import read_power_state
from lecture_copilot.store import Store
from lecture_copilot.trigger import score_segment

log = logging.getLogger(__name__)


class LectureSession:
    def __init__(self, settings: Settings, store: Store, bus: EventBus) -> None:
        self.s = settings
        self.store = store
        self.bus = bus
        self._lock = threading.Lock()
        self.lecture: dict[str, Any] | None = None
        self.worker: WhisperWorker | None = None
        self.capture: MicCapture | None = None
        self.hotkeys: Hotkeys | None = None
        self.paused = False
        self.badge = 0
        self._pause_gap: int | None = None
        self._mic_gap: int | None = None

    # -- status ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        power = read_power_state()
        return {
            "recording": self.lecture is not None,
            "paused": self.paused,
            "lecture": self.lecture,
            "elapsed_s": round(self.capture.seconds, 1) if self.capture else 0.0,
            "badge": self.badge,
            "asr": self.worker.status() if self.worker else None,
            "hotkeys": self.hotkeys.active if self.hotkeys else False,
            "power": {"state": power.label, "battery": power.battery_percent},
        }

    # -- lifecycle -------------------------------------------------------
    def start(self, course_id: str, language: str = "en", hotwords: str | None = None) -> dict[str, Any]:
        with self._lock:
            if self.lecture is not None:
                raise RuntimeError("a lecture is already recording")
            course = self.store.one("SELECT * FROM courses WHERE id=?", (course_id,))
            if course is None:
                raise KeyError("unknown course")
            if not course["capture_enabled"]:
                raise PermissionError("capture is disabled for this course")

            power = read_power_state()
            self.worker = WhisperWorker(self.s, self._on_segment, self._on_status, language=language, hotwords=hotwords)
            name, device, compute = self.worker.pick_model(power.on_ac)
            self.lecture = self.store.start_lecture(course_id, name, device, power.label, power.battery_percent)
            self.badge = 0
            self.paused = False

            # Capture starts before the model finishes loading: audio queues up
            # in the worker, so the first minute is not lost to a cold start.
            self.worker.start(name, device, compute)
            self.capture = MicCapture(self.s.sample_rate, self._on_frames, self._on_capture_error)
            try:
                self.capture.start()
            except Exception:
                self.worker.stop()
                self.store.end_lecture(self.lecture["id"], None, power.battery_percent)
                self.lecture, self.worker, self.capture = None, None, None
                raise

            self.hotkeys = Hotkeys({self.s.hotkey_flag: self.flag, self.s.hotkey_pause: self.toggle_pause})
            self.hotkeys.start()

            self.bus.publish({"type": "lecture_started", "lecture": self.lecture, "asr": self.worker.status()})
            return self.lecture

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self.lecture is None or self.worker is None or self.capture is None:
                raise RuntimeError("no lecture is recording")
            lecture_id = self.lecture["id"]
            if self.hotkeys:
                self.hotkeys.stop()
            if self._pause_gap is not None:
                self.store.close_gap(self._pause_gap, self.capture.seconds)
                self._pause_gap = None
            self.capture.stop()
            self.worker.stop()  # flushes the last chunk; blocks until transcribed
            power = read_power_state()
            self.store.end_lecture(lecture_id, self.worker.rtf_p95(), power.battery_percent)
            ended = self.store.get_lecture(lecture_id)
            self.bus.publish({"type": "lecture_ended", "lecture": ended})
            self.lecture, self.worker, self.capture, self.hotkeys = None, None, None, None
            self.paused = False
            return ended  # type: ignore[return-value]

    # -- user actions ----------------------------------------------------
    def flag(self) -> dict[str, Any] | None:
        if self.lecture is None or self.capture is None:
            return None
        t = self.capture.seconds
        w0 = max(0.0, t - self.s.flag_lookback_s)
        w1 = t + self.s.flag_lookahead_s
        flag_id = self.store.add_flag(self.lecture["id"], t, w0, w1)
        flag = {"id": flag_id, "lecture_id": self.lecture["id"], "t": round(t, 1), "window_t0": round(w0, 1), "window_t1": round(w1, 1)}
        self.bus.publish({"type": "flag", "flag": flag})
        return flag

    def toggle_pause(self) -> bool:
        if self.lecture is None or self.worker is None or self.capture is None:
            return False
        self.paused = not self.paused
        self.worker.set_paused(self.paused)
        t = self.capture.seconds
        if self.paused:
            self._pause_gap = self.store.add_gap(self.lecture["id"], t, "paused")
        elif self._pause_gap is not None:
            self.store.close_gap(self._pause_gap, t)
            self._pause_gap = None
        self.bus.publish({"type": "status", "event": "pause", "paused": self.paused, "t": round(t, 1)})
        return self.paused

    # -- callbacks from worker / capture threads -------------------------
    def _on_frames(self, frames: np.ndarray, index: int) -> None:
        if self.worker is not None:
            self.worker.feed(frames, index)

    def _on_capture_error(self, cause: str) -> None:
        if self.lecture is None or self.capture is None:
            return
        if self._mic_gap is None:
            self._mic_gap = self.store.add_gap(self.lecture["id"], self.capture.seconds, cause)
        self.bus.publish({"type": "gap", "cause": cause, "t": round(self.capture.seconds, 1)})

    def _on_segment(self, seg: Segment) -> None:
        if self.lecture is None:
            return
        trig = score_segment(seg.text)
        seg_id = self.store.add_segment(self.lecture["id"], seg.t0, seg.t1, seg.text, seg.conf, seg.model, trig.score, trig.terms)
        event = {
            "type": "segment",
            "segment": {
                "id": seg_id,
                "t0": round(seg.t0, 1),
                "t1": round(seg.t1, 1),
                "text": seg.text,
                "conf": seg.conf,
                "model": seg.model,
                "trigger": trig.terms if trig.hit else [],
            },
        }
        if trig.hit:
            self.badge += 1
            event["badge"] = self.badge
        self.bus.publish(event)

    def _on_status(self, event: dict[str, Any]) -> None:
        self.bus.publish(event)
