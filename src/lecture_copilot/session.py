"""The single active lecture session: wires the audio source, the Whisper
worker, hotkeys, the trigger filter and the timeline together (M1).

Two sources share one session (D26): the microphone during a live lecture,
or a recording the student plays back to catch up on a class they missed.
The clock is the audio clock in both cases: mic samples captured, or the
playback position the browser reports. Everything here runs on the laptop;
no network access is needed during class.
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
from lecture_copilot.replay import Clock, FileFeeder, PlaybackClock
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
        # source: "mic" | "replay"; the clock is the mic capture or the playback clock
        self.source = "mic"
        self.clock: Clock | None = None
        self.feeder: FileFeeder | None = None
        self._replay_name: str | None = None

    # -- status ----------------------------------------------------------
    def status(self) -> dict[str, Any]:
        power = read_power_state()
        return {
            "recording": self.lecture is not None,
            "source": self.source if self.lecture else None,
            "paused": self.paused,
            "lecture": self.lecture,
            "elapsed_s": round(clock.seconds, 1) if (clock := self.clock) else 0.0,
            "badge": self.badge,
            "asr": self.worker.status() if self.worker else None,
            "replay": self._replay_status(),
            "hotkeys": self.hotkeys.active if self.hotkeys else False,
            "power": {"state": power.label, "battery": power.battery_percent},
        }

    def _replay_status(self) -> dict[str, Any] | None:
        feeder, worker, clock = self.feeder, self.worker, self.clock
        if feeder is None or worker is None or not isinstance(clock, PlaybackClock):
            return None
        # Done once the whole file is fed and nothing is queued (chunks are at least 0.5 s).
        done = feeder.done and worker.status()["backlog_s"] < 0.25
        return {
            "filename": self._replay_name,
            "duration_s": round(feeder.duration_s, 1),
            "transcribed_s": round(feeder.duration_s if done else min(worker.transcribed_to_s, feeder.fed_s), 1),
            "done": done,
            "heard_s": round(clock.reach, 1),
        }

    # -- lifecycle -------------------------------------------------------
    def _open(self, course_id: str, language: str, hotwords: str | None, on_ac: bool) -> tuple[str, str, str]:
        """Shared preamble: refuse a second session, check the course, build the worker."""
        if self.lecture is not None:
            raise RuntimeError("a lecture is already recording")
        course = self.store.one("SELECT * FROM courses WHERE id=?", (course_id,))
        if course is None:
            raise KeyError("unknown course")
        if not course["capture_enabled"]:
            raise PermissionError("capture is disabled for this course")
        self.worker = WhisperWorker(self.s, self._on_segment, self._on_status, language=language, hotwords=hotwords)
        self.badge = 0
        self.paused = False
        return self.worker.pick_model(on_ac)

    def start(self, course_id: str, language: str = "en", hotwords: str | None = None) -> dict[str, Any]:
        with self._lock:
            power = read_power_state()
            name, device, compute = self._open(course_id, language, hotwords, power.on_ac)
            assert self.worker is not None
            self.lecture = self.store.start_lecture(course_id, name, device, power.label, power.battery_percent)
            self.source = "mic"

            # Capture starts before the model finishes loading: audio queues up
            # in the worker, so the first minute is not lost to a cold start.
            self.worker.start(name, device, compute)
            self.capture = MicCapture(self.s.sample_rate, self._on_frames, self._on_capture_error)
            self.clock = self.capture
            try:
                self.capture.start()
            except Exception:
                self.worker.stop()
                self.store.end_lecture(self.lecture["id"], None, power.battery_percent)
                self.lecture, self.worker, self.capture, self.clock = None, None, None, None
                raise

            self._start_hotkeys()
            self.bus.publish({"type": "lecture_started", "lecture": self.lecture, "asr": self.worker.status()})
            return self.lecture

    def start_replay(self, course_id: str, audio: np.ndarray, language: str = "en", filename: str = "recording") -> dict[str, Any]:
        """Transcribe a decoded recording while the browser plays it (D26).

        The worker gets the file as fast as it can take it (paced by its own
        backlog), so the transcript runs ahead of playback; the UI reveals
        each line when the audio reaches it. The session starts paused and
        the clock moves only on the browser's playback reports.
        """
        with self._lock:
            power = read_power_state()
            name, device, compute = self._open(course_id, language, None, power.on_ac)
            assert self.worker is not None
            self.lecture = self.store.start_lecture(
                course_id, name, device, power.label, power.battery_percent, source="replay", notes=f"replay of {filename}"
            )
            self.source = "replay"
            self._replay_name = filename
            self.paused = True
            self.clock = PlaybackClock()

            self.worker.start(name, device, compute)
            self.feeder = FileFeeder(audio, self.s.sample_rate, self.worker, cap_s=self.s.replay_backlog_cap_s)
            self.feeder.start()

            self._start_hotkeys()
            self.bus.publish(
                {"type": "lecture_started", "lecture": self.lecture, "asr": self.worker.status(), "replay": self._replay_status()}
            )
            return self.lecture

    def _start_hotkeys(self) -> None:
        self.hotkeys = Hotkeys({self.s.hotkey_flag: self.flag, self.s.hotkey_pause: self.toggle_pause})
        self.hotkeys.start()

    def stop(self) -> dict[str, Any]:
        with self._lock:
            if self.lecture is None or self.worker is None or self.clock is None:
                raise RuntimeError("no lecture is recording")
            lecture_id = self.lecture["id"]
            if self.hotkeys:
                self.hotkeys.stop()
            if self._pause_gap is not None:
                self.store.close_gap(self._pause_gap, self.clock.seconds)
                self._pause_gap = None
            if self.feeder is not None:
                self.feeder.stop()  # a replay stopped early is not transcribed past this point
            if self.capture is not None:
                self.capture.stop()
            self.worker.stop()  # flushes the last chunk; blocks until transcribed
            power = read_power_state()
            self.store.end_lecture(lecture_id, self.worker.rtf_p95(), power.battery_percent)
            ended = self.store.get_lecture(lecture_id)
            self.bus.publish({"type": "lecture_ended", "lecture": ended})
            self.lecture, self.worker, self.capture, self.hotkeys = None, None, None, None
            self.clock, self.feeder, self._replay_name = None, None, None
            self.paused = False
            self.source = "mic"
            return ended  # type: ignore[return-value]

    # -- user actions ----------------------------------------------------
    def flag(self, t: float | None = None) -> dict[str, Any] | None:
        """`t` is the playback position the browser saw at the click (replay);
        the hotkey and the live mic use the session clock."""
        lecture, clock = self.lecture, self.clock
        if lecture is None or clock is None:
            return None
        t = clock.seconds if t is None else max(0.0, float(t))
        w0 = max(0.0, t - self.s.flag_lookback_s)
        w1 = t + self.s.flag_lookahead_s
        flag_id = self.store.add_flag(lecture["id"], t, w0, w1)
        flag = {"id": flag_id, "lecture_id": lecture["id"], "t": round(t, 1), "window_t0": round(w0, 1), "window_t1": round(w1, 1)}
        self.bus.publish({"type": "flag", "flag": flag})
        return flag

    def toggle_pause(self, t: float | None = None) -> bool:
        lecture, worker, clock = self.lecture, self.worker, self.clock
        if lecture is None or worker is None or clock is None:
            return False
        self.paused = not self.paused
        if not isinstance(clock, PlaybackClock):
            # Live: the mic keeps running but nothing is transcribed; that is a gap in the record.
            worker.set_paused(self.paused)
            now = clock.seconds
            if self.paused:
                self._pause_gap = self.store.add_gap(lecture["id"], now, "paused")
            elif self._pause_gap is not None:
                self.store.close_gap(self._pause_gap, now)
                self._pause_gap = None
        else:
            # Replay: only the playback stops; the recording is complete, so there is no gap.
            if t is not None:
                clock.report(playing=not self.paused, t=t)
            elif self.paused:
                clock.pause()
            else:
                clock.play()
            now = clock.seconds
        self.bus.publish({"type": "status", "event": "pause", "paused": self.paused, "t": round(now, 1)})
        return self.paused

    def playback(self, playing: bool, t: float) -> dict[str, Any]:
        """The browser's audio element reports play, pause, seek and progress."""
        clock = self.clock
        if self.lecture is None or not isinstance(clock, PlaybackClock):
            raise RuntimeError("no recording is playing")
        clock.report(playing, t)
        if self.paused == playing:  # the state changed on the browser side (native controls, autoplay, end of file)
            self.paused = not playing
            self.bus.publish({"type": "status", "event": "pause", "paused": self.paused, "t": round(clock.seconds, 1)})
        return {"paused": self.paused, "elapsed_s": round(clock.seconds, 1)}

    # -- callbacks from worker / capture threads -------------------------
    def _on_frames(self, frames: np.ndarray, index: int) -> None:
        if self.worker is not None:
            self.worker.feed(frames, index)

    def _on_capture_error(self, cause: str) -> None:
        if self.lecture is None or self.clock is None:
            return
        if self._mic_gap is None:
            self._mic_gap = self.store.add_gap(self.lecture["id"], self.clock.seconds, cause)
        self.bus.publish({"type": "gap", "cause": cause, "t": round(self.clock.seconds, 1)})

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
