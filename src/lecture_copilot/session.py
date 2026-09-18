"""The single active lecture session: wires the audio source, the Whisper
worker, hotkeys, the trigger filter and the timeline together (M1).

Three sources share one session: the microphone during a live lecture, a
recording the student plays back to catch up on a class they missed (D26), or
a public lecture video imported from YouTube, slides included (D27). The
clock is the audio clock in all three: mic samples captured, or the playback
position the browser reports. Everything here runs on the laptop; no network
access is needed during class (a YouTube import is the one action that reaches
the internet, and only when the student asks for it).
"""

from __future__ import annotations

import logging
import tempfile
import threading
from pathlib import Path
from typing import Any

import numpy as np

from lecture_copilot.asr.worker import Segment, WhisperWorker
from lecture_copilot.bus import EventBus
from lecture_copilot.capture import MicCapture
from lecture_copilot.config import Settings
from lecture_copilot.hotkeys import Hotkeys
from lecture_copilot.llm import LlmGateway
from lecture_copilot.power import read_power_state
from lecture_copilot.replay import Clock, FileFeeder, PlaybackClock
from lecture_copilot.store import Store
from lecture_copilot.trigger import score_segment
from lecture_copilot.youtube import YoutubeMedia, detect_cuts, download, sample_frames

log = logging.getLogger(__name__)


def _cleanup(tmp: tempfile.TemporaryDirectory) -> None:
    try:
        tmp.cleanup()
    except OSError:
        log.warning("could not remove the temporary download directory %s", tmp.name, exc_info=True)


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
        # source: "mic" | "replay" | "youtube"; the clock is the mic capture or the playback clock
        self.source = "mic"
        self.clock: Clock | None = None
        self.feeder: FileFeeder | None = None
        self._replay_name: str | None = None
        # YouTube only (D27): the downloaded audio, held so the browser can fetch it once to play.
        self._playable_audio: bytes | None = None
        self._playable_audio_ext: str = "bin"
        self._slides_state: str | None = None  # youtube only: reading | ready | none
        self._slides_cancel = threading.Event()
        self._slides_thread: threading.Thread | None = None

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
            "slides": self._slides_state if self.lecture else None,
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

    def start_youtube(self, course_id: str, url: str, llm: LlmGateway, language: str = "en") -> dict[str, Any]:
        """Import a public lecture recording (D27): download audio and a
        capped-resolution video stream to a temp directory (D8, same rule as a
        deck upload: deleted as soon as it has been read), transcribe the audio
        exactly like a played-back file, and read the slide changes in the video
        with the vision model.

        This returns as soon as the audio is decoded and transcription is under
        way, so the student can start listening within seconds; reading the
        slides takes a minute or more and continues in the background, announcing
        itself with `slides_ready`. Each slide's alignment window is its exact
        on-screen interval, so there is no guessing which sentence goes with
        which slide, unlike the lexical matching a PDF deck needs.
        """
        with self._lock:
            power = read_power_state()
            name, device, compute = self._open(course_id, language, None, power.on_ac)
            assert self.worker is not None
            tmp = tempfile.TemporaryDirectory(prefix="lc-youtube-")
            handed_off = False  # the slide reader owns the temp directory once it starts
            try:
                from faster_whisper.audio import decode_audio

                media = download(url, Path(tmp.name), self.s.youtube_max_minutes, self.s.youtube_video_height)
                audio = decode_audio(str(media.audio_path), sampling_rate=self.s.sample_rate)
                playable = media.audio_path.read_bytes()
                ext = media.audio_path.suffix.lstrip(".").lower() or "bin"

                course = self.store.one("SELECT * FROM courses WHERE id=?", (course_id,))
                self.lecture = self.store.start_lecture(
                    course_id,
                    name,
                    device,
                    power.label,
                    power.battery_percent,
                    source="youtube",
                    notes=f"YouTube: {media.title}",
                    source_url=url,
                )
                self.source = "youtube"
                self.paused = True
                self.clock = PlaybackClock()
                self._replay_name = media.title
                self._playable_audio, self._playable_audio_ext = playable, ext
                self._slides_state = None

                self.worker.start(name, device, compute)
                self.feeder = FileFeeder(audio, self.s.sample_rate, self.worker, cap_s=self.s.replay_backlog_cap_s)
                self.feeder.start()
                self._start_hotkeys()

                if media.video_path is not None and llm.available:
                    self._slides_state = "reading"
                    self._slides_cancel.clear()
                    self._slides_thread = threading.Thread(
                        target=self._read_slides,
                        args=(media, llm, course, self.lecture, tmp),
                        name="slides-reader",
                        daemon=True,
                    )
                    self._slides_thread.start()
                    handed_off = True
                self.bus.publish(
                    {"type": "lecture_started", "lecture": self.lecture, "asr": self.worker.status(), "replay": self._replay_status()}
                )
                return {"lecture": self.lecture, "duration_s": media.duration_s, "slides": self._slides_state or "none"}
            except Exception:
                if self.lecture is None:
                    self.worker = None  # failed before anything else was touched: fully reset, like start()
                raise
            finally:
                if not handed_off:
                    _cleanup(tmp)

    def _read_slides(self, media: YoutubeMedia, llm: LlmGateway, course: dict, lecture: dict, tmp: tempfile.TemporaryDirectory) -> None:
        """Runs on its own thread. Best-effort: a failure here never touches the
        transcript already under way, and it stops early if the session ends."""
        from lecture_copilot.pipeline import slides_from_youtube

        summary: dict | None = None
        try:
            limit = self.s.slide_max_frames
            cuts = []
            # One more than the cap, so the summary can say "there were more"; also bounds the frames held in memory.
            frames = sample_frames(media.video_path, self.s.slide_sample_fps)
            for cut in detect_cuts(frames, self.s.slide_diff_threshold, self.s.slide_min_gap_s):
                if self._slides_cancel.is_set():
                    return
                cuts.append(cut)
                if len(cuts) > limit:
                    break

            def _progress(done: int, total: int) -> None:
                self.bus.publish(
                    {
                        "type": "progress",
                        "job": "youtube",
                        "lecture_id": lecture["id"],
                        "stage": "reading slides",
                        "done": done,
                        "total": total,
                    }
                )

            summary = slides_from_youtube(
                self.store,
                llm,
                lecture,
                course,
                media.video_id,
                media.title,
                cuts,
                media.duration_s,
                progress=_progress,
                max_frames=limit,
                cancelled=self._slides_cancel.is_set,
            )
        except Exception:
            log.warning("reading the slides of %s failed; the transcript stands on its own", media.video_id, exc_info=True)
        finally:
            _cleanup(tmp)
        if not self._slides_cancel.is_set():
            self._slides_state = "ready" if summary else "none"
            self.bus.publish({"type": "slides_ready", "lecture_id": lecture["id"], "deck": summary})

    def playable_audio(self) -> tuple[bytes, str] | None:
        if self.source != "youtube" or self._playable_audio is None:
            return None
        return self._playable_audio, self._playable_audio_ext

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
            self._slides_cancel.set()  # a slide reader still running is told to stop and drop what it has
            if self._slides_thread is not None:
                self._slides_thread.join(timeout=30)
                self._slides_thread = None
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
            self._playable_audio = None  # drop the downloaded audio from memory once the session ends
            self._slides_state = None
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
