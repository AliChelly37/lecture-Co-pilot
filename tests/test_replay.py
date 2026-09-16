"""Replay mode (D26) without a model or a microphone: the playback clock, the
paced file feeder, decoding, and the session/API path with the Whisper worker
stubbed."""

from __future__ import annotations

import io
import time
import wave
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lecture_copilot.replay import FileFeeder, PlaybackClock, decode_recording

SR = 16000


# -- clock -------------------------------------------------------------------
def test_playback_clock_follows_reports_and_runs_only_while_playing(monkeypatch: pytest.MonkeyPatch) -> None:
    now = [100.0]
    monkeypatch.setattr("lecture_copilot.replay.time.perf_counter", lambda: now[0])
    c = PlaybackClock()
    assert c.seconds == 0.0 and not c.playing

    c.report(playing=False, t=12.0)
    now[0] += 5
    assert c.seconds == 12.0  # paused: the clock stands still

    c.report(playing=True, t=12.0)
    now[0] += 2.5
    assert c.seconds == pytest.approx(14.5)  # playing: real time between reports

    c.pause()
    now[0] += 10
    assert c.seconds == pytest.approx(14.5)
    c.play()
    now[0] += 1
    assert c.seconds == pytest.approx(15.5)

    c.report(playing=True, t=3.0)  # a seek backwards
    assert c.seconds == pytest.approx(3.0)
    assert c.reach == pytest.approx(15.5)  # what was already heard stays heard
    now[0] += 20
    assert c.reach == pytest.approx(23.0)
    c.report(playing=False, t=-1.0)
    assert c.seconds == 0.0 and c.reach == pytest.approx(23.0)


# -- feeder ------------------------------------------------------------------
class FakeWorker:
    def __init__(self, backlog: float = 0.0) -> None:
        self.backlog = backlog
        self.fed: list[tuple[int, int]] = []  # (sample_index, frame_len)
        self.flushes = 0

    def status(self) -> dict:
        return {"backlog_s": self.backlog}

    def feed(self, frames: np.ndarray, index: int) -> None:
        self.fed.append((index, len(frames)))

    def flush(self) -> None:
        self.flushes += 1


def test_feeder_feeds_every_sample_in_order_then_flushes_the_tail() -> None:
    audio = np.arange(SR * 2 + 700, dtype=np.float32)  # not a multiple of the frame
    w = FakeWorker()
    f = FileFeeder(audio, SR, w)
    f.start()
    f.stop()  # joins; feeding 2 s takes milliseconds
    assert f.done and w.flushes == 1
    assert f.fed_s == pytest.approx(len(audio) / SR) and f.duration_s == pytest.approx(len(audio) / SR)
    assert [i for i, _ in w.fed] == list(range(0, len(audio), SR // 10))
    assert sum(n for _, n in w.fed) == len(audio) and w.fed[-1][1] == 700
    assert len(f.audio) == 0  # the decoded buffer is released once consumed


def test_feeder_waits_on_backlog_and_stops_promptly() -> None:
    audio = np.zeros(SR * 30, dtype=np.float32)
    w = FakeWorker(backlog=8.0)  # the worker is "full": nothing may be fed
    f = FileFeeder(audio, SR, w, cap_s=8.0)
    f.start()
    time.sleep(0.2)
    assert w.fed == [] and not f.done

    # Room for exactly five frames, then full again.
    w.status = lambda: {"backlog_s": 0.0 if len(w.fed) < 5 else 8.0}  # type: ignore[method-assign]
    time.sleep(0.2)
    assert len(w.fed) == 5 and f.fed_s == pytest.approx(0.5) and not f.done

    t0 = time.perf_counter()
    f.stop()
    assert time.perf_counter() - t0 < 1.0
    assert not f.done and w.flushes == 0  # stopped mid-file: nothing past this point is transcribed


# -- decoding ----------------------------------------------------------------
def wav_bytes(seconds: float, sr: int = SR) -> bytes:
    t = np.arange(int(seconds * sr)) / sr
    pcm = (0.3 * np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()


def test_decode_recording_from_memory_and_rejects_junk() -> None:
    audio = decode_recording(bytearray(wav_bytes(2.0, sr=44100)), SR)  # resampled to 16 kHz
    assert audio.dtype == np.float32 and abs(len(audio) - 2 * SR) < SR // 100
    with pytest.raises(ValueError):
        decode_recording(b"this is not audio", SR)
    with pytest.raises(ValueError):
        decode_recording(wav_bytes(0.2), SR)


# -- session + API with a stubbed worker ---------------------------------------
class StubWhisper:
    """Emits one segment per second of audio it is fed; never loads a model."""

    def __init__(self, settings, on_segment, on_status, language="en", hotwords=None) -> None:
        from lecture_copilot.asr.worker import Segment

        self.on_segment, self.paused, self.flushed = on_segment, False, False
        self._seg = Segment
        self.transcribed_to_s = 0.0

    def pick_model(self, on_ac: bool) -> tuple[str, str, str]:
        return ("stub", "cpu", "int8")

    def start(self, model, device, compute) -> None:
        pass

    def feed(self, frames, index) -> None:
        if index % SR == 0:
            t = index / SR
            self.on_segment(self._seg(t, t + 1.0, f"line at {int(t)} seconds", 0.9, "stub"))
        self.transcribed_to_s = (index + len(frames)) / SR

    def flush(self) -> None:
        self.flushed = True

    def set_paused(self, paused: bool) -> None:
        self.paused = paused

    def stop(self) -> None:
        pass

    def status(self) -> dict:
        return {
            "model": "stub",
            "device": "cpu",
            "compute": "int8",
            "loaded": True,
            "downgraded": False,
            "backlog_s": 0.0,
            "rtf_recent": None,
        }

    def rtf_p95(self) -> float | None:
        return 0.05


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from lecture_copilot import api, session
    from lecture_copilot.config import settings

    monkeypatch.setattr(settings, "data_dir", tmp_path / "data")
    monkeypatch.setattr(session, "WhisperWorker", StubWhisper)
    monkeypatch.setattr(session.Hotkeys, "start", lambda self: None)  # no global key hooks in tests
    with TestClient(api.app) as c:
        yield c


def upload(client: TestClient, course_id: str, data: bytes, filename: str = "friday.wav"):
    return client.post(
        "/api/lectures/replay",
        params={"course_id": course_id, "language": "en", "filename": filename},
        content=data,
        headers={"content-type": "application/octet-stream"},
    )


def test_replay_session_end_to_end(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from lecture_copilot.api import state
    from lecture_copilot.config import settings

    course = client.post("/api/courses", json={"name": "Signals", "timezone": "Europe/Paris"}).json()

    # Refused before any session starts: junk, nothing, too big.
    assert upload(client, course["id"], b"nope").status_code == 400
    assert upload(client, course["id"], b"").status_code == 400
    monkeypatch.setattr(settings, "replay_max_mb", 0)
    assert upload(client, course["id"], wav_bytes(1.5)).status_code == 413
    monkeypatch.setattr(settings, "replay_max_mb", 400)
    assert state.session.lecture is None

    r = upload(client, course["id"], wav_bytes(3.0), filename="C:\\fakepath\\friday.wav")
    assert r.status_code == 200, r.text
    started = r.json()
    lid = started["lecture"]["id"]
    assert started["lecture"]["source"] == "replay" and started["duration_s"] == pytest.approx(3.0)

    st = client.get("/api/status").json()
    assert st["recording"] and st["source"] == "replay" and st["paused"] is True  # paused until the browser plays
    assert st["replay"]["filename"] == "friday.wav" and st["replay"]["duration_s"] == pytest.approx(3.0)
    for _ in range(50):  # the stub transcribes as fast as it is fed
        if (st := client.get("/api/status").json())["replay"]["done"]:
            break
        time.sleep(0.02)
    assert st["replay"]["done"] and st["replay"]["transcribed_s"] == pytest.approx(3.0) and st["replay"]["heard_s"] == 0.0
    assert state.session.worker.flushed  # the file's tail was queued without waiting for Finish

    # Neither a second recording nor the mic can start while one is open.
    assert upload(client, course["id"], wav_bytes(2.0)).status_code == 409
    assert client.post("/api/lectures/start", json={"course_id": course["id"]}).status_code == 409

    # The browser reports playback; a clicked flag lands on the recording's timeline.
    assert client.post("/api/lectures/playback", json={"playing": True, "t": 1.0}).json()["paused"] is False
    flag = client.post("/api/lectures/flag", json={"t": 1.6}).json()
    assert flag["t"] == 1.6 and flag["window_t0"] == 0.0 and flag["window_t1"] == pytest.approx(16.6)
    hot = client.post("/api/lectures/flag").json()  # the F9 path: no t, the clock answers
    assert 1.0 <= hot["t"] < 3.0

    # Pause in replay pauses the clock, not the transcription, and records no gap.
    assert client.post("/api/lectures/pause", json={"t": 2.0}).json()["paused"] is True
    assert state.session.worker.paused is False
    assert client.get("/api/status").json()["elapsed_s"] == 2.0
    assert client.post("/api/lectures/playback", json={"playing": False, "t": 2.0}).json()["paused"] is True  # idempotent
    assert client.post("/api/lectures/pause").json()["paused"] is False

    ended = client.post("/api/lectures/stop").json()
    assert ended["status"] == "ended" and ended["source"] == "replay" and ended["notes"] == "replay of friday.wav"
    detail = client.get(f"/api/lectures/{lid}").json()
    assert [s["t0"] for s in detail["segments"]] == [0.0, 1.0, 2.0]  # the stub's one line per second
    assert len(detail["flags"]) == 2 and detail["gaps"] == []
    assert client.get("/api/lectures").json()[0]["source"] == "replay"
    assert client.get("/api/status").json()["recording"] is False

    # Playback reports are refused once nothing is open, and replays stay out of the live-capture stats.
    assert client.post("/api/lectures/playback", json={"playing": True, "t": 0}).status_code == 409
    assert client.get("/api/dashboard").json()["asr"]["recorded_hours"] == 0.0
