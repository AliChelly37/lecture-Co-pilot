"""YouTube import (D27) without a network call: slide-cut detection is pure
numpy and tested directly; frame decoding is tested against a small video this
test encodes itself; the pipeline and session/API paths use a fake vision
model and a monkeypatched download, the same way test_replay.py stubs the mic."""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lecture_copilot.youtube import Cut, YoutubeMedia, detect_cuts, sample_frames, to_jpeg

FIXTURES = Path(__file__).resolve().parents[1] / "eval" / "fixtures"


# -- detect_cuts: pure logic, no video -----------------------------------
def frame(color: int, size: tuple[int, int] = (48, 48)) -> np.ndarray:
    return np.full((*size, 3), color, dtype=np.uint8)


def test_detect_cuts_finds_slide_changes_and_ignores_small_noise() -> None:
    seq = [
        (0.0, frame(20)),
        (1.0, frame(22)),  # encoder noise: not a new slide
        (2.0, frame(220)),  # a real change, allowed even though it is soon after the start
        (2.3, frame(15)),  # a transient flicker right after a real cut: debounced
        (2.6, frame(220)),  # settles back: also within the debounce window
        (6.0, frame(220)),  # same slide, later: no diff at all
        (7.0, frame(40)),  # another real change, well past the debounce window
    ]
    cuts = list(detect_cuts(iter(seq), diff_threshold=0.3, min_gap_s=3.0))
    assert [round(c.t, 1) for c in cuts] == [0.0, 2.0, 7.0]
    assert cuts[0].frame[0, 0, 0] == 20


def test_detect_cuts_on_empty_input() -> None:
    assert list(detect_cuts(iter([]))) == []


def test_detect_cuts_always_keeps_the_first_frame() -> None:
    cuts = list(detect_cuts(iter([(0.0, frame(100))])))
    assert len(cuts) == 1 and cuts[0].t == 0.0


# -- sample_frames: a real, tiny video this test builds itself -----------
@pytest.fixture
def two_slide_video(tmp_path: Path) -> Path:
    av = pytest.importorskip("av")
    path = tmp_path / "slides.mp4"
    container = av.open(str(path), mode="w")
    try:
        stream = container.add_stream("mpeg4", rate=10)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        colors = [30] * 20 + [220] * 20  # 2 s of one slide, 2 s of another, at 10 fps
        for c in colors:
            vframe = av.VideoFrame.from_ndarray(frame(c, (64, 64)), format="rgb24").reformat(format="yuv420p")
            for packet in stream.encode(vframe):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()
    return path


def test_sample_frames_and_detect_cuts_on_a_real_video(two_slide_video: Path) -> None:
    frames = list(sample_frames(two_slide_video, fps=2.0))
    assert len(frames) >= 6  # ~4 s at 2 fps, allowing for rounding at the tail
    assert [t for t, _ in frames] == sorted(t for t, _ in frames)
    cuts = list(detect_cuts(iter(frames), diff_threshold=0.2, min_gap_s=1.0))
    assert len(cuts) == 2
    assert cuts[0].t < 1.0 and 1.5 <= cuts[1].t <= 2.5


def test_to_jpeg_round_trips() -> None:
    data = to_jpeg(frame(128, (16, 16)))
    assert data[:2] == b"\xff\xd8"  # a JPEG magic number, not raw pixels


# -- pipeline.slides_from_youtube: a fake vision model --------------------
def test_slides_from_youtube_keeps_real_slides_and_builds_exact_alignment(tmp_path: Path) -> None:
    from lecture_copilot.photos import BoardReading
    from lecture_copilot.pipeline import slides_from_youtube
    from lecture_copilot.store import Store

    store = Store(tmp_path / "d.sqlite3")
    course = store.create_course("Signals", "Europe/Paris")
    lecture = store.start_lecture(course["id"], "small", "cpu", "ac", None, source="youtube", source_url="https://youtu.be/abc")
    cuts = [Cut(0.0, frame(10)), Cut(30.0, frame(20)), Cut(60.0, frame(30)), Cut(90.0, frame(40))]

    readings = [
        BoardReading(content_kind="slide_projection", text="Title slide", latex=[], diagram_notes=[], legibility=0.9),
        BoardReading(content_kind="other", text="", latex=[], diagram_notes=[], legibility=0.0),  # a talking-head frame
        BoardReading(
            content_kind="slide_projection", text="Fugacity: effective pressure", latex=["f = phi P"], diagram_notes=[], legibility=0.8
        ),
        BoardReading(content_kind="slide_projection", text="Questions?", latex=[], diagram_notes=[], legibility=0.7),
    ]
    calls = []

    class FakeLlm:
        def structured(self, **kw):
            calls.append(kw["stage"])
            return readings[len(calls) - 1]

    progress: list[tuple[int, int]] = []
    summary = slides_from_youtube(
        store, FakeLlm(), lecture, course, "abc", "Intro to Signals", cuts, duration_s=120.0, progress=lambda d, t: progress.append((d, t))
    )

    assert calls == ["slide_ocr"] * 4
    assert progress[0] == (0, 4) and progress[-1] == (4, 4)
    assert summary == {"deck_id": summary["deck_id"], "slide_count": 3, "frames_read": 4, "skipped": 0}

    deck = store.get_deck(summary["deck_id"])
    assert deck["slides_text"] == ["Title slide", "Fugacity: effective pressure", "Questions?"]
    assert deck["filename"] == "Intro to Signals (slides recovered from the video)"
    assert store.get_lecture(lecture["id"])["deck_id"] == summary["deck_id"]

    windows = store.alignment(lecture["id"])
    # The skipped talking-head cut (t=30) leaves no gap: slide 1's window simply
    # runs until slide 2 actually appears, and every window is an exact timestamp (score 1.0).
    assert [(w["t0"], w["t1"], w["slide"], w["score"]) for w in windows] == [
        (0.0, 60.0, 1, 1.0),
        (60.0, 90.0, 2, 1.0),
        (90.0, 120.0, 3, 1.0),
    ]


def test_slides_from_youtube_returns_none_when_nothing_is_a_real_slide(tmp_path: Path) -> None:
    from lecture_copilot.photos import BoardReading
    from lecture_copilot.pipeline import slides_from_youtube
    from lecture_copilot.store import Store

    store = Store(tmp_path / "d.sqlite3")
    course = store.create_course("Signals", "Europe/Paris")
    lecture = store.start_lecture(course["id"], "small", "cpu", "ac", None, source="youtube")

    class FakeLlm:
        def structured(self, **kw):
            return BoardReading(content_kind="other", text="", latex=[], diagram_notes=[], legibility=0.0)

    assert slides_from_youtube(store, FakeLlm(), lecture, course, "x", "Talking head", [Cut(0.0, frame(1))], 60.0) is None
    assert store.get_lecture(lecture["id"])["deck_id"] is None


def test_slides_from_youtube_survives_a_model_error_on_one_frame(tmp_path: Path) -> None:
    from lecture_copilot.photos import BoardReading
    from lecture_copilot.pipeline import slides_from_youtube
    from lecture_copilot.store import Store

    store = Store(tmp_path / "d.sqlite3")
    course = store.create_course("Signals", "Europe/Paris")
    lecture = store.start_lecture(course["id"], "small", "cpu", "ac", None, source="youtube")

    class FlakyLlm:
        def __init__(self):
            self.n = 0

        def structured(self, **kw):
            self.n += 1
            if self.n == 1:
                raise RuntimeError("model timed out")
            return BoardReading(content_kind="slide_projection", text=f"slide {self.n}", latex=[], diagram_notes=[], legibility=0.9)

    summary = slides_from_youtube(store, FlakyLlm(), lecture, course, "x", "Flaky", [Cut(0.0, frame(1)), Cut(10.0, frame(2))], 20.0)
    assert summary["slide_count"] == 1  # the failed frame is skipped, not fatal
    assert store.get_deck(summary["deck_id"])["slides_text"] == ["slide 2"]


def test_slides_from_youtube_merges_a_repeated_slide_and_caps_the_model_calls(tmp_path: Path) -> None:
    from lecture_copilot.photos import BoardReading
    from lecture_copilot.pipeline import slides_from_youtube
    from lecture_copilot.store import Store

    store = Store(tmp_path / "d.sqlite3")
    course = store.create_course("Signals", "Europe/Paris")
    lecture = store.start_lecture(course["id"], "small", "cpu", "ac", None, source="youtube")
    texts = ["Slide A", "slide   a", "Slide B", "Slide C"]  # the camera cut away and back: A again, differently spaced
    calls = []

    class FakeLlm:
        def structured(self, **kw):
            calls.append(1)
            return BoardReading(content_kind="slide_projection", text=texts[len(calls) - 1], latex=[], diagram_notes=[], legibility=0.9)

    cuts = [Cut(float(i * 10), frame(i)) for i in range(4)]
    summary = slides_from_youtube(store, FakeLlm(), lecture, course, "x", "Repeats", cuts, 40.0, max_frames=3)

    assert len(calls) == 3  # the fourth cut is never sent to the model
    assert summary["skipped"] == 1 and summary["frames_read"] == 3 and summary["slide_count"] == 2
    assert store.get_deck(summary["deck_id"])["slides_text"] == ["Slide A", "Slide B"]
    # The repeat is absorbed: slide 1 runs until slide 2 appears, with no window for the duplicate.
    assert [(w["t0"], w["t1"], w["slide"]) for w in store.alignment(lecture["id"])] == [(0.0, 20.0, 1), (20.0, 40.0, 2)]


def test_no_cuts_is_a_cheap_no_op(tmp_path: Path) -> None:
    from lecture_copilot.pipeline import slides_from_youtube
    from lecture_copilot.store import Store

    store = Store(tmp_path / "d.sqlite3")
    course = store.create_course("Signals", "Europe/Paris")
    lecture = store.start_lecture(course["id"], "small", "cpu", "ac", None, source="youtube")

    class ShouldNotBeCalled:
        def structured(self, **kw):
            raise AssertionError("no cuts: the model should never be called")

    assert slides_from_youtube(store, ShouldNotBeCalled(), lecture, course, "x", "Empty", [], 20.0) is None


# -- session + API, with the download monkeypatched ------------------------
class StubWhisper:
    """Emits one segment per second of audio it is fed; never loads a model."""

    def __init__(self, settings, on_segment, on_status, language="en", hotwords=None) -> None:
        from lecture_copilot.asr.worker import Segment

        self.on_segment = on_segment
        self._seg = Segment
        self.transcribed_to_s = 0.0

    def pick_model(self, on_ac: bool) -> tuple[str, str, str]:
        return ("stub", "cpu", "int8")

    def start(self, model, device, compute) -> None:
        pass

    def feed(self, frames, index) -> None:
        sr = 16000
        if index % sr == 0:
            t = index / sr
            self.on_segment(self._seg(t, t + 1.0, f"line at {int(t)} seconds", 0.9, "stub"))
        self.transcribed_to_s = (index + len(frames)) / sr

    def flush(self) -> None:
        pass

    def set_paused(self, paused: bool) -> None:
        pass

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
    monkeypatch.setattr(settings, "llm_provider", "ollama")
    monkeypatch.setattr(session, "WhisperWorker", StubWhisper)
    monkeypatch.setattr(session.Hotkeys, "start", lambda self: None)
    with TestClient(api.app) as c:
        yield c


@pytest.fixture
def fake_media(tmp_path: Path, two_slide_video: Path) -> YoutubeMedia:
    audio_path = tmp_path / "audio.wav"
    shutil.copy(FIXTURES / "tts_lecture.wav", audio_path)
    return YoutubeMedia(
        video_id="abc123", title="Zero to One (guest lecture)", duration_s=140.0, audio_path=audio_path, video_path=two_slide_video
    )


def wait_for_slides(client: TestClient, want: str = "ready", timeout: float = 15.0) -> dict:
    import time

    t0 = time.time()
    while time.time() - t0 < timeout:
        st = client.get("/api/status").json()
        if st["slides"] == want:
            return st
        time.sleep(0.05)
    raise AssertionError(f"slides never became {want!r}: {client.get('/api/status').json()['slides']!r}")


def test_youtube_import_end_to_end(client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_media: YoutubeMedia) -> None:
    from lecture_copilot import session
    from lecture_copilot.api import state
    from lecture_copilot.photos import BoardReading

    tmp_dirs: list[Path] = []

    def fake_download(url, tmp_dir, max_minutes, height):
        tmp_dirs.append(tmp_dir)
        assert tmp_dir.is_dir()
        return fake_media

    monkeypatch.setattr(session, "download", fake_download)
    readings = iter(
        [
            BoardReading(content_kind="slide_projection", text="Zero to One", latex=[], diagram_notes=[], legibility=0.9),
            BoardReading(content_kind="slide_projection", text="Competition is for losers", latex=[], diagram_notes=[], legibility=0.9),
        ]
    )
    monkeypatch.setattr(state.llm, "structured", lambda **kw: next(readings))

    course = client.post("/api/courses", json={"name": "CS183", "timezone": "Europe/Paris"}).json()

    # No active session yet: refused without ever touching the (patched) download.
    assert client.post("/api/lectures/youtube", json={"course_id": course["id"], "url": "  "}).status_code == 400

    r = client.post("/api/lectures/youtube", json={"course_id": course["id"], "url": "https://youtu.be/abc123"})
    assert r.status_code == 200, r.text
    body = r.json()
    lid = body["lecture"]["id"]
    assert body["duration_s"] == 140.0
    assert body["slides"] == "reading"  # the response does not wait for the vision model

    st = client.get("/api/status").json()
    assert st["recording"] and st["source"] == "youtube" and st["paused"] is True
    assert st["replay"]["filename"] == "Zero to One (guest lecture)"

    # The slides arrive in the background; the temp download directory is gone once they have.
    wait_for_slides(client, "ready")
    assert not tmp_dirs[0].exists()

    # A second import, or the mic, cannot start while this one is open.
    assert client.post("/api/lectures/youtube", json={"course_id": course["id"], "url": "https://youtu.be/other"}).status_code == 409
    assert client.post("/api/lectures/start", json={"course_id": course["id"]}).status_code == 409

    # The browser fetches the downloaded audio once, the same way it plays an uploaded recording.
    audio = client.get(f"/api/lectures/{lid}/audio")
    assert audio.status_code == 200 and audio.headers["content-type"] == "audio/wav"
    assert audio.content == fake_media.audio_path.read_bytes()

    client.post("/api/lectures/playback", json={"playing": True, "t": 1.0})
    ended = client.post("/api/lectures/stop").json()
    assert ended["status"] == "ended" and ended["source"] == "youtube" and ended["source_url"] == "https://youtu.be/abc123"
    assert ended["notes"] == "YouTube: Zero to One (guest lecture)"

    detail = client.get(f"/api/lectures/{lid}").json()
    assert detail["deck"]["slide_count"] == 2
    assert [w["slide"] for w in detail["alignment"]] == [1, 2]
    assert len(detail["segments"]) > 0

    # The audio is released once the session ends.
    assert client.get(f"/api/lectures/{lid}/audio").status_code == 404
    # And a youtube import never counts toward the live-capture dashboard numbers (D26/D27).
    assert client.get("/api/dashboard").json()["asr"]["recorded_hours"] == 0.0


def test_youtube_import_without_a_model_still_produces_a_transcript(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_media: YoutubeMedia
) -> None:
    from lecture_copilot import session
    from lecture_copilot.api import state
    from lecture_copilot.config import settings

    monkeypatch.setattr(settings, "llm_provider", "anthropic")  # unconfigured: state.llm.available is False
    monkeypatch.setattr(settings, "anthropic_api_key", None)
    monkeypatch.setattr(session, "download", lambda url, tmp_dir, max_minutes, height: fake_media)

    def fail(**kw):
        raise AssertionError("no model configured: slide reading must not be attempted")

    monkeypatch.setattr(state.llm, "structured", fail)

    course = client.post("/api/courses", json={"name": "CS183", "timezone": "Europe/Paris"}).json()
    r = client.post("/api/lectures/youtube", json={"course_id": course["id"], "url": "https://youtu.be/abc123"})
    assert r.status_code == 200, r.text
    assert r.json()["slides"] == "none"

    detail = client.get(f"/api/lectures/{r.json()['lecture']['id']}").json()
    assert detail["deck"] is None and len(detail["segments"]) > 0
    client.post("/api/lectures/stop")


def test_youtube_download_error_leaves_no_session(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    from lecture_copilot import session
    from lecture_copilot.api import state

    def boom(url, tmp_dir, max_minutes, height):
        raise ValueError("that video is 240 minutes long; the limit is 180")

    monkeypatch.setattr(session, "download", boom)
    course = client.post("/api/courses", json={"name": "CS183", "timezone": "Europe/Paris"}).json()

    r = client.post("/api/lectures/youtube", json={"course_id": course["id"], "url": "https://youtu.be/toolong"})
    assert r.status_code == 400 and "180" in r.json()["detail"]
    assert state.session.lecture is None and state.session.worker is None
    # The session is clean: a normal lecture can start right after.
    assert client.post("/api/lectures/start", json={"course_id": course["id"]}).status_code == 200
    client.post("/api/lectures/stop")


def test_stopping_while_slides_are_being_read_stores_nothing_and_cleans_up(
    client: TestClient, monkeypatch: pytest.MonkeyPatch, fake_media: YoutubeMedia
) -> None:
    import time

    from lecture_copilot import session
    from lecture_copilot.api import state
    from lecture_copilot.photos import BoardReading

    tmp_dirs: list[Path] = []

    def fake_download(url, tmp_dir, max_minutes, height):
        tmp_dirs.append(tmp_dir)
        return fake_media

    def slow(**kw):
        time.sleep(0.4)  # long enough for stop() to arrive while the first frame is being read
        return BoardReading(content_kind="slide_projection", text="A slide", latex=[], diagram_notes=[], legibility=0.9)

    monkeypatch.setattr(session, "download", fake_download)
    monkeypatch.setattr(state.llm, "structured", slow)

    course = client.post("/api/courses", json={"name": "CS183", "timezone": "Europe/Paris"}).json()
    r = client.post("/api/lectures/youtube", json={"course_id": course["id"], "url": "https://youtu.be/abc123"})
    lid = r.json()["lecture"]["id"]
    assert r.json()["slides"] == "reading"

    assert client.post("/api/lectures/stop").status_code == 200  # cancels the reader and waits for it
    detail = client.get(f"/api/lectures/{lid}").json()
    assert detail["deck"] is None and detail["alignment"] == []
    assert not tmp_dirs[0].exists()
    assert client.get("/api/status").json()["slides"] is None
