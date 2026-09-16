"""The VAD chunker in WhisperWorker, tested without a model: speech detection
is stubbed so the test controls exactly where speech and silence fall."""

import numpy as np

from lecture_copilot.asr.worker import WhisperWorker
from lecture_copilot.config import Settings

SR = 16000


def make_worker(**overrides) -> WhisperWorker:
    s = Settings(sample_rate=SR, asr_min_silence_ms=500, asr_max_chunk_s=20.0, **overrides)
    w = WhisperWorker(s, on_segment=lambda seg: None, on_status=lambda ev: None)
    w._block_has_speech = lambda block: bool(np.abs(block).max() > 0)  # type: ignore[method-assign]
    return w


def feed(w: WhisperWorker, audio: np.ndarray, start_index: int = 0) -> None:
    frame = SR // 10
    for i in range(0, len(audio), frame):
        w.feed(audio[i : i + frame], start_index + i)


def queued(w: WhisperWorker) -> list[tuple[float, float]]:
    out = []
    while not w._chunks.empty():
        c = w._chunks.get_nowait()
        assert c is not None
        out.append((round(c.t0, 1), round(len(c.audio) / SR, 1)))
    return out


def test_pause_closes_chunk_and_silence_is_dropped() -> None:
    w = make_worker()
    speech = np.ones(3 * SR, dtype=np.float32)
    silence = np.zeros(6 * SR, dtype=np.float32)
    feed(w, np.concatenate([speech, silence]))
    chunks = queued(w)
    # One chunk: 3 s of speech plus the 0.5 s pause that closed it.
    assert chunks == [(0.0, 3.5)]
    # The 5.5 s of pure silence after it never reached the queue.
    assert w._pending == [] or w._pending_samples < 3 * SR


def test_flush_queues_speech_that_never_reached_a_pause() -> None:
    # The end of a replayed file: no silence will ever close this chunk.
    w = make_worker()
    feed(w, np.ones(3 * SR, dtype=np.float32))
    assert queued(w) == []
    w.flush()
    assert queued(w) == [(0.0, 3.0)]


def test_long_speech_is_force_cut_at_max_chunk() -> None:
    w = make_worker()
    feed(w, np.ones(45 * SR, dtype=np.float32))
    chunks = queued(w)
    assert chunks[:2] == [(0.0, 20.0), (20.0, 20.0)]


def test_clock_survives_dropped_silence() -> None:
    w = make_worker()
    audio = np.concatenate([np.zeros(10 * SR, dtype=np.float32), np.ones(2 * SR, dtype=np.float32), np.zeros(SR, dtype=np.float32)])
    feed(w, audio)
    chunks = queued(w)
    assert len(chunks) == 1
    t0, dur = chunks[0]
    # Speech began at 10 s; the chunk may include the last silent block before it.
    assert 9.0 <= t0 <= 10.0 and 2.0 <= dur <= 3.5


def test_paused_audio_is_skipped_but_clock_advances() -> None:
    w = make_worker()
    w.set_paused(True)
    feed(w, np.ones(5 * SR, dtype=np.float32))
    assert queued(w) == []
    w.set_paused(False)
    feed(w, np.concatenate([np.ones(2 * SR, dtype=np.float32), np.zeros(SR, dtype=np.float32)]), start_index=5 * SR)
    chunks = queued(w)
    assert len(chunks) == 1 and chunks[0][0] == 5.0
