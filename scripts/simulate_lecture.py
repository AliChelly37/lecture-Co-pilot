"""Feed a WAV file through the real WhisperWorker as if it were the live mic
(seed of the L1 replay harness). Validates VAD chunking, hallucination
filters and the trigger filter without a microphone.

  uv run python scripts/simulate_lecture.py eval/fixtures/tts_lecture.wav --speed 0
  (--speed 1 = real time, 0 = as fast as possible)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lecture_copilot.asr.worker import Segment, WhisperWorker
from lecture_copilot.config import Settings
from lecture_copilot.trigger import score_segment


def load_wav_16k(path: Path, sr: int) -> np.ndarray:
    from faster_whisper.audio import decode_audio

    return decode_audio(str(path), sampling_rate=sr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--speed", type=float, default=0.0, help="1 = real time, 0 = max speed")
    ap.add_argument("--model", default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    s = Settings()
    if args.model:
        s.asr_model_ac = s.asr_model_battery = args.model
    if args.device:
        s.asr_device = args.device

    segments: list[Segment] = []
    statuses: list[dict] = []

    def on_segment(seg: Segment) -> None:
        segments.append(seg)
        trig = score_segment(seg.text)
        mark = " <== " + ", ".join(trig.terms) if trig.hit else ""
        print(f"[{seg.t0:6.1f}-{seg.t1:6.1f}] ({seg.conf:.2f}) {seg.text}{mark}", flush=True)

    def on_status(ev: dict) -> None:
        statuses.append(ev)

    worker = WhisperWorker(s, on_segment, on_status, language="en")
    name, device, compute = worker.pick_model(on_ac=True)
    print(json.dumps({"model": name, "device": device, "compute": compute}), flush=True)
    worker.start(name, device, compute)

    audio = load_wav_16k(args.audio, s.sample_rate)
    frame = s.sample_rate // 10  # 100 ms, like the mic callback
    t_start = time.perf_counter()
    for i in range(0, len(audio), frame):
        worker.feed(audio[i : i + frame], i)
        if args.speed > 0:
            target = (i + frame) / s.sample_rate / args.speed
            lag = target - (time.perf_counter() - t_start)
            if lag > 0:
                time.sleep(lag)
    worker.stop()
    wall = time.perf_counter() - t_start

    print(
        json.dumps(
            {
                "audio_s": round(len(audio) / s.sample_rate, 1),
                "wall_s": round(wall, 1),
                "segments": len(segments),
                "badge_hits": sum(1 for seg in segments if score_segment(seg.text).hit),
                "rtf_p95": worker.rtf_p95(),
                "final_status": worker.status(),
            },
            indent=1,
        )
    )


if __name__ == "__main__":
    main()
