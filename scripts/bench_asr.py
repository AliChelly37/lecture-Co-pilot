"""M0: measure faster-whisper speed on this machine.

Usage:
  uv run python scripts/bench_asr.py --audio eval/fixtures/tts_lecture.wav \
      --configs cuda:large-v3-turbo:int8_float16 cpu:small:int8

Each config is device:model:compute_type. Prints one JSON line per config with
load time, transcription time, real-time factor (elapsed / audio seconds) and a
rough word error rate when a .txt reference sits next to the audio file.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path


def ensure_cuda_dlls() -> list[str]:
    """Make the pip-installed NVIDIA runtime DLLs loadable on Windows.

    faster-whisper (CTranslate2) needs cublas and cudnn on the DLL search path;
    the nvidia-*-cu12 wheels drop them under site-packages/nvidia/<lib>/bin.
    """
    added: list[str] = []
    if sys.platform != "win32":
        return added
    try:
        import nvidia  # type: ignore[import-not-found]
    except ImportError:
        return added
    for base in nvidia.__path__:
        for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
            bin_dir = Path(base) / sub / "bin"
            if bin_dir.is_dir():
                os.add_dll_directory(str(bin_dir))
                os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                added.append(str(bin_dir))
    return added


def audio_seconds(path: Path) -> float:
    import av  # bundled with faster-whisper

    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        if container.duration:
            return container.duration / 1_000_000
        return float(stream.duration * stream.time_base)


_norm_re = re.compile(r"[^a-z0-9' ]+")


def normalise(text: str) -> list[str]:
    return _norm_re.sub(" ", text.lower()).split()


def wer(ref: list[str], hyp: list[str]) -> float:
    # Levenshtein distance over words, O(len(ref) * len(hyp)) memory-light.
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, h in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h))
        prev = cur
    return prev[-1] / max(len(ref), 1)


def run(config: str, audio: Path, reference: str | None, beam: int) -> dict:
    from faster_whisper import WhisperModel

    device, model_name, compute = config.split(":")
    result: dict = {"config": config, "device": device, "model": model_name, "compute": compute}
    try:
        t0 = time.perf_counter()
        model = WhisperModel(model_name, device=device, compute_type=compute)
        result["load_s"] = round(time.perf_counter() - t0, 2)

        t1 = time.perf_counter()
        segments, info = model.transcribe(
            str(audio),
            beam_size=beam,
            vad_filter=True,
            condition_on_previous_text=False,
            word_timestamps=False,
        )
        text = " ".join(s.text.strip() for s in segments)
        elapsed = time.perf_counter() - t1
        secs = audio_seconds(audio)
        result.update(
            {
                "audio_s": round(secs, 1),
                "transcribe_s": round(elapsed, 2),
                "rtf": round(elapsed / secs, 3),
                "speedup_x": round(secs / elapsed, 1),
                "language": info.language,
                "chars": len(text),
            }
        )
        if reference:
            result["wer"] = round(wer(normalise(reference), normalise(text)), 3)
        result["sample"] = text[:160]
        del model
    except Exception as exc:  # report and keep going: a failing config is a finding
        result["error"] = f"{type(exc).__name__}: {exc}"[:300]
    return result


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True, type=Path)
    ap.add_argument("--configs", nargs="+", required=True)
    ap.add_argument("--beam", type=int, default=1, help="1 = greedy, what the live worker will use")
    args = ap.parse_args()

    dlls = ensure_cuda_dlls()
    print(json.dumps({"cuda_dll_dirs": dlls}), flush=True)

    ref_path = args.audio.with_suffix(".txt")
    reference = ref_path.read_text(encoding="utf-8") if ref_path.exists() else None

    for config in args.configs:
        print(json.dumps(run(config, args.audio, reference, args.beam)), flush=True)


if __name__ == "__main__":
    main()
