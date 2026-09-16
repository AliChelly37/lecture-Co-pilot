"""Make the pip-installed NVIDIA runtime DLLs loadable on Windows.

faster-whisper (CTranslate2) needs cublas and cudnn on the DLL search path; the
nvidia-*-cu12 wheels put them under site-packages/nvidia/<lib>/bin. Call once,
before importing faster_whisper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_done: list[str] | None = None


def ensure_cuda_dlls() -> list[str]:
    global _done
    if _done is not None:
        return _done
    added: list[str] = []
    if sys.platform == "win32":
        try:
            import nvidia  # type: ignore[import-not-found]
        except ImportError:
            nvidia = None
        if nvidia is not None:
            for base in nvidia.__path__:
                for sub in ("cublas", "cudnn", "cuda_nvrtc", "cuda_runtime"):
                    bin_dir = Path(base) / sub / "bin"
                    if bin_dir.is_dir():
                        os.add_dll_directory(str(bin_dir))
                        os.environ["PATH"] = f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}"
                        added.append(str(bin_dir))
    _done = added
    return added


def cuda_available() -> bool:
    ensure_cuda_dlls()
    try:
        import ctranslate2

        return ctranslate2.get_cuda_device_count() > 0
    except Exception:
        return False
