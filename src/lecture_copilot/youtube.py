"""YouTube import (D27): catching up on a lecture recording someone posted
publicly, slides included, when the deck itself was never shared.

`download` fetches audio and a capped-resolution video stream to a temp
directory (deleted by the caller right after processing, same rule as a deck
upload, D8); `sample_frames` decodes the video at a low, fixed rate; `detect_cuts`
is pure numpy logic that turns that frame stream into a short list of distinct
slides, so it is tested without a real video. Reading what a slide says is a
vision-model job (pipeline.slides_from_youtube); everything here is
deterministic and free.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

log = logging.getLogger(__name__)


@dataclass
class YoutubeMedia:
    video_id: str
    title: str
    duration_s: float
    audio_path: Path
    video_path: Path | None  # None if the video stream could not be fetched: audio-only import


@dataclass
class Cut:
    t: float
    frame: np.ndarray  # HxWx3 uint8, RGB


def download(url: str, tmp_dir: Path, max_minutes: float, video_height: int) -> YoutubeMedia:
    """Two separate downloads (audio, video-only) rather than one merged file:
    yt-dlp can merge them, but that needs the ffmpeg binary on PATH, and lecture
    audio should not depend on it. A file this browser can play but with no
    visible slides (a screen-share yt-dlp could not split, an audio-only
    source) still produces a transcript; only the deck is missing.
    """
    import yt_dlp

    common = _opts({"outtmpl": str(tmp_dir / "%(id)s.info")})
    try:
        with yt_dlp.YoutubeDL({**common, "skip_download": True}) as probe:
            info = probe.extract_info(url, download=False)
    except yt_dlp.utils.DownloadError as exc:
        raise ValueError(f"could not read that YouTube video ({_first_line(exc)})") from exc
    if info is None or info.get("_type") == "playlist":
        raise ValueError("that link is a playlist, not a single video")
    duration = float(info.get("duration") or 0)
    if duration <= 0:
        raise ValueError("could not determine the video's length (maybe a live stream?)")
    if duration > max_minutes * 60:
        raise ValueError(f"that video is {duration / 60:.0f} minutes long; the limit is {max_minutes:.0f}")

    video_id = info.get("id") or "video"
    title = info.get("title") or video_id

    # YouTube intermittently answers 403 for one stream URL and not the next: retry, then settle for a smaller stream.
    audio_path = _download_one(yt_dlp, url, tmp_dir / "audio.%(ext)s", ["bestaudio/best", "bestaudio/best", "worstaudio/worst"])
    try:
        video_path = _download_one(yt_dlp, url, tmp_dir / "video.%(ext)s", [f"bestvideo[height<={video_height}]/worst", "worstvideo/worst"])
    except ValueError:
        log.warning("no video-only stream for %s; importing audio only, without slides", video_id)
        video_path = None

    return YoutubeMedia(video_id=video_id, title=title, duration_s=duration, audio_path=audio_path, video_path=video_path)


def _opts(extra: dict) -> dict:
    opts = {"quiet": True, "no_warnings": True, "noprogress": True, "noplaylist": True, **extra}
    # yt-dlp solves YouTube's signature challenges with a JavaScript runtime and only enables Deno by
    # default; Node is usually already here (the UI build needs it), and without one some formats vanish.
    if not shutil.which("deno") and shutil.which("node"):
        opts["js_runtimes"] = {"node": {}}
    return opts


def _download_one(yt_dlp, url: str, out_template: Path, formats: list[str]) -> Path:
    last = "no format tried"
    for fmt in formats:
        try:
            with yt_dlp.YoutubeDL(_opts({"format": fmt, "outtmpl": str(out_template)})) as y:
                y.download([url])
        except yt_dlp.utils.DownloadError as exc:
            last = _first_line(exc)
            log.warning("download with format %r failed: %s", fmt, last)
            continue
        matches = sorted(out_template.parent.glob(out_template.name.replace("%(ext)s", "*")))
        if matches:
            return matches[0]
        last = f"download reported success but no file matched {out_template.name}"
    raise ValueError(last)


def _first_line(exc: Exception) -> str:
    return str(exc).splitlines()[0].removeprefix("ERROR: ")[:200]


def sample_frames(video_path: Path, fps: float) -> Iterator[tuple[float, np.ndarray]]:
    """Decode `video_path` at a fixed, low rate. A lecture recording rarely
    changes slides faster than once every few seconds, so 1 fps is plenty and
    keeps a two-hour video to a few thousand small comparisons."""
    import av

    with av.open(str(video_path)) as container:
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        next_t = 0.0
        step = 1.0 / fps if fps > 0 else 1.0
        for frame in container.decode(stream):
            t = float(frame.time or 0.0)
            if t + 1e-6 < next_t:
                continue
            next_t = t + step
            yield t, frame.to_ndarray(format="rgb24")


def _small_gray(frame: np.ndarray, size: tuple[int, int] = (64, 36)) -> np.ndarray:
    return np.asarray(Image.fromarray(frame).convert("L").resize(size), dtype=np.float32) / 255.0


def detect_cuts(
    frames: Iterator[tuple[float, np.ndarray]],
    diff_threshold: float = 0.045,
    min_gap_s: float = 3.0,
) -> Iterator[Cut]:
    """Yield one `Cut` per distinct slide: the first frame, then any frame far
    enough (in time and in pixels) from the slide currently on screen. Pure
    numpy over whatever `frames` provides, so it runs the same on a real
    decode or a synthetic test sequence."""
    last_small: np.ndarray | None = None
    last_cut_t = float("-inf")  # the forced first frame is a starting point, not a detected change: no debounce against it
    for t, frame in frames:
        small = _small_gray(frame)
        if last_small is None:
            yield Cut(t, frame)
            last_small = small
            continue
        diff = float(np.abs(small - last_small).mean())
        if diff >= diff_threshold and t - last_cut_t >= min_gap_s:
            yield Cut(t, frame)
            last_cut_t = t
        # Compared with the previous sampled frame, not the last cut: steady motion (a webcam in
        # the corner) stays under the threshold at every step. The cost is that a very slow
        # fade, changing less than the threshold per second, is not seen as a slide change.
        last_small = small


def to_jpeg(frame: np.ndarray, quality: int = 82) -> bytes:
    import io

    buf = io.BytesIO()
    Image.fromarray(frame).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
