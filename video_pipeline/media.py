"""Media probing: what an asset actually is.

Validation depends on facts about the source (is it a video or a still? is it
long enough? does it carry audio?), so those facts are read with ffprobe once
and cached.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

from . import ffmpeg
from .errors import AssetError

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".avif"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".m4v", ".avi", ".mpg", ".mpeg"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus"}

STILL_SUFFIXES = IMAGE_SUFFIXES | {".gif"}


@dataclass(frozen=True)
class MediaInfo:
    path: Path
    kind: str  # "video" | "image" | "audio" | "unknown"
    duration: float
    width: int
    height: int
    fps: float
    pix_fmt: str
    video_codec: str
    audio_codec: str
    sample_rate: int
    channels: int
    has_audio: bool
    has_video: bool

    @property
    def is_still(self) -> bool:
        return self.kind == "image"

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0

    def describe(self) -> str:
        if self.kind == "image":
            return f"image {self.width}x{self.height}"
        if self.kind == "audio":
            return f"audio {self.duration:.2f}s {self.sample_rate}Hz"
        parts = [f"video {self.width}x{self.height}"]
        if self.fps:
            parts.append(f"{self.fps:.3f}fps")
        if self.duration:
            parts.append(f"{self.duration:.2f}s")
        parts.append(self.pix_fmt or "?")
        if self.has_audio:
            parts.append(f"+audio {self.sample_rate}Hz/{self.channels}ch")
        return " ".join(parts)


def _fraction_to_float(value: str | None) -> float:
    if not value:
        return 0.0
    try:
        return float(Fraction(value))
    except (ValueError, ZeroDivisionError):
        return 0.0


def classify(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        return "image"
    if suffix in VIDEO_SUFFIXES:
        return "video"
    if suffix in AUDIO_SUFFIXES:
        return "audio"
    if suffix in {".gif"}:
        return "image"
    return "unknown"


def _probe_cached(path_str: str, stamp: tuple[int, int]) -> MediaInfo:
    return _probe_uncached(path_str)


@lru_cache(maxsize=512)
def probe(path_str: str) -> MediaInfo:
    """Probe an asset and cache the result.

    The cache key includes size and mtime so a regenerated test asset is
    re-probed instead of returning a stale answer.
    """
    path = Path(path_str)
    stat = path.stat()
    return _probe_cached(path_str, (stat.st_size, stat.st_mtime_ns))


def clear_cache() -> None:
    """Drop cached probe results (used by tests that rewrite fixtures)."""
    probe.cache_clear()


def _probe_uncached(path_str: str) -> MediaInfo:
    """Probe an asset once per process and cache the result."""
    path = Path(path_str)
    if not path.exists():
        raise AssetError("Asset does not exist.", asset=str(path))
    if not path.is_file():
        raise AssetError("Asset is not a regular file.", asset=str(path))
    if path.stat().st_size == 0:
        raise AssetError("Asset file is empty (0 bytes).", asset=str(path))

    suffix_kind = classify(path)
    data = ffmpeg.probe_json(path)
    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio = next((s for s in streams if s.get("codec_type") == "audio"), None)

    duration = 0.0
    for candidate in (
        fmt.get("duration"),
        (video or {}).get("duration"),
        (audio or {}).get("duration"),
    ):
        try:
            if candidate is not None:
                duration = float(candidate)
                if duration > 0:
                    break
        except (TypeError, ValueError):
            continue

    width = int((video or {}).get("width") or 0)
    height = int((video or {}).get("height") or 0)
    fps = _fraction_to_float((video or {}).get("avg_frame_rate")) or _fraction_to_float(
        (video or {}).get("r_frame_rate")
    )

    kind = suffix_kind
    if kind == "unknown":
        if video is not None:
            kind = "video"
        elif audio is not None:
            kind = "audio"

    # A .gif or .mp4 may in practice be a single still; duration 0 with one
    # frame is still-image behaviour for our purposes.
    if kind == "video" and duration <= 0 and fps <= 0:
        kind = "image"

    return MediaInfo(
        path=path,
        kind=kind,
        duration=duration,
        width=width,
        height=height,
        fps=fps,
        pix_fmt=str((video or {}).get("pix_fmt") or ""),
        video_codec=str((video or {}).get("codec_name") or ""),
        audio_codec=str((audio or {}).get("codec_name") or ""),
        sample_rate=int((audio or {}).get("sample_rate") or 0),
        channels=int((audio or {}).get("channels") or 0),
        has_audio=audio is not None,
        has_video=video is not None,
    )


def is_video_asset(path: Path) -> bool:
    return path.suffix.lower() in VIDEO_SUFFIXES or path.suffix.lower() in {".gif"}


def is_image_asset(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_SUFFIXES