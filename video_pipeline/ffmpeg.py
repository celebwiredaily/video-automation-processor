"""Thin, safe wrapper around the ffmpeg / ffprobe binaries.

Every command is built as a Python list and executed with ``shell=False``.
There is no code path in this package that builds a shell command string, so
filenames containing spaces, quotes, parentheses, ``$`` or Unicode are safe.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, Sequence

from .errors import CapabilityError, FFmpegError

_FFMPEG: str | None = None
_FFPROBE: str | None = None

# Filtergraph metacharacters. Filter arguments are strings, so a value that
# ends up inside one must be escaped. Absolute paths are never placed inside a
# filtergraph (see renderer.py), but this helper exists so that any future call
# site is safe by default rather than safe by memory.
_FILTERGRAPH_SPECIALS = "\\:,'[];"


def escape_filter_value(value: str) -> str:
    """Escape a value destined for use inside a filtergraph argument."""
    out = []
    for ch in value:
        if ch in _FILTERGRAPH_SPECIALS:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def ffmpeg_bin() -> str:
    global _FFMPEG
    if _FFMPEG is None:
        found = shutil.which("ffmpeg")
        if not found:
            raise CapabilityError(
                "ffmpeg was not found on PATH.",
                hint="Install ffmpeg (Debian/Ubuntu: apt-get install ffmpeg) "
                "or place a static build on PATH.",
            )
        _FFMPEG = found
    return _FFMPEG


def ffprobe_bin() -> str:
    global _FFPROBE
    if _FFPROBE is None:
        found = shutil.which("ffprobe")
        if not found:
            raise CapabilityError(
                "ffprobe was not found on PATH.",
                hint="ffprobe ships with ffmpeg; install the ffmpeg package.",
            )
        _FFPROBE = found
    return _FFPROBE


def _stderr_tail(stderr: str, lines: int = 40) -> str:
    if not stderr:
        return ""
    return "\n".join(stderr.strip().splitlines()[-lines:])


def run(
    args: Sequence[str],
    *,
    stage: str,
    timeout: int | None = None,
    log_path: Path | None = None,
    check: bool = True,
    cwd: str | Path | None = None,
    binary: bool = False,
) -> subprocess.CompletedProcess:
    """Run an ffmpeg-family command given as an argument array.

    ``binary=True`` captures stdout as bytes, which is required whenever the
    output is raw PCM or raw video rather than text.
    """
    argv = [str(a) for a in args]
    try:
        proc = subprocess.run(
            argv,
            capture_output=True,
            text=not binary,
            timeout=timeout,
            check=False,
            cwd=cwd,
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"Timed out after {timeout}s.",
            stage=stage,
            command=argv,
            returncode=None,
        ) from exc
    except FileNotFoundError as exc:
        raise CapabilityError(f"Executable not found: {argv[0]}") from exc

    if log_path is not None:
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            with log_path.open("a", encoding="utf-8", errors="replace") as handle:
                handle.write(f"\n$ {' '.join(argv)}\n")
                handle.write(
                    proc.stderr.decode("utf-8", "replace") if binary else (proc.stderr or "")
                )
        except OSError:
            # Never let diagnostics-writing fail a render.
            pass

    if check and proc.returncode != 0:
        stderr_text = (
            proc.stderr.decode("utf-8", "replace") if binary else (proc.stderr or "")
        )
        raise FFmpegError(
            f"{Path(argv[0]).name} failed.",
            stage=stage,
            command=argv,
            stderr_tail=_stderr_tail(stderr_text),
            returncode=proc.returncode,
        )
    return proc


def probe_json(path: Path, *, stage: str = "probe", extra: Iterable[str] = ()) -> dict:
    """Run ffprobe and return parsed JSON."""
    argv = [
        ffprobe_bin(),
        "-v", "error",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        *extra,
        str(path),
    ]
    proc = run(argv, stage=stage, timeout=120)
    try:
        return json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise FFmpegError(
            "Could not parse ffprobe output.",
            stage=stage,
            asset=str(path),
            stdout_tail=(proc.stdout or "")[-500:],
        ) from exc


def probe_value(path: Path, entries: str, *, stage: str = "probe") -> str:
    """Read a single ffprobe value, returning "" when unavailable."""
    argv = [
        ffprobe_bin(), "-v", "error",
        "-show_entries", entries,
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = run(argv, stage=stage, timeout=120, check=False)
    if proc.returncode != 0:
        return ""
    return (proc.stdout or "").strip()


class FilterCapabilities:
    """What this ffmpeg build can actually do, verified rather than assumed."""

    def __init__(self) -> None:
        proc = run(
            [ffmpeg_bin(), "-hide_banner", "-filters"],
            stage="capabilities",
            timeout=60,
            check=False,
        )
        text = proc.stdout or ""
        self.filters: set[str] = set()
        for line in text.splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.startswith("Filters:") and "->" in line:
                self.filters.add(parts[1])

        proc = run(
            [ffmpeg_bin(), "-hide_banner", "-encoders"],
            stage="capabilities",
            timeout=60,
            check=False,
        )
        self.encoders: set[str] = set()
        for line in (proc.stdout or "").splitlines():
            parts = line.split()
            if len(parts) >= 2 and not line.startswith("Encoders:"):
                self.encoders.add(parts[1])

        proc = run([ffmpeg_bin(), "-version"], stage="capabilities", timeout=60, check=False)
        self.version = ((proc.stdout or "").splitlines() or ["unknown"])[0]

    def require(self, *, filters: Sequence[str] = (), encoders: Sequence[str] = ()) -> None:
        missing_filters = sorted(f for f in filters if f not in self.filters)
        missing_encoders = sorted(e for e in encoders if e not in self.encoders)
        problems = []
        if missing_filters:
            problems.append(f"missing filters: {', '.join(missing_filters)}")
        if missing_encoders:
            problems.append(f"missing encoders: {', '.join(missing_encoders)}")
        if problems:
            raise CapabilityError(
                "The local ffmpeg build cannot run this pipeline.",
                problems="; ".join(problems),
                version=self.version,
                hint="Ubuntu runners ship an ffmpeg with libass and libx264. "
                "If libass is missing, install ffmpeg via apt rather than a "
                "minimal static build.",
            )


def check_capabilities(*, need_captions: bool) -> FilterCapabilities:
    caps = FilterCapabilities()
    filters = ["scale", "crop", "pad", "fps", "setsar", "format", "concat", "tpad"]
    encoders = ["libx264", "aac"]
    if need_captions:
        filters += ["ass", "subtitles"]
    caps.require(filters=filters, encoders=encoders)
    return caps


def make_even(value: float) -> int:
    """Round to the nearest positive even integer.

    yuv420p requires even dimensions; odd sizes make libx264 fail at encode
    time, far from the cause.
    """
    number = max(2, int(round(value)))
    return number if number % 2 == 0 else number + 1