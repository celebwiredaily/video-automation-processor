"""Deterministic synthetic test media.

Used by ``python -m video_pipeline prepare-test`` and by the pytest suite, so
CI exercises the real renderer against real encoded media rather than mocks.

Every clip is a flat, unmistakably identifiable colour with a moving element,
which lets tests assert *which* clip is on screen at a given time instead of
merely checking that FFmpeg exited zero.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from . import ffmpeg

# Distinct, unmistakable primaries plus dark/light neutrals for edge cases.
PALETTE = {
    "red": (0xE8, 0x10, 0x10),
    "blue": (0x10, 0x20, 0xE8),
    "green": (0x10, 0xD0, 0x20),
    "white": (0xF0, 0xF0, 0xF0),
    "grey": (0x80, 0x80, 0x80),
}


def base_luma(spec, channel: str) -> str:
    """Render one channel of a palette entry as a geq-compatible hex literal."""
    if isinstance(spec, tuple):
        index = {"r": 0, "g": 1, "b": 2}[channel]
        return f"0x{spec[index]:02X}"
    return str(spec)


def dominant_rgb(spec) -> tuple[int, int, int]:
    if isinstance(spec, tuple):
        return spec
    raise ValueError(f"No dominant colour known for {spec!r}")


def _run(args: Sequence[str], stage: str) -> None:
    ffmpeg.run(args, stage=stage, check=True, timeout=600)


def make_clip(
    path: Path,
    *,
    colour: str,
    seconds: float,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    moving: bool = True,
    audio: bool = False,
    audio_tone: int = 440,
) -> Path:
    """A solid-colour clip with optional motion and optional audio."""
    path.parent.mkdir(parents=True, exist_ok=True)

    # Base colour plus a travelling bright stripe. The stripe is baked into the
    # geq expression (evaluated per pixel per frame) so motion is guaranteed,
    # which is what lets the validator tell a playing clip from a frozen one.
    base = PALETTE.get(colour, colour)
    if moving:
        stripe = max(12, width // 16)
        step = max(3, width // 40)
        mask = f"lt(mod(X+N*{step}\\,W)\\,{stripe})"
        eq = (
            f"r='{base_luma(base, 'r')}+130*{mask}'"
            f":g='{base_luma(base, 'g')}+130*{mask}'"
            f":b='{base_luma(base, 'b')}+130*{mask}'"
        )
    else:
        eq = base

    args = [
        ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-f", "lavfi", "-i", f"nullsrc=s={width}x{height}:r={fps}:d={seconds}",
    ]
    if audio:
        args += [
            "-f", "lavfi", "-i",
            f"sine=frequency={audio_tone}:duration={seconds}:sample_rate=48000",
        ]
    args += [
        "-vf", f"geq={eq}",
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "18",
        "-pix_fmt", "yuv420p", "-r", str(fps),
    ]
    if audio:
        args += ["-c:a", "aac", "-b:a", "128k", "-ar", "48000", "-ac", "2", "-shortest"]
    else:
        args += ["-an"]
    args += [str(path)]
    _run(args, stage=f"make_clip[{colour}]")
    return path


def make_image(
    path: Path,
    *,
    colour: str = "green",
    width: int = 640,
    height: int = 360,
) -> Path:
    """A still image with a solid colour, used to test image handling."""
    path.parent.mkdir(parents=True, exist_ok=True)
    eq = (
        f"r='0x{dominant_rgb(PALETTE.get(colour, colour))[0]:02X}'"
        f":g='0x{dominant_rgb(PALETTE.get(colour, colour))[1]:02X}'"
        f":b='0x{dominant_rgb(PALETTE.get(colour, colour))[2]:02X}'"
    )
    _run(
        [
            ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-f", "lavfi", "-i", f"nullsrc=s={width}x{height}:r=1:d=1",
            "-vf", f"geq={eq}",
            "-frames:v", "1",
            str(path),
        ],
        stage="make_image",
    )
    return path


def make_voiceover(
    path: Path,
    *,
    seconds: float,
    frequency: int = 320,
    sample_rate: int = 48000,
    channels: int = 2,
) -> Path:
    """A spoken-word-shaped test signal.

    Bursts of tone with silence between them, so the audio validator sees real
    energy variation and can catch a truncated or silent render.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    # Bursts of tone with near-silence between them: real energy variation, so
    # the audio validator can distinguish a good render from a silent or
    # truncated one.
    period = 1.0
    gate = f"0.85*lt(mod(t\\,{period})\\,0.62)"
    _run(
        [
            ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-f", "lavfi",
            "-i", f"sine=frequency={frequency}:sample_rate={sample_rate}:duration={seconds}",
            "-af", f"volume=volume='{gate}':eval=frame",
            "-c:a", "pcm_s16le",
            "-ar", str(sample_rate),
            "-ac", str(channels),
            str(path),
        ],
        stage="make_voiceover",
    )
    return path


@dataclass
class SegmentSpec:
    asset: str
    start: float
    end: float
    fit: str = "contain"
    audio: str = "none"
    source_start: float | None = None
    on_short: str | None = None


def build_test_project(
    directory: Path,
    *,
    seconds: float = 6.0,
    width: int = 1280,
    height: int = 720,
    fps: int = 30,
    include_image: bool = True,
    include_portrait: bool = False,
    include_silent_clip: bool = True,
    clips: Iterable[tuple[str, float]] | None = None,
) -> list[Path]:
    """Write a complete, renderable synthetic project to ``directory``."""
    directory = Path(directory)
    if directory.exists():
        import shutil

        shutil.rmtree(directory)
    assets = directory / "assets"
    audio_dir = directory / "audio"
    captions_dir = directory / "captions"
    for path in (assets, audio_dir, captions_dir):
        path.mkdir(parents=True, exist_ok=True)

    created: list[Path] = []

    # Default: three flat clips, one per third of the timeline.
    if clips is None:
        third = seconds / 3.0
        clips = [("red", third), ("blue", third), ("green", third)]

    for colour, clip_seconds in clips:
        created.append(
            make_clip(
                assets / f"{colour}.mp4",
                colour=colour,
                seconds=max(1.0, clip_seconds),
                width=640,
                height=360,
                fps=fps,
            )
        )

    if include_image:
        created.append(make_image(assets / "still.jpg", colour="green"))

    if include_portrait:
        created.append(
            make_clip(
                assets / "portrait.mp4",
                colour="white",
                seconds=max(1.0, seconds / 4.0),
                width=360,
                height=640,
                fps=fps,
            )
        )

    if include_silent_clip:
        created.append(
            make_clip(
                assets / "silent.mp4",
                colour="grey",
                seconds=max(1.0, seconds / 4.0),
                width=640,
                height=360,
                fps=fps,
                audio=False,
            )
        )

    created.append(
        make_voiceover(audio_dir / "voiceover.wav", seconds=seconds)
    )

    # Timeline: lay the clips end to end, filling exactly `seconds`.
    entries: list[dict] = []
    cursor = 0.0
    sources = [f"{colour}.mp4" for colour, _ in clips]
    if include_image:
        sources.append("still.jpg")
    if include_portrait:
        sources.append("portrait.mp4")
    if include_silent_clip:
        sources.append("silent.mp4")

    span = seconds / len(sources)
    for asset in sources:
        end = cursor + span
        entries.append(
            {
                "start": round(cursor, 4),
                "end": round(end, 4),
                "asset": f"assets/{asset}",
                "fit": "contain",
            }
        )
        cursor = end
    # Absorb rounding drift so the timeline ends exactly on the voiceover.
    entries[-1]["end"] = round(seconds, 4)

    timeline_path = directory / "timeline.yaml"
    timeline_path.write_text(
        "timeline:\n"
        + "".join(
            f"  - start: {e['start']}\n"
            f"    end: {e['end']}\n"
            f"    asset: {e['asset']}\n"
            f"    fit: {e['fit']}\n"
            for e in entries
        ),
        encoding="utf-8",
    )
    created.append(timeline_path)

    # Captions: one cue per segment, so caption sampling has known intervals.
    cues = [
        {
            "start": e["start"],
            "end": e["end"],
            "text": Path(e["asset"]).stem.replace("_", " ").title(),
        }
        for e in entries
    ]
    captions_path = captions_dir / "segments.json"
    captions_path.write_text(json.dumps(cues, indent=2), encoding="utf-8")
    created.append(captions_path)

    config_path = directory / "config.yaml"
    config_path.write_text(
        "output:\n"
        f"  width: {width}\n"
        f"  height: {height}\n"
        f"  fps: {fps}\n"
        "duration:\n"
        "  target_seconds: 300\n"
        "  min_seconds: 1\n"
        "  max_seconds: 3600\n",
        encoding="utf-8",
    )
    created.append(config_path)

    project_path = directory / "project.yaml"
    project_path.write_text(
        f"name: {directory.name}\n"
        "voiceover: audio/voiceover.wav\n"
        "captions:\n"
        "  file: captions/segments.json\n"
        "storage:\n"
        "  backend: local\n",
        encoding="utf-8",
    )
    created.append(project_path)

    (directory / "script.md").write_text(
        "# Test project\n\nSynthetic narration used to exercise the renderer.\n",
        encoding="utf-8",
    )

    return created