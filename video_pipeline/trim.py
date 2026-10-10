"""Automatic source preparation: from an arbitrary source video to a segment of
exactly the configured duration.

The pipeline renders what its timeline points at. Nothing stops that timeline
from pointing at a raw 8-minute interview, so this module makes the opposite
choice explicit and repeatable: inspect a source with FFprobe, cut exactly
``seconds`` from timestamp 00:00:00, and verify the result with FFprobe again.

Rules this module enforces, deliberately:

* The source file is opened read-only and never rewritten. The prepared segment
  is written to a separate generated working directory.
* A source shorter than the target fails. The footage is not looped, duplicated
  or slowed down to fill the duration - that would be a fabricated video.
* No filters. No scaling, no zooms, no pans, no transitions, no colour
  changes, no aspect-ratio change. The only transform is a cut, plus a
  frame-rate conform when the target frame rate differs from the source's, and
  a plain audio re-encode when the source carries audio.
* Audio is carried through when the source has it and is *not* invented when it
  does not. A silent source is reported, not filled with a synthetic tone.
* The prepared segment is verified against the actual encoded file before it is
  handed to the renderer, and the rendered file is verified again afterwards.
  Nothing here trusts an FFmpeg exit code.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from . import ffmpeg
from .config import Config
from .errors import AssetError, ValidationError
from .media import MediaInfo, probe

# The default target. Projects may override it; the pipeline default is ~5
# minutes and this module exists precisely because "approximately five minutes"
# was not good enough.
DEFAULT_TARGET_SECONDS = 300.0

# Directory name used under a project's work directory for generated media.
SOURCE_DIRNAME = "source"
PREPARED_BASENAME = "source_{seconds}s"

# Text used for the generated captions when no transcript is available. A
# deterministic caption track keeps the caption/audio relationship testable
# without pulling a speech model into CI.
_PLACEHOLDER_LINES = [
    "This caption is generated",
    "from the voiceover timing",
    "so captions and audio",
    "share one clock by",
    "construction, not by hand",
    "tuned offsets.",
]


@dataclass(frozen=True)
class TrimResult:
    """What was cut, from what, and what was measured afterwards."""

    source: Path
    clip: Path
    source_duration: float
    clip_duration: float
    target_seconds: float
    fps: float
    frames: int
    has_audio: bool
    action: str  # "trimmed" | "verbatim"
    source_width: int
    source_height: int
    clip_width: int
    clip_height: int
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": str(self.source),
            "clip": str(self.clip),
            "source_duration_seconds": round(self.source_duration, 6),
            "clip_duration_seconds": round(self.clip_duration, 6),
            "target_seconds": round(self.target_seconds, 6),
            "action": self.action,
            "fps": round(self.fps, 6),
            "frames": self.frames,
            "has_audio": self.has_audio,
            "source_resolution": f"{self.source_width}x{self.source_height}",
            "clip_resolution": f"{self.clip_width}x{self.clip_height}",
            "warnings": list(self.warnings),
        }


# --------------------------------------------------------------------- probe


def frame_count(path: Path) -> int:
    """Count decoded video frames.

    ``nb_frames`` is a container hint and is absent or wrong often enough that
    trusting it would undermine the point of a strict check, so the frames are
    actually decoded and counted with ``-count_frames``.

    A count that cannot be obtained is an error, never a zero: returning 0 and
    letting a caller compare it against an expectation would turn an
    unmeasurable file into either a silent pass or a misleading failure.
    """
    argv = [
        ffmpeg.ffprobe_bin(), "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = ffmpeg.run(argv, stage="count_frames", timeout=1800, check=False)
    if proc.returncode != 0:
        raise ValidationError(
            "Could not count video frames.",
            file=str(path),
            stderr_tail=(proc.stderr or "").strip()[-400:],
            hint="ffprobe must support -count_frames.",
        )
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if line and line != "N/A":
            try:
                return int(line)
            except ValueError:
                continue
    raise ValidationError(
        "ffprobe reported no decodable frame count.", file=str(path)
    )


def stream_duration(path: Path, kind: str) -> float:
    """Duration of the first audio (``a``) or video (``v``) stream.

    Read from the stream rather than the container so a long audio tail or an
    MP4 edit list cannot hide a truncated video stream (or the reverse).
    """
    argv = [
        ffmpeg.ffprobe_bin(), "-v", "error",
        "-select_streams", f"{kind}:0",
        "-show_entries", "stream=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    proc = ffmpeg.run(argv, stage=f"probe_{kind}_duration", timeout=120, check=False)
    if proc.returncode != 0:
        return 0.0
    for line in (proc.stdout or "").splitlines():
        try:
            return float(line.strip())
        except ValueError:
            continue
    return 0.0


def probe_media(path: Path, *, stage: str = "probe_source") -> MediaInfo:
    """Probe a source and refuse anything that is not usable footage."""
    if not path.exists():
        raise AssetError("Source video does not exist.", source=str(path))
    info = probe(str(path))
    if info.kind == "audio" or not info.has_video:
        raise AssetError(
            "Source is not a video with a video stream.",
            source=str(path),
            kind=info.kind,
            hint="Point `source:` at a video file, not an audio-only asset.",
        )
    if info.duration <= 0:
        raise AssetError(
            "Source reports no duration; it may be corrupt or still-only.",
            source=str(path),
        )
    return info


# --------------------------------------------------------------------- locate


def locate_source(project_root: Path, explicit: str | None = None) -> Path:
    """Find the configured source video for a project.

    Order: an explicit path, then ``source:`` in ``project.yaml``, then the
    usual media locations. Ambiguity is an error rather than a coin flip.
    """
    root = Path(project_root)
    if explicit:
        candidate = Path(explicit)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate.resolve()
        raise AssetError("Configured source video not found.", source=str(candidate))

    configured = _source_from_project_yaml(root)
    if configured:
        candidate = Path(configured)
        if not candidate.is_absolute():
            candidate = root / candidate
        if candidate.is_file():
            return candidate.resolve()
        raise AssetError(
            "`source.video` in project.yaml does not exist.", source=str(candidate)
        )

    found = sorted(
        p
        for p in (*(root / "assets").glob("*"), *(root.glob("*.mp4")))
        if p.is_file() and p.suffix.lower() in {".mp4", ".mov", ".mkv", ".webm", ".m4v"}
    )
    if not found:
        raise AssetError(
            "No source video found.",
            searched=str(root),
            hint="Set `source: {video: <path>}` in project.yaml or pass --source.",
        )
    if len(found) > 1:
        raise AssetError(
            "Several candidate source videos found; configure one explicitly.",
            candidates=", ".join(p.name for p in found),
        )
    return found[0].resolve()


def _source_from_project_yaml(root: Path) -> str | None:
    path = root / "project.yaml"
    if not path.is_file():
        return None
    try:
        import yaml

        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return None
    block = data.get("source")
    if isinstance(block, str) and block.strip():
        return block.strip()
    if isinstance(block, dict):
        value = block.get("video") or block.get("path")
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


# ----------------------------------------------------------------------- trim


def prepare_segment(
    source: Path,
    dest_dir: Path,
    *,
    seconds: float = DEFAULT_TARGET_SECONDS,
    fps: int = 30,
    config: Config | None = None,
    stage_timeout: int = 3600,
) -> TrimResult:
    """Cut exactly ``seconds`` from the start of ``source`` into ``dest_dir``.

    Raises ValidationError when the source cannot supply the requested duration.
    The source is only ever read.
    """
    if seconds <= 0:
        raise ValidationError(
            "Requested source duration must be positive.", seconds=str(seconds)
        )

    source = Path(source)
    info = probe_media(source)
    config = config or Config.from_mapping({})
    out = config.output

    target_fps = float(fps or out.fps)
    one_frame = 1.0 / target_fps

    if info.duration + one_frame < seconds:
        raise ValidationError(
            "The source video is shorter than the requested duration.",
            source=source.name,
            source_duration=f"{info.duration:.3f}s",
            requested=f"{seconds:.3f}s",
            shortfall=f"{seconds - info.duration:.3f}s",
            hint="The source is not looped, duplicated or slowed down to reach the "
            "target: that would fabricate footage. Supply a source at least "
            f"{seconds:.0f}s long, or lower `duration.target_seconds`.",
        )

    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    clip = dest_dir / f"{PREPARED_BASENAME.format(seconds=_format_seconds(seconds))}.mp4"

    warnings: list[str] = []
    if not info.has_audio:
        warnings.append(
            "source has no audio stream; the prepared clip will have none either"
        )

    if abs(info.duration - seconds) <= one_frame:
        # Already the right length. Copy it byte for byte rather than running a
        # pointless re-encode that could only introduce generation loss.
        shutil.copy2(source, clip)
        action = "verbatim"
    else:
        action = "trimmed"
        _cut(source, clip, seconds=seconds, fps=target_fps, config=config,
             has_audio=info.has_audio, timeout=stage_timeout)
        if info.fps and abs(info.fps - target_fps) > 0.5:
            warnings.append(
                f"source runs at {info.fps:.3f}fps; the prepared segment was conformed "
                f"to {target_fps:.0f}fps so the segment is frame-exact"
            )

    clip_info = probe_media(clip, stage="probe_clip")
    frames = frame_count(clip)

    measured_tolerance = max(one_frame, 1e-3)
    if abs(clip_info.duration - seconds) > measured_tolerance:
        clip.unlink(missing_ok=True)
        raise ValidationError(
            "The prepared segment did not come out at the requested duration.",
            source=source.name,
            requested=f"{seconds:.3f}s",
            measured=f"{clip_info.duration:.3f}s",
            hint="This is a measured failure, not an estimate: the encoded file was "
            "probed and rejected.",
        )
    expected_frames = int(round(seconds * target_fps))
    if frames != expected_frames:
        clip.unlink(missing_ok=True)
        raise ValidationError(
            "The prepared segment has the wrong frame count.",
            source=source.name,
            expected_frames=expected_frames,
            measured_frames=frames,
            fps=f"{target_fps:.0f}",
        )
    if clip_info.has_audio != info.has_audio:
        clip.unlink(missing_ok=True)
        raise ValidationError(
            "Audio streams were not carried through the trim.",
            source_has_audio=str(info.has_audio),
            clip_has_audio=str(clip_info.has_audio),
        )
    if (clip_info.width, clip_info.height) != (info.width, info.height):
        clip.unlink(missing_ok=True)
        raise ValidationError(
            "The prepared segment changed resolution.",
            source=f"{info.width}x{info.height}",
            clip=f"{clip_info.width}x{clip_info.height}",
            hint="Trimming must not scale or crop the source.",
        )

    return TrimResult(
        source=source,
        clip=clip,
        source_duration=info.duration,
        clip_duration=clip_info.duration,
        target_seconds=seconds,
        fps=target_fps,
        frames=frames,
        has_audio=clip_info.has_audio,
        action=action,
        source_width=info.width,
        source_height=info.height,
        clip_width=clip_info.width,
        clip_height=clip_info.height,
        warnings=warnings,
    )


def _cut(
    source: Path,
    clip: Path,
    *,
    seconds: float,
    fps: float,
    config: Config,
    has_audio: bool,
    timeout: int,
) -> None:
    """Encode the first ``seconds`` of ``source``.

    ``-ss 0`` keeps the seek an input option and the cut frame-accurate. The
    output carries an explicit time base derived from the frame rate so the
    container duration lands on the target rather than near it.
    """
    out = config.output
    argv: list[str] = [
        ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
        "-ss", "0",
        "-i", str(source),
        "-t", f"{seconds:.6f}",
        "-map", "0:v:0",
    ]
    if has_audio:
        argv += ["-map", "0:a:0"]
    else:
        argv += ["-an"]

    argv += [
        "-c:v", out.video_codec,
        "-crf", str(config.intermediate_crf),
        "-preset", config.intermediate_preset,
        "-pix_fmt", out.pix_fmt,
        "-r", f"{fps:.6f}",
        # One time base unit == one millisecond divided by ... keep it an exact
        # multiple of the frame duration: 1/fps seconds in timescale units.
        "-video_track_timescale", str(int(round(fps * 1000))),
    ]
    if has_audio:
        argv += [
            "-c:a", out.audio_codec,
            "-b:a", out.audio_bitrate,
            "-ar", str(out.audio_rate),
            "-ac", str(out.audio_channels),
        ]
    argv += ["-movflags", "+faststart", str(clip)]
    ffmpeg.run(argv, stage="trim_source", timeout=timeout)


def _format_seconds(seconds: float) -> str:
    text = f"{seconds:.3f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


# ------------------------------------------------------------ strict measure


def measure_final(
    path: Path,
    *,
    target_seconds: float,
    fps: int,
    tolerance_seconds: float | None = None,
    require_audio: bool = True,
) -> dict[str, Any]:
    """Probe a rendered file and decide whether it meets a strict duration spec.

    The rendered MP4 is the authority. Three independent facts are read with
    FFprobe and all of them must agree:

    1. container duration within ``tolerance_seconds`` of the target;
    2. decoded frame count equal to ``round(target_seconds * fps)`` - this is
       what actually pins the length, because container timestamps are metadata;
    3. audio stream duration within tolerance, when audio is required.

    The default tolerance is one frame period. At 30 fps that is 33.3 ms, which
    is strict enough to fail a 5-minute render that came out at 299.7s or
    300.4s, and loose enough to tolerate nothing that is not a real frame
    boundary.
    """
    if tolerance_seconds is None:
        tolerance_seconds = 1.0 / float(fps)

    path = Path(path)
    if not path.is_file():
        raise ValidationError("Rendered video not found.", file=str(path))

    info = probe_media(path, stage="probe_final")
    frames = frame_count(path)
    expected_frames = int(round(target_seconds * fps))
    delta = info.duration - target_seconds

    video_duration = stream_duration(path, "v")
    audio_duration = stream_duration(path, "a") if info.has_audio else 0.0

    checks: list[dict[str, Any]] = [
        {
            "name": "container_duration",
            "measured": round(info.duration, 6),
            "expected": round(target_seconds, 6),
            "tolerance": round(tolerance_seconds, 6),
            "delta": round(delta, 6),
            "ok": abs(delta) <= tolerance_seconds,
        },
        {
            "name": "video_frame_count",
            "measured": frames,
            "expected": expected_frames,
            "tolerance": 0,
            "ok": frames == expected_frames,
        },
    ]
    if video_duration > 0:
        checks.append(
            {
                "name": "video_stream_duration",
                "measured": round(video_duration, 6),
                "expected": round(target_seconds, 6),
                "tolerance": round(tolerance_seconds, 6),
                "delta": round(video_duration - target_seconds, 6),
                "ok": abs(video_duration - target_seconds) <= tolerance_seconds,
            }
        )
    if info.has_audio:
        # One AAC frame of slack: audio is packetised, so a stream that ends
        # within a single frame of the target is not a timing error.
        audio_slack = tolerance_seconds + (1024.0 / info.sample_rate if info.sample_rate else 0.0)
        checks.append(
            {
                "name": "audio_duration",
                "measured": round(audio_duration, 6),
                "expected": round(target_seconds, 6),
                "tolerance": round(audio_slack, 6),
                "delta": round(audio_duration - target_seconds, 6),
                "ok": abs(audio_duration - target_seconds) <= audio_slack,
            }
        )
    elif require_audio:
        checks.append(
            {
                "name": "audio_present",
                "measured": False,
                "expected": True,
                "tolerance": 0,
                "ok": False,
            }
        )

    report = {
        "path": str(path),
        "target_seconds": round(target_seconds, 6),
        "fps": float(fps),
        "tolerance_seconds": round(tolerance_seconds, 6),
        "measured_duration": round(info.duration, 6),
        "measured_delta": round(delta, 6),
        "frames": frames,
        "expected_frames": expected_frames,
        "fps_measured": round(info.fps, 6),
        "width": info.width,
        "height": info.height,
        "video_codec": info.video_codec,
        "pix_fmt": info.pix_fmt,
        "video_stream_duration": round(video_duration, 6),
        "has_audio": info.has_audio,
        "audio_duration": round(audio_duration, 6),
        "checks": checks,
        "ok": all(check["ok"] for check in checks),
    }
    return report


def enforce_final(
    path: Path,
    *,
    target_seconds: float,
    fps: int,
    tolerance_seconds: float | None = None,
    require_audio: bool = True,
) -> dict[str, Any]:
    """``measure_final`` as a gate: writes nothing, raises on failure."""
    report = measure_final(
        path,
        target_seconds=target_seconds,
        fps=fps,
        tolerance_seconds=tolerance_seconds,
        require_audio=require_audio,
    )
    if not report["ok"]:
        failures = [c for c in report["checks"] if not c["ok"]]
        raise ValidationError(
            "The rendered video does not meet the strict duration requirement.",
            file=str(path),
            target=f"{target_seconds:.3f}s",
            measured=f"{report['measured_duration']:.3f}s",
            frames=f"{report['frames']} (expected {report['expected_frames']})",
            failed="; ".join(c["name"] for c in failures),
            hint="Every number here was read from the encoded file with FFprobe.",
        )
    return report


def write_report(report: dict[str, Any], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


# ------------------------------------------------------------ project wiring


def placeholder_cues(seconds: float, *, cue_seconds: float = 5.0) -> list[dict]:
    """Deterministic caption cues covering ``seconds``.

    Anchored to the same clock as the voiceover, so a caption is never visible
    outside its own audio window.
    """
    cues: list[dict] = []
    cursor = 0.0
    index = 0
    while cursor < seconds - 1e-6:
        end = min(seconds, cursor + cue_seconds)
        text = _PLACEHOLDER_LINES[index % len(_PLACEHOLDER_LINES)]
        cues.append(
            {
                "start": round(cursor, 3),
                "end": round(end, 3),
                "text": f"{text} ({index + 1})",
            }
        )
        cursor = end
        index += 1
    return cues


def extract_voiceover(clip: Path, destination: Path, *, seconds: float) -> Path:
    """Extract the prepared segment's own audio as the project's voiceover.

    This is the source's audio, not a replacement: the narration the timeline
    is synchronised against is the narration that is actually in the footage.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg.run(
        [
            ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-i", str(clip),
            "-vn",
            "-t", f"{seconds:.6f}",
            "-c:a", "pcm_s16le",
            "-ar", "48000",
            "-ac", "2",
            str(destination),
        ],
        stage="extract_voiceover",
        timeout=1800,
    )
    return destination


def build_source_project(
    project_dir: Path,
    *,
    source: Path | None = None,
    seconds: float = DEFAULT_TARGET_SECONDS,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> tuple[Path, TrimResult]:
    """Materialise a renderable project whose only visual source is the segment.

    The rendered video is therefore, by construction, the trimmed clip - there
    is no configuration in which the renderer can fall back to the untrimmed
    original, because the original is never referenced by the timeline.
    """
    from .testkit import make_source

    project_dir = Path(project_dir)
    if project_dir.exists():
        shutil.rmtree(project_dir)

    assets = project_dir / "assets"
    audio = project_dir / "audio"
    captions_dir = project_dir / "captions"
    for path in (assets, audio, captions_dir):
        path.mkdir(parents=True, exist_ok=True)

    if source is None:
        source = assets / "source.mp4"
        make_source(source, seconds=seconds * 1.6, fps=fps)

    config = Config.from_mapping(
        {
            "output": {"width": width, "height": height, "fps": fps},
            "duration": {
                "target_seconds": seconds,
                "min_seconds": 1.0,
                "max_seconds": 3600.0,
                "tolerance_seconds": 1.0 / fps,
            },
            "visual": {"default_fit": "contain", "on_short": "error"},
            "captions": {"enabled": True},
        }
    )

    work_dir = project_dir / "build"
    result = prepare_segment(
        source,
        work_dir / SOURCE_DIRNAME,
        seconds=seconds,
        fps=fps,
        config=config,
    )

    if not result.has_audio:
        raise ValidationError(
            "The prepared segment has no audio, so there is no narration to "
            "synchronise captions against.",
            source=result.source.name,
            hint="video_pipeline never invents audio. Supply a source with an "
            "audio track, or configure a voiceover explicitly.",
        )

    voiceover = extract_voiceover(result.clip, audio / "voiceover.wav", seconds=seconds)
    voiceover_duration = probe(str(voiceover)).duration
    if abs(voiceover_duration - seconds) > 1.0 / fps:
        raise ValidationError(
            "Extracted voiceover does not match the prepared segment.",
            expected=f"{seconds:.3f}s",
            measured=f"{voiceover_duration:.3f}s",
        )

    relative_clip = result.clip.relative_to(project_dir).as_posix()
    (project_dir / "timeline.yaml").write_text(
        "timeline:\n"
        "  - start: 0.0\n"
        f"    end: {_fmt(seconds)}\n"
        f"    asset: {relative_clip}\n"
        "    fit: contain\n"
        f"    label: {result.source.stem}\n",
        encoding="utf-8",
    )

    import json as _json

    cues = placeholder_cues(seconds)
    (captions_dir / "segments.json").write_text(
        _json.dumps(cues, indent=2), encoding="utf-8"
    )

    (project_dir / "config.yaml").write_text(
        "output:\n"
        f"  width: {width}\n"
        f"  height: {height}\n"
        f"  fps: {fps}\n"
        "duration:\n"
        f"  target_seconds: {_fmt(seconds)}\n"
        "  min_seconds: 1\n"
        "  max_seconds: 3600\n"
        f"  tolerance_seconds: {1.0 / fps:.6f}\n"
        "visual:\n"
        "  default_fit: contain\n"
        "  on_short: error\n"
        "captions:\n"
        "  enabled: true\n",
        encoding="utf-8",
    )

    (project_dir / "project.yaml").write_text(
        f"name: {project_dir.name}\n"
        "voiceover: audio/voiceover.wav\n"
        "captions:\n"
        "  file: captions/segments.json\n"
        "output_dir: output\n"
        "work_dir: build\n"
        f"source:\n"
        f"  video: {result.source.relative_to(project_dir).as_posix() if _inside(result.source, project_dir) else result.source}\n"
        f"  seconds: {_fmt(seconds)}\n"
        "storage:\n"
        "  backend: local\n",
        encoding="utf-8",
    )

    (project_dir / "script.md").write_text(
        "# Source-derived project\n\n"
        f"Visual source: `{result.source.name}` "
        f"({result.source_duration:.3f}s), trimmed automatically to "
        f"{result.target_seconds:.3f}s ({result.frames} frames at "
        f"{result.fps:.0f}fps). The prepared segment in `build/source/` is the "
        "only asset the timeline references.\n",
        encoding="utf-8",
    )

    return project_dir, result


def _inside(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _fmt(value: float) -> str:
    text = f"{float(value):.3f}".rstrip("0").rstrip(".")
    return text or "0"
