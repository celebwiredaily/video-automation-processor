"""Post-render validation of the actual rendered MP4.

The rendered file is the authority. FFmpeg's exit code and ffprobe's metadata
are necessary but not sufficient: a file can encode cleanly and still contain
black frames, the wrong clip at the wrong time, or no captions at all.

Checks performed
----------------
Technical (ffprobe)
    container readable, video stream present, audio stream present,
    dimensions, frame rate, codecs, duration within tolerance of the voiceover,
    audio duration, audio is not digitally silent.

Frame level (real decoded pixels)
    a deterministic sample of frames including the first and final frame,
    every visual timeline boundary, and - crucially - points deliberately
    chosen to fall *inside* known caption intervals plus points just outside
    them. Each sample is classified as black / blank / normal, and caption
    pixels are looked for inside the caption band only.

Everything is written to ``validation.json`` plus a diagnostics folder so a
failure in CI is diagnosable without reproducing it locally.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Sequence

from . import ffmpeg
from .config import Config
from .errors import ValidationError
from .media import probe
from .timeline import Timeline

# Defaults; the effective values come from Config.validation so thresholds are
# configurable rather than hard-coded behaviour.
DEFAULT_THRESHOLDS = {
    "black_mean_luma": 8.0,
    "black_max_luma": 24.0,
    "blank_stddev": 1.5,
    "blank_max_luma": 40.0,
    "caption_pixel_ratio": 0.0012,
    "frozen_diff": 0.002,
    "frozen_min_interval": 0.5,
    "silence_rms": 0.0015,
}


@dataclass
class FrameCheck:
    label: str
    time: float
    extracted: bool
    mean_luma: float | None = None
    max_luma: float | None = None
    stddev: float | None = None
    is_black: bool = False
    is_blank: bool = False
    caption_expected: bool = False
    caption_detected: bool = False
    caption_ratio: float = 0.0
    notes: list[str] = field(default_factory=list)
    image: str | None = None


@dataclass
class ValidationResult:
    path: str
    ok: bool
    file_size: int
    duration: float
    width: int
    height: int
    fps: float
    video_codec: str
    pix_fmt: str
    has_video: bool
    has_audio: bool
    audio_codec: str
    audio_sample_rate: int
    audio_channels: int
    audio_duration: float
    audio_rms: float | None
    audio_silent: bool | None
    expected_duration: float | None
    duration_delta: float | None
    frame_checks: list[FrameCheck] = field(default_factory=list)
    black_frame_count: int = 0
    blank_frame_count: int = 0
    frozen_pairs: list[str] = field(default_factory=list)
    caption_checks_total: int = 0
    caption_checks_passed: int = 0
    caption_false_positives: list[str] = field(default_factory=list)
    final_frame_ok: bool = False
    diagnostics_dir: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data

    def summary(self) -> str:
        state = "PASS" if self.ok else "FAIL"
        lines = [
            f"[{state}] {self.path}",
            f"  duration      : {self.duration:.3f}s"
            + (
                f"  (expected {self.expected_duration:.3f}s, delta {self.duration_delta:+.3f}s)"
                if self.expected_duration is not None and self.duration_delta is not None
                else ""
            ),
            f"  video         : {self.width}x{self.height} @ {self.fps:.3f}fps "
            f"{self.video_codec}/{self.pix_fmt}",
            f"  audio         : {self.audio_codec} {self.audio_sample_rate}Hz "
            f"{self.audio_channels}ch dur={self.audio_duration:.3f}s "
            f"rms={self.audio_rms if self.audio_rms is None else round(self.audio_rms, 5)}",
            f"  frames        : {len(self.frame_checks)} sampled, "
            f"{self.black_frame_count} black, {self.blank_frame_count} blank, "
            f"{len(self.frozen_pairs)} frozen pair(s)",
            f"  captions      : {self.caption_checks_passed}/{self.caption_checks_total} "
            f"caption-active samples confirmed"
            + (
                f", {len(self.caption_false_positives)} unexpected"
                if self.caption_false_positives
                else ""
            ),
            f"  final frame   : {'ok' if self.final_frame_ok else 'PROBLEM'}",
            f"  diagnostics   : {self.diagnostics_dir}",
        ]
        for warning in self.warnings:
            lines.append(f"  WARNING: {warning}")
        for error in self.errors:
            lines.append(f"  ERROR: {error}")
        return "\n".join(lines)


# --------------------------------------------------------------------- pixels


def _decode_frame_rgb(
    path: Path, at: float, width: int, height: int, *, from_end: float | None = None
) -> bytes | None:
    """Decode a single frame as raw RGB24 at the requested size.

    ``from_end`` uses ``-sseof`` instead of ``-ss``, which is the only reliable
    way to read the actual last frame of a file.
    """
    seek = (
        ["-sseof", f"{-from_end:.4f}"] if from_end is not None
        else ["-ss", f"{at:.4f}"]
    )
    proc = ffmpeg.run(
        [
            ffmpeg.ffmpeg_bin(), "-v", "error",
            *seek,
            "-i", str(path),
            "-frames:v", "1",
            "-vf", f"scale={width}:{height}",
            "-f", "rawvideo",
            "-pix_fmt", "rgb24",
            "-",
        ],
        stage="frame_decode",
        check=False,
        timeout=60,
        binary=True,
    )
    if proc.returncode != 0 or not proc.stdout:
        return None
    expected = width * height * 3
    return proc.stdout[:expected] if len(proc.stdout) >= expected else None


def _save_frame_jpeg(
    path: Path, at: float, out: Path, *, from_end: float | None = None
) -> bool:
    seek = (
        ["-sseof", f"{-from_end:.4f}"] if from_end is not None
        else ["-ss", f"{at:.4f}"]
    )
    proc = ffmpeg.run(
        [
            ffmpeg.ffmpeg_bin(), "-v", "error",
            *seek,
            "-i", str(path),
            "-frames:v", "1",
            "-q:v", "3",
            str(out),
            "-y",
        ],
        stage="frame_save",
        check=False,
        timeout=60,
    )
    return proc.returncode == 0 and out.exists() and out.stat().st_size > 100


def _luma_stats(rgb: bytes) -> tuple[float, float, float]:
    """Return (mean, max, stdev) of the Rec.601 luma plane."""
    total = 0.0
    peak = 0.0
    count = 0
    for i in range(0, len(rgb) - 2, 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        total += y
        if y > peak:
            peak = y
        count += 1
    if count == 0:
        return 0.0, 0.0, 0.0
    mean = total / count
    variance = 0.0
    for i in range(0, len(rgb) - 2, 3):
        r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
        y = 0.299 * r + 0.587 * g + 0.114 * b
        variance += (y - mean) ** 2
    return mean, peak, (variance / count) ** 0.5


def _caption_band_ratio(rgb: bytes, width: int, height: int, config: Config) -> float:
    """Fraction of pixels in the caption band matching the caption colour.

    The band is derived from the configured alignment and margin, so a
    non-bottom or non-centred caption style is still checked correctly.
    """
    caption = config.captions
    scaled = caption.scaled(height)
    colour = _parse_hex(caption.primary_color)

    margin_v = scaled.margin_v
    line_height = max(4, int(scaled.font_size * 1.35))
    if scaled.alignment in (1, 4, 7):  # top
        top, bottom = margin_v, margin_v + line_height * scaled.max_lines
    elif scaled.alignment in (3, 6, 9):  # middle
        band = line_height * scaled.max_lines
        top = max(0, (height - band) // 2)
        bottom = min(height, top + band)
    else:  # bottom
        bottom = height - margin_v
        top = max(0, bottom - line_height * scaled.max_lines)

    left = margin_v  # reuse as horizontal inset when bottom aligned
    left = scaled.margin_h if scaled.margin_h else 0
    right = width - left if left else width

    tolerance = 70
    hits = 0
    total = 0
    for y in range(max(0, top), min(height, bottom)):
        row = y * width * 3
        for x in range(max(0, left), min(width, right)):
            i = row + x * 3
            r, g, b = rgb[i], rgb[i + 1], rgb[i + 2]
            total += 1
            if (
                abs(r - colour[0]) <= tolerance
                and abs(g - colour[1]) <= tolerance
                and abs(b - colour[2]) <= tolerance
            ):
                hits += 1
    return (hits / total) if total else 0.0


def _parse_hex(value: str) -> tuple[int, int, int]:
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(c * 2 for c in text)
    try:
        return int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError:
        return 255, 255, 255


def _frame_diff(a: bytes, b: bytes, width: int, height: int, *, step: int = 11) -> float:
    """Mean absolute difference between two RGB buffers, normalised 0..1.

    A 2-D grid sample is used rather than a linear stride: a linear stride
    walks along one row and misses small moving objects elsewhere in the
    frame, which made genuinely playing clips look frozen.
    """
    if len(a) != len(b) or not a:
        return 1.0
    total = 0
    count = 0
    for y in range(0, height, step):
        row = y * width * 3
        for x in range(0, width, step):
            i = row + x * 3
            if i + 2 >= len(a):
                continue
            total += abs(a[i] - b[i]) + abs(a[i + 1] - b[i + 1]) + abs(a[i + 2] - b[i + 2])
            count += 3
    return (total / count) / 255.0 if count else 1.0


# ---------------------------------------------------------------------- audio


def _measure_audio_rms(path: Path, duration: float, windows: int = 8) -> float | None:
    """Mean RMS over sampled windows, on a 0..1 scale."""
    if duration <= 0:
        return None
    samples: list[float] = []
    for i in range(windows):
        centre = duration * (i + 0.5) / windows
        proc = ffmpeg.run(
            [
                ffmpeg.ffmpeg_bin(), "-v", "error",
                "-ss", f"{max(0.0, centre - 1.0):.3f}",
                "-t", "2",
                "-i", str(path),
                "-map", "0:a",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ar", "8000",
                "-ac", "1",
                "-",
            ],
            stage="audio_rms",
            check=False,
            timeout=60,
            binary=True,
        )
        if proc.returncode != 0 or not proc.stdout:
            continue
        raw = proc.stdout
        total = 0
        count = 0
        for j in range(0, len(raw) - 1, 2):
            value = int.from_bytes(raw[j : j + 2], "little", signed=True)
            total += (value / 32768.0) ** 2
            count += 1
        if count:
            samples.append((total / count) ** 0.5)
    if not samples:
        return None
    return sum(samples) / len(samples)


# ------------------------------------------------------------------ sampling


@dataclass
class SamplePoint:
    label: str
    time: float
    caption_expected: bool = False
    from_end: float | None = None


def build_sample_points(
    duration: float,
    timeline: Timeline | None,
    cues: Sequence[Any],
    config: Config,
) -> list[SamplePoint]:
    """Deterministic, caption-aware sampling plan.

    This is the fix for the weak "sampled arbitrary frames" approach: caption
    intervals are sampled deliberately at start/inside/end, and the gaps
    between them are sampled too, so captions appearing when they should not
    is also detected. Every sample carries whether a caption was expected at
    that instant, computed from the cue list rather than assumed.
    """
    points: list[SamplePoint] = []
    eps = 0.04

    points.append(SamplePoint("start", min(0.05, duration / 2)))
    if duration > 2:
        points.append(SamplePoint("early", duration * 0.05))

    # Visual boundaries: just inside each side of every timeline transition.
    if timeline is not None:
        for entry in timeline.entries:
            if entry.start > eps:
                points.append(SamplePoint(f"boundary_{entry.index}_in", entry.start + 0.10))
            if entry.end < duration - eps:
                points.append(SamplePoint(f"boundary_{entry.index}_out", entry.end - 0.10))

    # Caption-aware sampling: inside each caption at three points.
    for i, cue in enumerate(cues):
        span = cue.end - cue.start
        offset = min(0.08, span * 0.15)
        if span > 0.5:
            points.append(SamplePoint(f"cap{i}_start", cue.start + offset))
            points.append(SamplePoint(f"cap{i}_mid", (cue.start + cue.end) / 2))
            points.append(SamplePoint(f"cap{i}_end", cue.end - offset))
        else:
            points.append(SamplePoint(f"cap{i}_mid", (cue.start + cue.end) / 2))
        # The gap immediately after the caption, where nothing should show.
        if cue.end + 0.45 < duration:
            points.append(SamplePoint(f"gap{i}", cue.end + 0.30))

    points.append(SamplePoint("mid", duration * 0.5))
    points.append(SamplePoint("late", duration * 0.90))
    # The true final frame, read with -sseof so it cannot fall off the end.
    points.append(SamplePoint("final", 0.0, from_end=2.0 / max(1.0, config.output.fps)))

    for point in points:
        if point.from_end is None:
            point.time = max(0.0, min(point.time, max(0.0, duration - 0.02)))
        point.caption_expected = _caption_active_at(cues, point.time)

    points.sort(key=lambda p: (p.time, p.label))
    return points


def _caption_active_at(cues: Sequence[Any], time: float) -> bool:
    for cue in cues:
        if cue.start <= time < cue.end:
            return True
    return False


# ---------------------------------------------------------------- entry point


def validate_video(
    path: Path,
    *,
    expected_duration: float | None = None,
    duration_tolerance: float = 0.5,
    config: Config | None = None,
    timeline: Timeline | None = None,
    cues: Sequence[Any] | None = None,
    diagnostics_dir: Path | None = None,
    strict: bool = True,
) -> ValidationResult:
    """Validate a rendered MP4 against the rendered pixels, not just metadata."""
    if not path.exists():
        raise ValidationError("Output file does not exist.", file=str(path))
    if path.stat().st_size < 1024:
        raise ValidationError(
            "Output MP4 is too small to be a real video.",
            file=str(path),
            size=path.stat().st_size,
        )

    diagnostics_dir = diagnostics_dir or (path.parent / "diagnostics")
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = diagnostics_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    if config is None:
        config = Config.from_mapping({})

    thresholds = {
        **DEFAULT_THRESHOLDS,
        "black_mean_luma": config.validation.black_mean_luma,
        "black_max_luma": config.validation.black_max_luma,
        "blank_stddev": config.validation.blank_stddev,
        "blank_max_luma": config.validation.blank_max_luma,
        "caption_pixel_ratio": config.validation.caption_pixel_ratio,
        "frozen_diff": config.validation.frozen_diff,
        "frozen_min_interval": config.validation.frozen_min_interval,
        "silence_rms": config.validation.silence_rms,
    }

    errors: list[str] = []
    warnings: list[str] = []

    info = probe(str(path))
    if not info.has_video:
        raise ValidationError("Output MP4 has no video stream.", file=str(path))
    if info.duration <= 0:
        raise ValidationError(
            "ffprobe reported no duration; the file may be corrupt.",
            duration=str(info.duration),
        )

    delta = None
    if expected_duration is not None:
        delta = info.duration - expected_duration
        if abs(delta) > duration_tolerance:
            errors.append(
                f"duration mismatch: expected {expected_duration:.3f}s, "
                f"got {info.duration:.3f}s (delta {delta:+.3f}s, "
                f"tolerance {duration_tolerance:.3f}s)"
            )

    if info.width != config.output.width or info.height != config.output.height:
        errors.append(
            f"resolution is {info.width}x{info.height}, expected "
            f"{config.output.width}x{config.output.height}"
        )
    if abs(info.fps - config.output.fps) > 0.5:
        errors.append(
            f"frame rate is {info.fps:.3f}, expected {config.output.fps}"
        )

    if not info.has_audio:
        errors.append("output has no audio stream")
        warnings.append("audio validation skipped")

    # Audio duration + silence.
    audio_duration = 0.0
    rms: float | None = None
    silent: bool | None = None
    if info.has_audio:
        audio_duration = _audio_duration(path)
        if expected_duration is not None and abs(audio_duration - expected_duration) > (
            duration_tolerance + 0.5
        ):
            errors.append(
                f"audio duration {audio_duration:.3f}s does not match the "
                f"expected {expected_duration:.3f}s"
            )
        rms = _measure_audio_rms(path, info.duration)
        if rms is not None:
            silent = rms < thresholds["silence_rms"]
            if silent:
                errors.append(
                    f"output audio is effectively silent (rms={rms:.6f} "
                    f"< {thresholds['silence_rms']})"
                )
            else:
                warnings.append(f"output audio rms={rms:.5f}")

    # Frame-level checks.
    points = build_sample_points(info.duration, timeline, cues or [], config)
    width = config.output.width
    height = config.output.height

    frame_checks: list[FrameCheck] = []
    previous_rgb: bytes | None = None
    previous_label = ""
    previous_time: float | None = None
    black_count = 0
    blank_count = 0
    frozen_pairs: list[str] = []
    caption_pass = 0
    caption_total = 0
    false_positives: list[str] = []

    for point in points:
        from_end = point.from_end
        rgb = _decode_frame_rgb(
            path, point.time, width, height, from_end=from_end
        )
        check = FrameCheck(
            label=point.label,
            time=round(point.time, 4) if from_end is None else -1.0,
            extracted=False,
        )
        if rgb is None:
            check.notes.append("frame could not be decoded")
            frame_checks.append(check)
            continue

        check.extracted = True
        mean, peak, stddev = _luma_stats(rgb)
        check.mean_luma = round(mean, 3)
        check.max_luma = round(peak, 3)
        check.stddev = round(stddev, 3)

        # Legitimately dark footage exists in real B-roll, so a frame is only
        # called black when it is both very dark and very flat.
        if mean < thresholds["black_mean_luma"] and peak < thresholds["black_max_luma"]:
            check.is_black = True
            black_count += 1
            check.notes.append("frame is black")
        elif stddev < thresholds["blank_stddev"] and mean < thresholds["blank_max_luma"]:
            # Flat *and* dark: an empty-looking frame. A flat but bright frame
            # is real content (a solid-colour still, a title card) and is fine.
            check.is_blank = True
            blank_count += 1
            check.notes.append("frame is flat and empty")

        ratio = _caption_band_ratio(rgb, width, height, config)
        check.caption_expected = point.caption_expected
        check.caption_ratio = round(ratio, 6)
        check.caption_detected = ratio >= thresholds["caption_pixel_ratio"]

        if point.caption_expected:
            caption_total += 1
            if check.caption_detected:
                caption_pass += 1
            else:
                check.notes.append(
                    f"expected a caption here but found none (band ratio {ratio:.6f})"
                )
        elif check.caption_detected:
            false_positives.append(f"{point.label}@{point.time:.2f}s")

        # Frozen detection compares neighbouring samples only. Samples can sit
        # very close together (caption-interval sampling is deliberately dense),
        # and two reads a few milliseconds apart are identical by construction,
        # so a minimum separation is required before calling anything frozen.
        if previous_rgb is not None and previous_time is not None:
            separation = abs(point.time - previous_time)
            if separation >= thresholds["frozen_min_interval"]:
                diff = _frame_diff(previous_rgb, rgb, width, height)
                if diff < thresholds["frozen_diff"]:
                    frozen_pairs.append(
                        f"{previous_label}->{point.label}@{point.time:.2f}s "
                        f"(dt={separation:.2f}s)"
                    )

        previous_rgb = rgb
        previous_label = point.label
        previous_time = point.time

        # Persist a subset of frames as JPEGs for visual inspection.
        keep = (
            point.label in {"start", "final", "mid", "late", "early"}
            or (point.label.startswith("cap") and check.caption_expected)
            or check.is_black
            or check.is_blank
        )
        if keep:
            stamp = f"{point.time:09.3f}" if from_end is None else "final"
            name = f"{stamp}_{point.label}.jpg"
            if _save_frame_jpeg(path, point.time, frames_dir / name, from_end=from_end):
                check.image = f"frames/{name}"

        frame_checks.append(check)

    # Final-frame verdict is a hard requirement.
    final_checks = [c for c in frame_checks if c.label == "final"]
    final_frame_ok = False
    if final_checks:
        final = final_checks[0]
        final_frame_ok = final.extracted and not final.is_black and not final.is_blank
        if not final_frame_ok:
            errors.append(
                "final frame is not valid visual content "
                f"(black={final.is_black} blank={final.is_blank})"
            )

    # Aggregate frame verdicts.
    if black_count:
        errors.append(f"{black_count} sampled frame(s) were black")
    if blank_count:
        warnings.append(f"{blank_count} sampled frame(s) were flat/blank")
    if frozen_pairs:
        warnings.append(
            f"{len(frozen_pairs)} adjacent sampled frame pair(s) were near-identical "
            f"(expected when consecutive stills are used): {', '.join(frozen_pairs[:4])}"
        )

    # Caption verdicts.
    if caption_total:
        if caption_pass < caption_total:
            errors.append(
                f"captions missing in {caption_total - caption_pass} of "
                f"{caption_total} caption-active samples"
            )
    elif cues:
        errors.append("caption data was supplied but no caption samples were generated")

    if false_positives:
        warnings.append(
            f"caption-coloured pixels found at {len(false_positives)} sample(s) "
            f"with no caption expected: {', '.join(false_positives[:5])}"
        )

    result = ValidationResult(
        path=str(path),
        ok=not errors,
        file_size=path.stat().st_size,
        duration=info.duration,
        width=info.width,
        height=info.height,
        fps=info.fps,
        video_codec=info.video_codec,
        pix_fmt=info.pix_fmt,
        has_video=info.has_video,
        has_audio=info.has_audio,
        audio_codec=info.audio_codec,
        audio_sample_rate=info.sample_rate,
        audio_channels=info.channels,
        audio_duration=audio_duration,
        audio_rms=round(rms, 6) if rms is not None else None,
        audio_silent=silent,
        expected_duration=expected_duration,
        duration_delta=round(delta, 4) if delta is not None else None,
        frame_checks=frame_checks,
        black_frame_count=black_count,
        blank_frame_count=blank_count,
        frozen_pairs=frozen_pairs,
        caption_checks_total=caption_total,
        caption_checks_passed=caption_pass,
        caption_false_positives=false_positives,
        final_frame_ok=final_frame_ok,
        diagnostics_dir=str(diagnostics_dir),
        errors=errors,
        warnings=warnings,
    )

    _write_contact_sheet(frames_dir, diagnostics_dir)
    _write_report(result, diagnostics_dir / "validation.json")
    _write_ffprobe(path, diagnostics_dir)

    if errors and strict:
        result.ok = False
    return result


def _audio_duration(path: Path) -> float:
    value = ffmpeg.probe_value(
        path, "stream=duration:format=duration", stage="audio_duration"
    )
    for line in value.splitlines():
        try:
            return float(line)
        except ValueError:
            continue
    return 0.0


def _write_contact_sheet(frames_dir: Path, diagnostics_dir: Path) -> None:
    frames = sorted(frames_dir.glob("*.jpg"))
    if not frames:
        return
    listing = diagnostics_dir / "contact_frames.txt"
    listing.write_text(
        "\n".join(f"file 'frames/{f.name}'" for f in frames) + "\n", encoding="utf-8"
    )
    columns = 4
    rows = max(1, min(8, (len(frames) + columns - 1) // columns))
    ffmpeg.run(
        [
            ffmpeg.ffmpeg_bin(), "-y", "-v", "error",
            "-f", "concat", "-safe", "1", "-i", "contact_frames.txt",
            "-vf", f"scale=480:-1,tile={columns}x{rows}",
            "-frames:v", "1",
            "-q:v", "3",
            "contact_sheet.jpg",
        ],
        stage="contact_sheet",
        check=False,
        timeout=120,
        cwd=diagnostics_dir,
    )


def _write_report(result: ValidationResult, path: Path) -> None:
    path.write_text(json.dumps(result.to_dict(), indent=2), encoding="utf-8")


def _write_ffprobe(path: Path, diagnostics_dir: Path) -> None:
    data = ffmpeg.probe_json(path)
    (diagnostics_dir / "ffprobe.json").write_text(
        json.dumps(data, indent=2), encoding="utf-8"
    )