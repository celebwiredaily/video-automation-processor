"""Configuration model.

All tunables live here. Nothing in the renderer hard-codes a resolution, a
frame rate, a caption colour or an output name.

Defaults correspond to the supplied reference video
("How Sean Connery Escaped the Curse of James Bond.mp4"), measured with
ffprobe and by decoding its frames:

    1280x720, 30 fps, H.264 High / yuv420p, AAC stereo

and its caption presentation, measured from the decoded frames:

    yellow #F7FF00 fill, black outline, heavy bold-italic, bottom centre,
    baseline sitting ~39 px above the bottom edge, ~44 px em at 720p,
    single line of roughly 40-44 characters.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import ConfigError

# Caption geometry in the reference was measured at 720p. Keeping the reference
# height lets a 1080p project scale proportionally instead of silently
# producing captions that are half the intended size.
CAPTION_REFERENCE_HEIGHT = 720

DEFAULTS: dict[str, Any] = {
    "output": {
        "filename": "final.mp4",
        # What to do when the destination already has a file with this name:
        #   overwrite - replace it in place (default; keeps names deterministic)
        #   version   - upload as final_v2.mp4, final_v3.mp4, ...
        #   error     - refuse to upload
        "existing_file": "overwrite",
        "width": 1280,
        "height": 720,
        "fps": 30,
        "video_codec": "libx264",
        "crf": 20,
        "preset": "veryfast",
        "pix_fmt": "yuv420p",
        "profile": "high",
        "level": "4.0",
        "gop_seconds": 2,
        "audio_codec": "aac",
        "audio_bitrate": "192k",
        "audio_rate": 48000,
        "audio_channels": 2,
    },
    # Intermediates are encoded once more during the final pass (caption burn
    # forces a re-encode), so they are kept near-lossless to avoid stacking
    # two lossy generations.
    "intermediate": {"crf": 14, "preset": "veryfast"},
    "duration": {
        # Target is ~5 minutes. The renderer never forces a duration; the
        # voiceover decides it. These values only sanity-check the project.
        "target_seconds": 300.0,
        "tolerance_seconds": 0.25,
        "min_seconds": 30.0,
        "max_seconds": 1800.0,
        "warn_target_deviation_ratio": 0.5,
        "mismatch": "error",
    },
    "visual": {
        # "contain" is the default on purpose: supplied B-roll is finished
        # material that may already carry its own letterbox or framing, and
        # cropping to fill would destroy that composition.
        "default_fit": "contain",
        "letterbox_color": "black",
        # error | freeze | loop -- what to do when a source is shorter than the
        # requested interval. Default refuses to invent frozen frames.
        "on_short": "error",
        "allow_clip_audio": False,
    },
    "captions": {
        "enabled": True,
        "font_name": "Montserrat",
        # em size at 720p; scaled by height/720 for other resolutions.
        "font_size": 46,
        "auto_scale_font_size": True,
        "bold": True,
        "italic": True,
        "primary_color": "#F7FF00",
        "outline_color": "#000000",
        "outline": 4,
        "shadow": 0,
        "alignment": 2,
        "margin_v": 44,
        "margin_h": 80,
        "max_chars_per_line": 42,
        "max_lines": 2,
        "max_words_per_caption": 12,
        "min_caption_seconds": 0.30,
        "uppercase": False,
        "wrap_style": 0,
    },
    "fonts": {"directory": None},
    "paths": {"work_dir": None, "diagnostics": True},
    # Validation thresholds are configuration, not magic constants.
    "validation": {
        "require_voiceover": True,
        "duration_tolerance": 0.5,
        "black_mean_luma": 8.0,
        "black_max_luma": 24.0,
        "blank_stddev": 1.5,
        "blank_max_luma": 40.0,
        "caption_pixel_ratio": 0.0012,
        "frozen_diff": 0.002,
        "frozen_min_interval": 0.5,
        "silence_rms": 0.0015,
        "sample_count": 96,
    },
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base``."""
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


@dataclass(frozen=True)
class OutputConfig:
    filename: str
    existing_file: str
    width: int
    height: int
    fps: int
    video_codec: str
    crf: int
    preset: str
    pix_fmt: str
    profile: str
    level: str
    gop_seconds: int
    audio_codec: str
    audio_bitrate: str
    audio_rate: int
    audio_channels: int

    @property
    def gop_size(self) -> int:
        return max(1, int(round(self.gop_seconds * self.fps)))

    @property
    def size(self) -> tuple[int, int]:
        return self.width, self.height

    @property
    def aspect(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass(frozen=True)
class DurationConfig:
    target_seconds: float
    tolerance_seconds: float
    min_seconds: float
    max_seconds: float
    warn_target_deviation_ratio: float
    mismatch: str


@dataclass(frozen=True)
class VisualConfig:
    default_fit: str
    letterbox_color: str
    on_short: str
    allow_clip_audio: bool


@dataclass(frozen=True)
class CaptionConfig:
    enabled: bool
    font_name: str
    font_size: int
    auto_scale_font_size: bool
    bold: bool
    italic: bool
    primary_color: str
    outline_color: str
    outline: float
    shadow: float
    alignment: int
    margin_v: int
    margin_h: int
    max_chars_per_line: int
    max_lines: int
    max_words_per_caption: int
    min_caption_seconds: float
    uppercase: bool
    wrap_style: int

    def scaled_font_size(self, height: int) -> int:
        if not self.auto_scale_font_size:
            return self.font_size
        factor = height / CAPTION_REFERENCE_HEIGHT
        return max(8, int(round(self.font_size * factor)))

    def scaled(self, height: int) -> "CaptionConfig":
        """Return a copy with vertical metrics scaled to ``height``."""
        if not self.auto_scale_font_size:
            return self
        factor = height / CAPTION_REFERENCE_HEIGHT
        return CaptionConfig(
            enabled=self.enabled,
            font_name=self.font_name,
            font_size=self.scaled_font_size(height),
            auto_scale_font_size=False,
            bold=self.bold,
            italic=self.italic,
            primary_color=self.primary_color,
            outline_color=self.outline_color,
            outline=self.outline,
            shadow=self.shadow,
            alignment=self.alignment,
            margin_v=max(0, int(round(self.margin_v * factor))),
            margin_h=max(0, int(round(self.margin_h * factor))),
            max_chars_per_line=self.max_chars_per_line,
            max_lines=self.max_lines,
            max_words_per_caption=self.max_words_per_caption,
            min_caption_seconds=self.min_caption_seconds,
            uppercase=self.uppercase,
            wrap_style=self.wrap_style,
        )


@dataclass(frozen=True)
class ValidationConfig:
    require_voiceover: bool
    duration_tolerance: float
    black_mean_luma: float
    black_max_luma: float
    blank_stddev: float
    blank_max_luma: float
    caption_pixel_ratio: float
    frozen_diff: float
    frozen_min_interval: float
    silence_rms: float
    sample_count: int


@dataclass(frozen=True)
class Config:
    output: OutputConfig
    duration: DurationConfig
    visual: VisualConfig
    captions: CaptionConfig
    validation: ValidationConfig
    intermediate_crf: int
    intermediate_preset: str
    fonts_directory: Path | None
    work_dir: Path | None
    diagnostics: bool
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "Config":
        merged = deep_merge(DEFAULTS, data or {})
        try:
            output = OutputConfig(**merged["output"])
            duration = DurationConfig(**merged["duration"])
            visual = VisualConfig(**merged["visual"])
            captions = CaptionConfig(**merged["captions"])
            validation = ValidationConfig(**merged["validation"])
        except TypeError as exc:
            raise ConfigError(f"Invalid configuration: {exc}") from exc

        problems = []
        if output.width <= 0 or output.height <= 0:
            problems.append("output.width and output.height must be positive")
        if output.width % 2 or output.height % 2:
            problems.append(
                f"output dimensions must be even for {output.pix_fmt} "
                f"(got {output.width}x{output.height})"
            )
        if output.fps <= 0:
            problems.append("output.fps must be positive")
        if output.existing_file not in {"overwrite", "version", "error"}:
            problems.append(
                f"output.existing_file must be overwrite|version|error, "
                f"got {output.existing_file!r}"
            )
        if visual.default_fit not in {"contain", "cover", "stretch"}:
            problems.append(
                f"visual.default_fit must be contain|cover|stretch, "
                f"got {visual.default_fit!r}"
            )
        if visual.on_short not in {"error", "freeze", "loop"}:
            problems.append(
                f"visual.on_short must be error|freeze|loop, got {visual.on_short!r}"
            )
        if duration.mismatch not in {"error", "warn"}:
            problems.append(
                f"duration.mismatch must be error|warn, got {duration.mismatch!r}"
            )
        if problems:
            raise ConfigError(
                "Invalid configuration", problems="\n".join(f"- {p}" for p in problems)
            )

        return cls(
            output=output,
            duration=duration,
            visual=visual,
            captions=captions,
            validation=validation,
            intermediate_crf=int(merged["intermediate"]["crf"]),
            intermediate_preset=str(merged["intermediate"]["preset"]),
            fonts_directory=Path(merged["fonts"]["directory"]) if merged["fonts"].get("directory") else None,
            work_dir=Path(merged["paths"]["work_dir"]) if merged["paths"].get("work_dir") else None,
            diagnostics=bool(merged["paths"]["diagnostics"]),
            raw=merged,
        )