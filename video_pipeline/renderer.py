"""The renderer.

Three stages, each with a single job:

1. **conform** - every timeline entry becomes one normalised intermediate at
   identical codec, size, frame rate and sample aspect ratio. Doing this per
   entry means a bad asset fails with a precise error instead of corrupting one
   enormous filtergraph.
2. **concat** - the intermediates are joined with the concat demuxer and
   stream-copied. No re-encode, so this is effectively instant.
3. **mux** - the voiceover is mixed in and captions are burned with libass.

Supplied B-roll is treated as finished material: clips are trimmed only when
the timeline asks for it, and their internal cuts, zooms and effects are
preserved untouched. Still images are scaled to fit and letterboxed; nothing is
cropped or stretched unless the timeline explicitly requests it.
"""

from __future__ import annotations

import concurrent.futures
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from . import captions as captions_mod
from . import ffmpeg
from .config import Config
from .errors import AssetError, ValidationError
from .media import MediaInfo, is_image_asset, probe
from .timeline import Timeline, TimelineEntry

FONT_SUFFIXES = {".ttf", ".otf", ".ttc"}


@dataclass
class RenderResult:
    output_path: Path
    duration: float
    segment_count: int
    caption_count: int
    warnings: list[str]
    plan_path: Path


class Renderer:
    def __init__(
        self,
        *,
        project_root: Path,
        config: Config,
        work_dir: Path,
        output_path: Path,
        voiceover: Path | None,
        captions_file: Path | None,
        keep_intermediates: bool = False,
        jobs: int = 4,
        verbose: bool = True,
    ) -> None:
        self.project_root = project_root
        self.config = config
        self.work_dir = work_dir
        self.output_path = output_path
        self.voiceover = voiceover
        self.captions_file = captions_file
        self.keep_intermediates = keep_intermediates
        self.jobs = max(1, jobs)
        self.verbose = verbose

        self.segments_dir = work_dir / "segments"
        self.logs_dir = work_dir / "logs"
        self.fonts_dir = work_dir / "fonts"
        self.font_files: list[str] = []
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ setup

    def _log(self, message: str) -> None:
        if self.verbose:
            print(message, flush=True)

    def _stage_fonts(self) -> None:
        """Copy the caption font into the work directory.

        Fontconfig and libass resolve ``fontsdir`` relative to the process
        working directory, so a relative directory name keeps the filter
        argument free of characters that would need escaping.
        """
        source = self.config.fonts_directory or Path(__file__).parent / "assets" / "fonts"
        if not source.is_dir():
            raise AssetError(
                "Caption font directory not found.",
                directory=str(source),
                hint="Set fonts.directory in config.yaml.",
            )

        if self.fonts_dir.exists():
            shutil.rmtree(self.fonts_dir)
        self.fonts_dir.mkdir(parents=True, exist_ok=True)

        copied = []
        for font in sorted(source.iterdir()):
            if font.suffix.lower() in FONT_SUFFIXES:
                shutil.copy2(font, self.fonts_dir / font.name)
                copied.append(font.name)

        if not copied:
            raise AssetError(
                "No .ttf/.otf font files found for caption rendering.",
                directory=str(source),
                hint="Place the caption font in that directory.",
            )

        self.font_files = copied
        self._log(f"[setup] fonts: {', '.join(copied)}")

    # -------------------------------------------------------------- stage one

    def _scale_filter(self, entry: TimelineEntry, info: MediaInfo) -> str:
        """Build the scale/crop chain.

        The scaled size is computed here rather than left to FFmpeg's
        ``force_original_aspect_ratio`` so the result is identical across
        FFmpeg versions and can be asserted in tests.
        """
        width, height = self.config.output.size
        source_w = info.width or width
        source_h = info.height or height
        if source_w <= 0 or source_h <= 0:
            raise AssetError(
                "Asset has no usable dimensions.", asset=str(entry.asset)
            )

        if entry.fit == "stretch":
            return f"scale={width}:{height}"

        source_ratio = source_w / source_h
        target_ratio = width / height

        if entry.fit == "cover":
            if source_ratio > target_ratio:
                scaled_h = height
                scaled_w = ffmpeg.make_even(height * source_ratio)
            else:
                scaled_w = width
                scaled_h = ffmpeg.make_even(width / source_ratio)
            scaled_w = min(scaled_w, max(2, ffmpeg.make_even(width * 4)))
            scaled_h = min(scaled_h, max(2, ffmpeg.make_even(height * 4)))
            return (
                f"scale={scaled_w}:{scaled_h},"
                f"crop={width}:{height}:(iw-ow)/2:(ih-oh)/2"
            )

        # contain: scale to fit inside the frame, then letterbox.
        if source_ratio > target_ratio:
            scaled_w = width
            scaled_h = ffmpeg.make_even(width / source_ratio)
        else:
            scaled_h = height
            scaled_w = ffmpeg.make_even(height * source_ratio)
        scaled_w = max(2, min(scaled_w, width))
        scaled_h = max(2, min(scaled_h, height))
        color = self.config.visual.letterbox_color
        return (
            f"scale={scaled_w}:{scaled_h},"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color={color}"
        )

    def _conform_one(self, entry: TimelineEntry, info: MediaInfo, wants_audio: bool) -> Path:
        output = self.segments_dir / f"{entry.index:04d}.mp4"
        duration = entry.duration
        out = self.config.output
        mid = self.config.intermediate_crf

        argv: list[str] = [ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error"]

        # Frame-rate conform is part of the scale chain so the encoder sees a
        # constant frame rate regardless of the source.
        video_chain = [
            self._scale_filter(entry, info),
            "setsar=1",
            f"fps={out.fps}",
            f"format={out.pix_fmt}",
        ]

        is_still = is_image_asset(entry.asset) or info.kind == "image"

        if is_still:
            argv += ["-loop", "1", "-framerate", str(out.fps), "-i", str(entry.asset)]
        else:
            if entry.on_short == "loop":
                argv += ["-stream_loop", "-1"]
            argv += ["-i", str(entry.asset)]
            # Accurate (post-input) seeking so source_start is frame-accurate
            # rather than snapped to the nearest keyframe.
            if entry.source_start > 0:
                argv += ["-ss", f"{entry.source_start:.6f}"]

        if not is_still and entry.on_short == "freeze":
            video_chain.append(
                f"tpad=stop_mode=clone:stop_duration={duration + 1:.3f}"
            )

        argv += ["-t", f"{duration:.6f}"]

        if wants_audio:
            argv += [
                "-af",
                f"aformat=sample_fmts=fltp:sample_rates={out.audio_rate}:"
                f"channel_layouts={'stereo' if out.audio_channels == 2 else 'mono'}",
            ]
        else:
            argv += ["-an"]

        argv += [
            "-vf", ",".join(video_chain),
            "-c:v", out.video_codec,
            "-crf", str(mid),
            "-preset", self.config.intermediate_preset,
            "-pix_fmt", out.pix_fmt,
            "-r", str(out.fps),
            "-video_track_timescale", str(out.fps * 1000),
        ]

        if wants_audio:
            argv += [
                "-c:a", out.audio_codec,
                "-b:a", out.audio_bitrate,
                "-ar", str(out.audio_rate),
                "-ac", str(out.audio_channels),
            ]

        argv += ["-movflags", "+faststart", str(output)]

        ffmpeg.run(
            argv,
            stage=f"conform[{entry.index}]",
            log_path=self.logs_dir / "conform.log",
        )
        return output

    def render_segments(self, timeline: Timeline) -> list[Path]:
        self.segments_dir.mkdir(parents=True, exist_ok=True)
        for stale in self.segments_dir.glob("*.mp4"):
            stale.unlink()

        infos: dict[str, MediaInfo] = {}
        for entry in timeline.entries:
            infos.setdefault(str(entry.asset), probe(str(entry.asset)))

        wants_audio = bool(timeline.clip_audio_entries())
        outputs: list[Path] = []

        def task(entry: TimelineEntry) -> Path:
            return self._conform_one(entry, infos[str(entry.asset)], wants_audio)

        if self.jobs > 1 and len(timeline.entries) > 1:
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.jobs) as pool:
                for index, path in enumerate(pool.map(task, timeline.entries), 1):
                    outputs.append(path)
                    self._log(f"[conform] {index}/{len(timeline.entries)}")
        else:
            for index, entry in enumerate(timeline.entries, 1):
                outputs.append(task(entry))
                self._log(f"[conform] {index}/{len(timeline.entries)}")

        return outputs

    # -------------------------------------------------------------- stage two

    def concat_segments(self, segments: Sequence[Path]) -> Path:
        visuals = self.work_dir / "visuals.mp4"
        listing = self.work_dir / "concat.txt"

        # Relative names only: the concat demuxer list is never given a
        # project-controlled path, so quoting and traversal are moot.
        listing.write_text(
            "\n".join(f"file 'segments/{path.name}'" for path in segments) + "\n",
            encoding="utf-8",
        )

        argv = [
            ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-f", "concat", "-safe", "1",
            "-i", listing.name,
            "-c", "copy",
            "-movflags", "+faststart",
            visuals.name,
        ]
        ffmpeg.run(
            argv,
            stage="concat",
            log_path=self.logs_dir / "concat.log",
            cwd=self.work_dir,
        )
        return visuals

    # ------------------------------------------------------------ stage three

    def _duck_filter(self, intervals: Sequence[tuple[float, float]], volume: str = "0") -> str:
        """Attenuate audio outside/inside the given intervals.

        Used to silence the narration under interview footage and to gate the
        B-roll audio bed so only the intended entries are audible.
        """
        if not intervals:
            return "anull"
        terms = "+".join(
            f"between(t\\,{start:.3f}\\,{end:.3f})" for start, end in intervals
        )
        return f"volume=volume='if({terms},{volume},1)':eval=frame"

    def _mix_audio(
        self, timeline: Timeline, duration: float
    ) -> tuple[list[str], str]:
        """Build the audio side of the final pass.

        Inputs are, in order:
            0  visuals.mp4   (audio present only when a timeline entry asks
                              for `audio: clip`)
            1  voiceover     (only when configured and present)

        Returns ``(extra_input_args, audio_map)`` where ``audio_map`` is
        either a direct stream specifier (``1:a``) or a marker carrying a
        filter_complex fragment that :meth:`finalize` merges into the graph.
        """
        out = self.config.output
        wants_clip_audio = bool(timeline.clip_audio_entries())
        has_voiceover = bool(self.voiceover and self.voiceover.exists())

        if not has_voiceover and not wants_clip_audio:
            raise ValidationError(
                "There is no audio to produce.",
                voiceover=str(self.voiceover) if self.voiceover else "not configured",
                hint="Set `voiceover:` in project.yaml, or mark timeline entries "
                "with `audio: clip`.",
            )

        clip_intervals = [(e.start, e.end) for e in timeline.clip_audio_entries()]

        extra: list[str] = []
        index_bed: int | None = None
        index_voiceover: int | None = None

        if wants_clip_audio:
            # The conform stage gave every segment an identically-formatted
            # audio stream, so concat produced a bed that is already aligned
            # to the timeline. It rides on the existing visuals.mp4 input.
            index_bed = 0

        if has_voiceover:
            index_voiceover = 1
            extra += ["-i", str(self.voiceover.resolve())]

        # Single source: map it directly and let the output options handle
        # sample rate and channel layout. No filtergraph needed.
        if has_voiceover and not wants_clip_audio:
            return extra, f"{index_voiceover}:a"
        if wants_clip_audio and not has_voiceover:
            return extra, f"{index_bed}:a"

        channels = "stereo" if out.audio_channels == 2 else "mono"
        layout = (
            f"aformat=sample_fmts=fltp:sample_rates={out.audio_rate}:"
            f"channel_layouts={channels}"
        )

        chain = [
            f"[{index_voiceover}:a]{layout},"
            f"{self._duck_filter(clip_intervals)}[vo]"
        ]
        # Gate the bed so entries that did not request audio stay silent.
        chain.append(
            f"[{index_bed}:a]{layout},"
            f"{self._duck_filter(clip_intervals, volume='1')}[bed]"
        )
        chain.append(
            "[vo][bed]amix=inputs=2:duration=longest:dropout_transition=0:"
            "normalize=0[amixed]"
        )
        chain.append(
            f"[amixed]atrim=0:{duration:.6f},"
            "asetpts=N/SR/TB,alimiter=limit=0.97[aout]"
        )
        return extra, "GRAPH:" + ";".join(chain)

    def build_captions(self, duration: float) -> tuple[Path | None, int, list[str]]:
        config = self.config.captions
        if not config.enabled or self.captions_file is None:
            return None, 0, []

        cues, warnings = captions_mod.load_and_prepare(
            self.captions_file, config, video_duration=duration
        )
        out = self.config.output
        ass_path = self.work_dir / "captions.ass"
        captions_mod.write_ass(
            ass_path, cues, config, width=out.width, height=out.height
        )
        return ass_path, len(cues), warnings

    def finalize(
        self,
        timeline: Timeline,
        visuals: Path,
        duration: float,
    ) -> tuple[Path, int, list[str]]:
        out = self.config.output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        ass_path, caption_count, caption_warnings = self.build_captions(duration)

        extra_args, audio_map = self._mix_audio(timeline, duration)

        argv: list[str] = [
            ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin",
            "-loglevel", "error",
            "-i", "visuals.mp4",
            *extra_args,
        ]

        # The caption path is a fixed, project-controlled filename inside the
        # work directory, so nothing user-supplied reaches the filtergraph.
        if ass_path is not None:
            graph = "[0:v]ass=captions.ass:fontsdir=fonts:alpha=1[vout]"
        else:
            graph = "[0:v]null[vout]"

        if audio_map.startswith("GRAPH:"):
            argv += [
                "-filter_complex", graph + ";" + audio_map[len("GRAPH:"):],
                "-map", "[vout]",
                "-map", "[aout]",
            ]
        else:
            argv += [
                "-filter_complex", graph,
                "-map", "[vout]",
                "-map", audio_map,
            ]

        argv += [
            "-t", f"{duration:.6f}",
            "-c:v", out.video_codec,
            "-preset", out.preset,
            "-crf", str(out.crf),
            "-pix_fmt", out.pix_fmt,
            "-profile:v", out.profile,
            "-level:v", out.level,
            "-r", str(out.fps),
            "-g", str(out.gop_size),
            "-bf", "2",
            "-c:a", out.audio_codec,
            "-b:a", out.audio_bitrate,
            "-ar", str(out.audio_rate),
            "-ac", str(out.audio_channels),
            "-movflags", "+faststart",
            "-video_track_timescale", str(out.fps * 1000),
            str(self.output_path.resolve()),
        ]

        ffmpeg.run(
            argv,
            stage="mux+burn",
            log_path=self.logs_dir / "finalize.log",
            cwd=self.work_dir,
        )
        return self.output_path, caption_count, caption_warnings

    # ------------------------------------------------------------------- main

    def render(self, timeline: Timeline, duration: float) -> RenderResult:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self._stage_fonts()

        visual_total = timeline.total_duration
        tolerance = self.config.duration.tolerance_seconds

        # The voiceover is the timing backbone. Visuals must cover it, and any
        # overrun past the narration is trimmed rather than left hanging.
        if visual_total + tolerance < duration:
            short_by = duration - visual_total
            if self.config.visual.on_short == "error":
                raise ValidationError(
                    "Timeline is shorter than the voiceover.",
                    visual_duration=f"{visual_total:.3f}s",
                    voiceover_duration=f"{duration:.3f}s",
                    shortfall=f"{short_by:.3f}s",
                    hint="Extend the last timeline entry, add segments, or "
                    "configure `visual.on_short: freeze|loop`.",
                )
            # freeze/loop already extend each segment during conform, so the
            # materialised visuals will reach the voiceover length.
            self.warnings.append(
                f"Timeline is {short_by:.3f}s shorter than the voiceover; "
                f"visual.on_short={self.config.visual.on_short} will extend it."
            )
        elif visual_total - tolerance > duration:
            self.warnings.append(
                f"Timeline runs {visual_total - duration:.3f}s past the voiceover; "
                f"the render is trimmed to the narration length."
            )

        plan_path = self.write_plan(timeline, duration)

        segments = self.render_segments(timeline)
        visuals = self.concat_segments(segments)

        output_path, caption_count, warnings = self.finalize(timeline, visuals, duration)
        self.warnings.extend(warnings)

        if not self.keep_intermediates:
            shutil.rmtree(self.segments_dir, ignore_errors=True)
            visuals.unlink(missing_ok=True)
            (self.work_dir / "concat.txt").unlink(missing_ok=True)

        return RenderResult(
            output_path=output_path,
            duration=duration,
            segment_count=len(segments),
            caption_count=caption_count,
            warnings=self.warnings,
            plan_path=plan_path,
        )

    def write_plan(self, timeline: Timeline, duration: float) -> Path:
        plan_path = self.work_dir / "plan.json"
        plan = {
            "duration": round(duration, 4),
            "output": {
                "width": self.config.output.width,
                "height": self.config.output.height,
                "fps": self.config.output.fps,
                "crf": self.config.output.crf,
                "preset": self.config.output.preset,
            },
            "timeline": timeline.to_dict(),
            "fonts": self.font_files,
        }
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")
        return plan_path