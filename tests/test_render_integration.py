"""Integration tests that render real MP4s and inspect the rendered pixels.

These are deliberately media-level: they decode frames from the produced file
and assert which source clip is on screen at which time. A test that only
checked FFmpeg's exit code would pass even when the renderer put the wrong
clip at the wrong moment, which is exactly the failure this project has to
catch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_pipeline import media
from video_pipeline.cli import main
from video_pipeline.validation import (
    _decode_frame_rgb,
    _frame_diff,
    _luma_stats,
    validate_video,
)

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def dominant_channel(rgb: bytes, width: int, height: int) -> str:
    """Classify a frame by which channel dominates a grid of sample points."""
    # Sample a coarse grid, ignoring the centre where captions may sit.
    r_total = g_total = b_total = 0
    count = 0
    step_y = max(1, height // 12)
    step_x = max(1, width // 16)
    for y in range(0, height, step_y):
        row = y * width * 3
        for x in range(0, width, step_x):
            i = row + x * 3
            r_total += rgb[i]
            g_total += rgb[i + 1]
            b_total += rgb[i + 2]
            count += 1
    r = r_total / count
    g = g_total / count
    b = b_total / count
    # A neutral (grey) frame has all three channels close together.
    if max(r, g, b) - min(r, g, b) < 40:
        return "grey"
    if r >= g and r >= b:
        return "red"
    if g >= r and g >= b:
        return "green"
    return "blue"


# ------------------------------------------------------- the rendered project


def test_render_succeeds_and_produces_a_mp4(rendered):
    assert rendered.exists()
    assert rendered.stat().st_size > 10_000
    assert rendered.suffix == ".mp4"


def test_rendered_video_matches_the_configured_format(rendered):
    info = media.probe(str(rendered))
    assert (info.width, info.height) == (1280, 720)
    assert abs(info.fps - 30.0) < 0.05
    assert info.video_codec == "h264"
    assert info.pix_fmt == "yuv420p"      # YouTube-playable, not 4:4:4/10-bit
    assert info.audio_codec == "aac"
    assert info.sample_rate == 48000
    assert info.channels == 2
    assert info.has_video and info.has_audio


def test_rendered_duration_follows_the_voiceover(rendered, project_dir):
    voiceover = media.probe(str(project_dir / "audio" / "voiceover.wav"))
    info = media.probe(str(rendered))
    assert abs(info.duration - voiceover.duration) < 0.25


def test_visual_timeline_is_in_the_right_order(rendered):
    """The actual pixels at each point must match the declared source clip.

    This is the check that fails if clips are reordered, dropped, or shown at
    the wrong time - none of which any exit code would reveal.
    """
    expected = [
        (0.3, "red"),
        (1.8, "blue"),
        (3.0, "green"),
        (4.2, "green"),   # still.jpg is green
        (5.4, "grey"),    # silent.mp4 is grey
    ]
    for at, colour in expected:
        rgb = _decode_frame_rgb(rendered, at, 1280, 720)
        assert rgb is not None, f"could not decode a frame at {at}s"
        # Sample above the caption band so the caption colour cannot skew this.
        top = rgb[: int(720 * 0.6) * 1280 * 3]
        assert dominant_channel(top, 1280, int(720 * 0.6)) == colour, (
            f"expected {colour} at {at}s"
        )


def test_clip_boundaries_land_in_the_right_place(rendered):
    """Just before a cut is the old clip; just after is the new one."""
    # Timeline: red 0-1.2, blue 1.2-2.4, green 2.4-3.6, still 3.6-4.8, grey 4.8-6.0
    before = _decode_frame_rgb(rendered, 1.10, 1280, 720)
    after = _decode_frame_rgb(rendered, 1.30, 1280, 720)
    assert dominant_channel(before, 1280, int(720 * 0.6)) == "red"
    assert dominant_channel(after, 1280, int(720 * 0.6)) == "blue"


def test_final_frame_is_not_black_or_blank(rendered):
    result = validate_video(
        rendered,
        diagnostics_dir=rendered.parent / "diagnostics_finalcheck",
        strict=False,
    )
    final = [c for c in result.frame_checks if c.label == "final"]
    assert final, "a final-frame sample must exist"
    assert final[0].extracted
    assert not final[0].is_black
    assert not final[0].is_blank
    assert result.final_frame_ok


def test_no_black_frames_anywhere(rendered):
    result = validate_video(
        rendered,
        diagnostics_dir=rendered.parent / "diagnostics_blackcheck",
        strict=False,
    )
    assert result.black_frame_count == 0
    for check in result.frame_checks:
        if check.extracted:
            assert not check.is_black, f"black frame at {check.label}"


def test_captions_are_present_inside_every_caption_interval(rendered, project_dir):
    """Caption validation must sample *inside* known caption intervals."""
    from video_pipeline.captions import load_and_prepare
    from video_pipeline.config import Config

    config = Config.from_mapping({})
    cues, _ = load_and_prepare(
        project_dir / "captions" / "segments.json",
        config.captions,
        video_duration=media.probe(str(rendered)).duration,
    )
    assert cues, "the test project must have captions"

    result = validate_video(
        rendered,
        config=config,
        cues=cues,
        diagnostics_dir=rendered.parent / "diagnostics_captioncheck",
        strict=False,
    )
    assert result.caption_checks_total > 0, "caption samples must be generated"
    assert result.caption_checks_passed == result.caption_checks_total, (
        "every caption-active sample must show caption pixels"
    )
    # And the samples really were inside caption windows.
    expected_labels = {f"cap{i}" for i in range(len(cues))}
    seen = {c.label.rsplit("_", 1)[0] for c in result.frame_checks}
    assert expected_labels & seen, "caption-interval samples are missing"


def test_video_is_actually_moving_not_frozen(rendered):
    """Adjacent samples inside one clip must differ (the clips animate)."""
    a = _decode_frame_rgb(rendered, 0.30, 1280, 720)
    b = _decode_frame_rgb(rendered, 0.60, 1280, 720)
    assert _frame_diff(a, b, 1280, 720) > 0.002


def test_audio_is_present_and_not_silent(rendered):
    result = validate_video(
        rendered,
        diagnostics_dir=rendered.parent / "diagnostics_audiocheck",
        strict=False,
    )
    assert result.has_audio
    assert result.audio_rms is not None
    assert not result.audio_silent, "rendered audio must not be silent"


def test_validation_report_is_machine_readable(rendered):
    diagnostics = rendered.parent / "diagnostics"
    report = diagnostics / "validation.json"
    assert report.is_file()
    data = json.loads(report.read_text(encoding="utf-8"))
    for key in (
        "ok", "duration", "width", "height", "fps", "video_codec", "audio_codec",
        "frame_checks", "caption_checks_total", "black_frame_count", "errors",
    ):
        assert key in data, f"validation.json is missing {key}"
    assert (diagnostics / "contact_sheet.jpg").is_file()
    assert (diagnostics / "ffprobe.json").is_file()


def test_render_fails_loudly_when_the_voiceover_is_missing(tmp_path):
    """No voiceover must be an error, not a silently short render."""
    from video_pipeline.testkit import build_test_project

    root = tmp_path / "novoice"
    build_test_project(root, seconds=3.0)
    (root / "audio" / "voiceover.wav").unlink()
    assert main(["render", str(root)]) != 0


def test_render_fails_when_visuals_are_shorter_than_the_voiceover(tmp_path):
    """A timeline that cannot cover the narration must not be padded silently."""
    from video_pipeline.testkit import build_test_project

    root = tmp_path / "tooshort"
    build_test_project(root, seconds=6.0)
    # Keep the 6s voiceover but shrink the timeline to 2s.
    (root / "timeline.yaml").write_text(
        "timeline:\n  - start: 0.0\n    end: 2.0\n    asset: assets/red.mp4\n",
        encoding="utf-8",
    )
    media.clear_cache()
    assert main(["render", str(root)]) != 0