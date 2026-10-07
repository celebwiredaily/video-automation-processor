"""Unit tests: configuration, timeline validation and path safety."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from video_pipeline.config import Config, DEFAULTS, deep_merge
from video_pipeline.errors import ConfigError, ValidationError
from video_pipeline.timeline import Timeline


# ------------------------------------------------------------------ config


def test_defaults_match_the_reference_video():
    config = Config.from_mapping({})
    # Measured from the supplied reference video.
    assert (config.output.width, config.output.height) == (1280, 720)
    assert config.output.fps == 30
    assert config.output.video_codec == "libx264"
    assert config.output.pix_fmt == "yuv420p"
    assert config.output.audio_codec == "aac"
    assert config.duration.target_seconds == 300.0


def test_caption_defaults_match_the_reference_style():
    captions = Config.from_mapping({}).captions
    assert captions.primary_color == "#F7FF00"   # measured from the sample
    assert captions.bold and captions.italic
    assert captions.alignment == 2                 # bottom centre
    assert captions.font_size == 46


def test_caption_font_size_scales_with_resolution():
    captions = Config.from_mapping({}).captions
    assert captions.scaled_font_size(720) == 46
    assert captions.scaled_font_size(1080) == 69   # 46 * 1.5
    assert captions.scaled(1080).margin_v == 66


def test_deep_merge_preserves_unrelated_defaults():
    merged = deep_merge(DEFAULTS, {"output": {"crf": 18}})
    assert merged["output"]["crf"] == 18
    assert merged["output"]["width"] == 1280      # untouched default survives
    assert DEFAULTS["output"]["crf"] == 20        # source not mutated


@pytest.mark.parametrize(
    "bad",
    [
        {"output": {"width": 1281, "height": 720}},   # odd width breaks yuv420p
        {"output": {"fps": 0}},
        {"visual": {"default_fit": "squish"}},
        {"visual": {"on_short": "stretchy"}},
        {"duration": {"mismatch": "maybe"}},
    ],
)
def test_invalid_configuration_is_rejected(bad):
    with pytest.raises(ConfigError):
        Config.from_mapping(bad)


# ---------------------------------------------------------------- timeline


def _write_project(root: Path, timeline, **extra) -> Path:
    data = {"timeline": timeline, **extra}
    (root / "project.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")
    return root


def test_parses_a_simple_timeline(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "red.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    _write_project(
        tmp_path,
        [
            {"start": 0, "end": 2, "asset": "assets/red.mp4"},
            {"start": 2, "end": 4, "asset": "assets/red.mp4"},
        ],
    )
    from video_pipeline.project import load_project

    project = load_project(tmp_path)
    assert len(project.timeline.entries) == 2
    assert project.timeline.total_duration == pytest.approx(4.0)
    assert project.timeline.entries[0].asset.name == "red.mp4"


def test_rejects_overlapping_entries(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "red.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    _write_project(
        tmp_path,
        [
            {"start": 0, "end": 3, "asset": "assets/red.mp4"},
            {"start": 2, "end": 4, "asset": "assets/red.mp4"},
        ],
    )
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    assert any("overlaps" in p for p in exc.value.problems)


def test_rejects_gaps_between_entries(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "red.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    _write_project(
        tmp_path,
        [
            {"start": 0, "end": 2, "asset": "assets/red.mp4"},
            {"start": 3, "end": 4, "asset": "assets/red.mp4"},
        ],
    )
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    assert any("gap of" in p for p in exc.value.problems)


def test_rejects_a_timeline_that_does_not_start_at_zero(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "red.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    _write_project(tmp_path, [{"start": 1, "end": 3, "asset": "assets/red.mp4"}])
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    assert any("must start at 0" in p for p in exc.value.problems)


def test_rejects_source_range_beyond_asset_duration(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "red.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    _write_project(
        tmp_path,
        [{"start": 0, "end": 2, "asset": "assets/red.mp4", "source_start": 100}],
    )
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    joined = " ".join(exc.value.problems)
    assert "source_start" in joined
    assert "red.mp4" in joined


def test_short_asset_is_rejected_by_default(tmp_path):
    """A segment longer than its source must fail rather than silently freeze."""
    from video_pipeline.testkit import make_clip

    (tmp_path / "assets").mkdir()
    make_clip(tmp_path / "assets" / "short.mp4", colour="red", seconds=1.0)
    _write_project(tmp_path, [{"start": 0, "end": 5, "asset": "assets/short.mp4"}])

    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    joined = " ".join(exc.value.problems)
    assert "short.mp4" in joined
    assert "5.000" in joined and "1.000" in joined   # requested vs available
    assert "on_short" in joined                      # tells you how to fix it


def test_short_asset_is_allowed_when_on_short_is_configured(tmp_path):
    from video_pipeline.testkit import make_clip

    (tmp_path / "assets").mkdir()
    make_clip(tmp_path / "assets" / "short.mp4", colour="red", seconds=1.0)
    _write_project(
        tmp_path,
        [{"start": 0, "end": 3, "asset": "assets/short.mp4", "on_short": "loop"}],
    )
    from video_pipeline.project import load_project

    project = load_project(tmp_path)
    assert project.timeline.entries[0].on_short == "loop"


def test_path_traversal_is_refused(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    _write_project(
        tmp_path, [{"start": 0, "end": 2, "asset": "../outside.mp4"}]
    )
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    assert any("missing or outside" in p for p in exc.value.problems)


def test_absolute_asset_path_is_refused(tmp_path):
    (tmp_path / "assets").mkdir()
    _write_project(tmp_path, [{"start": 0, "end": 2, "asset": "/etc/passwd"}])
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError):
        load_project(tmp_path)


def test_unknown_keys_are_reported(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "red.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    _write_project(
        tmp_path,
        [{"start": 0, "end": 2, "asset": "assets/red.mp4", "zooom": 1}],
    )
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    assert any("unknown key" in p for p in exc.value.problems)


def test_audio_clip_without_audio_is_rejected(tmp_path):
    from video_pipeline.testkit import make_clip

    (tmp_path / "assets").mkdir()
    make_clip(tmp_path / "assets" / "mute.mp4", colour="red", seconds=3.0, audio=False)
    _write_project(
        tmp_path,
        [{"start": 0, "end": 3, "asset": "assets/mute.mp4", "audio": "clip"}],
    )
    from video_pipeline.project import load_project

    with pytest.raises(ValidationError) as exc:
        load_project(tmp_path)
    assert any("no audio stream" in p for p in exc.value.problems)


def test_timeline_can_live_in_a_separate_file(tmp_path, media_dir):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "red.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    (tmp_path / "project.yaml").write_text("name: split\n", encoding="utf-8")
    (tmp_path / "timeline.yaml").write_text(
        yaml.safe_dump(
            {"timeline": [{"start": 0, "end": 2, "asset": "assets/red.mp4"}]}
        ),
        encoding="utf-8",
    )
    from video_pipeline.project import load_project

    project = load_project(tmp_path)
    assert project.name == "split"
    assert len(project.timeline.entries) == 1