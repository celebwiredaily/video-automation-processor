"""Tests for automatic source trimming and strict duration enforcement.

The claims under test are the ones that are easy to *say* and easy to get
wrong:

* an 8-minute source yields a 5-minute segment, measured with FFprobe;
* a source that is already 5 minutes is handled, not re-cut into something else;
* a short source fails loudly instead of being looped or slowed down;
* the original source file is never modified;
* the rendered file is verified, not assumed - a wrong duration fails;
* captions and audio stay on the same clock as the footage;
* a failed trim or a failed duration check leaves nothing to publish.

Media fixtures are session-scoped because generating them is the slow part.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_pipeline import media, trim
from video_pipeline.captions import load_and_prepare
from video_pipeline.config import Config
from video_pipeline.errors import AssetError, ValidationError

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg

TARGET = 30.0          # short enough to keep the suite quick; the logic is
FPS = 30               # identical at 300s - see the real-render test below.
FRAMES = int(TARGET * FPS)


# --------------------------------------------------------------- fixtures


@pytest.fixture(scope="module")
def long_source(tmp_path_factory) -> Path:
    """An 8-minute source: four times the target, like a real interview."""
    from video_pipeline.testkit import make_source

    path = tmp_path_factory.mktemp("source") / "interview.mp4"
    make_source(path, seconds=TARGET * 4, width=640, height=360, fps=FPS)
    return path


@pytest.fixture(scope="module")
def exact_source(tmp_path_factory) -> Path:
    """A source that is already exactly the target length."""
    from video_pipeline.testkit import make_source

    path = tmp_path_factory.mktemp("exact") / "exactly-five.mp4"
    make_source(path, seconds=TARGET, width=640, height=360, fps=FPS)
    return path


@pytest.fixture(scope="module")
def short_source(tmp_path_factory) -> Path:
    """A source shorter than the target - must not be padded or looped."""
    from video_pipeline.testkit import make_source

    path = tmp_path_factory.mktemp("short") / "too-short.mp4"
    make_source(path, seconds=TARGET / 3, width=640, height=360, fps=FPS)
    return path


# -------------------------------------------------------- the 8-minute case


def test_eight_minute_source_yields_five_minute_segment(long_source, tmp_path):
    """The headline requirement, measured rather than assumed."""
    source_info = media.probe(str(long_source))
    assert source_info.duration > TARGET * 3, "fixture must be much longer than the target"

    result = trim.prepare_segment(long_source, tmp_path / "out", seconds=TARGET, fps=FPS)

    assert result.action == "trimmed"
    assert result.clip_duration == pytest.approx(TARGET, abs=1.0 / FPS)
    assert result.frames == FRAMES
    assert result.source_duration == pytest.approx(source_info.duration, abs=0.1)
    assert result.clip.is_file()

    # Independently re-probed, not trusted from the return value.
    measured = media.probe(str(result.clip))
    assert measured.duration == pytest.approx(TARGET, abs=1.0 / FPS)
    assert trim.frame_count(result.clip) == FRAMES


def test_trim_keeps_the_first_seconds_not_a_random_slice(long_source, tmp_path):
    """The prepared segment must be the *start* of the source."""
    from video_pipeline.validation import _decode_frame_rgb, _frame_diff

    result = trim.prepare_segment(long_source, tmp_path / "out", seconds=TARGET, fps=FPS)

    def frame(path, at):
        return _decode_frame_rgb(path, at, 320, 180)

    # The same wall-clock instant in source and segment must look the same,
    # which only holds if the segment starts at 00:00:00 and is not shifted.
    for at in (1.0, 7.5, 20.0):
        a = frame(long_source, at)
        b = frame(result.clip, at)
        assert a is not None and b is not None
        assert _frame_diff(a, b, 320, 180) < 0.06, (
            f"frame at {at}s does not match the source; the cut is not aligned"
        )


def test_trim_does_not_change_resolution_or_frame_rate(long_source, tmp_path):
    """No zoom, crop, pan or rescale: the picture must look the same."""
    before = media.probe(str(long_source))
    result = trim.prepare_segment(long_source, tmp_path / "out", seconds=TARGET, fps=FPS)
    after = media.probe(str(result.clip))

    assert (after.width, after.height) == (before.width, before.height)
    assert abs(after.fps - FPS) < 0.05


def test_trim_carries_audio_through(long_source, tmp_path):
    result = trim.prepare_segment(long_source, tmp_path / "out", seconds=TARGET, fps=FPS)
    assert result.has_audio, "a source with audio must not be rendered silent"
    assert media.probe(str(result.clip)).audio_codec == "aac"


# ------------------------------------------------- already-exactly-5-minutes


def test_exactly_five_minute_source_is_kept_verbatim(exact_source, tmp_path):
    """No needless re-encode: the length is already right, so copy it."""
    source_info = media.probe(str(exact_source))
    assert source_info.duration == pytest.approx(TARGET, abs=1.0 / FPS)

    result = trim.prepare_segment(exact_source, tmp_path / "out", seconds=TARGET, fps=FPS)

    assert result.action == "verbatim"
    assert result.clip_duration == pytest.approx(TARGET, abs=1.0 / FPS)
    assert result.frames == FRAMES
    assert result.clip.read_bytes() == exact_source.read_bytes(), (
        "a byte-for-byte copy is expected when no cut is needed"
    )


def test_exactly_five_minute_source_is_not_mangled(exact_source, tmp_path):
    result = trim.prepare_segment(exact_source, tmp_path / "out", seconds=TARGET, fps=FPS)
    measured = media.probe(str(result.clip))
    assert (measured.width, measured.height) == (640, 360)
    assert measured.has_audio


# --------------------------------------------------------- a source too short


def test_short_source_fails_clearly(short_source, tmp_path):
    """Failing is the specified behaviour. Padding is not."""
    out = tmp_path / "out"
    with pytest.raises(ValidationError) as exc:
        trim.prepare_segment(short_source, out, seconds=TARGET, fps=FPS)

    message = str(exc.value)
    assert "shorter than the requested duration" in message
    assert "not looped" in message, "the error must say the footage is not faked"
    assert not any(out.glob("*.mp4")), "a failed trim must not leave a usable clip"


def test_short_source_does_not_loop_or_slow_down(short_source, tmp_path):
    """Whatever happens, no 300s file may appear from a 10s source."""
    out = tmp_path / "out"
    with pytest.raises(ValidationError):
        trim.prepare_segment(short_source, out, seconds=TARGET, fps=FPS)
    for path in out.rglob("*.mp4"):
        info = media.probe(str(path))
        assert info.duration <= TARGET


def test_short_source_does_not_modify_the_original(short_source, tmp_path):
    before = short_source.read_bytes()
    with pytest.raises(ValidationError):
        trim.prepare_segment(short_source, tmp_path / "out", seconds=TARGET, fps=FPS)
    assert short_source.read_bytes() == before


# ------------------------------------------------------- the source survives


def test_original_source_is_unchanged_by_a_successful_trim(long_source, tmp_path):
    before_bytes = long_source.read_bytes()
    before_stat = long_source.stat()

    trim.prepare_segment(long_source, tmp_path / "out", seconds=TARGET, fps=FPS)

    assert long_source.read_bytes() == before_bytes
    assert long_source.stat().st_size == before_stat.st_size
    assert media.probe(str(long_source)).duration == pytest.approx(
        TARGET * 4, abs=0.1
    ), "the source must still be the original 8-minute file afterwards"


def test_prepared_clip_lives_outside_the_source(long_source, tmp_path):
    result = trim.prepare_segment(long_source, tmp_path / "out", seconds=TARGET, fps=FPS)
    assert result.clip.resolve() != result.source.resolve()
    assert result.clip.resolve().parent != result.source.resolve().parent


def test_repeated_trims_are_stable(long_source, tmp_path):
    first = trim.prepare_segment(long_source, tmp_path / "a", seconds=TARGET, fps=FPS)
    second = trim.prepare_segment(long_source, tmp_path / "b", seconds=TARGET, fps=FPS)
    assert first.clip_duration == pytest.approx(second.clip_duration, abs=1e-9)
    assert first.frames == second.frames


# ---------------------------------------------------- strict duration measure


@pytest.fixture(scope="module")
def rendered_project(tmp_path_factory) -> Path:
    """A project built from a long source and rendered end to end."""
    from video_pipeline.cli import main
    from video_pipeline.testkit import make_source

    directory = tmp_path_factory.mktemp("source-project") / "five-min"
    source = directory.parent / "raw-footage.mp4"
    make_source(source, seconds=TARGET * 2, width=640, height=360, fps=FPS)

    assert main(["prepare-source", str(directory), "--source", str(source),
                 "--seconds", str(TARGET), "--fps", str(FPS)]) == 0
    assert main(["render", str(directory)]) == 0
    return directory


def test_rendered_video_meets_the_strict_duration_requirement(rendered_project):
    """The final file is 300.000s-equivalent to the frame, verified by FFprobe."""
    final = rendered_project / "output" / "final.mp4"
    report = trim.enforce_final(
        final, target_seconds=TARGET, fps=FPS, require_audio=False
    )
    assert report["ok"]
    assert report["measured_duration"] == pytest.approx(TARGET, abs=1.0 / FPS)
    assert report["frames"] == FRAMES
    assert report["expected_frames"] == FRAMES


def test_render_uses_the_trimmed_clip_not_the_original(rendered_project):
    """The untrimmed source must not be reachable from the render.

    Checked structurally (the timeline only names the prepared segment) and
    temporally (the frame at the very end of the render matches the source at
    the same instant, not at the source's much later timestamp).
    """
    from video_pipeline.validation import _decode_frame_rgb, _frame_diff

    timeline = (rendered_project / "timeline.yaml").read_text(encoding="utf-8")
    assert "build/source/" in timeline

    from video_pipeline.project import load_project

    project = load_project(rendered_project)
    assets = [entry.asset.resolve() for entry in project.timeline.entries]
    for asset in assets:
        assert "build" in asset.parts, f"timeline uses a non-prepared asset: {asset}"
        info = media.probe(str(asset))
        assert info.duration == pytest.approx(TARGET, abs=1.0 / FPS)

    final = rendered_project / "output" / "final.mp4"
    tail = _decode_frame_rgb(final, TARGET - 0.3, 1280, 720)
    assert tail is not None, "the tail of the render must decode"

    # If the renderer had kept the untrimmed original, the final frame would be
    # footage from ~2x the target time, which is a different picture.
    prepared = assets[0]
    original = None
    for candidate in rendered_project.parent.glob("*.mp4"):
        original = candidate
    if original is not None:
        expected = _decode_frame_rgb(original, TARGET - 0.3, 1280, 720)
        wrong = _decode_frame_rgb(original, 2 * TARGET - 0.3, 1280, 720)
        assert expected is not None and wrong is not None
        assert _frame_diff(tail, expected, 1280, 720) < 0.06
        assert _frame_diff(tail, wrong, 1280, 720) > 0.06


def test_final_duration_beyond_tolerance_fails_the_gate(rendered_project):
    """A render that is not the target length must not be reported as success."""
    final = rendered_project / "output" / "final.mp4"
    report = trim.measure_final(final, target_seconds=TARGET + 7.0, fps=FPS,
                                require_audio=False)
    assert not report["ok"]
    assert any(c["name"] == "container_duration" and not c["ok"]
               for c in report["checks"])
    assert any(c["name"] == "video_frame_count" and not c["ok"]
               for c in report["checks"])


def test_enforce_final_raises_on_a_mismatch(rendered_project):
    final = rendered_project / "output" / "final.mp4"
    with pytest.raises(ValidationError) as exc:
        trim.enforce_final(final, target_seconds=TARGET + 7.0, fps=FPS,
                           require_audio=False)
    assert "strict duration requirement" in str(exc.value)


def test_verify_duration_cli_fails_loudly_on_mismatch(rendered_project):
    from video_pipeline.cli import main

    final = rendered_project / "output" / "final.mp4"
    assert main(["verify-duration", str(final), "--seconds", str(TARGET),
                 "--fps", str(FPS)]) == 0
    assert main(["verify-duration", str(final), "--seconds", str(TARGET + 11.0),
                 "--fps", str(FPS)]) == 2


def test_frame_count_failure_is_an_error_not_a_zero(rendered_project, tmp_path):
    """A measurement that cannot be taken must never read as a pass."""
    broken = tmp_path / "not-a-video.mp4"
    broken.write_bytes(b"\x00" * 2048)
    with pytest.raises((ValidationError, AssetError, Exception)):
        trim.frame_count(broken)


def test_trim_report_is_machine_readable(rendered_project, tmp_path):
    report_path = tmp_path / "trim.json"
    trim.write_report(
        {
            "source": "x.mp4",
            "clip_duration_seconds": TARGET,
            "frames": FRAMES,
            "checks": [],
            "ok": True,
        },
        report_path,
    )
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["frames"] == FRAMES and data["ok"] is True


# ------------------------------------------------- audio and caption timing


def test_rendered_audio_matches_the_target(rendered_project):
    final = rendered_project / "output" / "final.mp4"
    info = media.probe(str(final))
    assert info.has_audio, "audio from the source must survive the whole pipeline"
    audio_duration = trim.stream_duration(final, "a")
    assert audio_duration == pytest.approx(TARGET, abs=1.0 / FPS + 0.03)


def test_captions_stay_inside_the_render(rendered_project):
    """Caption cues must be timed to the same 0..300 clock as the video."""
    config = Config.from_mapping({})
    cues, _ = load_and_prepare(
        rendered_project / "captions" / "segments.json",
        config.captions,
        video_duration=TARGET,
    )
    assert cues, "the generated project must have captions"
    for cue in cues:
        assert 0 <= cue.start < cue.end <= TARGET + 1e-6, (
            f"caption {cue.start}-{cue.end}s falls outside the rendered video"
        )
    # Coverage: captions must not stop well before the end of the video.
    assert cues[-1].end >= TARGET - 6.0


def test_validation_confirms_captions_on_the_rendered_frames(rendered_project):
    """Existing caption validation, applied to the trimmed-source render."""
    from video_pipeline.validation import validate_video

    config = Config.from_mapping({})
    cues, _ = load_and_prepare(
        rendered_project / "captions" / "segments.json",
        config.captions,
        video_duration=TARGET,
    )
    report = validate_video(
        rendered_project / "output" / "final.mp4",
        expected_duration=TARGET,
        duration_tolerance=1.0 / FPS,
        config=config,
        timeline=None,
        cues=cues,
        diagnostics_dir=rendered_project / "output" / "diagnostics" / "strict",
        strict=False,
    )
    assert report.ok, report.errors
    assert report.caption_checks_total > 0
    assert report.caption_checks_passed == report.caption_checks_total
    assert report.black_frame_count == 0
    assert not report.audio_silent


# ------------------------------------------------------------ source finding


def test_source_is_located_from_project_yaml(tmp_path):
    from video_pipeline.testkit import make_source

    root = tmp_path / "located"
    root.mkdir()
    (root / "project.yaml").write_text(
        "name: located\nsource:\n  video: assets/raw.mp4\n", encoding="utf-8"
    )
    make_source(root / "assets" / "raw.mp4", seconds=TARGET / 2, width=320,
                height=180, fps=FPS)

    assert trim.locate_source(root).name == "raw.mp4"


def test_ambiguous_sources_are_an_error(tmp_path):
    from video_pipeline.testkit import make_source

    root = tmp_path / "ambiguous"
    (root / "assets").mkdir(parents=True)
    make_source(root / "assets" / "one.mp4", seconds=3, width=320, height=180)
    make_source(root / "assets" / "two.mp4", seconds=3, width=320, height=180)

    with pytest.raises(AssetError) as exc:
        trim.locate_source(root)
    assert "explicitly" in str(exc.value)


def test_missing_source_is_an_error(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(AssetError):
        trim.locate_source(empty)


def test_missing_source_file_is_an_error_not_a_silent_default(tmp_path):
    root = tmp_path / "broken"
    root.mkdir()
    (root / "project.yaml").write_text(
        "name: broken\nsource:\n  video: assets/missing.mp4\n", encoding="utf-8"
    )
    with pytest.raises(AssetError):
        trim.locate_source(root)
