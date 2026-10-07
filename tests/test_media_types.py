"""Media-type coverage: portrait, square, odd frame rates, no audio, stills.

Each case renders a real MP4 and asserts the output is normalised, so a source
that disagrees with the output format cannot silently produce a broken file.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from video_pipeline import media
from video_pipeline.cli import main
from video_pipeline.project import load_project
from video_pipeline.testkit import make_clip, make_image, make_voiceover

from .conftest import requires_ffmpeg

pytestmark = requires_ffmpeg


def _make_project(
    root: Path,
    assets: list[tuple[str, object]],
    *,
    seconds: float = 6.0,
    extra_entry: dict | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "assets").mkdir(exist_ok=True)
    (root / "audio").mkdir(exist_ok=True)
    (root / "captions").mkdir(exist_ok=True)
    (root / "output").mkdir(exist_ok=True)

    for name, maker in assets:
        maker(root / "assets" / name)

    make_voiceover(root / "audio" / "voiceover.wav", seconds=seconds)

    entries = [
        {
            "start": round(i * seconds / len(assets), 4),
            "end": round((i + 1) * seconds / len(assets), 4),
            "asset": f"assets/{name}",
            "fit": "contain",
        }
        for i, (name, _) in enumerate(assets)
    ]
    if extra_entry:
        entries.append(extra_entry)
    entries[-1]["end"] = round(seconds, 4)

    (root / "timeline.yaml").write_text(
        yaml.safe_dump({"timeline": entries}), encoding="utf-8"
    )
    (root / "project.yaml").write_text(
        "name: media-types\nvoiceover: audio/voiceover.wav\n"
        "captions:\n  file: captions/segments.json\n",
        encoding="utf-8",
    )
    (root / "captions" / "segments.json").write_text("[]", encoding="utf-8")
    return root


def assert_normalised(output: Path, expected_duration: float):
    info = media.probe(str(output))
    assert output.is_file()
    assert (info.width, info.height) == (1280, 720)
    assert abs(info.fps - 30.0) < 0.05
    assert info.pix_fmt == "yuv420p"
    assert info.video_codec == "h264"
    assert info.has_audio
    assert abs(info.duration - expected_duration) < 0.4


def test_portrait_video_is_letterboxed_not_stretched(tmp_path):
    root = _make_project(
        tmp_path / "portrait",
        [
            ("wide.mp4", lambda p: make_clip(p, colour="red", seconds=3.0)),
            ("tall.mp4", lambda p: make_clip(
                p, colour="blue", seconds=3.0, width=360, height=640
            )),
        ],
    )
    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 6.0)

    # The portrait frame must have pillarbox bars rather than being stretched
    # to full width: the leftmost column stays black.
    from video_pipeline.validation import _decode_frame_rgb

    rgb = _decode_frame_rgb(root / "output" / "final.mp4", 4.5, 1280, 720)
    assert rgb is not None
    left_column = [rgb[(320 * 1280 + x) * 3] for x in range(0, 200, 20)]
    assert max(left_column) < 40, "portrait source should be pillarboxed"


def test_square_video_is_letterboxed(tmp_path):
    root = _make_project(
        tmp_path / "square",
        [
            ("sq.mp4", lambda p: make_clip(
                p, colour="green", seconds=6.0, width=480, height=480
            )),
        ],
    )
    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 6.0)


def test_source_at_24fps_is_normalised_to_30(tmp_path):
    root = _make_project(
        tmp_path / "fps24",
        [
            ("f24.mp4", lambda p: make_clip(
                p, colour="red", seconds=6.0, fps=24
            )),
        ],
    )
    source = media.probe(str(root / "assets" / "f24.mp4"))
    assert abs(source.fps - 24.0) < 0.1
    assert main(["render", str(root)]) == 0
    out = media.probe(str(root / "output" / "final.mp4"))
    assert abs(out.fps - 30.0) < 0.05


def test_source_at_60fps_is_normalised_to_30(tmp_path):
    root = _make_project(
        tmp_path / "fps60",
        [
            ("f60.mp4", lambda p: make_clip(
                p, colour="blue", seconds=6.0, fps=60
            )),
        ],
    )
    assert main(["render", str(root)]) == 0
    out = media.probe(str(root / "output" / "final.mp4"))
    assert abs(out.fps - 30.0) < 0.05


def test_video_without_audio_does_not_break_the_render(tmp_path):
    root = _make_project(
        tmp_path / "silent",
        [
            ("mute.mp4", lambda p: make_clip(
                p, colour="grey", seconds=6.0, audio=False
            )),
        ],
    )
    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 6.0)


def test_video_with_audio_is_excluded_unless_requested(tmp_path):
    """B-roll audio must not leak into the narration track by default."""
    root = _make_project(
        tmp_path / "noaudio",
        [
            ("talk.mp4", lambda p: make_clip(
                p, colour="blue", seconds=6.0, audio=True
            )),
        ],
    )
    assert main(["render", str(root)]) == 0
    out = media.probe(str(root / "output" / "final.mp4"))
    assert out.has_audio  # the voiceover is still there


def test_still_image_renders_for_the_full_interval(tmp_path):
    root = _make_project(
        tmp_path / "still",
        [
            ("photo.jpg", lambda p: make_image(p, colour="green")),
        ],
        seconds=5.0,
    )
    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 5.0)


def test_mixed_video_and_image_project(tmp_path):
    root = _make_project(
        tmp_path / "mixed",
        [
            ("a.mp4", lambda p: make_clip(p, colour="red", seconds=3.0)),
            ("b.jpg", lambda p: make_image(p, colour="blue", width=1920, height=1080)),
            ("c.mp4", lambda p: make_clip(p, colour="green", seconds=3.0)),
        ],
    )
    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 6.0)


def test_source_trimming_with_source_start(tmp_path):
    """source_start must actually move the in-point of the source."""
    root = _make_project(
        tmp_path / "trim",
        [
            ("long.mp4", lambda p: make_clip(p, colour="red", seconds=6.0)),
        ],
        seconds=2.0,
        extra_entry=None,
    )
    (root / "timeline.yaml").write_text(
        yaml.safe_dump(
            {
                "timeline": [
                    {
                        "start": 0.0,
                        "end": 2.0,
                        "asset": "assets/long.mp4",
                        "source_start": 3.0,
                        "fit": "contain",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 2.0)


def test_cover_fit_crops_instead_of_letterboxing(tmp_path):
    root = _make_project(
        tmp_path / "cover",
        [
            ("tall.mp4", lambda p: make_clip(
                p, colour="red", seconds=6.0, width=360, height=640
            )),
        ],
    )
    timeline = yaml.safe_load((root / "timeline.yaml").read_text())
    timeline["timeline"][0]["fit"] = "cover"
    (root / "timeline.yaml").write_text(
        yaml.safe_dump(timeline), encoding="utf-8"
    )
    assert main(["render", str(root)]) == 0

    from video_pipeline.validation import _decode_frame_rgb

    rgb = _decode_frame_rgb(root / "output" / "final.mp4", 3.0, 1280, 720)
    assert rgb is not None
    left_column = [rgb[(320 * 1280 + x) * 3] for x in range(0, 200, 20)]
    assert max(left_column) > 40, "cover fit should fill the frame edge-to-edge"


def test_filenames_with_special_characters_work(tmp_path):
    """Spaces, quotes, parentheses and unicode must survive the pipeline."""
    root = tmp_path / "weird names"
    root.mkdir()
    (root / "assets").mkdir()
    (root / "audio").mkdir()
    (root / "captions").mkdir()
    (root / "output").mkdir()

    names = [
        "Sean Connery (1970s) 'interview'.mp4",
        "café — naïve clip.mp4",
        "b&w footage; part 2.mp4",
    ]
    colours = ["red", "blue", "green"]
    for name, colour in zip(names, colours):
        make_clip(root / "assets" / name, colour=colour, seconds=2.0)

    make_voiceover(root / "audio" / "voice over (final).wav", seconds=6.0)

    (root / "timeline.yaml").write_text(
        yaml.safe_dump(
            {
                "timeline": [
                    {
                        "start": i * 2.0,
                        "end": (i + 1) * 2.0,
                        "asset": f"assets/{name}",
                        "fit": "contain",
                    }
                    for i, name in enumerate(names)
                ]
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    (root / "project.yaml").write_text(
        "name: weird\nvoiceover: audio/voice over (final).wav\n",
        encoding="utf-8",
    )
    (root / "captions" / "segments.json").write_text("[]", encoding="utf-8")

    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 6.0)


def test_project_with_captions_disabled(tmp_path):
    root = _make_project(
        tmp_path / "nocaptions",
        [("a.mp4", lambda p: make_clip(p, colour="red", seconds=4.0))],
        seconds=4.0,
    )
    (root / "config.yaml").write_text("captions:\n  enabled: false\n", encoding="utf-8")
    assert main(["render", str(root)]) == 0
    assert_normalised(root / "output" / "final.mp4", 4.0)