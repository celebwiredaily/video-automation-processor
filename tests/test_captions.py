"""Unit tests: caption parsing, wrapping, timing and ASS generation."""

from __future__ import annotations

import json

import pytest

from video_pipeline.captions import (
    CaptionCue,
    build_ass,
    hex_to_ass,
    load_cues,
    prepare,
    sanitize_text,
    seconds_to_ass_time,
)
from video_pipeline.config import CaptionConfig, Config


SRT = """1
00:00:00,000 --> 00:00:02,400
Connery is far more fascinating than any

2
00:00:02,400 --> 00:00:05,000
of the actors who played him
"""


def write(tmp_path, name, content):
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ------------------------------------------------------------------ parsing


def test_loads_srt(tmp_path):
    cues = load_cues(write(tmp_path, "c.srt", SRT))
    assert len(cues) == 2
    assert cues[0].start == pytest.approx(0.0)
    assert cues[0].end == pytest.approx(2.4)
    assert cues[1].text == "of the actors who played him"


def test_loads_segments_json(tmp_path):
    path = write(
        tmp_path,
        "c.json",
        json.dumps([{"start": 1.0, "end": 2.0, "text": "hello"}]),
    )
    cues = load_cues(path)
    assert cues == [CaptionCue(1.0, 2.0, "hello")]


def test_loads_segments_json_wrapped_in_an_object(tmp_path):
    path = write(
        tmp_path,
        "c.json",
        json.dumps({"segments": [{"start": 0, "end": 1, "text": "hi"}]}),
    )
    assert len(load_cues(path)) == 1


def test_srt_timestamps_convert_correctly():
    assert seconds_to_ass_time(0.0) == "0:00:00.00"
    assert seconds_to_ass_time(2.4) == "0:00:02.40"
    assert seconds_to_ass_time(3661.25) == "1:01:01.25"


def test_cue_ordering_is_normalised(tmp_path):
    path = write(
        tmp_path,
        "c.json",
        json.dumps(
            [
                {"start": 5, "end": 6, "text": "second"},
                {"start": 1, "end": 2, "text": "first"},
            ]
        ),
    )
    cues = load_cues(path)
    assert [c.text for c in cues] == ["first", "second"]


# ----------------------------------------------------------------- sanitising


def test_ass_override_blocks_are_neutralised():
    """A stray brace would open an ASS override block and corrupt rendering."""
    assert sanitize_text("hello {\\p1} world") == "hello (p1) world"
    assert sanitize_text("back\\slash") == "backslash"
    assert sanitize_text("  spaced   out  ") == "spaced out"


# ----------------------------------------------------------------- preparing


def test_short_text_stays_on_one_line():
    config = CaptionConfig(**{**vars(Config.from_mapping({}).captions)})
    cues, warnings = prepare(
        [CaptionCue(0.0, 2.0, "Connery is fascinating")], config
    )
    assert len(cues) == 1
    assert "\n" not in cues[0].text
    assert warnings == []


def test_long_text_is_chunked_by_word_count():
    config = Config.from_mapping({}).captions
    words = " ".join(f"word{i}" for i in range(20))
    cues, warnings = prepare([CaptionCue(0.0, 8.0, words)], config)
    assert len(cues) > 1
    # Chunks are contiguous and cover the whole interval.
    assert cues[0].start == pytest.approx(0.0)
    assert cues[-1].end == pytest.approx(8.0)
    for a, b in zip(cues, cues[1:]):
        assert b.start == pytest.approx(a.end)
    assert any("split" in w for w in warnings)


def test_lines_are_wrapped_within_the_configured_width():
    config = Config.from_mapping({}).captions
    text = "the overwhelming stench from the very nearby restaurant district"
    cues, _ = prepare([CaptionCue(0.0, 5.0, text)], config)
    for line in cues[0].text.split("\n"):
        assert len(line) <= config.max_chars_per_line
    assert len(cues[0].text.split("\n")) <= config.max_lines


def test_cues_are_clamped_to_the_video_duration():
    config = Config.from_mapping({}).captions
    cues, warnings = prepare(
        [CaptionCue(0.0, 10.0, "runs past the end")],
        config,
        video_duration=5.0,
    )
    assert cues[0].end <= 5.0
    assert any("past the end" in w for w in warnings)


def test_non_positive_duration_is_dropped():
    config = Config.from_mapping({}).captions
    cues, warnings = prepare([CaptionCue(2.0, 2.0, "zero length")], config)
    assert cues == []
    assert any("non-positive" in w for w in warnings)


def test_overlapping_cues_are_separated():
    config = Config.from_mapping({}).captions
    cues, warnings = prepare(
        [CaptionCue(0.0, 3.0, "first"), CaptionCue(2.0, 5.0, "second")], config
    )
    for a, b in zip(cues, cues[1:]):
        assert b.start >= a.end - 1e-6
    assert any("overlaps" in w for w in warnings)


def test_uppercase_is_configurable():
    base = Config.from_mapping({}).captions
    shouty = CaptionConfig(**{**vars(base), "uppercase": True})
    cues, _ = prepare([CaptionCue(0.0, 2.0, "mixed case words")], shouty)
    assert cues[0].text == "MIXED CASE WORDS"


def test_empty_cue_text_is_ignored():
    config = Config.from_mapping({}).captions
    cues, _ = prepare([CaptionCue(0.0, 2.0, "   ")], config)
    assert cues == []


# ---------------------------------------------------------------- ASS output


def test_hex_to_ass_is_bgr_order():
    assert hex_to_ass("#F7FF00") == "&H0000FFF7"
    assert hex_to_ass("#FFFFFF") == "&H00FFFFFF"
    assert hex_to_ass("#000000") == "&H00000000"
    assert hex_to_ass("#fff") == "&H00FFFFFF"


def test_ass_header_declares_the_render_resolution():
    config = Config.from_mapping({}).captions
    ass = build_ass(
        [CaptionCue(0.0, 2.0, "hello")], config, width=1280, height=720
    )
    assert "PlayResX: 1280" in ass
    assert "PlayResY: 720" in ass
    assert "Montserrat" in ass
    assert "&H0000FFF7" in ass            # yellow fill
    assert ",1," in ass                    # BorderStyle 1 (outline+shadow)
    assert "Dialogue: 0,0:00:00.00,0:00:02.00,Default" in ass


def test_ass_escapes_newlines_as_line_breaks():
    config = Config.from_mapping({}).captions
    ass = build_ass(
        [CaptionCue(0.0, 2.0, "line one\nline two")], config, width=1280, height=720
    )
    assert r"line one\Nline two" in ass


def test_ass_font_size_scales_for_1080p():
    captions = Config.from_mapping({}).captions
    ass = build_ass(
        [CaptionCue(0.0, 2.0, "x")], captions, width=1920, height=1080
    )
    assert f"Default,Montserrat,{captions.scaled_font_size(1080)}," in ass