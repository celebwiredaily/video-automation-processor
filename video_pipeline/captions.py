"""Caption timeline -> ASS.

Caption *timing* comes from data (``captions/segments.json`` or an SRT file).
Caption *styling* is pure data in ``config.yaml``. This module turns one into
the other and writes an ASS file; libass (inside FFmpeg) does the drawing.

The style defaults were measured by decoding frames from the supplied
reference video rather than guessed.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .config import CaptionConfig
from .errors import ConfigError

SRT_TIMESTAMP = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})"
)


@dataclass(frozen=True)
class CaptionCue:
    start: float
    end: float
    text: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "text": self.text,
        }


# --------------------------------------------------------------------- loading


def load_cues(path: Path) -> list[CaptionCue]:
    """Load caption cues from segments.json or an SRT/VTT file."""
    if not path.exists():
        raise ConfigError("Caption file does not exist.", file=str(path))
    suffix = path.suffix.lower()
    if suffix in {".srt", ".vtt"}:
        return _load_srt(path)
    return _load_segments_json(path)


def _load_segments_json(path: Path) -> list[CaptionCue]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(
            "Caption JSON could not be read.", file=str(path), error=str(exc)
        ) from exc

    # Accept either a bare list or {"segments": [...]}.
    if isinstance(data, dict):
        data = data.get("segments") or data.get("cues") or []
    if not isinstance(data, list):
        raise ConfigError(
            "Caption JSON must be a list of {start, end, text} objects.",
            file=str(path),
            found=type(data).__name__,
        )

    cues: list[CaptionCue] = []
    for position, item in enumerate(data):
        if not isinstance(item, dict):
            raise ConfigError(
                f"Caption entry {position} is not an object.", file=str(path)
            )
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(
                f"Caption entry {position} needs numeric `start` and `end`.",
                file=str(path),
                detail=str(exc),
            ) from exc
        text = str(item.get("text") or "").strip()
        if text:
            cues.append(CaptionCue(start=start, end=end, text=text))

    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def _load_srt(path: Path) -> list[CaptionCue]:
    try:
        raw = path.read_text(encoding="utf-8-sig")
    except OSError as exc:
        raise ConfigError("Caption file could not be read.", file=str(path), error=str(exc)) from exc

    cues: list[CaptionCue] = []
    blocks = re.split(r"\n\s*\n", raw.replace("\r\n", "\n").strip())
    for block in blocks:
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        timing_index = next(
            (i for i, line in enumerate(lines) if SRT_TIMESTAMP.search(line)), None
        )
        if timing_index is None:
            continue
        match = SRT_TIMESTAMP.search(lines[timing_index])
        assert match is not None
        start = _ts_to_seconds(*match.group(1, 2, 3, 4))
        end = _ts_to_seconds(*match.group(5, 6, 7, 8))
        text = " ".join(lines[timing_index + 1 :]).strip()
        text = re.sub(r"<[^>]+>", "", text)
        if text:
            cues.append(CaptionCue(start=start, end=end, text=text))
    cues.sort(key=lambda c: (c.start, c.end))
    return cues


def _ts_to_seconds(hours: str, minutes: str, seconds: str, millis: str) -> float:
    return (
        int(hours) * 3600
        + int(minutes) * 60
        + int(seconds)
        + int(millis.ljust(3, "0")) / 1000.0
    )


# ---------------------------------------------------------------- preparation


def sanitize_text(text: str) -> str:
    """Make arbitrary narration safe to place in an ASS dialogue line.

    ``{`` and ``}`` open ASS override blocks and ``\\`` introduces escapes;
    neither can appear literally in caption text, so all three are removed
    rather than escaped (ASS has no escape for them).
    """
    cleaned = text.replace("{", "(").replace("}", ")").replace("\\", "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def prepare(
    cues: Sequence[CaptionCue],
    caption_config: CaptionConfig,
    *,
    video_duration: float | None = None,
) -> tuple[list[CaptionCue], list[str]]:
    """Chunk, wrap and clamp cues. Returns (cues, warnings)."""
    warnings: list[str] = []
    prepared: list[CaptionCue] = []
    max_chars = caption_config.max_chars_per_line
    max_lines = max(1, caption_config.max_lines)
    hard_limit = max_chars * max_lines

    for cue in cues:
        text = sanitize_text(cue.text)
        if not text:
            continue
        if cue.duration <= 0:
            warnings.append(
                f"caption at {cue.start:.2f}s has non-positive duration and was dropped"
            )
            continue

        pieces = _chunk_text(text, cue.start, cue.end, caption_config.max_words_per_caption, hard_limit)
        if not pieces:
            continue

        if len(pieces) > 1 and hard_limit:
            warnings.append(
                f"caption at {cue.start:.2f}s was split into {len(pieces)} on-screen "
                f"cues to respect max_words_per_caption/max_chars_per_line"
            )

        for piece_start, piece_end, piece_text in pieces:
            lines = _wrap(piece_text, max_chars, max_lines)
            prepared.append(
                CaptionCue(
                    start=piece_start,
                    end=piece_end,
                    text="\n".join(lines),
                )
            )

    # Clamp to the video, keeping cues inside the frame.
    clamped: list[CaptionCue] = []
    for cue in prepared:
        if video_duration is not None and cue.end > video_duration + 0.5:
            warnings.append(
                f"caption at {cue.start:.2f}s runs past the end of the video and was "
                f"trimmed to {video_duration:.2f}s"
            )
            cue = CaptionCue(cue.start, min(cue.end, video_duration), cue.text)
        if cue.duration < caption_config.min_caption_seconds:
            warnings.append(
                f"caption at {cue.start:.2f}s lasts {cue.duration:.3f}s, shorter than "
                f"min_caption_seconds ({caption_config.min_caption_seconds}); it may flash"
            )
        if cue.duration <= 0:
            continue
        clamped.append(cue)

    # Overlapping cues would render as overlapping text.
    ordered = sorted(clamped, key=lambda c: (c.start, c.end))
    deduped: list[CaptionCue] = []
    for cue in ordered:
        if deduped and cue.start < deduped[-1].end - 1e-3:
            warnings.append(
                f"caption at {cue.start:.2f}s overlaps the previous cue and was adjusted"
            )
            previous = deduped[-1]
            boundary = max(previous.start, cue.start)
            if cue.end - boundary >= caption_config.min_caption_seconds:
                deduped[-1] = CaptionCue(previous.start, boundary, previous.text)
                deduped.append(
                    CaptionCue(boundary, cue.end, cue.text)
                )
            else:
                previous_end = previous.end
                deduped[-1] = CaptionCue(
                    previous.start, max(previous.start + 0.05, previous_end), previous.text
                )
            continue
        deduped.append(cue)

    if caption_config.uppercase:
        deduped = [
            CaptionCue(cue.start, cue.end, cue.text.upper()) for cue in deduped
        ]

    return deduped, warnings


def _chunk_text(
    text: str,
    start: float,
    end: float,
    max_words: int,
    hard_char_limit: int,
) -> list[tuple[float, float, str]]:
    """Split one caption into on-screen chunks, timing them by character weight.

    Character-proportional timing keeps chunks contiguous and in sync with
    speech without needing word-level timestamps.
    """
    words = text.split()
    if not words:
        return []

    max_words = max(1, max_words)
    if len(words) <= max_words and len(text) <= hard_char_limit:
        return [(start, end, text)]

    groups: list[list[str]] = []
    current: list[str] = []
    current_len = 0
    for word in words:
        candidate_len = current_len + len(word) + (1 if current else 0)
        would_overflow = len(current) >= max_words or candidate_len > hard_char_limit
        if current and would_overflow:
            groups.append(current)
            current = [word]
            current_len = len(word)
        else:
            current.append(word)
            current_len = candidate_len
    if current:
        groups.append(current)

    if len(groups) == 1:
        return [(start, end, groups[0] and " ".join(groups[0]) or text)]

    weights = [sum(len(w) + 1 for w in group) for group in groups]
    total_weight = sum(weights) or 1
    span = max(0.01, end - start)

    pieces: list[tuple[float, float, str]] = []
    cursor = start
    for group, weight in zip(groups, weights):
        piece_duration = span * (weight / total_weight)
        piece_start = cursor
        piece_end = cursor + piece_duration
        pieces.append((piece_start, piece_end, " ".join(group)))
        cursor = piece_end

    # Absorb rounding drift into the final chunk so the caption block ends on
    # the original end time rather than a few centiseconds short.
    if pieces:
        last_start, _, last_text = pieces[-1]
        pieces[-1] = (last_start, end, last_text)
    return pieces


def _wrap(text: str, max_chars: int, max_lines: int) -> list[str]:
    """Greedy wrap that keeps line lengths close to even.

    Even line lengths matter visually: a 6-word line next to a 40-character
    line reads badly at this font size.
    """
    text = text.replace("\n", " ").strip()
    if not text:
        return []
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    words = text.split()
    lines: list[str] = []
    current: list[str] = []
    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
        else:
            current.append(word)
    if current:
        lines.append(" ".join(current))

    if len(lines) <= max_lines:
        return _balance(lines)

    # Too many lines: keep the first max_lines and keep merging the remainder
    # into the last one rather than dropping narration.
    head = lines[: max_lines - 1]
    tail = " ".join(lines[max_lines - 1 :])
    return _balance([*head, tail])


def _balance(lines: list[str]) -> list[str]:
    if len(lines) < 2:
        return lines
    target = sum(len(line) for line in lines) / len(lines)
    balanced = [lines[0]]
    for line in lines[1:]:
        if abs(len(line) - target) < abs(len(balanced[-1]) - target):
            balanced[-1] = f"{balanced[-1]} {line}"
        else:
            balanced.append(line)
    return balanced


# ------------------------------------------------------------------ ASS output


def hex_to_ass(value: str, alpha: int = 0) -> str:
    """#RRGGBB -> &HAABBGGRR (ASS stores BGR)."""
    text = value.strip().lstrip("#")
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ConfigError(f"Caption colour must be #RRGGBB, got {value!r}")
    try:
        r = int(text[0:2], 16)
        g = int(text[2:4], 16)
        b = int(text[4:6], 16)
    except ValueError as exc:
        raise ConfigError(f"Caption colour must be #RRGGBB, got {value!r}") from exc
    return f"&H{alpha:02X}{b:02X}{g:02X}{r:02X}"


def seconds_to_ass_time(value: float) -> str:
    centis = max(0, int(round(value * 100)))
    hours, remainder = divmod(centis, 360000)
    minutes, remainder = divmod(remainder, 6000)
    seconds, cs = divmod(remainder, 100)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}.{cs:02d}"


def build_ass(
    cues: Sequence[CaptionCue],
    caption_config: CaptionConfig,
    *,
    width: int,
    height: int,
) -> str:
    scaled = caption_config.scaled(height)
    style_name = "Default"

    header = (
        "[Script Info]\n"
        "; Generated by video_pipeline. Do not edit by hand.\n"
        "ScriptType: v4.00+\n"
        f"PlayResX: {width}\n"
        f"PlayResY: {height}\n"
        f"WrapStyle: {scaled.wrap_style}\n"
        "ScaledBorderAndShadow: yes\n"
        "YCbCr Matrix: TV.709\n"
        "\n"
        "[V4+ Styles]\n"
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding\n"
        f"Style: {style_name},{scaled.font_name},{scaled.font_size},"
        f"{hex_to_ass(scaled.primary_color)},{hex_to_ass(scaled.primary_color)},"
        f"{hex_to_ass(scaled.outline_color)},&H80000000,"
        f"{-1 if scaled.bold else 0},{-1 if scaled.italic else 0},0,0,"
        f"100,100,0,0,1,{scaled.outline:g},{scaled.shadow:g},{scaled.alignment},"
        f"{scaled.margin_h},{scaled.margin_h},{scaled.margin_v},1\n"
        "\n"
        "[Events]\n"
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
    )

    lines = []
    for cue in cues:
        text = cue.text.replace("\n", "\\N")
        lines.append(
            f"Dialogue: 0,{seconds_to_ass_time(cue.start)},"
            f"{seconds_to_ass_time(cue.end)},{style_name},,0,0,0,,{text}"
        )
    return header + "\n".join(lines) + ("\n" if lines else "")


def write_ass(
    path: Path,
    cues: Sequence[CaptionCue],
    caption_config: CaptionConfig,
    *,
    width: int,
    height: int,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        build_ass(cues, caption_config, width=width, height=height),
        encoding="utf-8",
    )
    return path


def load_and_prepare(
    captions_file: Path | None,
    caption_config: CaptionConfig,
    *,
    video_duration: float | None,
) -> tuple[list[CaptionCue], list[str]]:
    """Load raw cues and normalise them for rendering.

    An absent or empty caption set is a valid state - the video simply renders
    without captions. A malformed file is an error, because silently dropping
    captions the author expected would be worse than failing.
    """
    if captions_file is None:
        return [], []
    if not captions_file.exists():
        return [], [f"caption file not found: {captions_file} (rendering without captions)"]
    raw = load_cues(captions_file)
    if not raw:
        return [], ["caption file contained no cues; rendering without captions"]
    return prepare(raw, caption_config, video_duration=video_duration)