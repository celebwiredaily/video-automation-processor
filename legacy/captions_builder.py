"""
captions_builder.py

Converts segments JSON (start/end/text, from wts_parser.py) into an .ass
subtitle file.

Modified style to match image:
- Fixed to bottom center.
- White caption text, forced ALL CAPS.
- Heavy, thick black outline (outline=6.0).
- Default font: Anton.
- Default font size: 62.
- Tightened bottom margin (margin_v=0.04) so text sits low.

Usage:
python captions_builder.py segments.json captions.ass --resolution 1920x1080
"""

import argparse
import json
from pathlib import Path


def seconds_to_ass_time(t):
    """
    Convert seconds to ASS timestamp.
    """
    total_centis = int(round(t * 100))
    centis = total_centis % 100

    total_seconds = total_centis // 100
    s = total_seconds % 60

    total_minutes = total_seconds // 60
    m = total_minutes % 60

    h = total_minutes // 60

    return f"{h:01d}:{m:02d}:{s:02d}.{centis:02d}"


def auto_fontsize(width, height):
    """
    Auto-scale font size based on the smaller frame dimension.
    """
    base_dim = min(width, height)
    return max(28, round(base_dim * 0.058))


def auto_margins(width, height):
    """
    Horizontal and vertical margins as a fraction of resolution.
    """
    margin_lr = round(width * 0.09)
    # Reduced to 0.04 to place text closer to the bottom like the image
    margin_v = round(height * 0.04)

    return margin_lr, margin_v


def split_into_word_chunks(text, start, end, words_per_chunk=2):
    """
    Breaks a sentence-level segment's text into chunks of words_per_chunk words.
    """
    words = text.split()
    if not words:
        return []

    total_duration = max(0.01, end - start)
    weights = [len(w) + 1 for w in words]
    total_weight = sum(weights)

    chunks = []
    cursor = start

    for i in range(0, len(words), words_per_chunk):
        group_words = words[i:i + words_per_chunk]
        group_weight = sum(weights[i:i + words_per_chunk])

        chunk_duration = total_duration * (group_weight / total_weight)
        chunk_start = cursor
        chunk_end = cursor + chunk_duration

        chunks.append((chunk_start, chunk_end, " ".join(group_words)))
        cursor = chunk_end

    if chunks:
        last_start, _, last_text = chunks[-1]
        chunks[-1] = (last_start, end, last_text)

    return chunks


def build_ass(
    segments,
    width,
    height,
    fontsize,
    alignment,
    font,
    bold,
    margin_lr,
    margin_v,
    words_per_chunk=None,
    outline=6.0,
    shadow=0.0,
):
    bold_flag = 1 if bold else 0

    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {width}
PlayResY: {height}
WrapStyle: 0
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,{font},{fontsize},&H00FFFFFF,&H000000FF,&H00000000,&H64000000,{bold_flag},0,0,0,100,100,0,0,1,{outline},{shadow},{alignment},{margin_lr},{margin_lr},{margin_v},1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    lines = []

    for seg in segments:
        text = seg.get("text", "").strip()
        if not text:
            continue

        if words_per_chunk:
            chunk_list = split_into_word_chunks(
                text,
                seg["start"],
                seg["end"],
                words_per_chunk,
            )
        else:
            chunk_list = [(seg["start"], seg["end"], text)]

        for chunk_start, chunk_end, chunk_text in chunk_list:
            start = seconds_to_ass_time(chunk_start)
            end = seconds_to_ass_time(chunk_end)

            # Convert text to ALL CAPS to match the image style
            safe_text = chunk_text.upper().replace("\n", "\\N")

            lines.append(
                f"Dialogue: 0,{start},{end},Default,,0,0,0,,{safe_text}"
            )

    return header + "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "input_segments",
        help="Input JSON segments file."
    )

    parser.add_argument(
        "output_ass",
        help="Output .ass caption file."
    )

    # Changed default to landscape 1920x1080 to match image aspect ratio
    parser.add_argument(
        "--resolution",
        default="1920x1080",
        help="WIDTHxHEIGHT. Must match your final render resolution."
    )

    parser.add_argument(
        "--fontsize",
        type=int,
        default=62,
        help="Fixed font size in px. Default: 62."
    )

    parser.add_argument(
        "--auto-fontsize",
        action="store_true",
        help="Ignore fixed font size and auto-scale based on resolution."
    )

    parser.add_argument(
        "--font",
        default="Anton",
        help="Caption font. Default: Anton."
    )

    parser.add_argument(
        "--bold",
        action="store_true",
        default=False,
        help="Enable ASS bold."
    )

    parser.add_argument(
        "--no-bold",
        dest="bold",
        action="store_false"
    )

    parser.add_argument(
        "--alignment",
        type=int,
        default=2,
        help="ASS alignment code. 2=bottom-center."
    )

    parser.add_argument(
        "--margin-percent",
        type=float,
        default=None,
        help="Override horizontal safe margin fraction."
    )

    parser.add_argument(
        "--margin-v-percent",
        type=float,
        default=None,
        help="Override vertical margin fraction."
    )

    # Increased default outline thickness from 2.0 to 6.0 to match image
    parser.add_argument(
        "--outline",
        type=float,
        default=6.0,
        help="Outline thickness. Default: 6.0 for heavy black border."
    )

    parser.add_argument(
        "--shadow",
        type=float,
        default=0.0,
        help="Shadow strength. Default: 0."
    )

    parser.add_argument(
        "--words-per-caption",
        type=int,
        default=None,
        help="Show only this many words on screen at a time."
    )

    args = parser.parse_args()

    try:
        width, height = (
            int(x) for x in args.resolution.lower().split("x")
        )
    except Exception:
        parser.error("--resolution must be in WIDTHxHEIGHT format, e.g. 1920x1080")

    segments = json.loads(
        Path(args.input_segments).read_text(encoding="utf-8")
    )

    if args.auto_fontsize:
        fontsize = auto_fontsize(width, height)
    else:
        fontsize = args.fontsize

    margin_lr, margin_v = auto_margins(width, height)

    if args.margin_percent is not None:
        margin_lr = round(width * args.margin_percent)

    if args.margin_v_percent is not None:
        margin_v = round(height * args.margin_v_percent)

    ass_content = build_ass(
        segments,
        width,
        height,
        fontsize,
        args.alignment,
        args.font,
        args.bold,
        margin_lr,
        margin_v,
        words_per_chunk=args.words_per_caption,
        outline=args.outline,
        shadow=args.shadow,
    )

    Path(args.output_ass).write_text(
        ass_content,
        encoding="utf-8"
    )

    print(
        f"Wrote {len(segments)} caption events -> {args.output_ass} "
        f"(font={args.font}, fontsize={fontsize}, bold={args.bold}, "
        f"alignment={args.alignment}, outline={args.outline}, "
        f"shadow={args.shadow}, margin_lr={margin_lr}px, "
        f"margin_v={margin_v}px, uppercase=True)"
    )


if __name__ == "__main__":
    main()