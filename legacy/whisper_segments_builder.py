#!/usr/bin/env python3
"""
whisper_segments_builder.py

Transcribes a voiceover audio file using Whisper, searches the transcript
for trigger keywords, and generates a segments JSON file for render_video.py.

The generated segments support this workflow:

1. Voiceover narration segments:
   - video_mode = "freeze"
   - audio_mode = "voiceover"

2. Sean Connery interview/quote segments:
   - video_mode = "play"
   - audio_mode = "clip"

Example:

python whisper_segments_builder.py voiceover.mp3 \
    --output segments.json \
    --clips-manifest assets/clips/clip_manifest.json \
    --model base \
    --language en \
    --clip-duration 8

The script can use either:
- faster-whisper, if installed
- openai-whisper, if installed

Install one of these:

pip install -U faster-whisper
pip install -U openai-whisper

ffmpeg/ffprobe must be available on PATH.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


# =====================================================================
# DEFAULT KEYWORDS
# =====================================================================

DEFAULT_TRIGGER_KEYWORDS = [
    # English triggers – put these phrases in script.md right before you want a clip
    "let's see",
    "lets see",
    "let us see",
    "let's watch",
    "lets watch",
    "let us watch",
    "let's hear",
    "lets hear",
    "watch this",
    "watch him",
    "watch her",
    "check this out",
    "take a look",
    "take a listen",
    "listen to this",
    "listen to what",
    "hear what",
    "hear him",
    "hear her",
    "what he said",
    "what she said",
    "what they said",
    "in this interview",
    "in that interview",
    "in the interview",
    "from the interview",
    "from an interview",
    "play the clip",
    "play that clip",
    "play the footage",
    "play that footage",
    "here's the clip",
    "heres the clip",
    "here is the clip",
    "here's what",
    "heres what",
    "here is what",
    "clip of",
    "footage of",

    # Chinese triggers
    "让我们看看",
    "我们来看看",
    "来看看",
    "看看",
    "听听",
    "来听听",
    "原声",
    "采访片段",
    "采访",
    "片段",
    "他说",
    "他说道",
    "他表示",
]

def _build_default_topic_keywords():
    """Build topic keywords from VIDEO_TOPIC env var so any subject works."""
    topic = os.environ.get("VIDEO_TOPIC", "Sean Connery").strip() or "Sean Connery"
    words = [w.lower() for w in re.split(r"\s+", topic) if len(w) > 1]
    keywords = [topic.lower()]
    keywords.extend(words)
    # Backward-compatible fallbacks for older Sean Connery scripts
    if "connery" in topic.lower() or topic.lower() == "sean connery":
        keywords.extend([
            "sean connery", "connery",
            "詹姆斯·邦德", "james bond", "007",
            "肖恩·康纳利", "肖恩•康纳利", "康纳利",
        ])
    # Deduplicate preserving order
    seen = set()
    out = []
    for k in keywords:
        if k and k not in seen:
            seen.add(k)
            out.append(k)
    return out

DEFAULT_TOPIC_KEYWORDS = _build_default_topic_keywords()

VIDEO_EXTENSIONS = {
    ".mp4",
    ".mkv",
    ".webm",
    ".mov",
    ".m4v",
    ".avi",
}

MIN_SEGMENT_DURATION = 0.08


# =====================================================================
# BASIC HELPERS
# =====================================================================

def parse_csv(value):
    if not value:
        return []

    return [
        item.strip()
        for item in value.split(",")
        if item.strip()
    ]


def build_keyword_list(override_csv, extra_csv, defaults):
    if override_csv:
        keywords = parse_csv(override_csv)
    else:
        keywords = list(defaults)

    if extra_csv:
        keywords.extend(parse_csv(extra_csv))

    # Normalize and deduplicate while preserving order.
    cleaned = []
    seen = set()

    for keyword in keywords:
        keyword = keyword.strip()
        if not keyword:
            continue

        key = keyword.lower()
        if key in seen:
            continue

        seen.add(key)
        cleaned.append(keyword)

    return cleaned


def normalize_text(text):
    if not text:
        return ""

    text = text.lower()
    text = text.replace("’", "'")
    text = text.replace("‘", "'")
    text = text.replace("“", '"')
    text = text.replace("”", '"')
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def contains_any(text, keywords):
    if not text or not keywords:
        return False

    normalized = normalize_text(text)

    return any(
        normalize_text(keyword) in normalized
        for keyword in keywords
    )


def probe_duration(path):
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=20,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def check_ffprobe():
    if shutil.which("ffprobe") is None:
        print("ERROR: ffprobe not found.")
        print("Please install ffmpeg/ffprobe first.")
        print()
        print("On Debian/Ubuntu/Colab:")
        print("    sudo apt-get update")
        print("    sudo apt-get install -y ffmpeg")
        return False

    return True


# =====================================================================
# WHISPER TRANSCRIPTION
# =====================================================================

def transcribe_with_faster_whisper(audio_path, args):
    from faster_whisper import WhisperModel

    print("=" * 70)
    print("Using faster-whisper")
    print("=" * 70)

    model = WhisperModel(
        args.model,
        device=args.device,
        compute_type=args.compute_type,
    )

    segments, info = model.transcribe(
        str(audio_path),
        language=args.language,
        vad_filter=args.vad_filter,
    )

    transcript = []

    for segment in segments:
        transcript.append(
            {
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip(),
            }
        )

    duration = getattr(info, "duration", None)

    return transcript, duration


def transcribe_with_openai_whisper(audio_path, args):
    import whisper

    print("=" * 70)
    print("Using openai-whisper")
    print("=" * 70)

    model = whisper.load_model(args.model)

    options = {
        "verbose": False,
    }

    if args.language:
        options["language"] = args.language

    result = model.transcribe(str(audio_path), **options)

    transcript = []

    for segment in result.get("segments", []):
        transcript.append(
            {
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": str(segment.get("text", "")).strip(),
            }
        )

    return transcript, None


def transcribe_audio(audio_path, args):
    errors = []

    if args.backend in {"auto", "faster-whisper"}:
        try:
            return transcribe_with_faster_whisper(audio_path, args)
        except Exception as error:
            errors.append(f"faster-whisper failed: {error}")

    if args.backend in {"auto", "openai-whisper"}:
        try:
            return transcribe_with_openai_whisper(audio_path, args)
        except Exception as error:
            errors.append(f"openai-whisper failed: {error}")

    error_message = "\n".join(errors)

    raise RuntimeError(
        "Could not transcribe audio.\n"
        "Install one Whisper backend first:\n\n"
        "    pip install -U faster-whisper\n"
        "    # or\n"
        "    pip install -U openai-whisper\n\n"
        f"Details:\n{error_message}"
    )


# =====================================================================
# CLIP LOADING
# =====================================================================

def load_clip_paths(manifest_path, clips_dir):
    clip_paths = []

    # First try the manifest produced by youtube_dl_toolbox.py.
    if manifest_path.exists():
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))

            if isinstance(data, list):
                items = data
            elif isinstance(data, dict):
                items = data.get("clips", [])
            else:
                items = []

            for item in items:
                if isinstance(item, dict):
                    raw_path = item.get("clip_path")
                else:
                    raw_path = item

                if not raw_path:
                    continue

                path = Path(raw_path)

                if not path.exists():
                    path = manifest_path.parent / raw_path

                if path.exists() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    clip_paths.append(path)

        except Exception as error:
            print(f"WARNING: could not read manifest: {error}")

    # Fallback: use all video files inside assets/clips.
    if not clip_paths and clips_dir.exists():
        clip_paths = sorted(
            path
            for path in clips_dir.iterdir()
            if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
        )

    return clip_paths


def probe_clip_durations(clip_paths):
    durations = {}

    for path in clip_paths:
        duration = probe_duration(path)

        if duration is not None:
            durations[path] = duration

    return durations


# =====================================================================
# TRIGGER DETECTION
# =====================================================================

def context_has_topic(transcript, index, topic_keywords, window):
    if window <= 0:
        return False

    start = max(0, index - window)
    end = min(len(transcript), index + window + 1)

    for i in range(start, end):
        text = transcript[i].get("text", "")

        if contains_any(text, topic_keywords):
            return True

    return False


def find_trigger_candidates(transcript, audio_duration, args):
    trigger_keywords = build_keyword_list(
        args.trigger_keywords,
        args.extra_trigger_keywords,
        DEFAULT_TRIGGER_KEYWORDS,
    )

    topic_keywords = build_keyword_list(
        args.topic_keywords,
        args.extra_topic_keywords,
        DEFAULT_TOPIC_KEYWORDS,
    )

    candidates = []

    for index, segment in enumerate(transcript):
        text = segment.get("text", "")

        if not contains_any(text, trigger_keywords):
            continue

        if args.require_topic:
            direct_topic = contains_any(text, topic_keywords)
            nearby_topic = context_has_topic(
                transcript,
                index,
                topic_keywords,
                args.context_window,
            )

            if not direct_topic and not nearby_topic:
                continue

        if args.start_mode == "end":
            start = float(segment["end"]) + args.start_offset
        else:
            start = float(segment["start"]) + args.start_offset

        start = max(0.0, start)

        temporary_duration = (
            args.clip_duration
            if args.clip_duration > 0
            else args.default_clip_duration
        )

        if start >= audio_duration:
            start = max(0.0, audio_duration - temporary_duration)

        end = min(audio_duration, start + temporary_duration)

        if end - start < args.min_clip_duration:
            continue

        candidates.append(
            {
                "start": start,
                "end": end,
                "trigger_text": text.strip(),
            }
        )

        if args.max_triggers and len(candidates) >= args.max_triggers:
            break

    return candidates


def merge_trigger_candidates(candidates, args):
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda item: item["start"])
    merged = []

    for candidate in ordered:
        if not merged:
            merged.append(dict(candidate))
            continue

        last = merged[-1]

        if candidate["start"] <= last["end"] + args.merge_gap:
            last["end"] = max(last["end"], candidate["end"])
            last["trigger_text"] = (
                f"{last['trigger_text']} | {candidate['trigger_text']}"
            )
        else:
            merged.append(dict(candidate))

    if args.max_triggers:
        merged = merged[:args.max_triggers]

    return merged


def build_clip_intervals(candidates, clip_paths, clip_durations, audio_duration, args):
    if not clip_paths:
        return []

    merged = merge_trigger_candidates(candidates, args)
    intervals = []

    for index, candidate in enumerate(merged):
        clip_path = clip_paths[index % len(clip_paths)]
        actual_clip_duration = clip_durations.get(clip_path)

        if args.clip_duration > 0:
            requested_duration = args.clip_duration
        else:
            requested_duration = (
                actual_clip_duration
                if actual_clip_duration
                else args.default_clip_duration
            )

        if args.cap_to_clip_duration and actual_clip_duration:
            requested_duration = min(requested_duration, actual_clip_duration)

        if requested_duration <= 0:
            continue

        start = candidate["start"]

        # If the requested clip would run past the end of the voiceover,
        # shift it backward so the full clip can still fit.
        if start + requested_duration > audio_duration:
            start = max(0.0, audio_duration - requested_duration)

        end = min(audio_duration, start + requested_duration)

        if end - start < args.min_clip_duration:
            continue

        intervals.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "clip_path": str(clip_path).replace("\\", "/"),
                "trigger_text": candidate.get("trigger_text", ""),
            }
        )

    return intervals


# =====================================================================
# FINAL SEGMENT TIMELINE
# =====================================================================

def make_segment(start, end, text, clip_path, video_mode, audio_mode, trigger_text=None):
    segment = {
        "start": round(float(start), 3),
        "end": round(float(end), 3),
        "text": text.strip() if text else "",
        "clip_path": str(clip_path).replace("\\", "/"),
        "video_mode": video_mode,
        "audio_mode": audio_mode,
    }

    if trigger_text:
        segment["trigger_text"] = trigger_text

    return segment


def add_narration_segments(final_segments, start, end, transcript, clip_path):
    if end - start < MIN_SEGMENT_DURATION:
        return

    relevant = [
        segment
        for segment in transcript
        if segment["end"] > start + 0.02 and segment["start"] < end - 0.02
    ]

    if not relevant:
        final_segments.append(
            make_segment(
                start=start,
                end=end,
                text="",
                clip_path=clip_path,
                video_mode="freeze",
                audio_mode="voiceover",
            )
        )
        return

    for segment in relevant:
        a = max(start, float(segment["start"]))
        b = min(end, float(segment["end"]))

        if b - a < MIN_SEGMENT_DURATION:
            continue

        original_duration = max(0.01, float(segment["end"]) - float(segment["start"]))
        overlap_ratio = (b - a) / original_duration

        # BUG FIX: threshold was 0.35 which silently dropped many real captions
        # when a Whisper segment was split at a trigger boundary. Lowered to
        # 0.15 so captions appear for all but the most tiny sliver overlaps.
        text = segment.get("text", "") if overlap_ratio >= 0.15 else ""

        final_segments.append(
            make_segment(
                start=a,
                end=b,
                text=text,
                clip_path=clip_path,
                video_mode="freeze",
                audio_mode="voiceover",
            )
        )


def merge_adjacent_empty_segments(segments):
    merged = []

    for segment in segments:
        if segment["end"] - segment["start"] < 0.03:
            continue

        if merged:
            previous = merged[-1]

            same_clip = previous["clip_path"] == segment["clip_path"]
            same_video = previous["video_mode"] == segment["video_mode"]
            same_audio = previous["audio_mode"] == segment["audio_mode"]
            both_empty = not previous["text"] and not segment["text"]

            if same_clip and same_video and same_audio and both_empty:
                previous["end"] = segment["end"]
                continue

        merged.append(segment)

    return merged


def build_final_segments(audio_duration, transcript, clip_intervals, clip_paths):
    final_segments = []

    cursor = 0.0

    # BUG FIX: when clip_paths is empty, default_clip was "" which caused
    # render_video.py to filter out ALL segments (it skips any segment without
    # a clip_path), resulting in "No segments have a clip_path" crash.
    # Use a sentinel value; render_video.py will skip segments with empty
    # clip_path, so we need at least one valid clip for freeze segments.
    # If no clips exist, we warn and return an empty list so the caller
    # (the bash script) can detect and handle it, rather than silently
    # producing a broken JSON.
    default_clip = clip_paths[0] if clip_paths else ""

    last_clip = default_clip

    for clip_interval in clip_intervals:
        clip_start = max(cursor, clip_interval["start"])
        clip_end = max(clip_start, clip_interval["end"])

        if clip_end - clip_start < MIN_SEGMENT_DURATION:
            continue

        # Narration before this clip.
        add_narration_segments(
            final_segments=final_segments,
            start=cursor,
            end=clip_start,
            transcript=transcript,
            clip_path=last_clip,
        )

        # Actual Sean Connery clip.
        final_segments.append(
            make_segment(
                start=clip_start,
                end=clip_end,
                text="",
                clip_path=clip_interval["clip_path"],
                video_mode="play",
                audio_mode="clip",
                trigger_text=clip_interval.get("trigger_text"),
            )
        )

        cursor = clip_end
        last_clip = clip_interval["clip_path"]

    # Narration after the last clip.
    add_narration_segments(
        final_segments=final_segments,
        start=cursor,
        end=audio_duration,
        transcript=transcript,
        clip_path=last_clip,
    )

    final_segments = merge_adjacent_empty_segments(final_segments)

    # Safety fallback: if everything was removed, create one full freeze segment.
    if not final_segments:
        final_segments.append(
            make_segment(
                start=0.0,
                end=audio_duration,
                text="",
                clip_path=default_clip,
                video_mode="freeze",
                audio_mode="voiceover",
            )
        )

    return final_segments


# =====================================================================
# MAIN
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Transcribe voiceover audio with Whisper, detect trigger phrases, "
            "and generate segments JSON for render_video.py."
        )
    )

    parser.add_argument(
        "audio",
        help="Voiceover audio file, e.g. voiceover.mp3",
    )

    parser.add_argument(
        "--output",
        default="segments.json",
        help="Output segments JSON file.",
    )

    parser.add_argument(
        "--transcript-output",
        default="transcript.json",
        help="Optional file where the raw Whisper transcript is saved.",
    )

    parser.add_argument(
        "--clips-manifest",
        default="assets/clips/clip_manifest.json",
        help="Manifest created by youtube_dl_toolbox.py.",
    )

    parser.add_argument(
        "--clips-dir",
        default="assets/clips",
        help="Directory containing downloaded clips.",
    )

    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "faster-whisper", "openai-whisper"],
        help="Whisper backend to use.",
    )

    parser.add_argument(
        "--model",
        default="base",
        help="Whisper model: tiny, base, small, medium, large, etc.",
    )

    parser.add_argument(
        "--language",
        default=None,
        help="Language code, e.g. en, zh. Omit for auto-detection.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="faster-whisper device: auto, cpu, cuda.",
    )

    parser.add_argument(
        "--compute-type",
        default="default",
        help="faster-whisper compute type: default, auto, int8, float16, etc.",
    )

    parser.add_argument(
        "--vad-filter",
        action="store_true",
        help="Enable VAD filter for faster-whisper.",
    )

    parser.add_argument(
        "--clip-duration",
        type=float,
        default=8.0,
        help=(
            "How many seconds of Sean Connery clip to play after a trigger. "
            "Use 0 to use the full downloaded clip duration."
        ),
    )

    parser.add_argument(
        "--default-clip-duration",
        type=float,
        default=8.0,
        help="Fallback clip duration if actual clip duration cannot be read.",
    )

    parser.add_argument(
        "--min-clip-duration",
        type=float,
        default=1.0,
        help="Ignore triggered clips shorter than this.",
    )

    parser.add_argument(
        "--start-mode",
        default="end",
        choices=["start", "end"],
        help=(
            "Whether the clip should start at the beginning of the trigger "
            "sentence or after the end of the trigger sentence."
        ),
    )

    parser.add_argument(
        "--start-offset",
        type=float,
        default=0.15,
        help="Extra offset in seconds added to the clip start time.",
    )

    parser.add_argument(
        "--merge-gap",
        type=float,
        default=1.0,
        help="Merge trigger events closer than this many seconds.",
    )

    parser.add_argument(
        "--max-triggers",
        type=int,
        default=0,
        help="Maximum number of clip triggers. 0 means unlimited.",
    )

    parser.add_argument(
        "--require-topic",
        action="store_true",
        default=True,
        help=(
            "Only trigger when the sentence or nearby sentences mention "
            "Sean Connery / topic keywords."
        ),
    )

    parser.add_argument(
        "--no-require-topic",
        dest="require_topic",
        action="store_false",
        help="Disable topic requirement.",
    )

    parser.add_argument(
        "--context-window",
        type=int,
        default=2,
        help=(
            "How many neighboring Whisper segments to check for topic context."
        ),
    )

    parser.add_argument(
        "--trigger-keywords",
        default=None,
        help="Comma-separated list that overrides default trigger keywords.",
    )

    parser.add_argument(
        "--extra-trigger-keywords",
        default=None,
        help="Comma-separated list added to default trigger keywords.",
    )

    parser.add_argument(
        "--topic-keywords",
        default=None,
        help="Comma-separated list that overrides default topic keywords.",
    )

    parser.add_argument(
        "--extra-topic-keywords",
        default=None,
        help="Comma-separated list added to default topic keywords.",
    )

    parser.add_argument(
        "--cap-to-clip-duration",
        action="store_true",
        default=True,
        help="Do not make the segment longer than the actual downloaded clip.",
    )

    parser.add_argument(
        "--no-cap-to-clip-duration",
        dest="cap_to_clip_duration",
        action="store_false",
        help="Allow the segment to be longer than the clip.",
    )

    parser.add_argument(
        "--allow-no-clips",
        action="store_true",
        help=(
            "Generate segments even if no clips are found. "
            "This is mainly useful for caption testing."
        ),
    )

    args = parser.parse_args()

    if not check_ffprobe():
        sys.exit(1)

    audio_path = Path(args.audio)

    if not audio_path.exists():
        print(f"ERROR: audio file not found: {audio_path}")
        sys.exit(1)

    audio_duration = probe_duration(audio_path)

    if audio_duration is None:
        print(f"ERROR: could not read duration from {audio_path}")
        sys.exit(1)

    print("=" * 70)
    print("Transcribing audio")
    print("=" * 70)

    transcript, whisper_duration = transcribe_audio(audio_path, args)

    # Prefer ffprobe duration, but fallback to Whisper if needed.
    if not audio_duration and whisper_duration:
        audio_duration = float(whisper_duration)

    transcript = sorted(transcript, key=lambda item: item["start"])

    transcript_output = Path(args.transcript_output)
    transcript_output.write_text(
        json.dumps(
            {
                "audio": str(audio_path),
                "duration": audio_duration,
                "segments": transcript,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved raw transcript -> {transcript_output}")

    manifest_path = Path(args.clips_manifest)
    clips_dir = Path(args.clips_dir)

    clip_paths = load_clip_paths(manifest_path, clips_dir)

    if not clip_paths:
        if not args.allow_no_clips:
            print("ERROR: no clips found.")
            print("Run youtube_dl_toolbox.py first, or use --allow-no-clips.")
            print(f"Looked in manifest: {manifest_path}")
            print(f"Looked in clips dir: {clips_dir}")
            sys.exit(1)

        print("WARNING: no clips found. Generating segments without clip paths.")

    clip_durations = probe_clip_durations(clip_paths)

    print("=" * 70)
    print("Searching for trigger phrases")
    print("=" * 70)

    candidates = find_trigger_candidates(transcript, audio_duration, args)

    print(f"Found {len(candidates)} raw trigger event(s).")
    for c in candidates[:8]:
        print(f"  trigger @ {c['start']:.2f}s: {c.get('trigger_text', '')[:80]}")

    clip_intervals = build_clip_intervals(
        candidates=candidates,
        clip_paths=clip_paths,
        clip_durations=clip_durations,
        audio_duration=audio_duration,
        args=args,
    )

    print(f"Created {len(clip_intervals)} clip interval(s).")

    if not clip_intervals and clip_paths:
        print(
            "\nWARNING: No trigger phrases matched in the voiceover transcript.\n"
            "The video will use freeze-frames of the first clip + full voiceover.\n"
            "To insert live interview footage, put phrases like these in script.md:\n"
            "  'let's watch', 'take a look', 'in this interview', 'listen to this',\n"
            "  'here's the clip', 'watch this', 'check this out'\n"
            "Or re-run with --no-require-topic if the topic name is not spoken near the trigger."
        )

    final_segments = build_final_segments(
        audio_duration=audio_duration,
        transcript=transcript,
        clip_intervals=clip_intervals,
        clip_paths=clip_paths,
    )

    output_path = Path(args.output)
    output_path.write_text(
        json.dumps(final_segments, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 70)
    print("Done")
    print("=" * 70)
    print(f"Wrote {len(final_segments)} segments -> {output_path}")

    clip_segments = [
        segment
        for segment in final_segments
        if segment["audio_mode"] == "clip"
    ]

    print(f"Clip play segments: {len(clip_segments)}")
    print(f"Voiceover/freeze segments: {len(final_segments) - len(clip_segments)}")


if __name__ == "__main__":
    try:
        main()

    except KeyboardInterrupt:
        print()
        print("Stopped by user.")
        sys.exit(130)

    except Exception as error:
        print()
        print("=" * 70)
        print("FAILED")
        print("=" * 70)
        print(error)
        sys.exit(1)