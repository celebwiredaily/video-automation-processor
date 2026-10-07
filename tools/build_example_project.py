#!/usr/bin/env python3
"""Build the ~5-minute production-style integration project.

Assets are derived from the supplied reference video, which is the only real
footage available in this repository (the configured Google Drive folder is
empty - see the report).

What this produces:
  audio/voiceover.wav      real narration, ~5 minutes, taken from the reference
  assets/*.mp4             real B-roll sub-clips
  assets/*.jpg             real stills
  captions/segments.json   real transcript timings from faster-whisper

The reference video has its own captions burned into the bottom of the frame.
Those are cropped away when cutting the B-roll so the pipeline's captions are
the only ones on screen - otherwise two caption layers would overlap. That
crop is fixture preparation, not pipeline behaviour: video_pipeline never
crops for effect, only to fit the target canvas.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from video_pipeline import ffmpeg  # noqa: E402

REFERENCE = ROOT / "How Sean Connery Escaped the Curse of James Bond.mp4"
PROJECT = ROOT / "examples" / "sean-connery-5min"
TARGET_SECONDS = 300.0

# The reference's caption band occupies roughly the bottom 100px at 720p.
CAPTION_CROP = "crop=1280:620:0:0,scale=1280:720:flags=lanczos"


def run(args, stage):
    ffmpeg.run(args, stage=stage, timeout=1800)


def build_voiceover() -> Path:
    audio_dir = PROJECT / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    out = audio_dir / "voiceover.wav"
    print(f"[1/5] extracting {TARGET_SECONDS:.0f}s of real narration")
    run(
        [
            ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
            "-ss", "30",
            "-t", f"{TARGET_SECONDS}",
            "-i", str(REFERENCE),
            "-vn",
            "-af", "aresample=48000",
            "-c:a", "pcm_s16le",
            "-ac", "2",
            str(out),
        ],
        stage="extract_voiceover",
    )
    return out


def build_assets(duration: float) -> list[Path]:
    """Cut varied real sub-clips plus a few stills from the reference."""
    assets = PROJECT / "assets"
    assets.mkdir(parents=True, exist_ok=True)

    # 16 clips of ~19s each keeps every source longer than the slice it serves,
    # so the short-clip policy is never silently triggered.
    clip_count = 16
    clip_seconds = duration / clip_count
    made: list[Path] = []
    print(f"[2/5] cutting {clip_count} real B-roll clips of {clip_seconds:.1f}s")

    # Sample from spread-out points in the source so the timeline is visually
    # varied rather than 16 consecutive seconds of one scene.
    source_span = 560.0
    for i in range(clip_count):
        start = 20.0 + (source_span - clip_seconds) * (i / max(1, clip_count - 1))
        out = assets / f"broll_{i:02d}.mp4"
        run(
            [
                ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
                "-ss", f"{start:.2f}",
                "-t", f"{clip_seconds + 0.5:.2f}",
                "-i", str(REFERENCE),
                "-vf", CAPTION_CROP,
                "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
                "-pix_fmt", "yuv420p",
                str(out),
            ],
            stage=f"cut_broll_{i:02d}",
        )
        made.append(out)

    print(f"[3/5] extracting {3} real stills")
    for i, at in enumerate((75.0, 240.0, 410.0)):
        out = assets / f"still_{i:02d}.jpg"
        run(
            [
                ffmpeg.ffmpeg_bin(), "-y", "-hide_banner", "-nostdin", "-loglevel", "error",
                "-ss", f"{at:.2f}",
                "-i", str(REFERENCE),
                "-frames:v", "1",
                "-vf", CAPTION_CROP,
                "-q:v", "3",
                str(out),
            ],
            stage=f"extract_still_{i}",
        )
        made.append(out)

    return made


def build_captions(voiceover: Path, duration: float) -> Path:
    captions_dir = PROJECT / "captions"
    captions_dir.mkdir(parents=True, exist_ok=True)
    out = captions_dir / "segments.json"

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise SystemExit(
            "faster-whisper is required to build real caption timings.\n"
            "  pip install faster-whisper"
        )

    print("[4/5] transcribing the voiceover with faster-whisper (CPU)")
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(
        str(voiceover), language="en", vad_filter=True, beam_size=1
    )

    cues = []
    for seg in segments:
        text = (seg.text or "").strip()
        if not text:
            continue
        start = max(0.0, float(seg.start))
        end = min(duration, float(seg.end))
        if end <= start:
            continue
        cues.append({"start": round(start, 3), "end": round(end, 3), "text": text})

    out.write_text(json.dumps(cues, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"      {len(cues)} real caption cues written to {out}")
    return out


def write_project_files(duration: float) -> None:
    timeline: list[str] = ["timeline:"]
    span = duration / 16.0
    # Interleave three stills among the video clips so the timeline mixes media.
    still_at = {4: "still_00.jpg", 9: "still_01.jpg", 14: "still_02.jpg"}
    cursor = 0.0
    for i in range(16):
        start = round(cursor, 3)
        cursor = round(start + span, 3)
        asset = still_at.get(i, f"broll_{i:02d}.mp4")
        timeline += [
            f"  - start: {start}",
            f"    end: {cursor}",
            f"    asset: assets/{asset}",
            "    fit: contain",
            "    label: " + asset.rsplit(".", 1)[0].replace("_", " "),
        ]
    # Land exactly on the voiceover length.
    last = timeline[1:].index(f"    end: {cursor}") + 1
    timeline[last] = f"    end: {duration}"

    (PROJECT / "timeline.yaml").write_text("\n".join(timeline) + "\n", encoding="utf-8")

    (PROJECT / "config.yaml").write_text(
        "output:\n"
        "  width: 1280\n"
        "  height: 720\n"
        "  fps: 30\n"
        "  crf: 20\n"
        "  preset: veryfast\n"
        "duration:\n"
        "  target_seconds: 300\n"
        "  min_seconds: 60\n"
        "  max_seconds: 1800\n"
        "visual:\n"
        "  default_fit: contain\n"
        "  on_short: error\n"
        "captions:\n"
        "  enabled: true\n"
        "  font_name: Montserrat\n"
        "  font_size: 46\n"
        "  primary_color: '#F7FF00'\n"
        "  outline_color: '#000000'\n"
        "  outline: 4\n"
        "  margin_v: 44\n"
        "  max_chars_per_line: 42\n"
        "  max_lines: 2\n",
        encoding="utf-8",
    )

    (PROJECT / "project.yaml").write_text(
        "name: sean-connery-5min\n"
        "script: script.md\n"
        "voiceover: audio/voiceover.wav\n"
        "captions:\n"
        "  file: captions/segments.json\n"
        "output_dir: output\n"
        "work_dir: build\n"
        "storage:\n"
        "  backend: local\n"
        "  destination: assets\n",
        encoding="utf-8",
    )

    (PROJECT / "script.md").write_text(
        "# How Sean Connery Escaped the Curse of James Bond\n\n"
        "Narration is taken from the reference recording; the voiceover file is\n"
        "the authoritative copy. Captions are generated from the voiceover with\n"
        "faster-whisper and are therefore in sync with the audio by\n"
        "construction, not by hand-tuned offsets.\n",
        encoding="utf-8",
    )

    (PROJECT / "assets" / ".gitkeep").write_text("", encoding="utf-8")
    (PROJECT / "README.md").write_text(
        "# Example project (~5 minutes)\n\n"
        "A production-shaped project assembled from the reference footage.\n\n"
        "```bash\n"
        "python -m video_pipeline render examples/sean-connery-5min\n"
        "python -m video_pipeline validate examples/sean-connery-5min/output/final.mp4 \\\n"
        "    --project examples/sean-connery-5min\n"
        "```\n\n"
        "Assets in `assets/` are generated by `tools/build_example_project.py` and\n"
        "are not committed. Point `voiceover`, `captions/segments.json` and\n"
        "`timeline.yaml` at your own material to reuse this as a template.\n",
        encoding="utf-8",
    )


def main() -> int:
    if not REFERENCE.is_file():
        raise SystemExit(f"reference video not found: {REFERENCE}")

    if PROJECT.exists():
        shutil.rmtree(PROJECT)
    PROJECT.mkdir(parents=True)

    voiceover = build_voiceover()
    info = ffmpeg.probe_json(voiceover)
    duration = float(info["format"]["duration"])
    print(f"      voiceover duration {duration:.2f}s ({duration / 60:.2f} min)")

    build_assets(duration)
    build_captions(voiceover, duration)
    write_project_files(duration)

    print(f"[5/5] project ready at {PROJECT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())