"""
render_video.py

Renders the final video.
- Freezes the video frame during voiceover segments.
- Plays the video normally during interview / trigger segments.
- Mixes voiceover and original clip audio dynamically.
- Optional NVIDIA NVENC (h264_nvenc) for Colab T4 / GPU acceleration.

FIXES applied:
1. audio mapping no longer puts brackets around raw stream specifiers.
2. clips without audio fall back to voiceover instead of broken filter.
3. freeze path uses select=eq(n\\,0) so the first real frame is looped
   instead of producing black frames.
4. --encoder auto|cpu|nvenc  (auto tries NVENC when a GPU is present).
5. Aborts early with a clear message when no usable clip_path exists.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


def probe_duration(path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(path),
            ],
            capture_output=True, text=True, check=True, timeout=20,
        )
        return float(result.stdout.strip())
    except Exception:
        return None


def probe_has_audio(path):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-select_streams", "a",
                "-show_entries", "stream=index",
                "-of", "csv=p=0",
                str(path),
            ],
            capture_output=True, text=True, check=True, timeout=20,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def nvenc_available() -> bool:
    """Return True if ffmpeg can use h264_nvenc."""
    if shutil.which("ffmpeg") is None:
        return False
    try:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            capture_output=True, text=True, timeout=15,
        )
        return "h264_nvenc" in (r.stdout or "")
    except Exception:
        return False


def build_filter_complex(segments, width, height, fps=30, has_audio_map=None):
    has_audio_map = has_audio_map or {}

    video_filters = []
    audio_filters = []
    v_concat = ""

    voiceover_input_idx = len(segments)

    mute_intervals = []
    clip_audio_labels = []

    for i, seg in enumerate(segments):
        duration = max(0.05, float(seg["end"]) - float(seg["start"]))
        clip_duration = probe_duration(seg["clip_path"])

        video_mode = str(seg.get("video_mode", "play")).lower()

        if video_mode == "freeze":
            # Grab frame 0 explicitly, then loop it. Avoids black frames that
            # appear when loop runs before any frame has been decoded.
            stages = [
                r"select=eq(n\,0)",
                "loop=loop=-1:size=1:start=0",
                "setpts=N/FRAME_RATE/TB",
                f"trim=duration={duration:.3f}",
                f"scale={width}:{height}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}",
                "setsar=1",
                f"fps={fps}",
            ]
        else:
            stages = [
                f"scale={width}:{height}:force_original_aspect_ratio=increase",
                f"crop={width}:{height}",
                "setsar=1",
                f"trim=duration={duration:.3f}",
                "setpts=PTS-STARTPTS",
            ]
            if clip_duration is not None and clip_duration < duration:
                pad_amount = duration - clip_duration
                stages.append(f"tpad=stop_mode=clone:stop_duration={pad_amount:.3f}")
            stages.append(f"fps={fps}")

        video_filters.append(f"[{i}:v]" + ",".join(stages) + f"[v{i}]")
        v_concat += f"[v{i}]"

        audio_mode = str(seg.get("audio_mode", "voiceover")).lower()

        if audio_mode == "clip" and has_audio_map.get(i, False):
            mute_intervals.append((float(seg["start"]), float(seg["end"])))
            delay_ms = int(float(seg["start"]) * 1000)
            audio_filters.append(
                f"[{i}:a]aresample=44100,aformat=channel_layouts=stereo,"
                f"atrim=duration={duration:.3f},asetpts=PTS-STARTPTS,"
                f"adelay={delay_ms}|{delay_ms},apad[ca{i}]"
            )
            clip_audio_labels.append(f"[ca{i}]")

    n = len(segments)
    video_filters.append(f"{v_concat}concat=n={n}:v=1:a=0[vconcat]")

    if mute_intervals:
        expr = "+".join(f"between(t,{s:.3f},{e:.3f})" for s, e in mute_intervals)
        audio_filters.append(
            f"[{voiceover_input_idx}:a]volume=0:enable='{expr}'[vo_muted]"
        )
        vo_label = "[vo_muted]"
    else:
        vo_label = f"[{voiceover_input_idx}:a]"

    all_audio_labels = [vo_label] + clip_audio_labels

    if len(all_audio_labels) > 1:
        audio_filters.append(
            f"{''.join(all_audio_labels)}"
            f"amix=inputs={len(all_audio_labels)}:"
            f"duration=longest:dropout_transition=0:normalize=0[aout]"
        )
        audio_out = "[aout]"
    elif mute_intervals:
        audio_out = "[vo_muted]"
    else:
        # Raw stream specifier – NO brackets
        audio_out = f"{voiceover_input_idx}:a"

    return ";".join(video_filters + audio_filters), audio_out


def render(
    segments_path,
    voiceover_path,
    captions_path,
    output_path,
    width=1920,
    height=1080,
    crf=20,
    preset="veryfast",
    fps=30,
    encoder="auto",
):
    segments = json.loads(Path(segments_path).read_text(encoding="utf-8"))
    segments = [s for s in segments if s.get("clip_path")]

    if not segments:
        print(
            "ERROR: No segments have a valid clip_path.\n"
            "This usually means youtube_dl_toolbox.py failed to download any "
            "interview clips. Check assets/clips/clip_manifest.json and re-run "
            "with --force-clips after fixing VIDEO_TOPIC / network.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify every clip file actually exists on disk
    missing = [s["clip_path"] for s in segments if not Path(s["clip_path"]).is_file()]
    if missing:
        print("ERROR: These clip files are missing:", file=sys.stderr)
        for m in missing[:10]:
            print(f"  - {m}", file=sys.stderr)
        sys.exit(1)

    has_audio_map = {}
    for i, seg in enumerate(segments):
        if str(seg.get("audio_mode", "voiceover")).lower() == "clip":
            has_audio_map[i] = probe_has_audio(seg["clip_path"])
            if not has_audio_map[i]:
                print(
                    f"WARNING: segment {i} wants clip audio but "
                    f"{seg['clip_path']} has none – using voiceover instead."
                )

    filter_complex, audio_out = build_filter_complex(
        segments, width, height, fps=fps, has_audio_map=has_audio_map
    )

    captions_path_escaped = str(Path(captions_path)).replace("\\", "/").replace(":", "\\:")
    filter_complex += f";[vconcat]ass={captions_path_escaped}[vout]"

    # Encoder selection
    use_nvenc = False
    if encoder == "nvenc":
        use_nvenc = True
    elif encoder == "auto":
        use_nvenc = nvenc_available()

    if use_nvenc:
        print("Using NVIDIA NVENC (h264_nvenc) – good for Colab T4")
        vcodec = [
            "-c:v", "h264_nvenc",
            "-preset", "p4",          # p1 (fastest) … p7 (slowest/best)
            "-rc", "vbr",
            "-cq", str(max(15, crf - 2)),
            "-b:v", "0",
            "-profile:v", "high",
        ]
    else:
        if encoder == "nvenc":
            print("WARNING: NVENC requested but not available – falling back to libx264")
        print("Using libx264 (CPU)")
        vcodec = [
            "-c:v", "libx264",
            "-preset", preset,
            "-crf", str(crf),
        ]

    cmd = ["ffmpeg", "-y"]
    for seg in segments:
        cmd += ["-i", seg["clip_path"]]
    cmd += ["-i", voiceover_path]

    cmd += [
        "-filter_complex", filter_complex,
        "-map", "[vout]",
        "-map", audio_out,
        *vcodec,
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        "-shortest",
        str(output_path),
    ]

    print("Running ffmpeg...\n", " ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Done -> {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("segments")
    parser.add_argument("voiceover")
    parser.add_argument("captions_ass")
    parser.add_argument("output")
    parser.add_argument("--resolution", default="1920x1080")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--crf", type=int, default=20)
    parser.add_argument("--preset", default="veryfast")
    parser.add_argument(
        "--encoder",
        default="auto",
        choices=["auto", "cpu", "nvenc"],
        help="auto = try NVENC when available (Colab T4), else libx264",
    )
    args = parser.parse_args()

    width, height = (int(x) for x in args.resolution.lower().split("x"))

    render(
        args.segments,
        args.voiceover,
        args.captions_ass,
        args.output,
        width=width,
        height=height,
        crf=args.crf,
        preset=args.preset,
        fps=args.fps,
        encoder=args.encoder,
    )


if __name__ == "__main__":
    main()
