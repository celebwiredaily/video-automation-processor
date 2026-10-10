#!/usr/bin/env python3
"""Render validation.json as a Markdown summary.

Used by CI so a failed run is readable without opening the JSON artifact.
Prints to stdout; the caller decides where it goes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    if len(argv) < 2:
        print(
            "usage: summary.py <validation.json> [project-label] [duration-report.json]",
            file=sys.stderr,
        )
        return 2

    report_path = Path(argv[1])
    label = argv[2] if len(argv) > 2 else report_path.parent.parent.name
    duration_path = Path(argv[3]) if len(argv) > 3 else None

    if not report_path.is_file():
        print(f"## Render summary: {label}\n\n- validation.json not produced")
        return 0

    data = json.loads(report_path.read_text(encoding="utf-8"))
    state = "PASS" if data.get("ok") else "FAIL"
    print(f"## Render summary: {label}")
    print()
    print(f"- **result**: {state}")
    print(f"- duration: {data.get('duration', 0):.3f}s")
    print(f"- resolution: {data.get('width')}x{data.get('height')} @ {data.get('fps', 0):.3f}fps")
    print(f"- video: {data.get('video_codec')}/{data.get('pix_fmt')}")
    print(
        f"- audio: {data.get('audio_codec')} {data.get('audio_sample_rate')}Hz "
        f"{data.get('audio_channels')}ch, rms={data.get('audio_rms')}"
    )
    print(
        f"- frames sampled: {len(data.get('frame_checks', []))} "
        f"(black={data.get('black_frame_count')}, blank={data.get('blank_frame_count')}, "
        f"frozen pairs={len(data.get('frozen_pairs', []))})"
    )
    print(
        f"- captions confirmed: {data.get('caption_checks_passed')}"
        f"/{data.get('caption_checks_total')} caption-active samples"
    )
    print(f"- final frame valid: {data.get('final_frame_ok')}")

    errors = data.get("errors") or []
    warnings = data.get("warnings") or []
    if errors:
        print()
        print("### Errors")
        for item in errors:
            print(f"- {item}")
    if warnings:
        print()
        print("### Warnings")
        for item in warnings:
            print(f"- {item}")

    # Strict duration / frame-count measurement, when one was produced. This is
    # the number that decides whether the render is actually 300.000s.
    if duration_path is not None and duration_path.is_file():
        duration = json.loads(duration_path.read_text(encoding="utf-8"))
        state = "PASS" if duration.get("ok") else "FAIL"
        print()
        print(f"### Strict duration: {state}")
        print()
        print(
            f"- target: {duration.get('target_seconds')}s, measured: "
            f"{duration.get('measured_duration')}s "
            f"(delta {duration.get('measured_delta')}s, tolerance "
            f"{duration.get('tolerance_seconds')}s)"
        )
        print(
            f"- frames: {duration.get('frames')} decoded, expected "
            f"{duration.get('expected_frames')} at {duration.get('fps')}fps"
        )
        for check in duration.get("checks") or []:
            flag = "ok" if check.get("ok") else "FAILED"
            print(f"- {check.get('name')}: {check.get('measured')} ({flag})")
    elif duration_path is not None:
        print()
        print("### Strict duration\n\n- duration report not produced")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))