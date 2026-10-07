#!/usr/bin/env python3
"""Stage CI outputs into a single directory for `actions/upload-artifact`.

This is deliberately *not* part of the renderer. The renderer stays
storage-agnostic and keeps writing its local output where the project says;
this tool only collects what a CI run should publish, and refuses to stage
anything when validation did not pass.

    artifacts/
    ├── final.mp4
    ├── validation.json
    └── render-report.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ARTIFACT_NAME = "final-video"
FINAL_NAME = "final.mp4"
VALIDATION_NAME = "validation.json"
REPORT_NAME = "render-report.json"
MIN_VALIDATION_BYTES = 256


class StageError(Exception):
    """Raised when the artefacts cannot be staged.

    Extra keyword arguments are attached for the CLI to print as context, so
    failures say what was missing and what to do about it.
    """

    def __init__(self, message: str, **context) -> None:
        super().__init__(message)
        self.message = message
        self.context = context

    def __str__(self) -> str:
        if not self.context:
            return self.message
        details = "; ".join(f"{k}={v}" for k, v in self.context.items())
        return f"{self.message} ({details})"


def _resolve(project: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    return project / "output" / FINAL_NAME


def build_report(
    validation: dict,
    *,
    project_name: str,
    project_dir: Path,
    artifact_name: str = ARTIFACT_NAME,
) -> dict:
    """A compact, human- and machine-readable summary of the run."""
    return {
        "artifact_name": artifact_name,
        "project": project_name,
        "project_dir": str(project_dir),
        "result": "pass" if validation.get("ok") else "fail",
        "output": {
            "filename": FINAL_NAME,
            "file_size": validation.get("file_size"),
            "duration_seconds": validation.get("duration"),
            "width": validation.get("width"),
            "height": validation.get("height"),
            "fps": validation.get("fps"),
            "video_codec": validation.get("video_codec"),
            "pix_fmt": validation.get("pix_fmt"),
            "audio_codec": validation.get("audio_codec"),
            "audio_sample_rate": validation.get("audio_sample_rate"),
            "audio_channels": validation.get("audio_channels"),
            "audio_duration_seconds": validation.get("audio_duration"),
            "audio_rms": validation.get("audio_rms"),
        },
        "validation": {
            "frames_sampled": len(validation.get("frame_checks") or []),
            "black_frames": validation.get("black_frame_count"),
            "blank_frames": validation.get("blank_frame_count"),
            "frozen_pairs": len(validation.get("frozen_pairs") or []),
            "captions_confirmed": validation.get("caption_checks_passed"),
            "captions_sampled": validation.get("caption_checks_total"),
            "final_frame_ok": validation.get("final_frame_ok"),
            "errors": validation.get("errors") or [],
            "warnings": validation.get("warnings") or [],
        },
        "notes": [
            "Validation inspected decoded frames from the rendered MP4; it is "
            "not based on the exit code of the renderer or FFmpeg.",
            "MP4s are not committed to Git. This artefact is the only copy kept "
            "by the CI run and is subject to artefact retention.",
        ],
    }


def stage(
    project_dir: Path,
    artifact_dir: Path,
    *,
    video: str | None = None,
    diagnostics: str | None = None,
    artifact_name: str = ARTIFACT_NAME,
    allow_failed_validation: bool = False,
) -> dict:
    """Copy the final MP4 plus diagnostics into ``artifact_dir``.

    Raises StageError unless the video exists and validation passed, so a
    failed run cannot produce a complete artefact set.
    """
    project_dir = Path(project_dir).resolve()
    if not project_dir.is_dir():
        raise StageError(f"project directory not found: {project_dir}")

    diagnostics_dir = (
        Path(diagnostics)
        if diagnostics
        else project_dir / "output" / "diagnostics"
    )
    video_path = _resolve(project_dir, video)
    validation_path = diagnostics_dir / VALIDATION_NAME

    if not video_path.is_file():
        raise StageError(
            f"rendered video not found: {video_path}",
            hint="Render and validate the project before staging artefacts.",
        )
    if video_path.stat().st_size < MIN_VALIDATION_BYTES:
        raise StageError(
            f"rendered video is implausibly small ({video_path.stat().st_size} bytes)",
            file=str(video_path),
        )
    if not validation_path.is_file():
        raise StageError(
            f"validation.json not found: {validation_path}",
            hint="Validation must run before artefacts are staged; uploading an "
            "unvalidated render would defeat the gate.",
        )
    if validation_path.stat().st_size < MIN_VALIDATION_BYTES:
        raise StageError(f"validation.json is truncated: {validation_path}")

    try:
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise StageError(f"validation.json is not valid JSON: {exc}") from exc

    if not validation.get("ok") and not allow_failed_validation:
        errors = "; ".join(validation.get("errors") or []) or "unknown"
        raise StageError(
            "validation did not pass; refusing to stage artefacts",
            errors=errors,
            note="Run with --allow-failed-validation to stage anyway for debugging.",
        )

    project_name = project_dir.name
    project_file = project_dir / "project.yaml"
    if project_file.is_file():
        for line in project_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("name:"):
                project_name = line.split(":", 1)[1].strip().strip("\"'")
                break

    artifact_dir = Path(artifact_dir)
    if artifact_dir.exists():
        shutil.rmtree(artifact_dir)
    artifact_dir.mkdir(parents=True)

    shutil.copy2(video_path, artifact_dir / FINAL_NAME)
    shutil.copy2(validation_path, artifact_dir / VALIDATION_NAME)

    report = build_report(
        validation,
        project_name=project_name,
        project_dir=project_dir,
        artifact_name=artifact_name,
    )
    (artifact_dir / REPORT_NAME).write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    # Keep the visual diagnostics alongside the report when they exist; they are
    # what makes a failed review possible without a local re-render.
    frames = diagnostics_dir / "frames"
    if frames.is_dir():
        existing = sorted(frames.glob("*.jpg"))
        if existing:
            target = artifact_dir / "frames"
            target.mkdir()
            for frame in existing[:24]:  # keep the artefact a sane size
                shutil.copy2(frame, target / frame.name)
    contact = diagnostics_dir / "contact_sheet.jpg"
    if contact.is_file():
        shutil.copy2(contact, artifact_dir / "contact_sheet.jpg")

    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Stage a validated render for actions/upload-artifact.",
    )
    parser.add_argument("project", help="Project directory")
    parser.add_argument(
        "--artifact-dir",
        default="artifacts",
        help="Directory to stage into (default: artifacts)",
    )
    parser.add_argument("--video", default=None, help="Override the video path")
    parser.add_argument(
        "--diagnostics", default=None, help="Override the diagnostics directory"
    )
    parser.add_argument("--artifact-name", default=ARTIFACT_NAME)
    parser.add_argument(
        "--allow-failed-validation",
        action="store_true",
        help="Stage even when validation failed (debugging only)",
    )
    args = parser.parse_args(argv)

    artifact_dir = Path(args.artifact_dir)
    if not artifact_dir.is_absolute():
        artifact_dir = Path.cwd() / artifact_dir

    try:
        report = stage(
            Path(args.project),
            artifact_dir,
            video=args.video,
            diagnostics=args.diagnostics,
            artifact_name=args.artifact_name,
            allow_failed_validation=args.allow_failed_validation,
        )
    except StageError as exc:
        print(f"[STAGE] FAILED: {exc.message}", file=sys.stderr)
        for key, value in exc.context.items():
            print(f"         {key}: {value}", file=sys.stderr)
        return 1

    print(f"[STAGE] result   : {report['result']}")
    print(f"[STAGE] project  : {report['project']}")
    print(f"[STAGE] duration : {report['output']['duration_seconds']}s")
    print(f"[STAGE] video    : {report['output']['width']}x{report['output']['height']}"
          f" @ {report['output']['fps']}fps")
    print(f"[STAGE] captions : {report['validation']['captions_confirmed']}"
          f"/{report['validation']['captions_sampled']} confirmed")
    print(f"[STAGE] artefact : {ARTIFACT_NAME}")
    for path in sorted(artifact_dir.iterdir()):
        label = path.name + ("/" if path.is_dir() else "")
        print(f"[STAGE]   {label}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())