"""Command line interface.

    python -m video_pipeline doctor
    python -m video_pipeline fetch   <project>
    python -m video_pipeline render  <project> [--upload]
    python -m video_pipeline validate <video>
    python -m video_pipeline upload  <project> [--video PATH]
    python -m video_pipeline inspect <project>
    python -m video_pipeline probe   <asset>
    python -m video_pipeline prepare-test <dir>

Exit codes
    0  success
    2  configuration or validation problem
    3  asset problem
    4  ffmpeg failure
    5  environment / capability problem
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

from . import ffmpeg, media, storage, validation
from .captions import load_and_prepare
from .errors import PipelineError
from .project import load_project
from .renderer import Renderer


# --------------------------------------------------------------------- doctor


def cmd_doctor(args: argparse.Namespace) -> int:
    problems: list[str] = []
    notes: list[str] = []

    print("video-pipeline doctor")
    print("=" * 60)

    version = platform.python_version()
    py_ok = sys.version_info >= (3, 11)
    print(f"  python          : {version} ({'ok' if py_ok else 'TOO OLD, need >= 3.11'})")
    if not py_ok:
        problems.append("Python 3.11 or newer is required.")

    for label, finder in (("ffmpeg", ffmpeg.ffmpeg_bin), ("ffprobe", ffmpeg.ffprobe_bin)):
        try:
            binary = finder()
        except PipelineError as exc:
            print(f"  {label:<15}: MISSING")
            problems.append(f"{label} not found on PATH.")
            notes.append(str(exc.render()))
            continue
        proc = subprocess.run(
            [binary, "-version"], capture_output=True, text=True, check=False
        )
        first = (proc.stdout or "").splitlines()[:1]
        print(f"  {label:<15}: {first[0] if first else binary}")

    if not problems:
        try:
            caps = ffmpeg.check_capabilities(need_captions=True)
            print(f"  capabilities    : all required filters and encoders present")
            notes.append(caps.version)
        except PipelineError as exc:
            problems.append(str(exc.render()))

    font_dir = Path(__file__).parent / "assets" / "fonts"
    fonts = sorted(font_dir.glob("*")) if font_dir.is_dir() else []
    real_fonts = [f for f in fonts if f.suffix.lower() in {".ttf", ".otf", ".ttc"}]
    print(f"  caption fonts   : {len(real_fonts)} file(s) in {font_dir}")
    if not real_fonts:
        problems.append(
            f"No caption font found in {font_dir}; captions cannot be rendered."
        )
    for font in real_fonts:
        print(f"                     - {font.name}")

    # Optional dependency groups
    for module, why, install in (
        ("pytest", "running the test suite", "pip install pytest"),
        ("faster_whisper", "generating captions from a voiceover", "pip install faster-whisper"),
        ("googleapiclient", "authenticated Google Drive listing", "pip install google-api-python-client google-auth"),
        ("google.auth", "authenticated Google Drive listing", "pip install google-auth"),
    ):
        found = _module_available(module)
        print(f"  optional        : {module:<16} {'present' if found else 'absent'} ({why})")
        if not found:
            notes.append(f"{module} not installed - {install}")

    credentials = os.environ.get("GOOGLE_DRIVE_CREDENTIALS")
    api_key = os.environ.get("GOOGLE_API_KEY")
    print(f"  GOOGLE_DRIVE_CREDENTIALS : {'set' if credentials else 'not set'}")
    print(f"  GOOGLE_API_KEY           : {'set' if api_key else 'not set'}")
    if args.drive_folder:
        try:
            folder_id = storage.GoogleDriveStorage(args.drive_folder).folder_id
            print(f"  drive folder             : {folder_id}")
            if credentials:
                drive = storage.GoogleDriveStorage(
                    args.drive_folder,
                    credentials_file=Path(credentials),
                    api_key=api_key,
                )
                try:
                    found = drive.list_assets()
                    print(f"  drive listing            : {len(found)} file(s)")
                    for record in found[:20]:
                        print(f"                             {record.name} ({record.size} bytes)")
                except PipelineError as exc:
                    problems.append(f"Drive listing failed: {exc.message}")
            elif not api_key:
                notes.append(
                    "Drive folder given but neither GOOGLE_API_KEY nor "
                    "GOOGLE_DRIVE_CREDENTIALS is set; listing will fail."
                )
        except PipelineError as exc:
            problems.append(str(exc.render()))

    print("=" * 60)
    for note in notes:
        if note:
            print(f"  note: {note}")
    if problems:
        print("\nPROBLEMS FOUND:")
        for problem in problems:
            print(f"  - {problem}")
        return 5
    print("\nAll checks passed.")
    return 0


def _module_available(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


# ---------------------------------------------------------------------- fetch


def cmd_fetch(args: argparse.Namespace) -> int:
    """Stage assets locally, optionally pulling from Google Drive.

    Timeline validation is skipped: assets must be staged before a timeline
    referencing them can possibly validate.
    """
    try:
        project = load_project(Path(args.project), validate_timeline=False)
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    backend = str(project.storage.get("backend") or "local")
    exclude = _project_output_names(project)

    print(f"Staging assets for '{project.name}' via {backend} "
          f"(ignoring outputs: {', '.join(sorted(exclude))})")

    if backend in {"local", "none"}:
        destination = Path(project.storage.get("destination") or project.assets_dir)
        if not destination.is_absolute():
            destination = project.root / destination
        destination.mkdir(parents=True, exist_ok=True)
        staged, notes = storage.auto_stage(
            project.root, destination, exclude_names=exclude
        )
        for path in staged:
            print(f"  present  {path.name} ({path.stat().st_size} bytes)")
        for note in notes:
            print(f"  {note}")
        if not staged:
            print(
                "\nNo local assets found. If assets live in Google Drive, add:\n"
                "  storage:\n"
                "    backend: google_drive\n"
                "    input_folder_id: <folder-id-or-url>\n"
                "    output_folder_id: <folder-id-or-url>\n"
                "and set GOOGLE_API_KEY (public read) or "
                "GOOGLE_DRIVE_CREDENTIALS (authenticated).",
                file=sys.stderr,
            )
            return 3
        print(f"\n{len(staged)} asset(s) available locally.")
        return 0

    if backend in {"google_drive", "gdrive", "drive"}:
        try:
            drive = _build_project_storage(project)
            records = drive.list_assets()
            usable = [r for r in records if r.is_media
                      and r.name.lower() not in {n.lower() for n in exclude}]
            if not records:
                print(
                    "The Drive folder contains no files.",
                    file=sys.stderr,
                )
                print(
                    "This is a real result, not an error: public listings only "
                    "return items shared with the requesting account.",
                    file=sys.stderr,
                )
                return 0
            print(f"  found {len(records)} file(s) in Drive, "
                  f"{len(usable)} usable as inputs:")
            for record in usable:
                print(f"    {record.name:<44} {record.size:>12} bytes")
            for record in records:
                if record.name.lower() in {n.lower() for n in exclude}:
                    print(f"    (ignored output file) {record.name}")
            if not usable:
                print("\nNo usable input assets in the Drive folder.", file=sys.stderr)
                return 3
            staged = drive.stage_all(
                progress=lambda m: print(f"  {m}"), exclude=exclude
            )
            print(f"\nStaged {len(staged)} file(s) into {drive.destination}")
            return 0
        except PipelineError as exc:
            print(str(exc), file=sys.stderr)
            return exc.exit_code

    print(f"Unknown storage backend {backend!r}", file=sys.stderr)
    return 2



# --------------------------------------------------------------------- upload


def _project_output_names(project) -> set[str]:
    """Names that are produced outputs, never inputs."""
    names = {project.config.output.filename}
    diagnostics = project.config.output.filename.rsplit(".", 1)[0]
    for suffix in (".mp4", ".mov"):
        names.add(f"{diagnostics}{suffix}")
    return names


def _build_project_storage(project, *, for_upload: bool = False):
    """Create the storage backend described by the project's `storage:` block."""
    backend = str(project.storage.get("backend") or "local")
    folder = (
        project.storage.get("input_folder_id")
        or project.storage.get("folder")
    )
    output_folder = project.storage.get("output_folder_id")
    destination = Path(project.storage.get("destination") or project.assets_dir)
    if not destination.is_absolute():
        destination = project.root / destination
    return storage.build_storage(
        backend,
        root=project.root,
        folder=folder,
        input_folder=folder,
        output_folder=output_folder,
        destination=destination,
    )


def do_upload(project, video_path: Path) -> int:
    """Upload a validated video. Returns a process exit code.

    Refuses loudly rather than reporting a success the API did not confirm.
    """
    cfg = project.config
    size = video_path.stat().st_size if video_path.is_file() else 0
    print(f"\n[UPLOAD] {video_path.name} ({size} bytes)")
    sys.stdout.flush()

    if not video_path.is_file():
        print("[UPLOAD] nothing to upload", file=sys.stderr)
        return 3

    try:
        backend = _build_project_storage(project, for_upload=True)
    except PipelineError as exc:
        print(f"[UPLOAD] {exc.message}", file=sys.stderr)
        for key, value in exc.context.items():
            print(f"         {key}: {value}", file=sys.stderr)
        return exc.exit_code

    print(f"[UPLOAD] backend {backend.name}")
    sys.stdout.flush()
    # Label the confirmation with the backend that actually performed it, so a
    # local copy is never announced as a Drive upload.
    tag = "DRIVE" if backend.name == "google_drive" else backend.name.upper()
    try:
        result = backend.upload(
            video_path,
            name=cfg.output.filename,
            existing_file=cfg.output.existing_file,
        )
    except PipelineError as exc:
        print(f"[UPLOAD] FAILED: {exc.message}", file=sys.stderr)
        for key, value in exc.context.items():
            print(f"         {key}: {value}", file=sys.stderr)
        print(f"         local copy kept at {video_path}", file=sys.stderr)
        return exc.exit_code

    print(f"[{tag}] uploaded successfully (confirmed by the {backend.name} backend)")
    for line in result.describe().splitlines():
        print(f"[{tag}] {line}")
    sys.stdout.flush()
    return 0


def cmd_upload(args: argparse.Namespace) -> int:
    """Upload an already-rendered, already-validated video.

    Timeline validation is skipped: publishing a finished file does not depend
    on the input assets still being present locally.
    """
    try:
        project = load_project(Path(args.project), validate_timeline=False)
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    video = Path(args.video) if args.video else project.output_path
    if not video.is_file():
        print(f"[UPLOAD] {video} does not exist", file=sys.stderr)
        return 3
    return do_upload(project, video)


# --------------------------------------------------------------------- render


def cmd_render(args: argparse.Namespace) -> int:
    overrides = {}
    if args.config:
        overrides = json.loads(Path(args.config).read_text(encoding="utf-8"))
    if args.width:
        overrides.setdefault("output", {})["width"] = args.width
    if args.height:
        overrides.setdefault("output", {})["height"] = args.height
    if args.fps:
        overrides.setdefault("output", {})["fps"] = args.fps

    try:
        project = load_project(Path(args.project), overrides=overrides)
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    # Voiceover is the timing backbone.
    voiceover_duration = 0.0
    if project.voiceover and project.voiceover.is_file():
        voiceover_duration = media.probe(str(project.voiceover)).duration or 0.0
    elif project.config.validation.require_voiceover:
        print(
            "ERROR: `voiceover:` is configured but the file is missing.",
            file=sys.stderr,
        )
        return 3

    visual_duration = project.timeline.total_duration
    if voiceover_duration > 0:
        duration = voiceover_duration
    elif args.duration:
        duration = float(args.duration)
    elif visual_duration > 0:
        duration = visual_duration
    else:
        print("ERROR: cannot determine a duration for this project.", file=sys.stderr)
        return 2

    cfg = project.config
    if duration < cfg.duration.min_seconds or duration > cfg.duration.max_seconds:
        print(
            f"WARNING: duration {duration:.1f}s is outside the configured range "
            f"[{cfg.duration.min_seconds:.0f}s, {cfg.duration.max_seconds:.0f}s].",
            file=sys.stderr,
        )

    deviation = abs(duration - cfg.duration.target_seconds) / cfg.duration.target_seconds
    if deviation > cfg.duration.warn_target_deviation_ratio:
        print(
            f"NOTE: {duration:.1f}s differs from the {cfg.duration.target_seconds:.0f}s "
            f"target by {deviation:.0%}. The voiceover decides the runtime; this is "
            f"informational only.",
            file=sys.stderr,
        )

    print(f"Project      : {project.name}")
    print(f"Voiceover    : {project.voiceover or '(none)'} ({voiceover_duration:.2f}s)")
    print(f"Timeline     : {len(project.timeline.entries)} entries, {visual_duration:.2f}s")
    print(f"Target length: {duration:.2f}s")
    print(f"Output       : {project.output_path}")
    print()

    renderer = Renderer(
        project_root=project.root,
        config=cfg,
        work_dir=project.work_dir,
        output_path=project.output_path,
        voiceover=project.voiceover,
        captions_file=project.captions_file,
        keep_intermediates=args.keep_intermediates,
        jobs=args.jobs,
        verbose=True,
    )

    try:
        result = renderer.render(project.timeline, duration=duration)
    except PipelineError as exc:
        print("\nRENDER FAILED", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        print(f"\nffmpeg logs: {renderer.logs_dir}", file=sys.stderr)
        return exc.exit_code

    for warning in result.warnings:
        print(f"  warning: {warning}")

    # Post-render validation against the actual pixels.
    cues = []
    if cfg.captions.enabled and project.captions_file:
        cues, _ = load_and_prepare(
            project.captions_file, cfg.captions, video_duration=duration
        )

    diagnostics = project.output_path.parent / "diagnostics"
    try:
        report = validation.validate_video(
            result.output_path,
            expected_duration=duration,
            duration_tolerance=max(cfg.duration.tolerance_seconds, 0.5),
            config=cfg,
            timeline=project.timeline,
            cues=cues,
            diagnostics_dir=diagnostics,
        )
    except PipelineError as exc:
        print("\nVALIDATION FAILED TO RUN", file=sys.stderr)
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    print()
    print(report.summary())
    print()
    print(f"validation report : {diagnostics / 'validation.json'}")
    print(f"contact sheet     : {diagnostics / 'contact_sheet.jpg'}")
    print(f"rendered frames   : {diagnostics / 'frames'}")

    # Upload happens only after validation has passed. A failed render or a
    # failed validation must never publish a bad video.
    if args.upload:
        if not report.ok:
            print(
                "\\n[UPLOAD] SKIPPED because validation did not pass; "
                "the local file and diagnostics were kept.",
                file=sys.stderr,
            )
            return 2
        return do_upload(project, result.output_path)

    return 0 if report.ok else 2


# ------------------------------------------------------------------- validate


def cmd_validate(args: argparse.Namespace) -> int:
    path = Path(args.path).resolve()
    if not path.exists():
        print(f"File not found: {path}", file=sys.stderr)
        return 2

    config = None
    timeline = None
    cues = None
    expected = args.expected_duration
    if args.project:
        try:
            project = load_project(Path(args.project))
            config = project.config
            timeline = project.timeline
            if project.captions_file and config.captions.enabled:
                cues, _ = load_and_prepare(
                    project.captions_file,
                    config.captions,
                    video_duration=media.probe(str(path)).duration,
                )
            if expected is None and project.voiceover and project.voiceover.is_file():
                expected = media.probe(str(project.voiceover)).duration
        except PipelineError as exc:
            print(f"Could not load project {args.project}: {exc.message}", file=sys.stderr)

    diagnostics = Path(args.diagnostics) if args.diagnostics else path.parent / "diagnostics"
    try:
        report = validation.validate_video(
            path,
            expected_duration=expected,
            config=config,
            timeline=timeline,
            cues=cues,
            diagnostics_dir=diagnostics,
        )
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    print(report.summary())
    if report.ok:
        return 0
    return 2


# -------------------------------------------------------------------- inspect


def cmd_inspect(args: argparse.Namespace) -> int:
    try:
        project = load_project(Path(args.project))
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code

    cfg = project.config
    print(f"Project      : {project.name}  ({project.root})")
    print(f"Output       : {project.output_path}")
    print(f"Work dir     : {project.work_dir}")
    print(f"Storage      : {project.storage.get('backend', 'local')}")
    print(
        f"Format       : {cfg.output.width}x{cfg.output.height} @ {cfg.output.fps}fps "
        f"{cfg.output.video_codec}/{cfg.output.pix_fmt} + {cfg.output.audio_codec}"
    )
    print(
        f"Duration rule: voiceover decides; target {cfg.duration.target_seconds:.0f}s, "
        f"tolerance {cfg.duration.tolerance_seconds:.2f}s, "
        f"short-asset behaviour {cfg.visual.on_short}"
    )
    print(
        f"Captions     : {'on' if cfg.captions.enabled else 'off'}, "
        f"{cfg.captions.font_name} {cfg.captions.scaled_font_size(cfg.output.height)}px "
        f"{cfg.captions.primary_color} outline {cfg.captions.outline} "
        f"margin_v {cfg.captions.scaled(cfg.output.height).margin_v}"
    )
    print(f"Voiceover    : {project.voiceover or '(none)'}")
    if project.voiceover and project.voiceover.is_file():
        print(f"               {media.probe(str(project.voiceover)).describe()}")
    print(f"Caption file : {project.captions_file or '(none)'}")
    print(f"Script       : {project.script_file or '(none)'}")
    print()
    print(f"Timeline ({len(project.timeline.entries)} entries, "
          f"{project.timeline.total_duration:.2f}s total):")
    for entry in project.timeline.entries:
        info = media.probe(str(entry.asset))
        warnings = []
        if info.duration and entry.duration > info.duration + 0.05:
            warnings.append(
                f"SHORT: needs {entry.duration:.2f}s, has {info.duration:.2f}s"
            )
        if entry.wants_clip_audio and not info.has_audio:
            warnings.append("audio: clip requested but source has no audio")
        print(
            f"  [{entry.index:2d}] {entry.start:8.3f} -> {entry.end:8.3f} "
            f"({entry.duration:6.3f}s)  {entry.fit:<8} "
            f"audio={entry.audio:<5} on_short={entry.on_short:<6} "
            f"{entry.asset.name}  [{info.describe()}]"
        )
        for warning in warnings:
            print(f"       ! {warning}")
    return 0


# ---------------------------------------------------------------------- probe


def cmd_probe(args: argparse.Namespace) -> int:
    try:
        info = media.probe(args.path)
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    print(f"{info.path}")
    print(f"  kind     : {info.kind}")
    print(f"  duration : {info.duration:.3f}s")
    print(f"  video    : {info.width}x{info.height} @ {info.fps:.3f}fps "
          f"{info.video_codec} {info.pix_fmt}")
    print(f"  audio    : {info.audio_codec or 'none'} "
          f"{info.sample_rate}Hz {info.channels}ch")
    return 0


# --------------------------------------------------------------- prepare-test


def cmd_prepare_test(args: argparse.Namespace) -> int:
    from .testkit import build_test_project

    directory = Path(args.dir)
    try:
        created = build_test_project(directory, seconds=float(args.seconds))
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    for path in created:
        print(f"  created  {path}")
    print(f"\nTest project ready in {directory}")
    print(f"  python -m video_pipeline render {directory}")
    return 0


# ------------------------------------------------------------------------ main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="video_pipeline",
        description="Deterministic narrated-video rendering pipeline.",
    )
    sub = parser.add_subparsers(dest="command")

    p = sub.add_parser("doctor", help="Check the environment is able to render")
    p.add_argument("--drive-folder", default=None, help="Also probe this Drive folder")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("fetch", help="Stage assets locally (local disk or Google Drive)")
    p.add_argument("project")
    p.set_defaults(func=cmd_fetch)

    p = sub.add_parser("render", help="Render a project to MP4 and validate the result")
    p.add_argument("project")
    p.add_argument("--config", help="JSON file of config overrides")
    p.add_argument("--duration", type=float, help="Override duration when there is no voiceover")
    p.add_argument("--width", type=int)
    p.add_argument("--height", type=int)
    p.add_argument("--fps", type=int)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--keep-intermediates", action="store_true")
    p.add_argument(
        "--upload",
        action="store_true",
        help="After validation passes, upload the final MP4 to the configured "
        "storage backend (Google Drive when configured).",
    )
    p.set_defaults(func=cmd_render)

    p = sub.add_parser("validate", help="Validate a rendered MP4 against its frames")
    p.add_argument("path")
    p.add_argument("--project", help="Validate against this project's expectations")
    p.add_argument("--expected-duration", type=float)
    p.add_argument("--diagnostics", help="Where to write frames and the report")
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser(
        "upload",
        help="Upload an already-rendered, already-validated MP4 to storage",
    )
    p.add_argument("project")
    p.add_argument(
        "--video",
        default=None,
        help="Video to upload (default: <project>/output/final.mp4)",
    )
    p.set_defaults(func=cmd_upload)

    p = sub.add_parser("inspect", help="Show a project's resolved configuration")
    p.add_argument("project")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("probe", help="Probe one media file")
    p.add_argument("path")
    p.set_defaults(func=cmd_probe)

    p = sub.add_parser("prepare-test", help="Generate a deterministic synthetic project")
    p.add_argument("dir")
    p.add_argument("--seconds", type=float, default=6.0)
    p.set_defaults(func=cmd_prepare_test)

    args = parser.parse_args(argv)
    if not getattr(args, "func", None):
        parser.print_help()
        return 2

    try:
        return int(args.func(args))
    except PipelineError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())