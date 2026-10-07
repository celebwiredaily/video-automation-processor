"""Project loading and safe path resolution.

A project is a directory containing everything needed to render one video.
Nothing outside that directory is reachable through project-controlled paths.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .config import Config
from .errors import AssetError, ConfigError
from .timeline import Timeline

PROJECT_FILENAMES = ("project.yaml", "project.yml")


def load_yaml(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError("Could not read YAML file.", file=str(path), error=str(exc)) from exc
    try:
        # safe_load only: never construct arbitrary Python objects from data.
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        mark = getattr(exc, "problem_mark", None)
        location = f"line {mark.line + 1}" if mark is not None else "unknown position"
        raise ConfigError(
            f"YAML is not valid ({location}).",
            file=str(path),
            problem=str(getattr(exc, "problem", exc)),
        ) from exc
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ConfigError("Top level of the YAML file must be a mapping.", file=str(path))
    return data


@dataclass(frozen=True)
class Project:
    """A fully resolved, validated project, ready to render."""

    root: Path
    name: str
    config: Config
    timeline: Timeline
    voiceover: Path | None
    captions_file: Path | None
    script_file: Path | None
    output_dir: Path
    work_dir: Path
    storage: dict[str, Any]
    data: dict[str, Any]

    @property
    def assets_dir(self) -> Path:
        return self.root / "assets"

    @property
    def duration_hint(self) -> float:
        return self.timeline.total_duration

    @property
    def output_path(self) -> Path:
        return self.output_dir / self.config.output.filename

    @property
    def diagnostics_dir(self) -> Path:
        return self.output_dir / "diagnostics"

    def asset_path(self, relative: str) -> Path:
        return self.resolve_inside(relative)

    def resolve_inside(self, relative: str) -> Path:
        """Resolve a project-relative path, refusing to escape the project.

        Blocks ``..`` traversal, absolute paths, and anything that resolves
        outside the project root through a symlink.
        """
        if not isinstance(relative, str) or not relative.strip():
            raise AssetError("Asset path must be a non-empty string.", value=repr(relative))
        candidate = Path(relative)
        if candidate.is_absolute():
            raise AssetError(
                "Asset paths must be relative to the project directory.",
                value=relative,
            )
        root = self.root.resolve()
        resolved = (root / candidate).resolve()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise AssetError(
                "Asset path escapes the project directory.",
                value=relative,
                project=str(root),
            ) from None
        return resolved

    def relative_to_root(self, path: Path) -> str:
        try:
            return str(path.resolve().relative_to(self.root.resolve()))
        except ValueError:
            return str(path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "root": str(self.root),
            "voiceover": str(self.voiceover) if self.voiceover else None,
            "captions": str(self.captions_file) if self.captions_file else None,
            "output": str(self.output_path),
            "timeline": self.timeline.to_dict(),
        }


def find_project_dir(start: Path) -> Path:
    """Locate the project directory containing ``start``."""
    start = start.resolve()
    if start.is_file():
        start = start.parent
    for candidate in [start, *start.parents]:
        for name in PROJECT_FILENAMES:
            if (candidate / name).is_file():
                return candidate
    raise ConfigError(
        "No project file found.",
        searched_from=str(start),
        expected=" or ".join(PROJECT_FILENAMES),
    )


def load_project(
    project_dir: Path,
    *,
    overrides: dict[str, Any] | None = None,
    validate_timeline: bool = True,
) -> Project:
    """Load, validate and freeze a project from disk.

    Timeline and configuration may live inline in ``project.yaml`` or in
    sibling ``timeline.yaml`` / ``config.yaml`` files; siblings win when both
    are present, which keeps project metadata readable.

    ``validate_timeline=False`` is used by ``fetch``: assets have to be staged
    before a timeline referencing them can possibly validate, so asset staging
    must not depend on timeline validity.

    Raises ConfigError for structural problems and ValidationError with the
    complete list of problems for content problems.
    """
    root = find_project_dir(project_dir)

    project_file = next(root / name for name in PROJECT_FILENAMES if (root / name).is_file())
    data = load_yaml(project_file)

    # ---- configuration: inline `config:` merged with optional config.yaml
    config_data: dict[str, Any] = dict(data.get("config") or {})
    external_config = root / "config.yaml"
    if external_config.is_file():
        external = load_yaml(external_config)
        nested = external.get("config")
        if isinstance(nested, dict):
            from .config import deep_merge

            config_data = deep_merge(config_data, nested)
        else:
            for key, value in external.items():
                if key != "config":
                    config_data[key] = value
    if overrides:
        from .config import deep_merge

        config_data = deep_merge(config_data, overrides)

    config = Config.from_mapping(config_data)

    name = str(data.get("name") or root.name)

    voiceover_raw = data.get("voiceover")
    voiceover: Path | None = None
    if voiceover_raw:
        try:
            voiceover = _resolve_optional(root, str(voiceover_raw))
        except AssetError as exc:
            raise ConfigError(
                "voiceover path in project.yaml is not usable.", **exc.context
            ) from exc

    captions_block = data.get("captions")
    captions_raw: Any = None
    if isinstance(captions_block, dict):
        captions_raw = captions_block.get("file")
    elif isinstance(captions_block, str):
        captions_raw = captions_block
    if captions_raw is None:
        captions_raw = data.get("captions_file")
    captions_file: Path | None = None
    if captions_raw:
        try:
            captions_file = _resolve_optional(root, str(captions_raw))
        except AssetError as exc:
            raise ConfigError(
                "captions.file path in project.yaml is not usable.", **exc.context
            ) from exc

    script_file: Path | None = None
    if data.get("script"):
        script_file = _resolve_optional(root, str(data["script"]))

    output_raw = data.get("output_dir") or "output"
    try:
        output_dir = _resolve_inside_unchecked(root, str(output_raw))
    except AssetError as exc:
        raise ConfigError("output_dir is not usable.", **exc.context) from exc

    work_raw = data.get("work_dir") or (config.work_dir and str(config.work_dir)) or "build"
    work_dir = (
        Path(str(work_raw))
        if Path(str(work_raw)).is_absolute()
        else _resolve_inside_unchecked(root, str(work_raw))
    )

    # ---- timeline: inline, sibling timeline.yaml, or timeline.json
    timeline_raw = data.get("timeline")
    if timeline_raw is None:
        for candidate in ("timeline.yaml", "timeline.yml", "timeline.json"):
            path = root / candidate
            if path.is_file():
                loaded = load_yaml(path) if path.suffix != ".json" else load_yaml(path)
                timeline_raw = loaded.get("timeline", loaded)
                break

    if timeline_raw is None:
        if not validate_timeline:
            timeline = Timeline(entries=())
        else:
            raise ConfigError(
                "No timeline found.",
                searched_from=str(root),
                hint="Add a `timeline:` block to project.yaml or create timeline.yaml.",
            )
    else:
        timeline = Timeline.from_entries(
            timeline_raw, root=root, config=config, validate=validate_timeline
        )

    storage_cfg = data.get("storage") or {}
    if not isinstance(storage_cfg, dict):
        raise ConfigError("`storage:` must be a mapping.")

    return Project(
        root=root,
        name=name,
        config=config,
        timeline=timeline,
        voiceover=voiceover,
        captions_file=captions_file,
        script_file=script_file,
        output_dir=output_dir,
        work_dir=work_dir,
        storage=storage_cfg,
        data=data,
    )


def _resolve_inside_unchecked(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute():
        return candidate
    resolved = (root / candidate).resolve()
    root_resolved = root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError:
        # Directories the pipeline creates (output/, build/) may be configured
        # anywhere, but a project file pointing outside is a mistake worth
        # surfacing loudly rather than silently allowing.
        raise AssetError(
            "Configured directory escapes the project directory.",
            value=relative,
            project=str(root_resolved),
        ) from None
    return resolved


def _resolve_optional(root: Path, relative: str) -> Path | None:
    path = _resolve_inside_unchecked(root, relative)
    return path if path.exists() else None