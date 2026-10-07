"""The visual timeline: data in, validated plan out.

Editing decisions live entirely in ``project.yaml``. This module parses them,
checks that they are physically possible, and produces a deterministic list of
segments for the renderer. It performs no media work of its own.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import Config
from .errors import ValidationError
from .media import MediaInfo, is_image_asset, probe

FIT_MODES = {"contain", "cover", "stretch"}
ON_SHORT_MODES = {"error", "freeze", "loop"}
AUDIO_MODES = {"none", "clip"}


@dataclass(frozen=True)
class TimelineEntry:
    """One continuous visual interval on the finished timeline."""

    index: int
    start: float
    end: float
    asset: Path
    fit: str
    audio: str
    on_short: str
    source_start: float
    source_end: float | None
    label: str

    @property
    def duration(self) -> float:
        return self.end - self.start

    @property
    def has_source_range(self) -> bool:
        return self.source_end is not None or self.source_start > 0.0

    @property
    def source_duration(self) -> float | None:
        if self.source_end is None:
            return None
        return self.source_end - self.source_start

    @property
    def wants_clip_audio(self) -> bool:
        return self.audio == "clip"

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "index": self.index,
            "start": round(self.start, 4),
            "end": round(self.end, 4),
            "duration": round(self.duration, 4),
            "asset": str(self.asset),
            "fit": self.fit,
            "audio": self.audio,
            "on_short": self.on_short,
        }
        if self.source_start:
            data["source_start"] = round(self.source_start, 4)
        if self.source_end is not None:
            data["source_end"] = round(self.source_end, 4)
        if self.label:
            data["label"] = self.label
        return data


@dataclass(frozen=True)
class Timeline:
    entries: tuple[TimelineEntry, ...]

    @property
    def total_duration(self) -> float:
        return sum(e.duration for e in self.entries)

    @property
    def start(self) -> float:
        return self.entries[0].start if self.entries else 0.0

    @property
    def end(self) -> float:
        return self.entries[-1].end if self.entries else 0.0

    def clip_audio_entries(self) -> list[TimelineEntry]:
        return [e for e in self.entries if e.wants_clip_audio]

    def boundaries(self) -> list[float]:
        times = {e.start for e in self.entries}
        times.update(e.end for e in self.entries)
        return sorted(times)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_duration": round(self.total_duration, 4),
            "entries": [e.to_dict() for e in self.entries],
        }

    # ---------------------------------------------------------------- parsing

    @classmethod
    def from_entries(
        cls,
        raw: Any,
        *,
        root: Path,
        config: Config,
        validate: bool = True,
    ) -> "Timeline":
        if not isinstance(raw, list) or not raw:
            if not validate:
                return cls(entries=())
            raise ValidationError(
                "`timeline:` must be a non-empty list of entries.",
                hint="Example:\n"
                "  timeline:\n"
                "    - start: 0\n"
                "      end: 8.5\n"
                "      asset: assets/clip001.mp4",
            )

        problems: list[str] = []
        parsed: list[TimelineEntry] = []
        defaults = config.visual

        for position, item in enumerate(raw):
            entry, issues = _parse_entry(
                item,
                index=len(parsed),
                root=root,
                config=config,
                default_fit=defaults.default_fit,
                default_on_short=defaults.on_short,
            )
            problems.extend(issues)
            if entry is not None:
                parsed.append(entry)

        if not parsed:
            if not validate:
                return cls(entries=())
            raise ValidationError("No usable timeline entries.", problems=problems or None)

        parsed.sort(key=lambda e: (e.start, e.end))
        if validate:
            problems.extend(_check_structure(parsed, config))

            if problems:
                raise ValidationError(
                    "The timeline cannot be rendered as written.",
                    problems=problems,
                    project=str(root),
                    entries=len(raw),
                )

        return cls(entries=tuple(parsed))


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _parse_entry(
    item: Any,
    *,
    index: int,
    root: Path,
    config: Config,
    default_fit: str,
    default_on_short: str,
) -> tuple[TimelineEntry | None, list[str]]:
    problems: list[str] = []
    where = f"timeline[{index}]"

    if not isinstance(item, dict):
        problems.append(f"{where}: entry must be a mapping with `start`, `end` and `asset`")
        return None, problems

    unknown = set(item) - {
        "start",
        "end",
        "asset",
        "fit",
        "audio",
        "on_short",
        "source_start",
        "source_end",
        "label",
    }
    if unknown:
        problems.append(f"{where}: unknown key(s): {', '.join(sorted(unknown))}")

    start = _as_float(item.get("start"))
    end = _as_float(item.get("end"))
    asset_raw = item.get("asset")

    if start is None:
        problems.append(f"{where}: `start` is missing or not a number")
    if end is None:
        problems.append(f"{where}: `end` is missing or not a number")

    if start is not None and end is not None:
        if start < 0:
            problems.append(f"{where}: `start` ({start}) must be >= 0")
        if end <= start:
            problems.append(
                f"{where}: `end` ({end}) must be greater than `start` ({start})"
            )

    if not asset_raw:
        problems.append(f"{where}: `asset` is required")
        return None, problems

    asset = _safe_asset(root, str(asset_raw))
    if asset is None:
        problems.append(f"{where}: asset {asset_raw!r} is missing or outside the project")
        return None, problems

    fit = str(item.get("fit", default_fit)).lower()
    if fit not in FIT_MODES:
        problems.append(f"{where}: `fit` must be one of {sorted(FIT_MODES)}, got {fit!r}")
        fit = default_fit

    audio = str(item.get("audio", "none")).lower()
    if audio not in AUDIO_MODES:
        problems.append(f"{where}: `audio` must be one of {sorted(AUDIO_MODES)}, got {audio!r}")
        audio = "none"

    on_short = str(item.get("on_short", default_on_short)).lower()
    if on_short not in ON_SHORT_MODES:
        problems.append(
            f"{where}: `on_short` must be one of {sorted(ON_SHORT_MODES)}, got {on_short!r}"
        )
        on_short = default_on_short

    source_start = _as_float(item.get("source_start")) or 0.0
    source_end = _as_float(item.get("source_end"))
    if source_start < 0:
        problems.append(f"{where}: `source_start` must be >= 0")
    if source_end is not None and source_end <= source_start:
        problems.append(
            f"{where}: `source_end` ({source_end}) must be greater than "
            f"`source_start` ({source_start})"
        )

    if problems:
        return None, problems

    assert start is not None and end is not None

    entry = TimelineEntry(
        index=index,
        start=start,
        end=end,
        asset=asset,
        fit=fit,
        audio=audio,
        on_short=on_short,
        source_start=source_start,
        source_end=source_end,
        label=str(item.get("label") or asset.name),
    )
    return entry, problems


def _safe_asset(root: Path, relative: str) -> Path | None:
    candidate = Path(relative)
    if candidate.is_absolute():
        return None
    root_resolved = root.resolve()
    try:
        resolved = (root_resolved / candidate).resolve()
        resolved.relative_to(root_resolved)
    except (ValueError, OSError):
        return None
    return resolved if resolved.is_file() else None


def _check_structure(entries: Sequence[TimelineEntry], config: Config) -> list[str]:
    problems: list[str] = []

    if entries[0].start > 1e-6:
        problems.append(
            f"timeline starts at {entries[0].start:.3f}s; the first entry must start at 0 "
            "(the video would begin with unexplained black)"
        )

    for previous, current in zip(entries, entries[1:]):
        if current.start < previous.end - 1e-6:
            problems.append(
                f"timeline[{current.index}] overlaps timeline[{previous.index}]: "
                f"{current.start:.3f}s < {previous.end:.3f}s"
            )
        elif current.start > previous.end + 1e-3:
            gap = current.start - previous.end
            problems.append(
                f"gap of {gap:.3f}s between timeline[{previous.index}] "
                f"(ends {previous.end:.3f}s) and timeline[{current.index}] "
                f"(starts {current.start:.3f}s); fill it or the video will "
                f"contain {gap:.3f}s of black"
            )

    problems.extend(_check_sources(entries, config))
    return problems


def _check_sources(entries: Sequence[TimelineEntry], config: Config) -> list[str]:
    problems: list[str] = []
    tolerance = config.duration.tolerance_seconds

    for entry in entries:
        try:
            info = probe(str(entry.asset))
        except Exception as exc:  # surfaced below as a problem, not a crash
            problems.append(f"timeline[{entry.index}]: cannot read {entry.asset.name}: {exc}")
            continue

        label = f"timeline[{entry.index}] ({entry.asset.name})"

        if info.kind == "audio":
            problems.append(f"{label}: asset is an audio file, not a visual")
            continue

        if entry.wants_clip_audio and not info.has_audio:
            problems.append(
                f"{label}: `audio: clip` was requested but {entry.asset.name} has no "
                f"audio stream"
            )

        if info.kind == "image" or is_image_asset(entry.asset):
            # Stills have no duration to run out of; they fill any interval.
            if entry.has_source_range and entry.source_end is not None:
                problems.append(
                    f"{label}: `source_start`/`source_end` do not apply to a still image"
                )
            continue

        available = info.duration
        if available <= 0:
            problems.append(f"{label}: {entry.asset.name} reports zero duration")
            continue

        source_start = entry.source_start
        source_end = entry.source_end if entry.source_end is not None else source_start + entry.duration

        if entry.source_start >= available - 1e-3:
            problems.append(
                f"{label}: source_start {source_start:.3f}s is at or past the end of "
                f"{entry.asset.name} ({available:.3f}s)"
            )
            continue

        # A looping entry can legitimately ask for more than the source holds.
        if entry.on_short == "loop":
            needed = entry.duration
        else:
            if source_end > available + tolerance:
                problems.append(
                    f"{label}: requested source range {source_start:.3f}-{source_end:.3f}s "
                    f"exceeds {entry.asset.name} duration of {available:.3f}s. "
                    f"Shorten the interval, reduce source_start, or set "
                    f"`on_short: freeze|loop` on this entry."
                )
                continue

            needed = source_end - source_start
            if needed > entry.duration + 1e-3:
                problems.append(
                    f"{label}: source range {needed:.3f}s is longer than the "
                    f"{entry.duration:.3f}s timeline interval"
                )
                continue

        shortfall = entry.duration - (available - source_start)
        if entry.on_short == "error" and shortfall > 0.05:
            problems.append(
                f"{label}: {entry.asset.name} has {available - source_start:.3f}s "
                f"available after source_start={source_start:.3f}s but the timeline "
                f"needs {entry.duration:.3f}s (short by {shortfall:.3f}s). "
                f"Set `on_short: freeze` or `on_short: loop` to handle this "
                f"explicitly, or shorten the interval."
            )

    return problems


def media_summary(entries: Iterable[TimelineEntry]) -> list[MediaInfo]:
    seen: dict[str, MediaInfo] = {}
    for entry in entries:
        key = str(entry.asset)
        if key not in seen:
            seen[key] = probe(key)
    return list(seen.values())