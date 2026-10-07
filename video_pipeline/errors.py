"""Error taxonomy.

Every error carries enough context to answer, without reproducing the run:
which project, which asset, which stage, what failed.
"""

from __future__ import annotations

from typing import Any


class PipelineError(Exception):
    """Base class for every error this pipeline raises on purpose."""

    exit_code = 1

    def __init__(self, message: str, **context: Any) -> None:
        self.message = message
        self.context = {k: v for k, v in context.items() if v is not None}
        super().__init__(self.render())

    def render(self) -> str:
        if not self.context:
            return self.message
        details = "\n".join(f"  {key}: {value}" for key, value in self.context.items())
        return f"{self.message}\n{details}"


class ConfigError(PipelineError):
    """project.yaml / config.yaml is missing, malformed or internally invalid."""

    exit_code = 2


class ValidationError(PipelineError):
    """The project cannot be rendered as described.

    Carries every problem found, so the author can fix them in one pass
    instead of discovering them one render at a time.
    """

    exit_code = 2

    def __init__(self, message: str, problems: list[str] | None = None, **context: Any) -> None:
        self.problems = problems or []
        super().__init__(message, **context)

    def render(self) -> str:
        base = super().render()
        if not self.problems:
            return base
        listing = "\n".join(f"  [{i}] {p}" for i, p in enumerate(self.problems, 1))
        return f"{base}\n{len(self.problems)} problem(s):\n{listing}"


class AssetError(PipelineError):
    """A source asset is missing, unreadable, or outside the project."""

    exit_code = 3


class FFmpegError(PipelineError):
    """An FFmpeg or FFprobe invocation failed.

    ``stderr_tail`` is preserved verbatim so the real cause is never hidden.
    """

    exit_code = 4

    def __init__(
        self,
        message: str,
        *,
        stage: str | None = None,
        command: list[str] | None = None,
        stderr_tail: str | None = None,
        returncode: int | None = None,
        **context: Any,
    ) -> None:
        self.stage = stage
        self.command = command
        self.stderr_tail = stderr_tail
        self.returncode = returncode
        super().__init__(message, **context)

    def render(self) -> str:
        parts = [PipelineError.render(self)]
        if self.stage:
            parts.append(f"  stage: {self.stage}")
        if self.command:
            parts.append(f"  command: {self.command}")
        if self.returncode is not None:
            parts.append(f"  exit code: {self.returncode}")
        if self.stderr_tail:
            parts.append(f"  ffmpeg stderr (last 40 lines):\n{self.stderr_tail}")
        return "\n".join(parts)


class CapabilityError(PipelineError):
    """The local FFmpeg build is missing something the pipeline requires."""

    exit_code = 5