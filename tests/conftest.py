"""Shared pytest fixtures.

Media fixtures are session-scoped: generating them once keeps the suite fast
enough to run on every push while still exercising the real renderer.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from video_pipeline import media
from video_pipeline.config import Config
from video_pipeline.testkit import (
    build_test_project,
    make_clip,
    make_image,
    make_voiceover,
)


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg/ffprobe not installed"
)


@pytest.fixture(scope="session", autouse=True)
def _clear_probe_cache():
    media.clear_cache()
    yield
    media.clear_cache()


@pytest.fixture(scope="session")
def config() -> Config:
    return Config.from_mapping({})


@pytest.fixture(scope="session")
def media_dir(tmp_path_factory) -> Path:
    """A handful of small assets shared by several tests."""
    root = tmp_path_factory.mktemp("media")
    make_clip(root / "red.mp4", colour="red", seconds=4.0)
    make_clip(root / "blue.mp4", colour="blue", seconds=4.0)
    make_clip(root / "green.mp4", colour="green", seconds=4.0)
    make_image(root / "still.jpg", colour="green")
    make_voiceover(root / "voice.wav", seconds=4.0)
    return root


@pytest.fixture(scope="session")
def project_dir(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("project") / "demo"
    build_test_project(root, seconds=6.0)
    return root


@pytest.fixture(scope="session")
def rendered(project_dir) -> Path:
    """Render the synthetic project once; several tests assert on the result."""
    from video_pipeline.cli import main

    code = main(["render", str(project_dir)])
    assert code == 0, "synthetic render should succeed"
    return project_dir / "output" / "final.mp4"