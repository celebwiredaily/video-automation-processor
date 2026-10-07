"""Storage abstraction tests.

Covers LocalStorage fully offline and GoogleDriveStorage's URL/credential
handling without ever contacting Google.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from video_pipeline.errors import AssetError
from video_pipeline.storage import (
    AssetRecord,
    GoogleDriveStorage,
    LocalStorage,
    _extract_folder_id,
    build_storage,
)

FOLDER_URL = (
    "https://drive.google.com/drive/folders/1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO?usp=sharing"
)


def test_local_storage_lists_and_copies(tmp_path):
    source = tmp_path / "src"
    source.mkdir()
    (source / "clip001.mp4").write_bytes(b"video")
    (source / "image001.jpg").write_bytes(b"image")
    (source / "notes.txt").write_bytes(b"ignore me")

    storage = LocalStorage(source)
    records = storage.list_assets()
    names = {r.name for r in records}
    assert names == {"clip001.mp4", "image001.jpg", "notes.txt"}

    video = next(r for r in records if r.name == "clip001.mp4")
    assert video.is_video and not video.is_image
    assert video.is_media

    destination = tmp_path / "staged" / "clip001.mp4"
    storage.download(video, destination)
    assert destination.read_bytes() == b"video"


def test_local_storage_missing_asset_raises(tmp_path):
    storage = LocalStorage(tmp_path)
    record = AssetRecord("gone.mp4", 0, "", str(tmp_path / "gone.mp4"), "local")
    with pytest.raises(AssetError):
        storage.download(record, tmp_path / "out.mp4")


@pytest.mark.parametrize(
    "value",
    [
        "1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO",
        FOLDER_URL,
        "https://drive.google.com/open?id=1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO",
    ],
)
def test_folder_id_is_extracted_from_urls(value):
    assert _extract_folder_id(value) == "1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO"


def test_bad_folder_value_is_rejected():
    with pytest.raises(AssetError):
        _extract_folder_id("not a folder id")


def test_drive_download_url_is_built_from_the_file_id():
    drive = GoogleDriveStorage(FOLDER_URL)
    url = drive.download_url("FILEID123")
    assert "FILEID123" in url
    assert "export=download" in url


def test_public_listing_without_api_key_explains_itself():
    drive = GoogleDriveStorage(FOLDER_URL)
    with pytest.raises(AssetError) as exc:
        drive.list_assets()
    assert "API key" in str(exc.value)
    assert "GOOGLE_API_KEY" in str(exc.value)


def test_missing_credentials_file_is_reported_clearly(tmp_path):
    drive = GoogleDriveStorage(FOLDER_URL, credentials_file=tmp_path / "nope.json")
    with pytest.raises(AssetError) as exc:
        drive.list_assets()
    assert "credentials file not found" in str(exc.value)


def test_build_storage_factory(tmp_path):
    local = build_storage("local", root=tmp_path)
    assert isinstance(local, LocalStorage)

    drive = build_storage(
        "google_drive", root=tmp_path, folder=FOLDER_URL,
        destination=tmp_path / "assets",
    )
    assert isinstance(drive, GoogleDriveStorage)

    with pytest.raises(AssetError):
        build_storage("dropbox", root=tmp_path)


def test_build_storage_requires_a_folder_for_drive(tmp_path):
    with pytest.raises(AssetError) as exc:
        build_storage("google_drive", root=tmp_path)
    message = str(exc.value)
    assert "no storage folder is set" in message
    assert "input_folder_id" in message
    assert "output_folder_id" in message


def test_storage_metadata_does_not_leak_secrets():
    drive = GoogleDriveStorage(FOLDER_URL, api_key="super-secret-key")
    record = AssetRecord("clip.mp4", 10, "video/mp4", "ID", "google_drive")
    text = str(drive.metadata(record))
    assert "super-secret-key" not in text


def test_renderer_never_receives_a_drive_url(tmp_path, media_dir):
    """The renderer must work purely from local paths."""
    from video_pipeline.project import load_project

    project_dir = tmp_path / "p"
    (project_dir / "assets").mkdir(parents=True)
    (project_dir / "assets" / "a.mp4").write_bytes(
        (media_dir / "red.mp4").read_bytes()
    )
    (project_dir / "project.yaml").write_text(
        "name: p\n"
        "storage:\n"
        "  backend: google_drive\n"
        f"  folder: {FOLDER_URL}\n"
        "timeline:\n"
        "  - start: 0\n    end: 1\n    asset: assets/a.mp4\n",
        encoding="utf-8",
    )
    project = load_project(project_dir)
    # Storage config is recorded, but every timeline asset resolves to a real
    # local file inside the project - never a Drive URL handed to FFmpeg.
    assert project.storage["backend"] == "google_drive"
    for entry in project.timeline.entries:
        assert entry.asset.exists()
        assert "drive.google.com" not in str(entry.asset)
        assert entry.asset.resolve().is_relative_to(project_dir.resolve())