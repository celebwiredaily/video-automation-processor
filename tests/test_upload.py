"""Tests for uploading a validated video to storage.

Unit tests use a fake Google Drive service so no real credentials are needed
and nothing touches the network. The behaviours under test are the ones that
make an upload trustworthy: it happens only after validation passes, it never
reports success the API did not confirm, and it never treats a produced output
as an input asset.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from video_pipeline import storage
from video_pipeline.errors import AssetError, ConfigError, ValidationError
from video_pipeline.storage import (
    AssetRecord,
    GoogleDriveStorage,
    LocalStorage,
    UploadResult,
    _next_version_name,
    guess_mime_type,
)


# ---------------------------------------------------------------- fake Drive


class FakeHttpError(Exception):
    """Stands in for googleapiclient.errors.HttpError."""

    def __init__(self, status: int, message: str = "error"):
        super().__init__(message)
        self.resp = type("Resp", (), {"status": status})()
        self.status = status


class FakeFiles:
    def __init__(self, drive: "FakeDrive"):
        self.drive = drive

    def list(self, q=None, fields=None, pageSize=None, pageToken=None):
        self.drive.calls.append(("list", q))
        parts = [p.strip() for p in (q or "").split(" and ")]
        folder = None
        name_filter = None
        for part in parts:
            if " in parents" in part:
                folder = part.split("'")[1]
            elif part.startswith("name = "):
                name_filter = part[len("name = "):].strip().strip("'")
        if self.drive.list_error is not None:
            raise self.drive.list_error
        files = [
            {"id": f["id"], "name": f["name"], "size": str(f["size"]),
             "mimeType": f["mime"]}
            for f in self.drive._store.values()
            if folder is None or folder in f["parents"]
        ]
        if name_filter:
            files = [f for f in files if f["name"] == name_filter]
        return FakeCall({"files": files})

    def create(self, body=None, media_body=None, fields=None):
        self.drive.calls.append(("create", body))
        if self.drive.create_error is not None:
            raise self.drive.create_error
        record = {
            "id": f"id-{len(self.drive._store) + 1}",
            "name": body["name"],
            "size": self.drive.last_upload_size,
            "mime": self.drive.last_upload_mime,
            "parents": list(body.get("parents") or []),
        }
        self.drive._store[record["id"]] = record
        return FakeCall({
            "id": record["id"],
            "name": record["name"],
            "size": str(record["size"]),
            "mimeType": record["mime"],
            "webViewLink": f"https://drive.google.com/file/d/{record['id']}/view",
        })

    def update(self, fileId=None, media_body=None):
        self.drive.calls.append(("update", fileId))
        if self.drive.update_error is not None:
            raise self.drive.update_error
        record = self.drive._store[fileId]
        record["size"] = self.drive.last_upload_size
        record["mime"] = self.drive.last_upload_mime
        return FakeCall({"id": fileId, "name": record["name"]})

    def get_media(self, fileId=None):
        self.drive.calls.append(("get_media", fileId))
        record = self.drive._store.get(fileId)
        if record is None:
            raise FakeHttpError(404, "not found")
        return FakeMedia(bytes(record["size"]))

    def get(self, fileId=None, fields=None):
        self.drive.calls.append(("get", fileId))
        if self.drive.get_error is not None:
            raise self.drive.get_error
        if fileId not in self.drive._store:
            raise FakeHttpError(404, "not found")
        record = self.drive._store[fileId]
        return FakeCall({
            "id": record["id"],
            "name": record["name"],
            "size": str(record["size"]),
            "mimeType": record["mime"],
            "parents": record["parents"],
            "webViewLink": f"https://drive.google.com/file/d/{record['id']}/view",
        })


class FakeMedia:
    def __init__(self, payload: bytes):
        self._payload = payload

    def execute(self):
        return self._payload


class FakeCall:
    def __init__(self, payload):
        self._payload = payload

    def execute(self):
        return self._payload


class FakeDrive:
    """Minimal in-memory stand-in for the Drive v3 service."""

    def __init__(self, folder: str = "FOLDERID0000000001"):
        self._store: dict[str, dict] = {}
        self.folder = folder
        self.calls: list[tuple] = []
        self.list_error = None
        self.create_error = None
        self.update_error = None
        self.get_error = None
        self.last_upload_size = 0
        self.last_upload_mime = ""
        self._files = FakeFiles(self)

    def files(self):  # noqa: A003 - mirrors the real client shape
        return self._files


class FakeMediaBody:
    """Stand-in for MediaFileUpload; the fake records size/mime separately."""

    def __init__(self, path, mimetype=None):
        self.path = str(path)
        self.mimetype = mimetype


class UploadStorage(GoogleDriveStorage):
    """GoogleDriveStorage with auth and the media body stubbed out.

    Uses the injected media_body_factory seam, so these tests need no
    googleapiclient install - which is exactly how normal CI runs them.
    """

    media_body_factory = staticmethod(
        lambda source, mime: FakeMediaBody(source, mime)
    )

    def __init__(self, drive: FakeDrive, *, credentials: bool = True, folder="FOLDERID0000000001"):
        super().__init__(folder, credentials_file=Path("dummy.json") if credentials else None)
        self._fake = drive
        self._service = drive
        self._service_write = drive

    def _write_service(self):
        if not self.credentials_file:
            raise AssetError(
                "Google Drive upload requires authenticated access.",
                hint="Set GOOGLE_DRIVE_CREDENTIALS.",
            )
        return self._fake

    def _auth_service(self):
        return self._fake if self.credentials_file else None

    def list_assets(self):
        records = []
        for record in self._fake._store.values():
            records.append(AssetRecord(
                name=record["name"], size=record["size"],
                mime_type=record["mime"], source_id=record["id"],
                storage=self.name,
            ))
        return records


def _video(tmp_path: Path, name: str = "final.mp4", size: int = 2048) -> Path:
    """A file that looks real enough to upload (non-empty, .mp4)."""
    path = tmp_path / name
    path.write_bytes(b"\x00" * size)
    return path


def _uploadable(drive: FakeDrive, tmp_path: Path, **kwargs) -> UploadStorage:
    store = UploadStorage(drive, **kwargs)
    # Record what a real MediaFileUpload would have sent.
    original_upload = store.upload

    def upload_with_size(source, *, name=None, existing_file="overwrite"):
        if Path(source).is_file():
            drive.last_upload_size = Path(source).stat().st_size
            drive.last_upload_mime = guess_mime_type(Path(source))
        return original_upload(source, name=name, existing_file=existing_file)

    store.upload = upload_with_size  # type: ignore[method-assign]
    return store


# --------------------------------------------------------------- mime types


def test_mp4_is_always_video_mp4():
    assert guess_mime_type(Path("final.mp4")) == "video/mp4"
    assert guess_mime_type(Path("a.MP4")) == "video/mp4"
    assert guess_mime_type(Path("x.txt")) in {"text/plain", "application/octet-stream"}


def test_version_names_are_deterministic():
    assert _next_version_name("final.mp4", set()) == "final_v2.mp4"
    assert _next_version_name("final.mp4", {"final_v2.mp4"}) == "final_v3.mp4"
    assert _next_version_name("final.mp4", {"final_v2.mp4", "final_v3.mp4"}) == "final_v4.mp4"
    # Never produces Drive-style "final (1).mp4".
    assert "(1)" not in _next_version_name("final.mp4", {"final.mp4"})


# ------------------------------------------------------------ upload success


def test_successful_upload_returns_real_metadata(tmp_path):
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)
    video = _video(tmp_path)

    result = store.upload(video, name="final.mp4", existing_file="overwrite")

    assert isinstance(result, UploadResult)
    assert result.file_id, "a Drive file id must be returned"
    assert result.name == "final.mp4"
    assert result.size == 2048
    assert result.mime_type == "video/mp4"
    assert result.folder_id == "FOLDERID0000000001"
    assert result.web_url and result.file_id in result.web_url
    # Verify the file really is in the fake Drive, in the right folder.
    stored = drive._store[result.file_id]
    assert stored["name"] == "final.mp4"
    assert stored["mime"] == "video/mp4"
    assert stored["size"] == 2048
    assert stored["parents"] == ["FOLDERID0000000001"]


def test_upload_verifies_by_reading_the_file_back(tmp_path):
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)
    result = store.upload(_video(tmp_path), name="final.mp4")
    assert ("get", result.file_id) in drive.calls, "must re-read to verify"


def test_upload_to_a_separate_output_folder(tmp_path):
    drive = FakeDrive()
    store = UploadStorage(drive, folder="INPUTFOLDER00001", credentials=True)
    store.output_folder_id = "OUTPUTFOLDER00001"
    original = store.upload

    def with_size(source, *, name=None, existing_file="overwrite"):
        drive.last_upload_size = source.stat().st_size
        drive.last_upload_mime = guess_mime_type(source)
        return original(source, name=name, existing_file=existing_file)

    store.upload = with_size  # type: ignore[method-assign]
    result = store.upload(_video(tmp_path), name="final.mp4")

    assert drive._store[result.file_id]["parents"] == ["OUTPUTFOLDER00001"]
    assert result.folder_id == "OUTPUTFOLDER00001"


def test_input_and_output_may_be_the_same_folder(tmp_path):
    drive = FakeDrive()
    store = UploadStorage(drive, folder="SAMEFOLDER00001")
    assert store.input_folder_id == "SAMEFOLDER00001"
    assert store.output_folder_id == "SAMEFOLDER00001"


# ------------------------------------------------------------ upload failure


def test_permission_failure_is_reported_as_permission_not_success(tmp_path):
    drive = FakeDrive()
    drive.create_error = FakeHttpError(403, "Insufficient Permission")
    store = _uploadable(drive, tmp_path)

    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4")

    message = str(exc.value)
    assert "does not have write permission" in message
    assert "FOLDERID0000000001" in message
    assert "client_email" in message


def test_authentication_failure_is_reported_clearly(tmp_path):
    drive = FakeDrive()
    drive.create_error = FakeHttpError(401, "Invalid Credentials")
    store = _uploadable(drive, tmp_path)

    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4")
    assert "authentication failed" in str(exc.value).lower()


def test_upload_without_credentials_refuses(tmp_path):
    store = UploadStorage(FakeDrive(), credentials=False)
    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4")
    assert "requires authenticated access" in str(exc.value)


def test_generic_failure_does_not_look_like_success(tmp_path):
    drive = FakeDrive()
    drive.create_error = FakeHttpError(500, "backend error")
    store = _uploadable(drive, tmp_path)
    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4")
    assert "upload failed" in str(exc.value).lower()
    assert not drive._store, "nothing should have been stored"


def test_empty_file_is_refused(tmp_path):
    empty = tmp_path / "final.mp4"
    empty.write_bytes(b"")
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)
    with pytest.raises(AssetError) as exc:
        store.upload(empty, name="final.mp4")
    assert "empty file" in str(exc.value)


def test_missing_file_is_refused(tmp_path):
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)
    with pytest.raises(AssetError):
        store.upload(tmp_path / "nope.mp4", name="final.mp4")


def test_verification_failure_raises_rather_than_reporting_success(tmp_path):
    """If the read-back does not match, that is a failure."""
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)

    real_upload = store.upload

    def corrupting(source, *, name=None, existing_file="overwrite"):
        result = real_upload(source, name=name, existing_file=existing_file)
        # Corrupt the stored record after upload to simulate a bad write.
        drive._store[result.file_id]["mime"] = "application/octet-stream"
        # Re-run verification by calling _verify directly.
        store._verify(result.file_id, name, source.stat().st_size, "video/mp4")
        return result

    store.upload = corrupting  # type: ignore[method-assign]
    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4")
    assert "verification failed" in str(exc.value).lower()
    assert "mimeType" in str(exc.value)


# ------------------------------------------------------- duplicate policies


def test_overwrite_replaces_the_existing_file(tmp_path):
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)

    first = store.upload(_video(tmp_path, size=1000), name="final.mp4")
    second = store.upload(_video(tmp_path, size=4096), name="final.mp4")

    assert second.file_id == first.file_id, "must not create a second file"
    assert len(drive._store) == 1
    assert drive._store[first.file_id]["size"] == 4096
    assert second.replaced_existing is True


def test_version_policy_creates_deterministic_versions(tmp_path):
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)

    names = [store.upload(_video(tmp_path), name="final.mp4",
                          existing_file="version").name for _ in range(3)]
    assert names == ["final.mp4", "final_v2.mp4", "final_v3.mp4"]
    assert len(drive._store) == 3
    assert not any("(1)" in n for n in names)


def test_error_policy_refuses_to_overwrite(tmp_path):
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)
    store.upload(_video(tmp_path), name="final.mp4", existing_file="error")

    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4", existing_file="error")
    assert "already contains" in str(exc.value)
    assert "existing_file" in str(exc.value)


def test_unknown_policy_is_rejected(tmp_path):
    drive = FakeDrive()
    store = _uploadable(drive, tmp_path)
    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4", existing_file="clobber")
    assert "existing_file policy" in str(exc.value)


# ------------------------------------------------------------- local upload


def test_local_storage_upload_copies(tmp_path):
    source = _video(tmp_path, "out.mp4", size=333)
    store = LocalStorage(tmp_path / "dest")
    result = store.upload(source, name="final.mp4")
    assert (tmp_path / "dest" / "final.mp4").read_bytes() == source.read_bytes()
    assert result.size == 333
    assert result.mime_type == "video/mp4"


def test_local_storage_honours_the_version_policy(tmp_path):
    source = _video(tmp_path, "out.mp4")
    store = LocalStorage(tmp_path / "dest")
    assert store.upload(source, name="final.mp4", existing_file="version").name == "final.mp4"
    assert store.upload(source, name="final.mp4", existing_file="version").name == "final_v2.mp4"


def test_upload_only_if_supported(tmp_path):
    class NoUpload(LocalStorage):
        supports_upload = False

        def upload(self, source, *, name=None, existing_file="overwrite"):
            return storage.Storage.upload(self, source, name=name, existing_file=existing_file)

    with pytest.raises(AssetError) as exc:
        NoUpload(tmp_path).upload(_video(tmp_path))
    assert "cannot upload" in str(exc.value)


# ------------------------------------------- output files are never inputs


def test_output_file_is_not_downloaded_as_an_input(tmp_path):
    assets = tmp_path / "assets"
    assets.mkdir()
    (assets / "clip001.mp4").write_bytes(b"clip")
    (assets / "final.mp4").write_bytes(b"previous render")
    (assets / "voiceover.wav").write_bytes(b"audio")

    staged, notes = storage.auto_stage(
        tmp_path, assets, exclude_names={"final.mp4"}
    )
    staged_names = {p.name for p in staged}
    assert "clip001.mp4" in staged_names
    assert "final.mp4" not in staged_names, "produced output must not be an input"
    assert any("final.mp4" in n for n in notes)


def test_drive_staging_excludes_output_names(tmp_path):
    drive = FakeDrive()
    drive._store["a"] = {"id": "a", "name": "broll.mp4", "size": 10,
                        "mime": "video/mp4", "parents": ["FOLDERID0000000001"]}
    drive._store["b"] = {"id": "b", "name": "final.mp4", "size": 20,
                        "mime": "video/mp4", "parents": ["FOLDERID0000000001"]}
    store = UploadStorage(drive)
    target = tmp_path / "staged"
    target.mkdir()

    staged = store.stage_all(target, exclude={"final.mp4"})
    assert set(staged) == {"broll.mp4"}


# ---------------------------------------- validation gates the upload


def test_validation_failure_prevents_upload(tmp_path, monkeypatch, capsys):
    """A render whose validation fails must never publish anything."""
    from video_pipeline import cli

    project_dir = tmp_path / "proj"
    (project_dir / "assets").mkdir(parents=True)
    (project_dir / "audio").mkdir()
    (project_dir / "captions").mkdir()
    (project_dir / "output").mkdir()
    (project_dir / "assets" / "a.mp4").write_bytes(b"x")
    (project_dir / "project.yaml").write_text(
        "name: p\nvoiceover: audio/voiceover.wav\n",
        encoding="utf-8",
    )

    class FakeReport:
        ok = False

        def summary(self):
            return "FAIL"

    uploaded: list[str] = []

    def fake_upload(project, video_path):
        uploaded.append(str(video_path))
        return 0

    monkeypatch.setattr(
        cli.validation, "validate_video", lambda *a, **k: FakeReport()
    )
    monkeypatch.setattr(cli, "do_upload", fake_upload)

    code = cli.main(["render", str(project_dir), "--upload"])
    assert code != 0, "must exit non-zero when validation fails"
    assert uploaded == [], "nothing may be uploaded when validation fails"


def test_upload_only_runs_after_a_passing_validation(tmp_path, monkeypatch, media_dir):
    from video_pipeline import cli

    project_dir = tmp_path / "proj"
    (project_dir / "assets").mkdir(parents=True)
    (project_dir / "output").mkdir()
    (project_dir / "assets" / "a.mp4").write_bytes((media_dir / "red.mp4").read_bytes())
    (project_dir / "project.yaml").write_text(
        "name: p\nconfig:\n  validation:\n    require_voiceover: false\n",
        encoding="utf-8",
    )
    (project_dir / "timeline.yaml").write_text(
        "timeline:\n  - start: 0\n    end: 2\n    asset: assets/a.mp4\n",
        encoding="utf-8",
    )

    order: list[str] = []

    class FakeResult:
        output_path = project_dir / "output" / "final.mp4"
        warnings: list[str] = []
        segment_count = 0
        caption_count = 0

    class FakeRenderer:
        def __init__(self, *a, **k):
            pass

        def render(self, timeline, duration):
            order.append("render")
            return FakeResult()

        logs_dir = project_dir / "build"

    class FakeReport:
        ok = True

        def summary(self):
            order.append("validate")
            return "PASS"

    monkeypatch.setattr(cli, "Renderer", FakeRenderer)
    monkeypatch.setattr(cli.media, "probe", lambda p: type("I", (), {"duration": 5.0})())
    monkeypatch.setattr(
        cli.validation, "validate_video", lambda *a, **k: FakeReport()
    )
    monkeypatch.setattr(
        cli, "do_upload",
        lambda project, video: (order.append("upload"), 0)[1],
    )

    cli.main(["render", str(project_dir), "--upload"])
    assert order == ["render", "validate", "upload"], (
        f"upload must come last, got {order}"
    )


def test_no_upload_when_flag_absent(tmp_path, monkeypatch, media_dir):
    from video_pipeline import cli

    project_dir = tmp_path / "proj"
    (project_dir / "assets").mkdir(parents=True)
    (project_dir / "output").mkdir()
    (project_dir / "assets" / "a.mp4").write_bytes((media_dir / "red.mp4").read_bytes())
    (project_dir / "project.yaml").write_text(
        "name: p\nconfig:\n  validation:\n    require_voiceover: false\n",
        encoding="utf-8",
    )
    (project_dir / "timeline.yaml").write_text(
        "timeline:\n  - start: 0\n    end: 2\n    asset: assets/a.mp4\n",
        encoding="utf-8",
    )

    class FakeResult:
        output_path = project_dir / "output" / "final.mp4"
        warnings: list[str] = []
        segment_count = 0
        caption_count = 0

    class FakeRenderer:
        def __init__(self, *a, **k):
            pass

        def render(self, timeline, duration):
            return FakeResult()

        logs_dir = project_dir / "build"

    class FakeReport:
        ok = True

        def summary(self):
            return "PASS"

    monkeypatch.setattr(cli, "Renderer", FakeRenderer)
    monkeypatch.setattr(cli.media, "probe", lambda p: type("I", (), {"duration": 5.0})())
    monkeypatch.setattr(cli.validation, "validate_video", lambda *a, **k: FakeReport())

    def explode(*a, **k):
        raise AssertionError("upload must not run without --upload")

    monkeypatch.setattr(cli, "do_upload", explode)
    assert cli.main(["render", str(project_dir)]) == 0


# ---------------------------------------------------- factory + config


def test_factory_supports_input_and_output_folders(tmp_path):
    store = storage.build_storage(
        "google_drive",
        root=tmp_path,
        input_folder="INFOLDER0000001",
        output_folder="OUTFOLDER000001",
    )
    assert isinstance(store, GoogleDriveStorage)
    assert store.input_folder_id == "INFOLDER0000001"
    assert store.output_folder_id == "OUTFOLDER000001"


def test_factory_single_folder_is_used_for_both(tmp_path):
    store = storage.build_storage(
        "google_drive", root=tmp_path, folder="ONEFOLDER0000001"
    )
    assert store.input_folder_id == "ONEFOLDER0000001"
    assert store.output_folder_id == "ONEFOLDER0000001"


def test_factory_rejects_drive_without_any_folder(tmp_path):
    with pytest.raises(AssetError) as exc:
        storage.build_storage("google_drive", root=tmp_path)
    assert "no storage folder is set" in str(exc.value)


def test_output_filename_policy_is_configurable():
    from video_pipeline.config import Config

    for policy in ("overwrite", "version", "error"):
        assert Config.from_mapping(
            {"output": {"existing_file": policy}}
        ).output.existing_file == policy
    with pytest.raises(ConfigError):
        Config.from_mapping({"output": {"existing_file": "nope"}})


def test_no_credentials_are_printed_in_errors(tmp_path):
    drive = FakeDrive()
    drive.create_error = FakeHttpError(403, "Insufficient Permission")
    store = _uploadable(drive, tmp_path)
    with pytest.raises(AssetError) as exc:
        store.upload(_video(tmp_path), name="final.mp4")
    text = str(exc.value)
    assert "ya29" not in text            # no token fragment
    assert "private_key" not in text
    assert "GOOGLE_API_KEY" not in text


# --------------------------------------- optional real integration test


@pytest.mark.skipif(
    not Path(storage.os.environ.get("GOOGLE_DRIVE_CREDENTIALS", "")).is_file(),
    reason="GOOGLE_DRIVE_CREDENTIALS is not set to a real file",
)
def test_real_drive_round_trip(tmp_path):
    """One genuine upload, only when real credentials are present.

    Never faked: without credentials the test is skipped, and with them the
    Drive API must confirm the file.
    """
    store = GoogleDriveStorage(
        "1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO",
        credentials_file=Path(storage.os.environ["GOOGLE_DRIVE_CREDENTIALS"]),
    )
    payload = tmp_path / "final.mp4"
    payload.write_bytes(b"\x00" * 4096)
    result = store.upload(payload, name="video_pipeline_probe.mp4",
                          existing_file="version")
    assert result.file_id
    assert result.mime_type == "video/mp4"
    assert result.size == 4096