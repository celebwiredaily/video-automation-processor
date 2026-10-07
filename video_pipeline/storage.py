"""Storage abstraction.

Rendering never knows where assets came from. A storage backend resolves
logical asset names to *local* paths inside a working directory; everything
downstream of staging is plain filesystem work.

    Google Drive ──▶ GoogleDriveStorage ──▶ local staging dir ──▶ FFmpeg
    local disk   ──▶ LocalStorage        ──▶ local staging dir ──▶ FFmpeg

Credentials are read from the environment only:

    GOOGLE_DRIVE_CREDENTIALS   path to a service-account JSON file
    GOOGLE_API_KEY             optional API key for public-folder listing

Nothing in this module prints a token or writes a secret to disk.
"""

from __future__ import annotations

import json
import mimetypes
import os
import re
import shutil
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

from .errors import AssetError

FOLDER_ID_RE = re.compile(r"^[A-Za-z0-9_-]{10,}$")
USER_AGENT = "video-pipeline/1.0 (asset staging)"

# Media extensions the pipeline understands, used to filter folder listings.
MEDIA_EXTENSIONS = {
    ".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi",
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff",
    ".wav", ".mp3", ".m4a", ".aac", ".flac", ".ogg", ".opus",
    ".srt", ".vtt", ".json", ".txt", ".md",
}


@dataclass(frozen=True)
class AssetRecord:
    """One remote object, independent of how it was discovered."""

    name: str
    size: int
    mime_type: str
    source_id: str
    storage: str

    @property
    def is_media(self) -> bool:
        return Path(self.name).suffix.lower() in MEDIA_EXTENSIONS

    @property
    def is_video(self) -> bool:
        return Path(self.name).suffix.lower() in {".mp4", ".mov", ".m4v", ".mkv", ".webm", ".avi"}

    @property
    def is_image(self) -> bool:
        return Path(self.name).suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class UploadResult:
    """Confirmation returned by a storage backend after an upload.

    Every field is populated from what the backend actually reported. The CLI
    only prints "uploaded successfully" when this object exists, and only a
    backend that made a real API call creates one.
    """

    name: str
    source_path: str
    size: int
    mime_type: str
    folder_id: str
    backend: str
    file_id: str | None = None
    web_url: str | None = None
    replaced_existing: bool = False

    def describe(self) -> str:
        lines = [
            f"name       : {self.name}",
            f"size       : {self.size} bytes",
            f"mime type  : {self.mime_type}",
            f"destination: folder {self.folder_id} ({self.backend})",
            f"replaced   : {'yes' if self.replaced_existing else 'no'}",
        ]
        if self.file_id:
            lines.append(f"file id    : {self.file_id}")
        if self.web_url:
            lines.append(f"url        : {self.web_url}")
        return "\n".join(lines)


def guess_mime_type(path: Path) -> str:
    """MIME type for an upload. MP4 is pinned because browsers differ."""
    suffix = path.suffix.lower()
    if suffix == ".mp4":
        return "video/mp4"
    guessed, _ = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


class Storage(ABC):
    """Resolve remote assets to local files, and store local results remotely."""

    name: str = "storage"
    supports_upload: bool = False

    @abstractmethod
    def list_assets(self) -> list[AssetRecord]:
        """Return everything this backend can see."""

    @abstractmethod
    def download(self, record: AssetRecord, destination: Path) -> Path:
        """Fetch one asset into ``destination`` and return the local path."""

    def upload(
        self,
        source: Path,
        *,
        name: str | None = None,
        existing_file: str = "overwrite",
    ) -> UploadResult:
        """Store a local file. Backends without write access refuse clearly."""
        raise AssetError(
            f"The {self.name} backend cannot upload files.",
            hint="Use `backend: google_drive` with GOOGLE_DRIVE_CREDENTIALS to "
            "upload, or drop --upload to render locally only.",
        )

    def find(self, name: str) -> AssetRecord | None:
        """Return the record named ``name``, or None. Case-insensitive."""
        lowered = name.lower()
        for record in self.list_assets():
            if record.name.lower() == lowered:
                return record
        return None

    def exists(self, name: str) -> bool:
        return self.find(name) is not None

    def metadata(self, record: AssetRecord) -> dict:
        return {
            "name": record.name,
            "size": record.size,
            "mime_type": record.mime_type,
            "source_id": record.source_id,
            "storage": record.storage,
        }


class LocalStorage(Storage):
    """Assets already on disk. The default, and works fully offline."""

    name = "local"
    supports_upload = True

    def __init__(self, root: Path) -> None:
        self.root = Path(root)

    def list_assets(self) -> list[AssetRecord]:
        if not self.root.is_dir():
            return []
        records = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                records.append(
                    AssetRecord(
                        name=path.name,
                        size=path.stat().st_size,
                        mime_type="",
                        source_id=str(path),
                        storage=self.name,
                    )
                )
        return records

    def download(self, record: AssetRecord, destination: Path) -> Path:
        source = Path(record.source_id)
        if not source.is_file():
            raise AssetError("Local asset is missing.", asset=str(source))
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)
        return destination

    def upload(
        self,
        source: Path,
        *,
        name: str | None = None,
        existing_file: str = "overwrite",
    ) -> UploadResult:
        """Copy the file into this backend's root, honouring the same policy."""
        source = Path(source)
        if not source.is_file():
            raise AssetError("Nothing to upload.", file=str(source))

        target_name = name or source.name
        target = self.root / target_name
        self.root.mkdir(parents=True, exist_ok=True)

        replaced = False
        if target.exists():
            if existing_file == "error":
                raise AssetError(
                    "Destination already contains a file with this name.",
                    name=target_name,
                    directory=str(self.root),
                    hint="Set output.existing_file to overwrite or version.",
                )
            if existing_file == "version":
                target = self._next_version(target)
            replaced = existing_file == "overwrite"

        shutil.copy2(source, target)
        return UploadResult(
            name=target.name,
            source_path=str(source),
            size=target.stat().st_size,
            mime_type=guess_mime_type(target),
            folder_id=str(self.root),
            backend=self.name,
            file_id=str(target),
            web_url=str(target),
            replaced_existing=replaced,
        )

    @staticmethod
    def _next_version(path: Path) -> Path:
        """final.mp4 -> final_v2.mp4 -> final_v3.mp4 ... deterministic."""
        match = re.match(r"^(?P<stem>.+?)(?P<ext>\.[A-Za-z0-9]+)$", path.name)
        if not match:
            return path
        stem, ext = match.group("stem"), match.group("ext")
        index = 2
        while True:
            candidate = path.with_name(f"{stem}_v{index}{ext}")
            if not candidate.exists():
                return candidate
            index += 1


class GoogleDriveStorage(Storage):
    """Google Drive backend.

    Two access modes:

    * **Public folder** (no credentials) - the folder must be shared
      "anyone with the link". Files are listed through the Drive public
      listing endpoint and downloaded through ``uc?export=download``.
    * **Authenticated** - set ``GOOGLE_DRIVE_CREDENTIALS`` to a service
      account JSON file. This is needed for private folders and is the
      supported path for CI, using GitHub Secrets.

    The public folder URL is only ever an *input* here; the renderer itself
    never sees a Drive URL.
    """

    name = "google_drive"
    supports_upload = True

    # Seam for tests: constructing the upload payload is the only part of the
    # upload path that needs googleapiclient installed, so it is isolated here.
    # Everything the tests care about - policy, verification, error
    # classification - runs without the optional dependency. The default is
    # resolved lazily because the helper is defined further down this module.
    media_body_factory = staticmethod(
        lambda source, mime: _drive_media_body(source, mime)
    )

    def __init__(
        self,
        folder: str,
        *,
        credentials_file: Path | None = None,
        api_key: str | None = None,
        destination: Path | None = None,
        output_folder: str | None = None,
    ) -> None:
        self.folder_id = _extract_folder_id(folder)
        # Input and output folders may be the same; that is a supported
        # configuration rather than an accident, so both ids are kept.
        self.output_folder_id = (
            _extract_folder_id(output_folder) if output_folder else self.folder_id
        )
        self.input_folder_id = self.folder_id
        self.credentials_file = credentials_file
        self.api_key = api_key
        self.destination = Path(destination) if destination else Path("assets")
        self._service = None
        self._service_write = None

    # ------------------------------------------------------------- discovery

    def _build_service(self, write: bool):
        # Check the credentials path before importing the client libraries: a
        # wrong path is more actionable than "module not installed".
        if not self.credentials_file:
            raise AssetError(
                "Google Drive upload requires authenticated access."
                if write
                else "Authenticated Google Drive access needs a credentials file.",
                hint="Set GOOGLE_DRIVE_CREDENTIALS to a service-account JSON "
                "file. Public links allow reading shared files but never "
                "writing, so uploading is not possible anonymously.",
            )
        if not Path(self.credentials_file).is_file():
            raise AssetError(
                "Google Drive credentials file not found.",
                path=str(self.credentials_file),
                hint="Set GOOGLE_DRIVE_CREDENTIALS to a readable JSON file.",
            )

        try:
            from googleapiclient.discovery import build  # type: ignore
            from google.oauth2.service_account import Credentials  # type: ignore
        except ImportError as exc:
            raise AssetError(
                "Authenticated Google Drive access needs extra dependencies.",
                detail="pip install google-api-python-client google-auth google-auth-httplib2",
                cause=str(exc),
            ) from exc

        # Write access needs the full drive scope; readonly is not enough.
        scope = (
            "https://www.googleapis.com/auth/drive"
            if write
            else "https://www.googleapis.com/auth/drive.readonly"
        )
        try:
            credentials = Credentials.from_service_account_file(
                str(self.credentials_file), scopes=[scope]
            )
        except (ValueError, KeyError, TypeError) as exc:
            # A malformed key file otherwise surfaces as a raw ASN.1 traceback.
            raise AssetError(
                "Google Drive credentials file is not a valid service-account "
                "JSON file.",
                path=str(self.credentials_file),
                detail=f"{type(exc).__name__}: {exc}"[:200],
                hint="Expected the JSON downloaded from the Google Cloud "
                "console, including an unescaped PEM `private_key`.",
            ) from exc
        except Exception as exc:  # noqa: BLE001 - google-auth raises broadly
            raise self._classify_upload_error(exc) from exc
        return build("drive", "v3", credentials=credentials)

    def _auth_service(self):
        if self._service is not None:
            return self._service
        if not self.credentials_file:
            return None
        self._service = self._build_service(write=False)
        return self._service

    def _write_service(self):
        if self._service_write is not None:
            return self._service_write
        self._service_write = self._build_service(write=True)
        return self._service_write

    # ----------------------------------------------------------------- upload

    def upload(
        self,
        source: Path,
        *,
        name: str | None = None,
        existing_file: str = "overwrite",
    ) -> UploadResult:
        """Upload a file to the destination folder and verify it landed.

        Only ever returns after the Drive API has confirmed the file, so a
        caller can treat a returned UploadResult as proof of success.
        """
        source = Path(source)
        if not source.is_file():
            raise AssetError("Nothing to upload.", file=str(source))
        if source.stat().st_size == 0:
            raise AssetError("Refusing to upload an empty file.", file=str(source))

        target_name = name or source.name
        mime_type = guess_mime_type(source)

        if existing_file not in {"overwrite", "version", "error"}:
            raise AssetError(
                f"Unknown existing_file policy {existing_file!r}.",
                hint="Use overwrite, version or error.",
            )

        service = self._write_service()

        # Every call against the API can raise, including the lookups below:
        # the OAuth token is refreshed lazily on the first request, so an
        # invalid grant surfaces here rather than at service construction.
        try:
            existing = self._find_in_output(target_name, service)
        except Exception as exc:  # noqa: BLE001
            raise self._classify_upload_error(exc) from exc

        if existing and existing_file == "error":
            raise AssetError(
                "Destination folder already contains a file with this name.",
                name=target_name,
                folder_id=self.output_folder_id,
                existing_file_id=existing["id"],
                hint="Set output.existing_file to overwrite or version.",
            )

        target_name, replaced = target_name, False
        if existing and existing_file == "version":
            try:
                taken = self._names_in_output(service)
                target_name = _next_version_name(target_name, taken)
                existing = self._find_in_output(target_name, service)
            except Exception as exc:  # noqa: BLE001
                raise self._classify_upload_error(exc) from exc

        media_body = self.media_body_factory(source, mime_type)

        try:
            if existing:
                request = service.files().update(
                    fileId=existing["id"], media_body=media_body
                )
                response = request.execute()
                replaced = True
            else:
                request = service.files().create(
                    body={"name": target_name, "parents": [self.output_folder_id]},
                    media_body=media_body,
                    fields="id, name, size, mimeType, webViewLink, webContentLink",
                )
                response = request.execute()
        except Exception as exc:  # noqa: BLE001 - classify, then re-raise clearly
            raise self._classify_upload_error(exc) from exc

        file_id = response.get("id") if isinstance(response, dict) else None
        if not file_id:
            raise AssetError(
                "Google Drive upload did not return a file id; treating this as "
                "a failure rather than a success.",
                response=repr(response)[:200],
            )

        # Verify by reading the file back, so success is never assumed.
        verified = self._verify(file_id, target_name, source.stat().st_size, mime_type)

        return UploadResult(
            name=verified["name"],
            source_path=str(source),
            size=verified["size"],
            mime_type=verified["mime_type"],
            folder_id=self.output_folder_id,
            backend=self.name,
            file_id=file_id,
            web_url=verified.get("web_url"),
            replaced_existing=replaced,
        )

    def _find_in_output(self, name: str, service) -> dict | None:
        response = (
            service.files()
            .list(
                q=f"'{self.output_folder_id}' in parents and name = "
                f"'{name.replace(chr(39), '')}' and trashed = false",
                fields="files(id, name, size, mimeType)",
                pageSize=10,
            )
            .execute()
        )
        files = response.get("files") or []
        return files[0] if files else None

    def _names_in_output(self, service) -> set[str]:
        response = (
            service.files()
            .list(
                q=f"'{self.output_folder_id}' in parents and trashed = false",
                fields="files(name)",
                pageSize=1000,
            )
            .execute()
        )
        return {item.get("name", "") for item in response.get("files") or []}

    def _verify(
        self, file_id: str, expected_name: str, expected_size: int, expected_mime: str
    ) -> dict:
        """Read the uploaded file back and confirm it is what we sent."""
        service = self._write_service()
        try:
            info = service.files().get(
                fileId=file_id, fields="id, name, size, mimeType, parents, webViewLink"
            ).execute()
        except Exception as exc:  # noqa: BLE001
            raise AssetError(
                "Upload verification failed: could not read the file back from "
                "Google Drive.",
                file_id=file_id,
                detail=str(exc),
            ) from exc

        problems = []
        name = info.get("name") or ""
        if name != expected_name:
            problems.append(f"name is {name!r}, expected {expected_name!r}")
        size = int(info.get("size") or 0)
        if size != expected_size:
            problems.append(f"size is {size}, expected {expected_size}")
        mime_type = info.get("mimeType") or ""
        if mime_type and mime_type != expected_mime:
            problems.append(f"mimeType is {mime_type!r}, expected {expected_mime!r}")
        parents = info.get("parents") or []
        if self.output_folder_id not in parents:
            problems.append(
                f"file is not in the destination folder {self.output_folder_id}"
            )
        if problems:
            raise AssetError(
                "Upload verification failed.",
                file_id=file_id,
                problems="; ".join(problems),
            )

        return {
            "name": name,
            "size": size,
            "mime_type": mime_type,
            "web_url": info.get("webViewLink")
            or (
                f"https://drive.google.com/file/d/{file_id}/view"
                if file_id
                else None
            ),
        }

    # ---------------------------------------------------------------- helpers

    def _classify_upload_error(self, exc: BaseException) -> AssetError:
        """Turn a Drive API failure into a message that names the real problem.

        A 403/404 on the destination folder almost always means the account
        cannot write there, and saying so is far more useful than echoing the
        API's wording.
        """
        status = _http_status(exc)
        detail = str(exc)[:300]

        if status in (401,):
            return AssetError(
                "Google Drive authentication failed: the credentials were "
                "rejected (HTTP 401).",
                detail=detail,
                hint="Check that GOOGLE_DRIVE_CREDENTIALS points at a valid "
                "service-account JSON file.",
            )
        if status in (403, 404):
            return AssetError(
                "Google Drive upload failed: the authenticated account does not "
                "have write permission to the destination folder.",
                folder_id=self.output_folder_id,
                status=status,
                detail=detail,
                hint="Share the folder with the service account's client_email "
                "and give it Editor access, or point output_folder_id at a "
                "folder it can write.",
            )
        if status == 403 or "forbidden" in detail.lower():
            return AssetError(
                "Google Drive upload failed: the authenticated account does not "
                "have write permission to the destination folder.",
                folder_id=self.output_folder_id,
                detail=detail,
                hint="Share the folder with the service account's client_email "
                "and give it Editor access.",
            )
        if _is_auth_error(exc):
            return AssetError(
                "Google Drive authentication failed during upload.",
                detail=detail,
                hint="Check that GOOGLE_DRIVE_CREDENTIALS points at a valid "
                "service-account JSON file.",
            )
        return AssetError(
            "Google Drive upload failed.",
            folder_id=self.output_folder_id,
            status=status,
            detail=detail,
        )

    def list_assets(self) -> list[AssetRecord]:
        service = self._auth_service()
        if service is not None:
            return self._list_authenticated(service)
        return self._list_public()

    def _list_authenticated(self, service) -> list[AssetRecord]:
        query = f"'{self.folder_id}' in parents and trashed = false"
        records: list[AssetRecord] = []
        page_token = None
        while True:
            response = (
                service.files()
                .list(
                    q=query,
                    fields=(
                        "nextPageToken, files(id, name, size, mimeType, "
                        "md5Checksum)"
                    ),
                    pageSize=1000,
                    pageToken=page_token,
                )
                .execute()
            )
            for item in response.get("files", []):
                if item.get("mimeType") == "application/vnd.google-apps.folder":
                    continue
                records.append(
                    AssetRecord(
                        name=item["name"],
                        size=int(item.get("size") or 0),
                        mime_type=item.get("mimeType", ""),
                        source_id=item["id"],
                        storage=self.name,
                    )
                )
            page_token = response.get("nextPageToken")
            if not page_token:
                break
        return records

    def _list_public(self) -> list[AssetRecord]:
        """List a publicly shared folder without credentials.

        Drive's public listing endpoint is the documented way to read a folder
        shared "anyone with the link". Requires an API key, which is a public
        identifier rather than a secret.
        """
        if not self.api_key:
            raise AssetError(
                "Public Google Drive listing needs an API key.",
                hint="Set GOOGLE_API_KEY, or set GOOGLE_DRIVE_CREDENTIALS for "
                "authenticated access.",
            )
        url = (
            "https://www.googleapis.com/drive/v3/files"
            f"?q=%27{urllib.parse.quote(self.folder_id)}%27+in+parents"
            "&fields=files(id,name,size,mimeType)&pageSize=1000"
            f"&key={urllib.parse.quote(self.api_key)}"
        )
        payload = self._http_json(url)
        return [
            AssetRecord(
                name=item["name"],
                size=int(item.get("size") or 0),
                mime_type=item.get("mimeType", ""),
                source_id=item["id"],
                storage=self.name,
            )
            for item in payload.get("files", [])
            if item.get("mimeType") != "application/vnd.google-apps.folder"
        ]

    # ------------------------------------------------------------- retrieval

    def download_url(self, file_id: str) -> str:
        if self._auth_service() is not None:
            raise NotImplementedError(
                "Authenticated downloads go through get_media(); see stage_all()."
            )
        return (
            "https://drive.usercontent.google.com/download"
            f"?id={urllib.parse.quote(file_id)}&export=download"
        )

    def download(self, record: AssetRecord, destination: Path) -> Path:
        destination.parent.mkdir(parents=True, exist_ok=True)
        service = self._auth_service()
        if service is not None:
            data = service.files().get_media(fileId=record.source_id).execute()
            destination.write_bytes(data)
        else:
            self._http_download(self.download_url(record.source_id), destination)

        if not destination.is_file() or destination.stat().st_size == 0:
            destination.unlink(missing_ok=True)
            raise AssetError(
                "Drive download produced an empty file.",
                asset=record.name,
                file_id=record.source_id,
                hint="A large file may need virus-scan confirmation; open the "
                "share link in a browser once to confirm access.",
            )
        return destination

    # ------------------------------------------------------------- staging

    def stage_all(
        self,
        destination: Path | None = None,
        *,
        wanted: set[str] | None = None,
        exclude: set[str] | None = None,
        progress=None,
    ) -> dict[str, Path]:
        """Download assets into a local directory. Returns name -> local path.

        ``exclude`` prevents a rendered output (for example ``final.mp4``)
        living in the same Drive folder from being downloaded as an input.
        """
        target = Path(destination or self.destination)
        target.mkdir(parents=True, exist_ok=True)

        records = self.list_assets()
        if wanted is not None:
            records = [r for r in records if r.name in wanted]
        skipped = {name.lower() for name in (exclude or set())}
        if skipped:
            records = [r for r in records if r.name.lower() not in skipped]

        staged: dict[str, Path] = {}
        for record in records:
            local = target / record.name
            if local.is_file() and local.stat().st_size == record.size and record.size > 0:
                staged[record.name] = local
                if progress:
                    progress(f"skip   {record.name} (already staged)")
                continue
            self.download(record, local)
            staged[record.name] = local
            if progress:
                progress(f"fetch  {record.name} ({local.stat().st_size} bytes)")
        return staged

    # ---------------------------------------------------------------- helpers

    def _http_json(self, url: str) -> dict:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            raise AssetError(
                "Google Drive request failed.",
                status=exc.code,
                url=url.split("key=")[0] + "key=<redacted>",
                hint="Check GOOGLE_API_KEY and that the folder is shared "
                "anyone-with-the-link.",
            ) from exc
        except urllib.error.URLError as exc:
            raise AssetError(
                "Could not reach Google Drive.", url="drive api", cause=str(exc.reason)
            ) from exc

    def _http_download(self, url: str, destination: Path) -> None:
        request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                content_type = response.headers.get("Content-Type", "")
                payload = response.read()
        except urllib.error.HTTPError as exc:
            raise AssetError(
                "Google Drive download failed.",
                status=exc.code,
                url=url.split("id=")[0] + "id=<redacted>",
            ) from exc
        except urllib.error.URLError as exc:
            raise AssetError(
                "Could not reach Google Drive.", url="drive download",
                cause=str(exc.reason),
            ) from exc

        # Google answers large-file downloads with an HTML confirmation page.
        if b"<html" in payload[:2048].lower() and "text/html" in content_type:
            raise AssetError(
                "Drive returned an HTML page instead of the file.",
                hint="The file is probably too large for a direct download and "
                "needs authenticated access or a virus-scan confirmation. "
                "Provide GOOGLE_DRIVE_CREDENTIALS.",
            )
        destination.write_bytes(payload)


def _http_status(exc: BaseException) -> int | None:
    """Best-effort HTTP status from a googleapiclient HttpError."""
    resp = getattr(exc, "resp", None)
    status = getattr(resp, "status", None)
    if isinstance(status, int):
        return status
    return None


def _is_auth_error(exc: BaseException) -> bool:
    try:
        from google.auth.exceptions import GoogleAuthError  # type: ignore

        return isinstance(exc, GoogleAuthError)
    except ImportError:
        return False


def _drive_media_body(source: Path, mime_type: str):
    """Build the upload payload without importing the Drive library eagerly."""
    try:
        from googleapiclient.http import MediaFileUpload  # type: ignore
    except ImportError as exc:
        raise AssetError(
            "Google Drive upload needs extra dependencies.",
            detail="pip install google-api-python-client google-auth google-auth-httplib2",
            cause=str(exc),
        ) from exc
    return MediaFileUpload(str(source), mimetype=mime_type, resumable=False)


def _next_version_name(name: str, taken: set[str]) -> str:
    """final.mp4 -> final_v2.mp4 -> final_v3.mp4 ... deterministic, never 'final (1).mp4'."""
    match = re.match(r"^(?P<stem>.+?)(?P<ext>\.[A-Za-z0-9]+)$", name)
    if not match:
        return name
    stem, ext = match.group("stem"), match.group("ext")
    index = 2
    while f"{stem}_v{index}{ext}" in taken:
        index += 1
    return f"{stem}_v{index}{ext}"


def _extract_folder_id(folder: str) -> str:
    """Accept a bare id, a /folders/<id> URL, or an ?id=<id> URL."""
    value = folder.strip()
    if FOLDER_ID_RE.match(value) and "/" not in value:
        return value
    match = re.search(r"/folders/([A-Za-z0-9_-]+)", value)
    if match:
        return match.group(1)
    match = re.search(r"[?&]id=([A-Za-z0-9_-]+)", value)
    if match:
        return match.group(1)
    raise AssetError(
        "Could not determine a Google Drive folder id.",
        value=folder,
        hint="Pass the bare folder id or the full folder URL.",
    )


# --------------------------------------------------------------------- factory


def build_storage(
    backend: str,
    *,
    root: Path,
    folder: str | None = None,
    destination: Path | None = None,
    output_folder: str | None = None,
    input_folder: str | None = None,
) -> Storage:
    """Create a storage backend from configuration.

    ``input_folder``/``output_folder`` allow separate source and destination
    folders. When only ``folder`` is given it is used for both, which supports
    a single folder holding inputs and rendered outputs.
    """
    backend = (backend or "local").lower()
    if backend == "local":
        # Honour an explicit upload destination; default to the project root.
        return LocalStorage(Path(destination) if destination else root)
    if backend in {"google_drive", "gdrive", "drive"}:
        source = input_folder or folder
        if not source:
            raise AssetError(
                "storage.backend is google_drive but no storage folder is set.",
                hint="Add `storage:` with `backend: google_drive` and either "
                "`input_folder_id:` (assets to read) or `folder:` for a single "
                "shared folder. Add `output_folder_id:` for where rendered "
                "videos are uploaded.",
            )
        credentials = os.environ.get("GOOGLE_DRIVE_CREDENTIALS")
        api_key = os.environ.get("GOOGLE_API_KEY")
        return GoogleDriveStorage(
            source,
            credentials_file=Path(credentials) if credentials else None,
            api_key=api_key,
            destination=destination or (root / "assets"),
            output_folder=output_folder,
        )
    raise AssetError(
        f"Unknown storage backend {backend!r}.",
        hint="Supported: local, google_drive",
    )


def auto_stage(
    project_root: Path,
    assets_dir: Path,
    *,
    exclude_names: set[str] | None = None,
) -> tuple[list[Path], list[str]]:
    """Best-effort local staging of whatever is present in the assets folder.

    ``exclude_names`` keeps produced outputs (for example a previously uploaded
    ``final.mp4`` living in the same Drive folder) from being pulled back in as
    input assets.
    """
    storage = LocalStorage(assets_dir)
    notes: list[str] = []
    staged: list[Path] = []
    excluded = {name.lower() for name in (exclude_names or set())}
    for record in storage.list_assets():
        if not record.is_media:
            continue
        if record.name.lower() in excluded:
            notes.append(f"skipped {record.name} (declared as an output file)")
            continue
        local = assets_dir / record.name
        if local.is_file():
            staged.append(local)
        else:
            storage.download(record, local)
            staged.append(local)
    if not staged:
        notes.append(f"No assets found in {assets_dir}")
    return staged, notes