"""Tests for CI artifact staging.

These verify the staging behaviour and the workflow definitions. They do not
and cannot perform a real GitHub Actions upload: `actions/upload-artifact` is a
GitHub-hosted action and only runs on a real runner. What is asserted here is
that the staging gate refuses to prepare an artifact from a failed render, and
that the workflows are wired to upload exactly when they should.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from stage_artifacts import (  # noqa: E402
    ARTIFACT_NAME,
    FINAL_NAME,
    REPORT_NAME,
    VALIDATION_NAME,
    StageError,
    build_report,
    stage,
)

WORKFLOWS = sorted((ROOT / ".github" / "workflows").glob("*.yml"))


# --------------------------------------------------------------- fixtures


def _validation(ok: bool = True, **overrides) -> dict:
    data = {
        "ok": ok,
        "file_size": 41568272,
        "duration": 300.0,
        "width": 1280,
        "height": 720,
        "fps": 30.0,
        "video_codec": "h264",
        "pix_fmt": "yuv420p",
        "audio_codec": "aac",
        "audio_sample_rate": 48000,
        "audio_channels": 2,
        "audio_duration": 300.0,
        "audio_rms": 0.056,
        "frame_checks": [{"label": "start"}, {"label": "final"}],
        "black_frame_count": 0,
        "blank_frame_count": 0,
        "frozen_pairs": [],
        "caption_checks_passed": 463,
        "caption_checks_total": 463,
        "caption_false_positives": [],
        "final_frame_ok": True,
        "errors": [] if ok else ["final frame is not valid visual content"],
        "warnings": ["output audio rms=0.056"],
    }
    data.update(overrides)
    return data


def make_project(
    tmp_path: Path,
    *,
    validation: dict | None = None,
    with_video: bool = True,
    with_validation: bool = True,
    name: str = "demo",
) -> Path:
    project = tmp_path / name
    (project / "output" / "diagnostics" / "frames").mkdir(parents=True)
    (project / "project.yaml").write_text(f"name: {name}\n", encoding="utf-8")

    if with_video:
        (project / "output" / FINAL_NAME).write_bytes(b"\x00" * 4096)
    if with_validation:
        (project / "output" / "diagnostics" / VALIDATION_NAME).write_text(
            json.dumps(validation if validation is not None else _validation()),
            encoding="utf-8",
        )
    return project


# ------------------------------------------------------- successful staging


def test_stage_collects_the_expected_files(tmp_path):
    project = make_project(tmp_path)
    (project / "output" / "diagnostics" / "frames" / "00000.050_start.jpg").write_bytes(
        b"jpeg"
    )
    (project / "output" / "diagnostics" / "contact_sheet.jpg").write_bytes(b"jpg")

    artifacts = tmp_path / "artifacts"
    report = stage(project, artifacts)

    assert report["result"] == "pass"
    assert (artifacts / FINAL_NAME).is_file()
    assert (artifacts / VALIDATION_NAME).is_file()
    assert (artifacts / REPORT_NAME).is_file()
    assert (artifacts / "contact_sheet.jpg").is_file()
    assert (artifacts / "frames").is_dir()

    staged = {p.name for p in artifacts.iterdir()}
    assert {FINAL_NAME, VALIDATION_NAME, REPORT_NAME} <= staged


def test_render_report_summarises_the_validation(tmp_path):
    project = make_project(tmp_path)
    artifacts = tmp_path / "artifacts"
    stage(project, artifacts)

    report = json.loads((artifacts / REPORT_NAME).read_text(encoding="utf-8"))
    assert report["artifact_name"] == ARTIFACT_NAME
    assert report["project"] == "demo"
    assert report["result"] == "pass"
    assert report["output"]["duration_seconds"] == 300.0
    assert report["output"]["width"] == 1280
    assert report["output"]["fps"] == 30.0
    assert report["validation"]["captions_confirmed"] == 463
    assert report["validation"]["frames_sampled"] == 2
    assert report["validation"]["final_frame_ok"] is True
    assert any("not committed to Git" in n for n in report["notes"])


def test_staged_video_is_byte_identical_to_the_render(tmp_path):
    project = make_project(tmp_path)
    payload = b"\x11" * 9000
    (project / "output" / FINAL_NAME).write_bytes(payload)
    artifacts = tmp_path / "artifacts"
    stage(project, artifacts)
    assert (artifacts / FINAL_NAME).read_bytes() == payload


def test_stage_cleans_a_previous_staging_directory(tmp_path):
    project = make_project(tmp_path)
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "stale.mp4").write_bytes(b"old")
    stage(project, artifacts)
    assert not (artifacts / "stale.mp4").exists()


def test_frame_copies_are_capped(tmp_path):
    project = make_project(tmp_path)
    frames = project / "output" / "diagnostics" / "frames"
    for i in range(40):
        (frames / f"f{i:03d}.jpg").write_bytes(b"jpg")
    artifacts = tmp_path / "artifacts"
    stage(project, artifacts)
    assert len(list((artifacts / "frames").glob("*.jpg"))) == 24


def test_explicit_video_and_diagnostics_overrides(tmp_path):
    project = make_project(tmp_path, with_video=False)
    alt = tmp_path / "elsewhere"
    alt.mkdir()
    (alt / "movie.mp4").write_bytes(b"\x00" * 4096)
    alt_diag = tmp_path / "diag"
    alt_diag.mkdir()
    (alt_diag / VALIDATION_NAME).write_text(json.dumps(_validation()))

    artifacts = tmp_path / "artifacts"
    stage(project, artifacts, video=str(alt / "movie.mp4"), diagnostics=str(alt_diag))
    assert (artifacts / FINAL_NAME).is_file()


# ------------------------------------------------------------- the gate


def test_failed_validation_refuses_to_stage(tmp_path):
    project = make_project(tmp_path, validation=_validation(ok=False))
    artifacts = tmp_path / "artifacts"
    with pytest.raises(StageError) as exc:
        stage(project, artifacts)
    assert "validation did not pass" in str(exc.value)
    assert "final frame is not valid" in exc.value.context.get("errors", "")
    assert not artifacts.exists(), "no artifact directory may be created"


def test_failed_validation_can_be_staged_for_debugging(tmp_path):
    project = make_project(tmp_path, validation=_validation(ok=False))
    artifacts = tmp_path / "artifacts"
    report = stage(project, artifacts, allow_failed_validation=True)
    assert report["result"] == "fail"
    assert (artifacts / FINAL_NAME).is_file()


def test_missing_video_refuses_to_stage(tmp_path):
    project = make_project(tmp_path, with_video=False)
    with pytest.raises(StageError) as exc:
        stage(project, tmp_path / "artifacts")
    assert "rendered video not found" in str(exc.value)


def test_missing_validation_refuses_to_stage(tmp_path):
    """Uploading an unvalidated render would defeat the gate."""
    project = make_project(tmp_path, with_validation=False)
    artifacts = tmp_path / "artifacts"
    with pytest.raises(StageError) as exc:
        stage(project, artifacts)
    assert "validation.json not found" in str(exc.value)
    assert not artifacts.exists()


def test_truncated_video_refuses_to_stage(tmp_path):
    project = make_project(tmp_path)
    (project / "output" / FINAL_NAME).write_bytes(b"\x00" * 10)
    with pytest.raises(StageError) as exc:
        stage(project, tmp_path / "artifacts")
    assert "implausibly small" in str(exc.value)


def test_corrupt_validation_json_refuses_to_stage(tmp_path):
    project = make_project(tmp_path)
    # Full-size but unparseable: must be caught as bad JSON, not a short file.
    (project / "output" / "diagnostics" / VALIDATION_NAME).write_text(
        '{"ok": true, ' + " " * 400 + "not json"
    )
    with pytest.raises(StageError) as exc:
        stage(project, tmp_path / "artifacts")
    assert "not valid JSON" in str(exc.value)


def test_truncated_validation_json_refuses_to_stage(tmp_path):
    project = make_project(tmp_path)
    (project / "output" / "diagnostics" / VALIDATION_NAME).write_text("{nope")
    with pytest.raises(StageError) as exc:
        stage(project, tmp_path / "artifacts")
    assert "truncated" in str(exc.value)


def test_missing_project_directory_refuses_to_stage(tmp_path):
    with pytest.raises(StageError):
        stage(tmp_path / "nope", tmp_path / "artifacts")


def test_build_report_handles_a_minimal_validation_dict():
    report = build_report({}, project_name="x", project_dir=Path("/tmp"))
    assert report["result"] == "fail"
    assert report["validation"]["frames_sampled"] == 0


# ------------------------------------------------------------- workflows


def test_workflows_exist():
    names = {p.name for p in WORKFLOWS}
    assert {"ci.yml", "render.yml", "render-upload.yml"} <= names


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_yaml_is_valid(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert "jobs" in data and data["jobs"]


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflows_declare_read_only_permissions(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert data.get("permissions") == {"contents": "read"}, (
        f"{path.name} must not request write permissions"
    )


@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflows_set_artifact_retention(path):
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    env = data.get("env") or {}
    assert env.get("ARTIFACT_RETENTION_DAYS"), "retention must be configurable"
    assert int(env["ARTIFACT_RETENTION_DAYS"]) >= 1


def _workflow(workflow: str) -> dict:
    return yaml.safe_load(
        (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    )


def _resolve(value, workflow: str):
    """Expand a `${{ env.NAME }}` reference using the workflow's env block."""
    if not isinstance(value, str) or "{{" not in value:
        return value
    env = _workflow(workflow).get("env") or {}
    for key, replacement in env.items():
        value = value.replace(f"${{{{ env.{key} }}}}", str(replacement))
    return value


def _steps(workflow: str) -> list[dict]:
    data = yaml.safe_load((ROOT / ".github" / "workflows" / workflow).read_text())
    steps: list[dict] = []
    for job in data["jobs"].values():
        steps.extend(job.get("steps") or [])
    return steps


@pytest.mark.parametrize("workflow", ["ci.yml", "render.yml", "render-upload.yml"])
def test_artifact_is_named_final_video(workflow):
    """The upload name must resolve to final-video, indirection included."""
    assert _workflow(workflow).get("env", {}).get("ARTIFACT_NAME") == "final-video"
    uploads = [
        s for s in _steps(workflow)
        if str(s.get("uses", "")).startswith("actions/upload-artifact")
        and _resolve(s.get("with", {}).get("name"), workflow) == "final-video"
    ]
    assert uploads, f"{workflow} must upload a final-video artifact"


@pytest.mark.parametrize("workflow", ["ci.yml", "render.yml", "render-upload.yml"])
def test_artifact_upload_uses_configured_retention(workflow):
    uploads = [
        s for s in _steps(workflow)
        if str(s.get("uses", "")).startswith("actions/upload-artifact")
        and _resolve(s.get("with", {}).get("name"), workflow) == "final-video"
    ]
    assert uploads
    for upload in uploads:
        assert _resolve(upload["with"].get("retention-days"), workflow) == "7"


@pytest.mark.parametrize("workflow", ["ci.yml", "render.yml", "render-upload.yml"])
def test_artifact_upload_runs_only_after_staging_and_validation(workflow):
    """No unconditional `if:` guard: failure must skip the upload."""
    steps = _steps(workflow)
    names = [s.get("name") or s.get("uses") or "" for s in steps]

    stage_index = next(
        i for i, n in enumerate(names)
        if n == "Stage artifact files"
    )
    upload_index = next(
        i for i, s in enumerate(steps)
        if str(s.get("uses", "")).startswith("actions/upload-artifact")
        and _resolve(s.get("with", {}).get("name"), workflow) == "final-video"
    )

    assert upload_index > stage_index, "upload must come after staging"

    upload = steps[upload_index]
    # No `if: always()` / `failure()` on the success-path upload.
    assert "if" not in upload, (
        "the final-video upload must not be forced to run on failure"
    )
    assert upload["with"].get("if-no-files-found") == "error"


@pytest.mark.parametrize("workflow", ["ci.yml", "render.yml"])
def test_render_and_validate_precede_staging(workflow):
    """The order must be render -> validate -> stage, never the reverse."""
    names = [s.get("name") or "" for s in _steps(workflow)]
    stage_at = next(i for i, n in enumerate(names) if n == "Stage artifact files")
    render_steps = [
        i for i, n in enumerate(names)
        if "render" in n.lower() and "artifact" not in n.lower()
    ]
    assert render_steps, f"{workflow} must have a render step"
    assert max(render_steps) < stage_at, (
        f"{workflow}: rendering must complete before artefacts are staged"
    )


@pytest.mark.parametrize("workflow", ["ci.yml", "render.yml"])
def test_normal_workflows_never_touch_google_drive(workflow):
    text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
    for forbidden in ("--upload", "video_pipeline upload", "GOOGLE_DRIVE_CREDENTIALS"):
        assert forbidden not in text, (
            f"{workflow} must not require Drive; found {forbidden!r}"
        )


def test_normal_workflows_do_not_install_drive_dependencies():
    for workflow in ("ci.yml", "render.yml"):
        text = (ROOT / ".github" / "workflows" / workflow).read_text(encoding="utf-8")
        assert "google-api-python-client" not in text
        assert "google-auth" not in text


def test_drive_workflow_treats_missing_credentials_as_skipped_not_failed():
    text = (
        ROOT / ".github" / "workflows" / "render-upload.yml"
    ).read_text(encoding="utf-8")
    assert "available=false" in text
    # The Drive step is gated on the credential check, and the artifact upload
    # comes before it so a missing secret still yields final-video.
    names = [s.get("name") or "" for s in _steps("render-upload.yml")]
    drive_at = names.index("Upload to Google Drive")
    artifact_at = names.index("Upload final-video artifact")
    assert artifact_at < drive_at


def test_no_credentials_are_committed():
    patterns = ("private_key", "BEGIN PRIVATE KEY", "ya29.", "client_secret")
    for path in WORKFLOWS:
        text = path.read_text(encoding="utf-8")
        for pattern in patterns:
            assert pattern not in text, f"{path.name} contains {pattern!r}"