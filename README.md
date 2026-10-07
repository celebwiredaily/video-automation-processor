# video_pipeline

Deterministic Python + FFmpeg pipeline for narrated YouTube videos.

```
project data + voiceover + prepared B-roll + caption timing
                        ↓
                     FFmpeg
                        ↓
                  final.mp4 (1280x720, 30fps)
                        ↓
        real output validation (frames, captions, audio)
```

Editing decisions live in `timeline.yaml`. Python validates and plans; FFmpeg
decodes, conforms, concatenates, mixes and encodes. Supplied B-roll is treated
as **finished material** — the pipeline organises it and adds captions, it does
not add zooms, pans, transitions or other invented effects.

---

## Contents

1. [Install](#1-install)
2. [Project structure](#2-project-structure)
3. [Quick start](#3-quick-start)
4. [Timeline format](#4-timeline-format)
5. [Caption format](#5-caption-format)
6. [Configuration](#6-configuration)
7. [Short clip behaviour](#7-short-clip-behaviour)
8. [Duration: the voiceover decides](#8-duration-the-voiceover-decides)
9. [Validation](#9-validation)
10. [Diagnostics](#10-diagnostics)
11. [CLI](#11-cli)
12. [Google Drive](#12-google-drive)
12b. [Uploading to Google Drive](#12b-uploading-to-google-drive)
13. [TTS](#13-tts)
14. [Tests](#14-tests)
15. [GitHub Actions](#15-github-actions)
15b. [GitHub Actions Artifacts](#15b-github-actions-artifacts)
16. [The reference video](#16-the-reference-video)
17. [Troubleshooting](#17-troubleshooting)
18. [Security](#18-security)
19. [Legacy scripts](#19-legacy-scripts)

---

## 1. Install

Requires **Python 3.11+** and **FFmpeg with libass, libx264 and AAC**.

```bash
git clone <this repo>
cd video-automation-processor

python3 -m venv .venv && source .venv/bin/activate
pip install -e .
```

FFmpeg:

```bash
# Debian / Ubuntu (GitHub runners already include this)
sudo apt-get update && sudo apt-get install -y ffmpeg

# Verify - the pipeline needs libass for caption burning
ffmpeg -filters | grep -E ' (ass|subtitles) '
ffmpeg -encoders | grep libx264
```

macOS: `brew install ffmpeg`. Windows: `winget install Gyan.FFmpeg`.

Check everything at once:

```bash
python -m video_pipeline doctor
```

`doctor` reports Python and FFmpeg versions, required filters/encoders, caption
fonts, optional dependencies and Google Drive configuration, and exits non-zero
if anything mandatory is missing.

Optional extras:

```bash
pip install faster-whisper     # generate caption timings from a voiceover
pip install pytest             # run the test suite
pip install google-api-python-client google-auth   # authenticated Drive
```

---

## 2. Project structure

```
my-video/
├── project.yaml          # metadata, voiceover, captions, storage
├── timeline.yaml         # the visual timeline (the only place edits live)
├── config.yaml           # optional: output/caption/validation settings
├── script.md             # optional narration source
├── audio/
│   └── voiceover.wav     # the timing backbone
├── assets/
│   ├── broll_00.mp4
│   ├── still_01.jpg
│   └── …
├── captions/
│   └── segments.json     # prepared caption timing
├── output/
│   ├── final.mp4         # the local render; not committed to Git
│   └── diagnostics/      # frames, contact sheet, validation.json
└── build/                # intermediates, ffmpeg logs, plan.json
```

`project.yaml`, `timeline.yaml` and `config.yaml` may be combined into a single
`project.yaml`; separate files are recommended because they keep editing data
readable and diffable.

---

## 3. Quick start

```bash
# Environment check
python -m video_pipeline doctor

# Render a project
python -m video_pipeline render my-video/

# Validate the produced file against its actual frames
python -m video_pipeline validate my-video/output/final.mp4 --project my-video/

# See the resolved configuration and timeline before rendering
python -m video_pipeline inspect my-video/
```

A runnable ~5-minute example lives in `examples/sean-connery-5min`. Its assets
are generated, not committed:

```bash
python tools/build_example_project.py          # needs the reference video
python -m video_pipeline render examples/sean-connery-5min
```

---

## 4. Timeline format

The timeline is a list of contiguous intervals covering the whole video. All
times are seconds on the finished timeline.

```yaml
timeline:
  - start: 0.0
    end: 18.75
    asset: assets/broll_00.mp4
    fit: contain          # contain | cover | stretch

  - start: 18.75
    end: 37.5
    asset: assets/broll_01.mp4
    fit: contain
    source_start: 12.0   # begin 12s into the source
    label: interview two

  - start: 37.5
    end: 56.25
    asset: assets/still_00.jpg
    fit: contain          # fills the interval; stills have no duration limit
```

### Entry fields

| Field | Default | Meaning |
|---|---|---|
| `start` | required | Interval start on the finished timeline |
| `end` | required | Interval end; must be > `start` |
| `asset` | required | Path relative to the project directory |
| `fit` | `contain` | How to reach the output frame — see below |
| `source_start` | `0` | In-point within the source media |
| `source_end` | `end-start` after `source_start` | Out-point within the source |
| `audio` | `none` | `none` (drop source audio) or `clip` (keep it) |
| `on_short` | `error` | What to do if the source is too short |
| `label` | asset name | Free-text note, shown in `inspect` |

### `fit` modes

| Mode | Behaviour | Use for |
|---|---|---|
| `contain` | Scale to fit inside the frame, letterbox the remainder | **Default.** Preserves a finished composition and its existing framing |
| `cover` | Scale to fill the frame, crop the overflow | When cropping is preferred over bars |
| `stretch` | Force to exact dimensions | Avoid — distorts |

`contain` is the default deliberately. Finished B-roll often already carries
letterboxing or a deliberate crop; `cover` would destroy that composition.

Aspect ratio is never distorted by `contain` or `cover`.

### Rules enforced before rendering

* The first entry must start at `0` (otherwise the video opens on black).
* Entries may not overlap.
* Gaps are rejected — they would render as black.
* Every asset must exist inside the project directory, be probeable, and be
  long enough for its interval (see §7).
* Absolute paths and `..` traversal are refused.
* Unknown keys are reported rather than ignored.

Every problem is reported at once, so a broken timeline can be fixed in one
pass instead of one render at a time.

---

## 5. Caption format

Caption timing is **prepared data**. The renderer accepts it without knowing or
caring whether it came from Whisper, a human, or anything else. TTS and ASR stay
outside the renderer.

`captions/segments.json`:

```json
[
  { "start": 0.0, "end": 2.4, "text": "Connery is far more fascinating" },
  { "start": 2.4, "end": 5.1, "text": "than any of the actors who played him" }
]
```

SRT and WebVTT are also accepted directly:

```yaml
captions:
  file: captions/transcript.srt
```

### Caption styling

Style is configuration, not code. The renderer converts cues to ASS and burns
them with libass, which keeps all styling logic in testable Python and keeps
project paths out of filtergraph strings.

```yaml
captions:
  enabled: true
  font_name: Montserrat
  font_size: 46             # em size at 720p; scales with output height
  auto_scale_font_size: true
  bold: true
  italic: true
  primary_color: "#F7FF00"  # measured from the reference video
  outline_color: "#000000"
  outline: 4
  shadow: 0
  alignment: 2              # ASS: 1-3 bottom, 5-7 middle, 9-11 top
  margin_v: 44              # distance from the bottom edge
  margin_h: 80
  max_chars_per_line: 42
  max_lines: 2
  max_words_per_caption: 12
  min_caption_seconds: 0.30
  uppercase: false
```

### What the renderer does to your cues

* Sorts them by time and drops empty text.
* Splits long cues into on-screen chunks, distributing time by character weight
  so chunks stay contiguous and in sync with speech.
* Wraps lines to `max_chars_per_line`, balancing line lengths, capped at
  `max_lines`. Remaining text merges into the last line rather than being lost.
* Strips `{`, `}` and `\`, which would otherwise open ASS override blocks.
* Clamps cues to the video duration and separates overlaps.
* Applies `uppercase` if enabled.

### Generating captions with Whisper

```bash
pip install faster-whisper
python - <<'PY'
from faster_whisper import WhisperModel
model = WhisperModel("small", device="cpu", compute_type="int8")
segments, _ = model.transcribe("my-video/audio/voiceover.wav", language="en")
import json
cues = [{"start": round(s.start,3), "end": round(s.end,3), "text": s.text.strip()}
        for s in segments if s.text.strip()]
json.dump(cues, open("my-video/captions/segments.json","w"), indent=2)
PY
```

---

## 6. Configuration

`config.yaml` is optional and deep-merges over the inline `config:` block. Every
value below is a default; none is hard-coded in the renderer.

```yaml
output:
  filename: final.mp4
  existing_file: overwrite        # overwrite | version | error
  width: 1280
  height: 720
  fps: 30
  video_codec: libx264
  crf: 20                    # lower = better quality, bigger file
  preset: veryfast          # CPU-friendly
  pix_fmt: yuv420p           # required for broad playback / YouTube
  profile: high
  level: "4.0"
  gop_seconds: 2             # 2s keyframes help YouTube seeking
  audio_codec: aac
  audio_bitrate: 192k
  audio_rate: 48000
  audio_channels: 2

intermediate:
  crf: 14                    # near-lossless conform stage
  preset: veryfast

duration:
  target_seconds: 300        # ~5 minutes; advisory only
  tolerance_seconds: 0.25    # voiceover vs timeline
  min_seconds: 30
  max_seconds: 1800
  warn_target_deviation_ratio: 0.5
  mismatch: error            # error | warn

visual:
  default_fit: contain
  letterbox_color: black
  on_short: error            # error | freeze | loop
  allow_clip_audio: false

validation:
  require_voiceover: true
  duration_tolerance: 0.5
  black_mean_luma: 8.0
  black_max_luma: 24.0
  blank_stddev: 1.5
  blank_max_luma: 40.0
  caption_pixel_ratio: 0.0012
  frozen_diff: 0.002
  frozen_min_interval: 0.5
  silence_rms: 0.0015

paths:
  work_dir: null
  diagnostics: true

fonts:
  directory: null            # null = bundled fonts
```

CLI overrides: `render <project> --width 1920 --height 1080 --fps 30 --config overrides.json`.

---

## 7. Short clip behaviour

When a timeline interval is longer than its source media, behaviour is explicit
and configurable. **The default refuses**, so a frozen frame is never produced
silently.

| `on_short` | Behaviour |
|---|---|
| `error` *(default)* | Render fails, naming the asset, requested duration and available duration |
| `freeze` | Holds the last frame for the remainder |
| `loop` | Loops the source until the interval is filled |

```
ERROR: The timeline cannot be rendered as written.
  1 problem(s):
  [1] timeline[3] (clip004.mp4): clip004.mp4 has 2.400s available after
      source_start=0.000s but the timeline needs 8.000s (short by 5.600s).
      Set `on_short: freeze` or `on_short: loop` to handle this explicitly,
      or shorten the interval.
```

Per-entry override:

```yaml
  - start: 0.0
    end: 12.0
    asset: assets/short_interview.mp4
    on_short: loop
```

Stills (`.jpg`, `.png`, …) have no duration limit and fill any interval.

---

## 8. Duration: the voiceover decides

The voiceover is the timing backbone. `duration.target_seconds: 300` is
**advisory** — the pipeline never pads or trims audio to hit it.

* Voiceover present → its duration is the render length.
* Timeline shorter than the voiceover → error (`visual.on_short`), never a
  silently short video.
* Timeline longer than the voiceover → trimmed to the narration length, with a
  warning.
* No voiceover → timeline length is used, or `--duration`.

Deviation from the 5-minute target is reported as an informational note.

---

## 9. Validation

`validate` inspects the **rendered frames**, not just metadata. FFmpeg exit 0 and
ffprobe success are explicitly treated as insufficient.

### Technical (ffprobe)
container readable · video stream · audio stream · dimensions · frame rate ·
codecs · pixel format · duration against the voiceover (configurable
tolerance) · audio duration · audio not digitally silent (RMS over sampled
windows).

### Frame level (decoded pixels)
Frames are decoded and classified as **black** (very dark *and* flat) or
**blank** (flat *and* dark). Legitimate dark or flat-but-bright footage does not
trip these.

The sampling plan is deterministic and caption-aware:

* first frame, an early frame, midpoint, late frame
* the true final frame (read with `-sseof`)
* **both sides of every visual timeline boundary**
* **three points inside every caption interval** — start, midpoint, end
* a point in the gap after each caption

### Caption validation
For each sample the validator knows whether a caption *should* be there, then
looks for caption-coloured pixels **inside the caption band only** (the band is
derived from `alignment`, `margin_v` and `font_size`, so a non-bottom style is
checked correctly).

* every caption-active sample must contain the caption colour → **error** if not
* caption-coloured pixels where no caption was expected → **warning**

This is what makes "captions: 463/463 caption-active samples confirmed" a real
result rather than a hopeful one.

### Exit codes
`0` pass · `2` validation failure · `3` asset problem · `4` ffmpeg failure ·
`5` environment problem.

---

## 10. Diagnostics

Every render and every `validate` run writes to `output/diagnostics/`:

```
output/diagnostics/
├── validation.json      machine-readable report
├── ffprobe.json         raw stream metadata
├── contact_sheet.jpg    tiled sample frames
├── contact_frames.txt
└── frames/              individual sampled JPEGs
```

`validation.json` contains duration, size, dimensions, FPS, codecs, audio
details, every frame check, caption check totals, black/blank/frozen counts,
final-frame verdict, and separate `errors` / `warnings` lists.

ffmpeg logs from each stage are kept in `build/logs/` during a render.

---

## 11. CLI

```
python -m video_pipeline doctor [--drive-folder ID]
python -m video_pipeline fetch   <project>
python -m video_pipeline render  <project> [--width N --height N --fps N --duration S]
                                   [--upload]
python -m video_pipeline validate <video> [--project P] [--expected-duration S]
python -m video_pipeline upload  <project> [--video PATH]
python -m video_pipeline inspect <project>
python -m video_pipeline probe   <asset>
python -m video_pipeline prepare-test <dir> [--seconds 6]
```

```bash
python -m video_pipeline doctor
python -m video_pipeline fetch   my-video
python -m video_pipeline render  my-video
python -m video_pipeline validate my-video/output/final.mp4 --project my-video

# publish a validated render to Google Drive
python -m video_pipeline render  my-video --upload
python -m video_pipeline upload  my-video
```

---

## 12. Google Drive

Storage is an adapter. **The renderer only ever sees local file paths**; the
Drive URL is an input to staging, never a media path handed to FFmpeg.

```
Google Drive ─▶ GoogleDriveStorage ─▶ local assets/ ─▶ FFmpeg ─▶ final.mp4
local disk   ─▶ LocalStorage      ─▶ local assets/ ─▶ FFmpeg ─▶ final.mp4
```

### Project configuration

```yaml
storage:
  backend: google_drive       # local | google_drive
  folder: 1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO    # bare id or full URL
  destination: assets
```

### Staging

```bash
python -m video_pipeline fetch my-video
```

Downloads into `destination`, skips files already present at the right size,
prints each transfer, and fails loudly on an empty or truncated download.

### Public folder (no credentials)

A folder shared "anyone with the link" can be listed and downloaded with an API
key. The key is a public identifier, not a secret.

```bash
export GOOGLE_API_KEY=...        # from Google Cloud console
python -m video_pipeline doctor --drive-folder 1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO
python -m video_pipeline fetch my-video
```

> **Note on the folder used for the integration test.** The configured folder
> `1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO` is shared `anyoneWithLink` but currently
> **contains no files**, so nothing could be downloaded from it. The Drive
> adapter is implemented and tested (URL/id extraction, listing, download,
> metadata, credential handling); the 5-minute integration test therefore
> uses locally staged assets derived from the reference video. `fetch` reports
> an empty folder as a real result rather than a fake success.

### Private folder (authenticated)

Service-account credentials, referenced only by environment variable:

```bash
export GOOGLE_DRIVE_CREDENTIALS=/path/to/service-account.json
python -m video_pipeline fetch my-video
```

```bash
pip install google-api-python-client google-auth
```

In CI, set the secret as `GOOGLE_DRIVE_CREDENTIALS` and write it to a file in a
run step. Credentials are never written to project data or logs.

### Fully offline

With no Drive configuration at all, `backend: local` and
`python -m video_pipeline render my-video/` work with no network.

---

## 12b. Uploading to Google Drive

A validated render can be published to Drive in one step. **Upload happens only
after validation passes** — a failed render is never published.

```bash
# Local only: render + validate. No credentials, no network.
python -m video_pipeline render project/

# Render + validate + upload to Drive.
python -m video_pipeline render project/ --upload

# Upload an already-rendered, already-validated file.
python -m video_pipeline upload project/
python -m video_pipeline upload project/ --video other/final.mp4
```

### The order is guaranteed

```
fetch assets → render → validate → IF PASS → upload → verify
                                     └── else: no upload, diagnostics kept, exit non-zero
```

### Configuration

```yaml
storage:
  backend: google_drive
  input_folder_id:  "1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO"   # assets to read
  output_folder_id: "1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO"   # where videos go

output:
  filename: final.mp4
  existing_file: overwrite        # overwrite | version | error
```

`folder:` is accepted as a shorthand for "use this folder for both", which is
the normal single-folder setup. The public folder URL is never hardcoded — the
folder id is configuration, and the renderer only ever sees local paths.

### Output filename and duplicates

A Drive folder holding both inputs and outputs means a previously rendered
`final.mp4` sits next to the source media. Two things prevent that from causing
trouble:

* **Fetch ignores produced outputs.** `output.filename` (and `.mp4`/`.mov`
  variants) is excluded when staging inputs, so the previous render is never
  downloaded as B-roll.
* **Upload applies an explicit policy** — no `final (1).mp4` clutter.

| `output.existing_file` | Behaviour |
|---|---|
| `overwrite` *(default)* | Updates the existing file in place. Same name, one file, no duplicates |
| `version` | Uploads `final_v2.mp4`, `final_v3.mp4`, … deterministic and never Drive's `(1)` suffixes |
| `error` | Refuses to upload and exits non-zero |

### Authentication

Uploading **requires** authenticated access with write permission. Public
sharing supports reading shared files; it can never write, so an anonymous
upload is refused rather than silently faked.

```bash
# Local / server
export GOOGLE_DRIVE_CREDENTIALS=/secure/path/service-account.json

# In CI, set the repository secret GOOGLE_DRIVE_CREDENTIALS to the same JSON
```

```bash
pip install google-api-python-client google-auth
```

Requirements:

1. A service account (or OAuth client with Drive scope).
2. The **destination folder shared with the service account's `client_email`**
   and granted **Editor**. Without this you get:

```
[UPLOAD] FAILED: Google Drive upload failed: the authenticated account does not
         have write permission to the destination folder.
         folder_id: 1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO
         status: 403
         hint: Share the folder with the service account's client_email
         and give it Editor access, or point output_folder_id at a folder it can write.
```

### Upload verification

An upload is only reported as successful after the Drive API has confirmed it.
The pipeline:

1. calls `files().create()` (or `files().update()` for overwrite),
2. requires a returned **file id**,
3. **reads the file back** with `files().get()`,
4. checks the **name**, **size**, **`video/mp4` MIME type**, and that the file
   is in the destination folder.

Any mismatch raises and the command exits non-zero. There is no code path that
prints "uploaded successfully" without a confirmed API response.

### Output

```
[UPLOAD] final.mp4 (41568272 bytes)
[UPLOAD] backend google_drive
[DRIVE] uploaded successfully (confirmed by the google_drive backend)
[DRIVE] name       : final.mp4
[DRIVE] size       : 41568272 bytes
[DRIVE] mime type  : video/mp4
[DRIVE] destination: folder 1WHyz5VyisoUtBe_PKAYX3up0GvZno_cO (google_drive)
[DRIVE] replaced   : yes
[DRIVE] file id    : 1AbC...
[DRIVE] url        : https://drive.google.com/file/d/1AbC.../view
```

Retrieve it later with the file id:

```bash
gdown "https://drive.google.com/uc?id=<FILE_ID>"
# or
curl -L -o final.mp4 "https://drive.google.com/uc?export=download&id=<FILE_ID>"
```

### Exit codes

Non-zero on render failure, validation failure, authentication failure,
inaccessible destination, upload failure, or failed verification. The local file
and `output/diagnostics/` are always preserved on failure.

---

## 13. TTS

TTS is a separate stage that produces `audio/voiceover.wav`. The renderer never
calls it, so it is never mandatory.

```bash
python -m video_pipeline render my-video/
```

`legacy/elevenlabs_tts.py` remains available as an optional script→voiceover
step (ElevenLabs with a local Kokoro fallback). Install its dependencies up
front rather than letting it `pip install` mid-run, which is unreliable in CI.

---

## 14. Tests

```bash
pip install pytest
pytest                      # full suite (~4 minutes, CPU)
pytest -q tests/test_captions.py tests/test_config_timeline.py   # fast, no media
```

| File | Covers |
|---|---|
| `tests/test_config_timeline.py` | config defaults, merging, invalid config, overlap/gap/short-asset/traversal rejection, split timeline files |
| `tests/test_captions.py` | SRT/JSON parsing, wrapping, chunking, clamping, ASS generation, colour conversion, override-block escaping |
| `tests/test_storage.py` | LocalStorage, Drive URL/id handling, credential errors, secret leakage, renderer never sees a Drive URL |
| `tests/test_artifacts.py` | artifact staging: expected files, report contents, byte-identical video, frame cap, and the gate (failed/missing/truncated/corrupt validation refuses to stage). Plus workflow YAML validity, `contents: read` permissions, `final-video` naming, retention wiring, render-before-stage ordering, and that `ci.yml`/`render.yml` never touch Drive |
| `tests/test_upload.py` | successful upload + returned metadata, read-back verification, auth failure, permission failure, overwrite/version/error policies, validation gating the upload, output files excluded from inputs. Uses a fake Drive service; **no real credentials needed**. A real round-trip test runs only when `GOOGLE_DRIVE_CREDENTIALS` points at a real file |
| `tests/test_render_integration.py` | real MP4 render, format checks, **which clip is on screen at which time**, boundary placement, caption-interval sampling, audio not silent, `validation.json`, failure cases |
| `tests/test_media_types.py` | portrait, square, 24/60 fps, stills, no-audio clips, mixed media, `cover` vs `contain`, filenames with spaces/quotes/brackets/unicode |

Tests generate their own media, so no proprietary assets are required and CI
needs no fixtures. The integration tests decode frames and assert the dominant
colour per time window — a wrong clip, a late cut or a wrong duration fails even
though every exit code is 0.

---

## 15. GitHub Actions

`.github/workflows/ci.yml` — CPU only, no GPU, no Docker:

1. checkout
2. install Python 3.12
3. install FFmpeg (with libass) and the bundled fonts
4. `pip install -e .`
5. unit tests
6. render + integration tests (real MP4s, frame inspection)
7. deterministic render + validation
8. stage artifact files
9. upload the `final-video` artifact
10. upload diagnostics on failure

`.github/workflows/render.yml` — `workflow_dispatch` to render a named project
(validate only; no Google credentials needed).

`.github/workflows/render-upload.yml` — `workflow_dispatch` to render, validate
**and upload to Drive** using the `GOOGLE_DRIVE_CREDENTIALS` secret. Kept
separate so ordinary CI never needs credentials. The credential JSON is written
to a `0600` file outside the workspace, masked in the log, and deleted when the
job ends.

```yaml
# "Project" must be an existing directory under projects/.
# It is passed as an environment variable, never interpolated into a shell
# command, so it cannot inject anything.
```

Secrets come from repository or environment secrets and are never printed or
committed. `render-upload.yml` exits early with a clear message if
`GOOGLE_DRIVE_CREDENTIALS` is not configured, rather than pretending to upload.

Rendered MP4s are uploaded as artifacts. Note GitHub's artifact storage quota
(500 MB on Free, 1 GB on Pro) — set a short `retention-days`, or publish long
renders to your own storage instead.

---

## 15b. GitHub Actions Artifacts

When a CI render succeeds **and passes validation**, the run publishes a
GitHub Actions artifact named **`final-video`**.

### Contents

```
artifacts/
├── final.mp4              the rendered, validated video
├── validation.json        machine-readable validation report
├── render-report.json     compact summary of the run
├── contact_sheet.jpg      tiled sample frames (when produced)
└── frames/                individual sampled frames (capped at 24)
```

### Order

```
checkout → install → render → validation → stage → upload artifact
```

`actions/upload-artifact` only runs when every previous step succeeded, so a
failed render or a failed validation uploads **nothing**. Staging is enforced
twice over:

1. `tools/stage_artifacts.py` exits non-zero unless `validation.json` reports
   `ok: true`.
2. The workflow's upload step has no `if: always()`, so it is skipped on failure.

A separate `failure-diagnostics` artifact (diagnostics + FFmpeg logs) is still
uploaded on failure so a broken run can be investigated.

Validation is unchanged and remains the gate: it inspects decoded frames from
the rendered MP4 — not the renderer's exit code, not the FFmpeg exit code, not
mere file existence.

### MP4s are not committed to Git

Rendered videos are deliberately **not** in version control. `.gitignore`
excludes `*.mp4` and every project's `output/` directory, so the repository
stays text-only. The artifact is the copy CI keeps.

### Retention

Retention is configurable per workflow:

```yaml
env:
  ARTIFACT_RETENTION_DAYS: "7"
  ARTIFACT_NAME: final-video
```

Artifacts are deleted automatically after that period. For anything you intend
to keep long-term, publish to Google Drive instead (see below).

### Downloading an artifact

```
GitHub → the repository → Actions → select the workflow run
      → Artifacts (bottom of the run summary) → final-video → download
```

From the CLI, for a specific run:

```bash
gh run download <run-id> --name final-video
```

### Staging manually

The staging tool works outside CI, which is useful when reproducing a CI
failure locally:

```bash
python -m video_pipeline render project/
python tools/stage_artifacts.py project --artifact-dir artifacts
ls artifacts/
```

It refuses to stage a render whose validation did not pass:

```
[STAGE] FAILED: validation did not pass; refusing to stage artefacts
         errors: final frame is not valid visual content
```

### Google Drive remains optional

Google Drive upload is a **separate, optional capability** and is independent
of artifact storage:

| Workflow | Google credentials | Publishes |
|---|---|---|
| `ci.yml` | never needed | `final-video` artifact |
| `render.yml` | never needed | `final-video` artifact |
| `render-upload.yml` | optional; a missing secret **skips** the Drive step | `final-video` artifact, plus Drive when configured |

Neither `ci.yml` nor `render.yml` mentions Google Drive, installs its
dependencies, or passes `--upload`, so ordinary CI works without any cloud
account. In `render-upload.yml` a missing `GOOGLE_DRIVE_CREDENTIALS` secret is
reported as a notice and the Drive step is skipped — it does not fail the run,
and the artifact is still produced.

---

## 16. The reference video

`How Sean Connery Escaped the Curse of James Bond.mp4` is the visual ground
truth. Measured with ffprobe and by decoding frames:

| Property | Measured |
|---|---|
| Resolution / FPS | 1280x720, 30 |
| Duration | 581.5 s |
| Video | H.264 High, yuv420p |
| Audio | AAC, 44.1 kHz, stereo |
| Caption colour | `#F7FF00` (247, 255, 0) |
| Caption face | heavy **bold italic** |
| Caption outline | black, ~4 px |
| Caption position | bottom centre |
| Cap-height band | y 633–680 (baseline ≈667, bottom margin ≈39–44) |
| Lines | effectively single-line, ≈40–44 characters |
| Coverage | captions present ~93 % of runtime |

These measurements became the defaults in `video_pipeline/config.py`. They were
derived from the sample, not assumed — note they differ from the older
`legacy/captions_builder.py` defaults (white, Anton, ALL CAPS), which did not
match the supplied video.

---

## 17. Troubleshooting

**`ffmpeg was not found on PATH`** — install FFmpeg, or put a static build on
PATH. `doctor` reports which filters/encoders are missing.

**`The local ffmpeg build cannot run this pipeline`** — the `ass` filter means
libass is required. Ubuntu's `ffmpeg` package includes it; many minimal static
builds do not.

**`Timeline is shorter than the voiceover`** — extend the timeline, or set
`on_short: freeze|loop` on the affected entries.

**`Duration mismatch`** — the rendered duration differs from the voiceover beyond
`validation.duration_tolerance`. Usually a container/timing artefact; raise the
tolerance if the delta is small and consistent.

**`captions missing in N of M caption-active samples`** — inspect
`output/diagnostics/frames/`. Either the caption colour is far from the
configured `primary_color` (raise `caption_pixel_ratio` sensitivity carefully),
captions were disabled, or the ASS file was not produced (check
`build/captions.ass`).

**`output audio is effectively silent`** — the voiceover path is wrong, or
`voiceover:` points at a silent file. Check with
`ffmpeg -i audio/voiceover.wav -af volumedetect -f null -`.

**Caption position looks wrong at a new resolution** — `font_size` and
`margin_v` scale with height when `auto_scale_font_size` is true; verify the
generated `build/captions.ass`.

**Upload says "does not have write permission"** — share the destination folder
with the service account's `client_email` and grant Editor. A folder being
readable publicly does not make it writable.

**Upload says "requires authenticated access"** — public sharing supports reading
only. Set `GOOGLE_DRIVE_CREDENTIALS`.

**Drive folder "contains no files"** — that is reported as a real result. Public
listings only return items shared with the requesting account; use
`GOOGLE_DRIVE_CREDENTIALS` for an authenticated listing.

**Renders are slow** — `preset: veryfast` is already CPU-friendly. For drafts use
`--width 1280 --height 720 --config` with `crf: 28` and `preset: ultrafast`.

---

## 18. Security

* **No shell.** Every FFmpeg invocation is a `subprocess` argument array with
  `shell=False`. There is no code path that builds a shell command string.
* **Filenames are safe.** Spaces, quotes, parentheses, `&`, `;` and Unicode all
  work; there are tests for this.
* **Path traversal is refused.** Asset paths must resolve inside the project
  directory; absolute paths and `..` are rejected.
* **No project path reaches a filtergraph.** The caption ASS file and font
  directory use fixed names inside the work directory.
* **No secrets in the repo.** `.env`, service-account JSON, `token.pickle` and
  similar are gitignored. Drive credentials come from environment variables
  only, are never written to project data, and are never logged. Access tokens
  and API keys are redacted from error messages.
* **Inputs are treated as untrusted.** Malformed media is rejected by ffprobe
  during validation rather than fed blindly into a filtergraph.
* **CI hardening.** Workflows declare `permissions: contents: read`, use minimal
  `workflow_dispatch` inputs passed as environment variables, and pin
  third-party actions to commit SHAs.

---

## 19. Legacy scripts

`legacy/` holds pre-existing scripts kept for reference only. They are **not**
imported, executed or supported — `video_pipeline` is the canonical renderer.

* `whisper_segments_builder.py` — superseded; the renderer accepts prepared
  caption timing from any source.
* `elevenlabs_tts.py` — optional TTS stage, still useful.
* `render_video.py` — superseded; it built one enormous `filter_complex` with
  every clip as an input, which does not scale and yields unreadable errors.
* `captions_builder.py` — superseded; its style defaults did not match the
  supplied reference video.