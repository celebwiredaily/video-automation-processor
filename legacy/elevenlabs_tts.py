"""
elevenlabs_tts.py
Reads a plain-text/markdown script file and generates a voiceover MP3.

Primary engine: ElevenLabs TTS API.
Fallback engine: Kokoro (local, no API key, no credits) -- triggered
automatically if ElevenLabs returns a quota/credit-exceeded error.

Assumes the ElevenLabs API key is in an env var named ELEVENLABS_API_KEY --
override with --api-key-env if it's set under a different name.

Usage:
    python elevenlabs_tts.py script.md voiceover.mp3
    python elevenlabs_tts.py script.md voiceover.mp3 --voice-id YOUR_VOICE_ID

    # Pre-flight check only -- generates a few seconds of test audio with
    # both engines (whichever are reachable) and exits. Used by
    # run_ffmpeg_pipeline.sh to fail fast before the rest of the pipeline
    # runs, rather than discovering a broken engine mid-render.
    python elevenlabs_tts.py --test-only

    # Force Kokoro even if ElevenLabs would work (e.g. you know you're
    # out of credits and don't want to wait on a failed API call first)
    python elevenlabs_tts.py script.md voiceover.mp3 --force-kokoro
"""
import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

import requests

ELEVENLABS_TTS_URL = "https://api.elevenlabs.io/v1/text-to-speech/{voice_id}"

DEFAULT_VOICE_ID = "VlUmeC1Uzj3NnwiVR9K9"  # your chosen ElevenLabs voice
DEFAULT_MODEL_ID = "eleven_multilingual_v2"
DEFAULT_KOKORO_VOICE = "am_fenrir"  # American English preset; swap for another Kokoro voice if preferred
DEFAULT_KOKORO_LANG = "a"          # 'a' = American English

TEST_SENTENCE = "This is a test of the text to speech engine."


# ---------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------

def clean_script_text(raw_text):
    """
    Strips basic markdown formatting from script.md so it isn't read aloud
    literally (e.g. "hashtag hashtag intro", "asterisk asterisk important
    asterisk asterisk"). Keeps this deliberately simple/conservative --
    only strips syntax, never rewrites wording.
    """
    text = raw_text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)   # markdown headers
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)                  # **bold**
    text = re.sub(r"\*(.+?)\*", r"\1", text)                      # *italic*
    text = re.sub(r"^[-*]\s+", "", text, flags=re.MULTILINE)      # bullet markers
    text = re.sub(r"\n{3,}", "\n\n", text)                        # collapse extra blank lines
    return text.strip()


class QuotaExceededError(Exception):
    """Raised specifically when ElevenLabs reports the account is out of
    credits/quota -- as opposed to any other failure (bad key, network
    error, bad request, etc.), which should surface as a real error
    rather than silently triggering a fallback."""
    pass


# ---------------------------------------------------------------------
# ElevenLabs engine
# ---------------------------------------------------------------------

def generate_elevenlabs(text, output_path, voice_id, model_id, api_key,
                         stability=0.5, similarity_boost=0.75):
    url = ELEVENLABS_TTS_URL.format(voice_id=voice_id)
    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }
    payload = {
        "text": text,
        "model_id": model_id,
        "voice_settings": {
            "stability": stability,
            "similarity_boost": similarity_boost,
        },
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=120)

    if not resp.ok:
        # ElevenLabs signals quota/credit exhaustion in a few different ways
        # depending on the endpoint/plan:
        #   - HTTP 402 Payment Required -- the actual status seen in practice
        #     for the TTS endpoint when credits run out
        #   - HTTP 401 with a JSON body like
        #     {"detail": {"status": "quota_exceeded", "message": "..."}}
        #   - HTTP 429 (rate/quota limiting)
        # 402 and 429 are unambiguous on their own. For 401 specifically we
        # still check the body for "quota"/"credit" wording, since a 401 can
        # also just mean a bad/expired API key -- which should NOT silently
        # fall back (you'd want to know your key is wrong, not get quietly
        # switched to a different voice).
        is_quota_error = False
        if resp.status_code in (402, 429):
            is_quota_error = True
        elif resp.status_code == 401:
            try:
                body = resp.json()
                detail = body.get("detail", {})
                status = detail.get("status", "") if isinstance(detail, dict) else ""
                message = (detail.get("message", "") if isinstance(detail, dict) else str(detail)) or ""
                if "quota" in status.lower() or "quota" in message.lower() or "credit" in message.lower():
                    is_quota_error = True
            except ValueError:
                pass  # non-JSON error body -- treat as a normal error below

        if is_quota_error:
            raise QuotaExceededError(
                f"ElevenLabs quota/credits exceeded (HTTP {resp.status_code}): {resp.text[:300]}"
            )
        resp.raise_for_status()  # any other error -- raise normally, no fallback

    Path(output_path).write_bytes(resp.content)
    return output_path


# ---------------------------------------------------------------------
# Kokoro engine (fallback)
# ---------------------------------------------------------------------

def ensure_kokoro_installed():
    """Installs kokoro + soundfile if not already importable. Safe to call
    every run -- pip no-ops instantly if already satisfied."""
    try:
        import kokoro  # noqa: F401
        import soundfile  # noqa: F401
        return
    except ImportError:
        print("Kokoro not found -- installing (first run only, ~1-2 min)...", file=sys.stderr)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "kokoro", "soundfile", "--break-system-packages"],
            check=True,
        )


def generate_kokoro(text, output_path, voice=DEFAULT_KOKORO_VOICE, lang=DEFAULT_KOKORO_LANG):
    ensure_kokoro_installed()

    from kokoro import KPipeline
    import numpy as np
    import soundfile as sf

    pipeline = KPipeline(lang_code=lang)

    audio_chunks = []
    for _, _, audio in pipeline(text, voice=voice):
        audio_chunks.append(audio)

    if not audio_chunks:
        raise RuntimeError("Kokoro produced no audio output.")

    full_audio = np.concatenate(audio_chunks)

    out = Path(output_path)
    if out.suffix.lower() == ".wav":
        sf.write(str(out), full_audio, 24000)
    else:
        # Kokoro outputs raw PCM/wav -- write a temp wav, then let ffmpeg
        # transcode to whatever the caller actually asked for (mp3, etc.)
        # so the rest of the pipeline (which expects voiceover.mp3) doesn't
        # need to know or care which engine produced the audio.
        tmp_wav = out.with_suffix(".kokoro_tmp.wav")
        sf.write(str(tmp_wav), full_audio, 24000)
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(tmp_wav), "-codec:a", "libmp3lame", "-qscale:a", "2", str(out)],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        tmp_wav.unlink(missing_ok=True)

    return output_path


# ---------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------

def generate_voiceover(text, output_path, voice_id, model_id, api_key,
                        stability, similarity_boost, force_kokoro=False,
                        kokoro_voice=DEFAULT_KOKORO_VOICE, kokoro_lang=DEFAULT_KOKORO_LANG):
    if force_kokoro:
        print("Using Kokoro (--force-kokoro set).")
        generate_kokoro(text, output_path, voice=kokoro_voice, lang=kokoro_lang)
        return "kokoro"

    if not api_key:
        print("No ElevenLabs API key set -- using Kokoro directly.", file=sys.stderr)
        generate_kokoro(text, output_path, voice=kokoro_voice, lang=kokoro_lang)
        return "kokoro"

    try:
        generate_elevenlabs(text, output_path, voice_id, model_id, api_key,
                             stability=stability, similarity_boost=similarity_boost)
        return "elevenlabs"
    except QuotaExceededError as e:
        print(f"WARNING: {e}", file=sys.stderr)
        print("Falling back to Kokoro (local, no credits needed)...", file=sys.stderr)
        generate_kokoro(text, output_path, voice=kokoro_voice, lang=kokoro_lang)
        return "kokoro"


# ---------------------------------------------------------------------
# Pre-flight test mode
# ---------------------------------------------------------------------

def run_test(voice_id, model_id, api_key, kokoro_voice, kokoro_lang):
    """
    Generates a couple seconds of test audio with whichever engine(s) are
    reachable and reports pass/fail for each, without touching your real
    script or voiceover file. Exits non-zero if BOTH engines fail (if at
    least one works, the pipeline can still run -- the fallback logic in
    generate_voiceover will pick it automatically).
    """
    import tempfile

    results = {}

    # Test ElevenLabs (only if a key is present -- otherwise it's expected
    # to be unavailable and shouldn't count as a failure)
    if api_key:
        tmp_path = tempfile.mktemp(suffix=".mp3")
        try:
            generate_elevenlabs(TEST_SENTENCE, tmp_path, voice_id, model_id, api_key)
            size = Path(tmp_path).stat().st_size
            results["elevenlabs"] = ("OK", f"{size} bytes")
        except QuotaExceededError as e:
            results["elevenlabs"] = ("QUOTA_EXCEEDED", str(e)[:150])
        except Exception as e:
            results["elevenlabs"] = ("FAIL", str(e)[:150])
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        results["elevenlabs"] = ("SKIPPED", "no API key set")

    # Test Kokoro (always -- it's the fallback, needs to actually work)
    tmp_path = tempfile.mktemp(suffix=".mp3")
    try:
        generate_kokoro(TEST_SENTENCE, tmp_path, voice=kokoro_voice, lang=kokoro_lang)
        size = Path(tmp_path).stat().st_size
        results["kokoro"] = ("OK", f"{size} bytes")
    except Exception as e:
        results["kokoro"] = ("FAIL", str(e)[:150])
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    print("\n=== TTS Pre-flight Test ===")
    for engine, (status, detail) in results.items():
        print(f"  {engine:12s} {status:15s} {detail}")
    print("===========================\n")

    kokoro_ok = results.get("kokoro", ("FAIL",))[0] == "OK"
    elevenlabs_ok = results.get("elevenlabs", ("FAIL",))[0] == "OK"

    if not kokoro_ok and not elevenlabs_ok:
        print("ERROR: both engines failed -- pipeline cannot generate a voiceover.", file=sys.stderr)
        sys.exit(1)
    if not kokoro_ok:
        print("WARNING: Kokoro fallback is broken. If ElevenLabs runs out of "
              "credits mid-run, generation will fail instead of falling back.", file=sys.stderr)

    print("Pre-flight check passed.")
    sys.exit(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("script_path", nargs="?", help="Path to script.md")
    parser.add_argument("output_path", nargs="?", help="Path to write voiceover.mp3")
    parser.add_argument("--voice-id", default=DEFAULT_VOICE_ID,
                         help="ElevenLabs voice ID.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--api-key-env", default="ELEVENLABS_API_KEY",
                         help="Name of the env var holding your ElevenLabs API key")
    parser.add_argument("--stability", type=float, default=0.5)
    parser.add_argument("--similarity-boost", type=float, default=0.75)
    parser.add_argument("--kokoro-voice", default=DEFAULT_KOKORO_VOICE,
                         help="Kokoro preset voice to use for fallback (or --force-kokoro)")
    parser.add_argument("--kokoro-lang", default=DEFAULT_KOKORO_LANG,
                         help="Kokoro language code (default 'a' = American English)")
    parser.add_argument("--force-kokoro", action="store_true",
                         help="Skip ElevenLabs entirely and use Kokoro directly")
    parser.add_argument("--test-only", action="store_true",
                         help="Run a pre-flight test of both engines and exit (no script/output needed)")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env)

    if args.test_only:
        run_test(args.voice_id, args.model_id, api_key, args.kokoro_voice, args.kokoro_lang)
        return  # unreachable, run_test always exits

    if not args.script_path or not args.output_path:
        print("ERROR: script_path and output_path are required unless --test-only is set.", file=sys.stderr)
        sys.exit(1)

    raw_text = Path(args.script_path).read_text(encoding="utf-8")
    text = clean_script_text(raw_text)

    if not text:
        print(f"ERROR: {args.script_path} produced no text after cleanup.", file=sys.stderr)
        sys.exit(1)

    print(f"Generating voiceover ({len(text)} chars) -> {args.output_path}")
    engine_used = generate_voiceover(
        text, args.output_path, args.voice_id, args.model_id, api_key,
        stability=args.stability, similarity_boost=args.similarity_boost,
        force_kokoro=args.force_kokoro,
        kokoro_voice=args.kokoro_voice, kokoro_lang=args.kokoro_lang,
    )

    out = Path(args.output_path)
    if not out.exists() or out.stat().st_size == 0:
        print(f"ERROR: {args.output_path} was not created or is empty.", file=sys.stderr)
        sys.exit(1)

    print(f"Done ({engine_used}). Wrote {out.stat().st_size} bytes -> {args.output_path}")


if __name__ == "__main__":
    main()