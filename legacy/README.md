# Legacy scripts (not part of the renderer)

These files predate `video_pipeline/` and are kept only for reference. They are
**not imported, executed, or supported**. `video_pipeline/` is the canonical
implementation.

* `whisper_segments_builder.py` - superseded. Caption timing is now prepared
  outside the renderer (any tool may emit `captions/segments.json`), so the
  renderer stays decoupled from Whisper.
* `elevenlabs_tts.py` - still useful as an optional TTS step that produces
  `audio/voiceover.wav`. Note it installs `kokoro` on demand at runtime; for CI
  install dependencies up front instead.
* `render_video.py` - superseded. It built one enormous `filter_complex` with
  every clip as an input, which does not scale and produces unreadable errors.
  `video_pipeline` renders in three stages instead.
* `captions_builder.py` - superseded. Its style defaults (white, Anton, ALL
  CAPS) do not match the supplied reference video; the measured style lives in
  `video_pipeline/config.py`.
