"""Deterministic Python + FFmpeg pipeline for narrated YouTube videos.

Design rules:

* Python owns data, validation, planning and orchestration.
* FFmpeg owns every decode, transform and encode.
* No FFmpeg command is ever built as a shell string. Every invocation is a
  subprocess argument array.
* Supplied B-roll is treated as finished material. The pipeline organises and
  conforms it; it never invents cinematic effects.
"""

__version__ = "1.0.0"

__all__ = ["__version__"]