"""Opt-in observation of the compiler, and the browser for what it records.

Importing this package is what arms recording: ``recording.py`` registers into the observer hook
``codegen/aot.py`` owns, so the dependency runs backend-to-frontend and nothing in the compiler
imports ``viz`` (``docs/dev/codebase.md``).
"""

from __future__ import annotations

from typing import Any

from .recording import (
  capture,
  clear_recordings,
  load_recordings,
  recording_dir,
  recording_path,
  recordings,
  unvisualize_function,
  visualize_function,
)

visualize = visualize_function


def serve(*args: Any, **kwargs: Any) -> None:
  from .serve import serve as _serve

  _serve(*args, **kwargs)


__all__ = [
  "capture",
  "clear_recordings",
  "load_recordings",
  "recording_dir",
  "recording_path",
  "recordings",
  "serve",
  "unvisualize_function",
  "visualize",
  "visualize_function",
]
