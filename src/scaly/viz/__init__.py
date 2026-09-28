"""Opt-in observation of the compiler, and the browser for what it records.

Importing this package is what arms recording: ``recording.py`` registers into the observer hook
``codegen/aot.py`` owns, so the dependency runs backend-to-frontend and nothing in the compiler
imports ``viz`` (``docs/dev/codebase.md``).
"""

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
from .serve import serve

visualize = visualize_function


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
