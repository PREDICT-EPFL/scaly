from __future__ import annotations

from typing import Any

from ._recording import (
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
