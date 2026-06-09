"""Opt-in IR recording for the Alloy visualization server."""

from __future__ import annotations

import json
import os
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterator

from alloy.assembly import expr_graph, program_graph, render_expr_assembly, render_program_assembly
from alloy.program import PNode, format_program

if TYPE_CHECKING:
  from alloy.function import Function

_RECORDING_LOCK = threading.Lock()
_VIZ_TARGETS: dict[int, str | None] = {}
_RECORDINGS: list[dict[str, Any]] = []


def recording_dir() -> Path:
  override = os.environ.get("ALLOY_VIZ_DIR")
  if override:
    return Path(override).expanduser()
  xdg = os.environ.get("XDG_CACHE_HOME")
  base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
  return base / "alloy" / "viz"


def recording_path() -> Path:
  return recording_dir() / "recordings.json"


def visualize_function(fun: Function, *, label: str | None = None) -> Function:
  """Mark ``fun`` for visualization on future AOT/JIT renders.

  Capturing is exact-object opt-in: no Function is recorded unless it was passed
  here (or to ``capture``). The function itself is returned so callers can write
  ``f = visualize_function(f)``.
  """
  with _RECORDING_LOCK:
    _VIZ_TARGETS[id(fun)] = label
  return fun


def unvisualize_function(fun: Function) -> None:
  with _RECORDING_LOCK:
    _VIZ_TARGETS.pop(id(fun), None)


@contextmanager
def capture(fun: Function, *, label: str | None = None) -> Iterator[Function]:
  visualize_function(fun, label=label)
  try:
    yield fun
  finally:
    unvisualize_function(fun)


def is_visualized(fun: Function) -> bool:
  with _RECORDING_LOCK:
    return id(fun) in _VIZ_TARGETS


def clear_recordings(*, disk: bool = False) -> None:
  with _RECORDING_LOCK:
    _RECORDINGS.clear()
  if disk:
    recording_path().unlink(missing_ok=True)


def recordings() -> list[dict[str, Any]]:
  with _RECORDING_LOCK:
    return list(_RECORDINGS)


def load_recordings(path: str | os.PathLike[str] | None = None) -> list[dict[str, Any]]:
  p = Path(path) if path is not None else recording_path()
  if not p.exists():
    return []
  return json.loads(p.read_text())


def _append_disk(recording: dict[str, Any]) -> None:
  path = recording_path()
  path.parent.mkdir(parents=True, exist_ok=True)
  data = []
  if path.exists():
    try:
      data = json.loads(path.read_text())
    except json.JSONDecodeError:
      data = []
  data.append(recording)
  tmp = path.with_suffix(path.suffix + ".tmp")
  tmp.write_text(json.dumps(data, indent=2))
  tmp.replace(path)


@dataclass
class VisualizationRecording:
  fun: Function
  label: str | None = None
  recording_id: str = field(default_factory=lambda: uuid.uuid4().hex)
  started_at: float = field(default_factory=time.time)
  steps: list[dict[str, Any]] = field(default_factory=list)

  @property
  def display_name(self) -> str:
    return self.label or self.fun.name

  def add_semantic(self) -> None:
    self.steps.append(
      {
        "name": "semantic",
        "phase": "semantic dialect",
        "dialect": "sem",
        "assembly": render_expr_assembly(self.fun),
        "graph": expr_graph(self.fun),
      }
    )

  def add_program(self, name: str, root: PNode) -> None:
    self.steps.append(
      {
        "name": name,
        "phase": "program dialect",
        "dialect": "prog",
        "assembly": render_program_assembly(root),
        "listing": format_program(root),
        "graph": program_graph(root),
      }
    )

  def add_code(self, source: str) -> None:
    self.steps.append({"name": "generated C", "phase": "generated code", "dialect": "c", "code": source})

  def finish(self, *, error: str | None = None) -> None:
    recording = {
      "id": self.recording_id,
      "name": self.display_name,
      "function": self.fun.name,
      "started_at": self.started_at,
      "error": error,
      "steps": self.steps,
    }
    with _RECORDING_LOCK:
      _RECORDINGS.append(recording)
      _append_disk(recording)


def begin_recording(fun: Function) -> VisualizationRecording | None:
  with _RECORDING_LOCK:
    if id(fun) not in _VIZ_TARGETS:
      return None
    label = _VIZ_TARGETS[id(fun)]
  recording = VisualizationRecording(fun, label)
  recording.add_semantic()
  return recording


__all__ = [
  "VisualizationRecording",
  "begin_recording",
  "capture",
  "clear_recordings",
  "is_visualized",
  "load_recordings",
  "recording_dir",
  "recording_path",
  "recordings",
  "unvisualize_function",
  "visualize_function",
]
