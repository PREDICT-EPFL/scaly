"""The build hook of scaly's first-party distributions: each ships the slice of ``src/`` that
``distributions.toml`` gives it.

``scripts/distributions.py`` writes the slice into the manifest's hook table as ``paths`` (under
``src/``) and ``exclude`` (the paths inside those that another distribution ships), and copies this
file beside every manifest in ``packages/``, since an sdist has to carry its own hook. In the
repository the tree is ``src/`` beside the root manifest and two levels above one in ``packages/``.
An sdist carries its slice at ``src/`` beside its manifest, which is where this hook puts it, so a
wheel rebuilt from the sdist ships the same files. An editable install ships no files: the manifest's
``dev-mode-dirs`` puts the repository's ``src/`` on the path, for every distribution alike.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def slice_files(src: Path, paths: list[str], exclude: list[str]) -> list[str]:
  """The files under ``src`` that ``paths`` name, less ``exclude``, as paths relative to ``src``;
  bytecode, caches and hidden files are never shipped, and a path that does not exist raises."""
  out: list[str] = []
  for entry in paths:
    root = src / entry
    if root.is_file():
      found = [root]
    elif root.is_dir():
      found = sorted(path for path in root.rglob("*") if path.is_file())
    else:
      raise FileNotFoundError(f"distribution path {entry!r} is not under {src}")
    for path in found:
      rel = path.relative_to(src).as_posix()
      if any(part == "__pycache__" or part.startswith(".") for part in rel.split("/")) or path.suffix in (".pyc", ".pyo"):
        continue
      if not any(rel == other or rel.startswith(f"{other}/") for other in exclude):
        out.append(rel)
  return out


class SliceHook(BuildHookInterface):
  """Force-include the distribution's slice: at its import path in a wheel, under ``src/`` in an sdist."""

  PLUGIN_NAME = "custom"

  def initialize(self, version: str, build_data: dict[str, Any]) -> None:
    if version == "editable":
      return
    root = Path(self.root)
    src = root / "src" if (root / "src").is_dir() else root.parents[1] / "src"
    prefix = "src/" if self.target_name == "sdist" else ""
    for rel in slice_files(src, list(self.config["paths"]), list(self.config.get("exclude", []))):
      build_data["force_include"][str(src / rel)] = prefix + rel
