"""The requirements an example declares, a script in its PEP 723 header and a notebook in its metadata, and which of them this environment lacks: what the example runner skips on."""

from __future__ import annotations

import json
import re
import tomllib
from dataclasses import dataclass
from functools import cache
from importlib.metadata import PackageNotFoundError, distribution, entry_points
from pathlib import Path
from typing import Any

NOTEBOOK_KEY = "scaly"
"""The notebook metadata key holding ``requires-python`` and ``dependencies``, as a script's header does."""

# The reference expression of PEP 723 for an inline metadata block.
_BLOCK = re.compile(r"(?m)^# /// (?P<type>[a-zA-Z0-9-]+)$\s(?P<content>(^#(| .*)$\s)+)^# ///$")
_NAME = re.compile(r"^\s*(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)\s*(\[(?P<extras>[^\]]*)\])?")


@dataclass(frozen=True)
class Requirements:
  """What an example needs: the Python versions it runs on and the distributions it imports."""

  requires_python: str | None
  dependencies: tuple[str, ...]


def script_requirements(text: str) -> Requirements | None:
  """The ``script`` block of a PEP 723 header in ``text``; ``None`` without one. Raises on two."""
  blocks = [m for m in _BLOCK.finditer(text) if m.group("type") == "script"]
  if len(blocks) > 1:
    raise ValueError("more than one '# /// script' block")
  if not blocks:
    return None
  toml = "".join(line[2:] if line.startswith("# ") else line[1:] for line in blocks[0].group("content").splitlines(keepends=True))
  data = tomllib.loads(toml)
  return Requirements(data.get("requires-python"), tuple(data.get("dependencies", ())))


def notebook_requirements(notebook: dict[str, Any]) -> Requirements | None:
  """The requirements in a notebook's metadata (under ``NOTEBOOK_KEY``); ``None`` without them."""
  data = notebook.get("metadata", {}).get(NOTEBOOK_KEY)
  if data is None:
    return None
  return Requirements(data.get("requires-python"), tuple(data.get("dependencies", ())))


def requirements(path: Path) -> Requirements | None:
  """The requirements ``path`` declares: a notebook's metadata, else a script's header."""
  if path.suffix == ".ipynb":
    return notebook_requirements(json.loads(path.read_text()))
  return script_requirements(path.read_text())


def parse(requirement: str) -> tuple[str, frozenset[str]]:
  """A requirement's normalized distribution name and its extras (``"scaly[experimental]"``)."""
  m = _NAME.match(requirement)
  if m is None:
    raise ValueError(f"not a requirement: {requirement!r}")
  extras = frozenset(e.strip() for e in (m.group("extras") or "").split(",") if e.strip())
  return _normalize(m.group("name")), extras


def _normalize(name: str) -> str:
  return re.sub(r"[-_.]+", "-", name).lower()


def _installed(name: str) -> bool:
  try:
    distribution(name)
  except PackageNotFoundError:
    return False
  return True


def _extra_names(name: str, extra: str) -> list[str]:
  """The distributions ``name[extra]`` adds; none for an extra the distribution does not define."""
  out = []
  for req in distribution(name).requires or ():
    body, _, marker = req.partition(";")
    if re.search(rf"""extra\s*==\s*["']{re.escape(extra)}["']""", marker):
      out.append(parse(body)[0])
  return out


def unmet(reqs: Requirements) -> list[str]:
  """Why each distribution ``reqs`` names, or one its extras add, is missing here; empty when all are installed."""
  missing = []
  for req in reqs.dependencies:
    name, extras = parse(req)
    if not _installed(name):
      missing.append(f"{name} is not installed")
      continue
    missing += [f"{dep} ({name}[{extra}]) is not installed" for extra in sorted(extras) for dep in _extra_names(name, extra) if not _installed(dep)]
  return missing


@cache
def _methods_by_distribution() -> dict[str, tuple[str, ...]]:
  out: dict[str, list[str]] = {}
  for ep in entry_points(group="scaly.methods"):
    if ep.dist is not None:
      out.setdefault(_normalize(ep.dist.name), []).append(ep.name)
  return {name: tuple(sorted(methods)) for name, methods in out.items()}


def methods(reqs: Requirements) -> list[str]:
  """The methods (``"opt.piqp"``) that the distributions ``reqs`` names beside ``scaly`` register,
  where installed: what a runner marks the example with, so it runs only where they load. The
  methods ``scaly`` itself brings are always there."""
  table = _methods_by_distribution()
  names = {parse(req)[0] for req in reqs.dependencies} - {"scaly"}
  return sorted({m for name in names for m in table.get(name, ())})


__all__ = ["NOTEBOOK_KEY", "Requirements", "methods", "notebook_requirements", "parse", "requirements", "script_requirements", "unmet"]
