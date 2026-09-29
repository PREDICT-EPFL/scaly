"""The distributions built from this repository, read from ``distributions.toml``: which one owns a
source file, a module or a test, what each may import, and the manifests that declare them.

Run it to write the manifests from the table: ``packages/<name>/`` and ``meta/scaly/`` whole, and
the marked block at the top of the root ``pyproject.toml``. ``--check`` writes nothing and lists the
stale files instead, exiting nonzero when there is one.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
TABLE = ROOT / "distributions.toml"
REPOSITORY = "repository"
"""The owner of the tests that belong to no distribution (``[repository]`` in the table)."""

BEGIN = "# BEGIN generated from distributions.toml by scripts/distributions.py; edit the table and rerun it"
END = "# END generated from distributions.toml"
HEADER = "# Generated from distributions.toml by scripts/distributions.py; edit the table and rerun it.\n"


@dataclass(frozen=True)
class Distribution:
  """One distribution: a first-party one released at the table's version (``lockstep``), or a solver
  plugin versioned on its own, whose dependencies and import package come from its manifest."""

  name: str
  manifest: str
  lockstep: bool
  tests: tuple[str, ...]
  description: str = ""
  readme: str | None = None
  paths: tuple[str, ...] = ()
  dependencies: tuple[str, ...] = ()
  extras: dict[str, tuple[str, ...]] = field(default_factory=dict)
  packages: tuple[str, ...] = ()


@dataclass(frozen=True)
class Table:
  """``distributions.toml``, read."""

  version: str
  common: dict[str, Any]
  distributions: dict[str, Distribution]
  repository_tests: tuple[str, ...]
  entry_points: dict[str, dict[str, str]]
  scripts: dict[str, str]

  @property
  def lockstep(self) -> list[Distribution]:
    return [d for d in self.distributions.values() if d.lockstep]


def requirement_name(requirement: str) -> str:
  """The normalized distribution name a requirement names: ``"scaly[piqp]>=1"`` names ``scaly``."""
  match = re.match(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip())
  if match is None:
    raise ValueError(f"not a requirement: {requirement!r}")
  return re.sub(r"[-_.]+", "-", match.group(0)).lower()


def _plugin(name: str, entry: dict[str, Any]) -> Distribution:
  project = tomllib.loads((ROOT / entry["manifest"] / "pyproject.toml").read_text())
  wheel = project.get("tool", {}).get("hatch", {}).get("build", {}).get("targets", {}).get("wheel", {})
  return Distribution(
    name=name,
    manifest=entry["manifest"],
    lockstep=False,
    tests=tuple(entry["tests"]),
    dependencies=tuple(project["project"].get("dependencies", ())),
    extras={extra: tuple(reqs) for extra, reqs in project["project"].get("optional-dependencies", {}).items()},
    packages=tuple(Path(package).name for package in wheel.get("packages", ())),
  )


@cache
def load(path: Path = TABLE) -> Table:
  """The table at ``path``, read once."""
  raw = tomllib.loads(path.read_text())
  dists: dict[str, Distribution] = {}
  for name, entry in raw["distributions"].items():
    dists[name] = Distribution(
      name=name,
      manifest=entry.get("manifest", f"packages/{name}"),
      lockstep=True,
      tests=tuple(entry.get("tests", ())),
      description=entry["description"],
      readme=entry.get("readme"),
      paths=tuple(entry.get("paths", ())),
      dependencies=tuple(entry.get("dependencies", ())),
      extras={extra: tuple(reqs) for extra, reqs in entry.get("extras", {}).items()},
    )
  for name, entry in raw.get("plugins", {}).items():
    dists[name] = _plugin(name, entry)
  return Table(
    version=raw["version"],
    common=raw["common"],
    distributions=dists,
    repository_tests=tuple(raw.get("repository", {}).get("tests", ())),
    entry_points=raw.get("entry-points", {}),
    scripts=raw.get("scripts", {}),
  )


def _covers(path: str, owned: str) -> bool:
  return path == owned or path.startswith(f"{owned}/")


def source_owner(rel: str, table: Table | None = None) -> str:
  """The distribution that ships ``rel``, a path under ``src/`` (``"scaly/ocp/altro.py"``): the one
  that lists it most specifically."""
  table = table or load()
  best = max(((len(p), d.name) for d in table.lockstep for p in d.paths if _covers(rel, p)), default=None)
  if best is None:
    raise LookupError(f"src/{rel} belongs to no distribution in distributions.toml")
  return best[1]


def module_path(module: str) -> str:
  """The file of ``module`` under ``src/``: ``scaly.opt.ipm`` is ``scaly/opt/ipm/__init__.py``."""
  base = module.replace(".", "/")
  for candidate in (f"{base}.py", f"{base}/__init__.py"):
    if (SRC / candidate).is_file():
      return candidate
  raise LookupError(f"no module {module} under src/")


def module_owner(module: str, table: Table | None = None) -> str:
  """The distribution that ships ``module``: a plugin by its import package, otherwise by its file."""
  table = table or load()
  top = module.split(".")[0]
  for d in table.distributions.values():
    if top in d.packages:
      return d.name
  return source_owner(module_path(module), table)


def owner_tests(table: Table | None = None) -> dict[str, tuple[str, ...]]:
  """Each owner's test paths: every distribution's, and the repository's own."""
  table = table or load()
  return {**{d.name: d.tests for d in table.distributions.values()}, REPOSITORY: table.repository_tests}


def nodeid_owner(nodeid: str, table: Table | None = None) -> str | None:
  """The owner of a pytest node ID, the one that lists its file most specifically; ``None`` when no
  one does."""
  path = nodeid.split("::", 1)[0]
  best = max(((len(p), owner) for owner, paths in owner_tests(table).items() for p in paths if _covers(path, p)), default=None)
  return None if best is None else best[1]


def reachable(name: str, table: Table | None = None) -> set[str]:
  """The distributions whose modules a module of ``name`` may import: itself, what it depends on and
  what its extras add, and what those depend on in turn."""
  table = table or load()
  first = table.distributions[name]
  seen = {name}
  stack = [requirement_name(r) for r in (*first.dependencies, *(r for reqs in first.extras.values() for r in reqs))]
  while stack:
    dep = stack.pop()
    if dep in seen or dep not in table.distributions:
      continue
    seen.add(dep)
    stack += [requirement_name(r) for r in table.distributions[dep].dependencies]
  return seen


def third_party(names: set[str], table: Table | None = None) -> set[str]:
  """The distributions from outside this table that ``names`` depend on, directly or by their extras."""
  table = table or load()
  out = set()
  for name in names:
    d = table.distributions[name]
    for requirement in (*d.dependencies, *(r for reqs in d.extras.values() for r in reqs)):
      if requirement_name(requirement) not in table.distributions:
        out.add(requirement_name(requirement))
  return out


def slice_of(name: str, table: Table | None = None) -> tuple[list[str], list[str]]:
  """What the build hook of ``name`` ships: its paths, and the paths inside them that another
  distribution lists more specifically."""
  table = table or load()
  paths = list(table.distributions[name].paths)
  exclude = sorted({p for d in table.lockstep if d.name != name for p in d.paths for own in paths if p != own and _covers(p, own)})
  return paths, exclude


def entry_points_of(name: str, table: Table | None = None) -> dict[str, dict[str, str]]:
  """The entry points ``name`` declares: those whose module it ships."""
  table = table or load()
  out: dict[str, dict[str, str]] = {}
  for group, entries in table.entry_points.items():
    for key, value in entries.items():
      if module_owner(value.split(":")[0], table) == name:
        out.setdefault(group, {})[key] = value
  return out


def scripts_of(name: str, table: Table | None = None) -> dict[str, str]:
  """The console scripts ``name`` declares: those whose module it ships."""
  table = table or load()
  return {key: value for key, value in table.scripts.items() if module_owner(value.split(":")[0], table) == name}


# --- manifests --------------------------------------------------------------------------------------


def _key(key: str) -> str:
  return key if re.fullmatch(r"[A-Za-z0-9_-]+", key) else json.dumps(key)


def _value(value: Any) -> str:
  if isinstance(value, bool):
    return "true" if value else "false"
  if isinstance(value, str):
    return json.dumps(value, ensure_ascii=False)
  if isinstance(value, dict):
    return "{ " + ", ".join(f"{_key(k)} = {_value(v)}" for k, v in value.items()) + " }"
  if isinstance(value, list | tuple):
    items = [_value(v) for v in value]
    inline = "[" + ", ".join(items) + "]"
    return inline if len(inline) <= 100 else "[\n" + "".join(f"  {item},\n" for item in items) + "]"
  raise TypeError(f"cannot write {value!r} as TOML")


def _table(header: str, body: dict[str, Any]) -> str:
  return "\n".join([f"[{header}]", *(f"{_key(k)} = {_value(v)}" for k, v in body.items())]) + "\n"


def _pin(requirement: str, owner: str, table: Table) -> str:
  """A bare first-party requirement pinned to the table's version; anything else as written."""
  name = requirement_name(requirement)
  bare = re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", requirement.strip()) is not None
  first_party = name in table.distributions and table.distributions[name].lockstep
  return f"{requirement}=={table.version}" if bare and first_party and name != owner else requirement


def render(name: str, table: Table | None = None) -> str:
  """The manifest of ``name``: its whole ``pyproject.toml``, or for the root the generated block."""
  table = table or load()
  d = table.distributions[name]
  common = table.common
  project: dict[str, Any] = {"name": name, "version": table.version, "description": d.description}
  if d.readme:
    project["readme"] = "README.md"
  project |= {
    "requires-python": common["requires-python"],
    "dependencies": [_pin(r, name, table) for r in d.dependencies],
    "license": common["license"],
    "license-files": ["LICENSE.md"],
    "authors": common["authors"],
    "keywords": common["keywords"],
    "classifiers": common["classifiers"],
  }
  sections = [_table("project", project), _table("project.urls", common["urls"])]
  if d.extras:
    sections.append(_table("project.optional-dependencies", {extra: [_pin(r, name, table) for r in reqs] for extra, reqs in d.extras.items()}))
  for group, entries in entry_points_of(name, table).items():
    sections.append(_table(f"project.entry-points.{_key(group)}", entries))
  if scripts := scripts_of(name, table):
    sections.append(_table("project.scripts", scripts))
  sections.append(_table("build-system", {"requires": ["hatchling"], "build-backend": "hatchling.build"}))
  wheel: dict[str, Any] = {"bypass-selection": True}
  if d.paths:
    wheel["dev-mode-dirs"] = [Path(os.path.relpath(SRC, ROOT / d.manifest)).as_posix()]
  sections.append(_table("tool.hatch.build.targets.wheel", wheel))
  if d.manifest == ".":
    # The root's sdist is its manifest, readme, license and hook, and the slice the hook adds.
    sections.append(_table("tool.hatch.build.targets.sdist", {"exclude": ["*"]}))
  if d.paths:
    paths, exclude = slice_of(name, table)
    sections.append(_table("tool.hatch.build.hooks.custom", {"paths": paths, **({"exclude": exclude} if exclude else {})}))
  sources = sorted(
    {requirement_name(r) for r in (*d.dependencies, *(r for reqs in d.extras.values() for r in reqs))} & set(table.distributions) - {name}
  )
  if sources and d.manifest != ".":
    sections.append(_table("tool.uv.sources", {source: {"workspace": True} for source in sources}))
  return "\n".join(sections)


def outputs(table: Table | None = None) -> dict[Path, str]:
  """Every file the table determines, with its content: each manifest, and beside the ones outside
  the root the copies an sdist needs (the license, the readme, the build hook)."""
  table = table or load()
  out: dict[Path, str] = {}
  for d in table.lockstep:
    text = render(d.name, table)
    if d.manifest == ".":
      current = (ROOT / "pyproject.toml").read_text()
      start, end = current.find(BEGIN), current.find(END)
      if start < 0 or end < start:
        raise ValueError(f"the root pyproject.toml has no {BEGIN!r} ... {END!r} block")
      out[ROOT / "pyproject.toml"] = f"{current[:start]}{BEGIN}\n{text}{END}{current[end + len(END) :]}"
      continue
    directory = ROOT / d.manifest
    out[directory / "pyproject.toml"] = HEADER + "\n" + text
    out[directory / "LICENSE.md"] = (ROOT / "LICENSE.md").read_text()
    if d.readme:
      out[directory / "README.md"] = (ROOT / d.readme).read_text()
    if d.paths:
      out[directory / "hatch_build.py"] = (ROOT / "hatch_build.py").read_text()
  return out


def stale(table: Table | None = None) -> list[Path]:
  """The generated files whose content differs from what the table says."""
  return [path for path, text in outputs(table).items() if not path.is_file() or path.read_text() != text]


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
  parser.add_argument("--check", action="store_true", help="write nothing; list the stale files and exit nonzero if there is one")
  args = parser.parse_args(argv)
  if args.check:
    paths = stale()
    for path in paths:
      print(f"stale: {path.relative_to(ROOT)}")
    return 1 if paths else 0
  for path, text in outputs().items():
    if not path.is_file() or path.read_text() != text:
      path.parent.mkdir(parents=True, exist_ok=True)
      path.write_text(text)
      print(f"wrote {path.relative_to(ROOT)}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
