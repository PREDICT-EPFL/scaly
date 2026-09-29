"""Run one distribution's tests against its wheel alone.

Builds the wheels of the distribution and of the first-party distributions it depends on, installs
them into a fresh environment with only their declared dependencies and the test tools, and runs the
tests ``distributions.toml`` gives it from there, so an import it does not declare fails:

    uv run scripts/isolation.py scaly-numerics
    uv run scripts/isolation.py scaly-piqp --against lowest   # the oldest scaly-numerics its range allows

A plugin runs ``--against head`` (the workspace's wheels, the default), ``lowest`` or ``highest`` (the
first-party distributions it depends on from the package index, oldest or newest in its range).
CI's isolation jobs run this once per distribution.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import tomllib
from collections.abc import Iterable, Sequence
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run as a file: `scripts` is a package of the root
from scripts import distributions as dist  # noqa: E402

ROOT = dist.ROOT
TEST_TOOLS = ("pytest", "pytest-xdist")
"""What a test run needs beside the distributions, at the dev group's specifiers."""


def closure(name: str, table: dist.Table) -> list[str]:
  """``name`` and the first-party distributions its declared dependencies reach, extras left out."""
  seen: list[str] = []
  stack = [name]
  while stack:
    current = stack.pop()
    if current in seen:
      continue
    seen.append(current)
    stack += [dep for dep in map(dist.requirement_name, table.distributions[current].dependencies) if dep in table.distributions]
  return sorted(seen)


def third_party_requirements(names: Iterable[str], table: dist.Table) -> list[str]:
  """The requirements from outside the table that ``names`` declare, with their specifiers."""
  return sorted({req for name in names for req in table.distributions[name].dependencies if dist.requirement_name(req) not in table.distributions})


def test_tools() -> list[str]:
  """The test tools at the specifiers of the root's dev group."""
  group = tomllib.loads((ROOT / "pyproject.toml").read_text())["dependency-groups"]["dev"]
  specs = {dist.requirement_name(req): req for req in group if isinstance(req, str)}
  return [specs.get(tool, tool) for tool in TEST_TOOLS]


def run(cmd: Sequence[str | Path], **kwargs) -> None:
  print("+", " ".join(map(str, cmd)), flush=True)
  subprocess.run(list(map(str, cmd)), check=True, cwd=kwargs.pop("cwd", ROOT), **kwargs)


def build_wheels(names: Iterable[str], out: Path) -> None:
  """Build the wheel of each distribution in ``names`` into ``out``."""
  for name in names:
    run(["uv", "build", "--package", name, "--wheel", "--out-dir", out, "--quiet"])


def make_env(where: Path, first_party: Sequence[str], wheels: Path, *, index: Sequence[str] = (), resolution: str = "highest") -> Path:
  """A fresh environment at ``where`` with ``index`` (third-party requirements, first-party ones
  resolved from the package index) installed first, then the first-party wheels in ``wheels`` by
  name, from those wheels alone; returns its Python."""
  run(["uv", "venv", where, "--python", f"{sys.version_info.major}.{sys.version_info.minor}", "--quiet"])
  python = where / "bin" / "python"
  env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
  if index:
    run(["uv", "pip", "install", "--python", python, "--resolution", resolution, "--prerelease", "allow", "--quiet", *index], env=env)
  if first_party:
    # Only the wheels: the package index has an older `scaly` of the same version, which must not stand in.
    run(["uv", "pip", "install", "--python", python, "--no-index", "--find-links", wheels, "--quiet", *first_party], env=env)
  return python


def run_tests(python: Path, tests: Sequence[str], pytest_args: Sequence[str] = (), *, require_methods: bool = False) -> int:
  """Run ``tests`` with ``python`` from the repository root, the node-ID check off (the collection is
  not the workspace's), every method a test names required to load when ``require_methods``, and
  return pytest's exit code."""
  env = {k: v for k, v in os.environ.items() if k not in ("VIRTUAL_ENV", "PYTHONPATH")}
  env["SCALY_NODEID_BASELINES"] = "off"
  if require_methods:
    env["SCALY_REQUIRE_METHODS"] = "1"
  cmd = [str(python), "-m", "pytest", "-p", "no:cacheprovider", "-q", *pytest_args, *tests]
  print("+", " ".join(cmd), flush=True)
  return subprocess.run(cmd, cwd=ROOT, env=env, check=False).returncode


def extras_closure(name: str, table: dist.Table) -> list[str]:
  """The first-party distributions ``name``'s extras reach: its tests may use what it declares as optional."""
  out: set[str] = set()
  for reqs in table.distributions[name].extras.values():
    for req in reqs:
      dep = dist.requirement_name(req)
      if dep in table.distributions and dep != name:
        out |= set(closure(dep, table))
  return sorted(out)


def ignored(name: str, table: dist.Table) -> list[str]:
  """The test paths inside ``name``'s that another owner lists more specifically, which its run leaves out."""
  own = table.distributions[name].tests
  others = [path for owner, paths in dist.owner_tests(table).items() if owner != name for path in paths]
  return sorted(path for path in others for mine in own if path != mine and path.startswith(f"{mine}/"))


def plan(name: str, against: str, table: dist.Table) -> tuple[list[str], list[str], list[str]]:
  """What to build here, what to install from the wheels, and what from the package index, for
  ``name``'s tests: the distribution with its extras, what it depends on, and scaly-testing (the
  pytest plugin)."""
  target = table.distributions[name]
  needed = sorted(set(closure(name, table)) | set(extras_closure(name, table)) | set(closure("scaly-testing", table)))
  optional = sorted({req for reqs in target.extras.values() for req in reqs if dist.requirement_name(req) not in table.distributions})
  index = [*third_party_requirements(needed, table), *optional, *test_tools()]
  if against == "head":
    return needed, needed, index
  if target.lockstep:
    raise SystemExit(f"--against {against} is for a solver plugin; {name} releases with the others")
  # The plugin from here; what it depends on from the index at the end of its range (scaly-testing
  # at the version that resolves beside it).
  first_party = [r for r in target.dependencies if dist.requirement_name(r) in table.distributions]
  return [name], [name], [*index, *first_party, "scaly-testing"]


def main(argv: list[str] | None = None) -> int:
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("distribution")
  parser.add_argument("--against", choices=("head", "lowest", "highest"), default="head")
  parser.add_argument("--keep", type=Path, help="build and install here, and leave it, instead of a temporary directory")
  parser.add_argument("pytest_args", nargs="*", help="passed to pytest, after --")
  args = parser.parse_args(argv)
  table = dist.load()
  if args.distribution not in table.distributions:
    parser.error(f"unknown distribution {args.distribution!r}; distributions.toml has {', '.join(sorted(table.distributions))}")
  build, first_party, index = plan(args.distribution, args.against, table)
  resolution = {"head": "highest", "lowest": "lowest-direct", "highest": "highest"}[args.against]
  with tempfile.TemporaryDirectory(prefix="scaly-isolation-") as tmp:
    work = args.keep or Path(tmp)
    work.mkdir(parents=True, exist_ok=True)
    build_wheels(build, work / "wheels")
    python = make_env(work / "env", first_party, work / "wheels", index=index, resolution=resolution)
    tests = table.distributions[args.distribution].tests
    skip = [f"--ignore={path}" for path in ignored(args.distribution, table)]
    return run_tests(python, tests, [*skip, *args.pytest_args])


if __name__ == "__main__":
  sys.exit(main())
