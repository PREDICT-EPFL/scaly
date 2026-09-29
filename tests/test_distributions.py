"""The distributions against ``distributions.toml``: the manifests are what it generates, every file
under ``src/scaly`` lands in exactly one wheel, a wheel rebuilt from its sdist ships the same files
and declares the same entry points, and the workspace installs every distribution as its manifest
declares it. The wheels are built here with hatchling, the build backend a release uses."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import tarfile
import textwrap
import zipfile
from importlib.metadata import distribution
from pathlib import Path

import numpy
import pytest
import scipy
from hatchling.builders.sdist import SdistBuilder
from hatchling.builders.wheel import WheelBuilder

import scaly as sc
from scaly.testing import examples
from scripts.distributions import ROOT, SRC, entry_points_of, load, owner_tests, scripts_of, source_owner, stale

TABLE = load()
LOCKSTEP = [d.name for d in TABLE.lockstep]


def _hook():
  spec = importlib.util.spec_from_file_location("scaly_slice_hook", ROOT / "hatch_build.py")
  assert spec is not None and spec.loader is not None
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  return module


def _tree() -> set[str]:
  """Every file under ``src/scaly`` a distribution could ship: what the build hook reads."""
  return set(_hook().slice_files(SRC, ["scaly"], []))


def _wheel(directory: Path, out: Path) -> Path:
  return Path(next(iter(WheelBuilder(str(directory)).build(directory=str(out), versions=["standard"]))))


def _sdist(directory: Path, out: Path) -> Path:
  return Path(next(iter(SdistBuilder(str(directory)).build(directory=str(out), versions=["standard"]))))


def _files(wheel: Path) -> set[str]:
  with zipfile.ZipFile(wheel) as archive:
    return {name for name in archive.namelist() if ".dist-info/" not in name}


def _entry_points(wheel: Path) -> dict[str, dict[str, str]]:
  """The wheel's ``entry_points.txt``: ``[group]`` headers, then ``name = value`` lines."""
  with zipfile.ZipFile(wheel) as archive:
    name = next((n for n in archive.namelist() if n.endswith(".dist-info/entry_points.txt")), None)
    text = archive.read(name).decode() if name is not None else ""
  out: dict[str, dict[str, str]] = {}
  group = ""
  for line in filter(None, (line.strip() for line in text.splitlines())):
    if line.startswith("["):
      group = line.strip("[]")
      out[group] = {}
    else:
      key, _, value = line.partition("=")
      out[group][key.strip()] = value.strip()
  return out


def _declared(name: str) -> dict[str, dict[str, str]]:
  out = dict(entry_points_of(name, TABLE))
  if scripts := scripts_of(name, TABLE):
    out["console_scripts"] = scripts
  return out


@pytest.fixture(scope="module")
def wheels(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
  out = tmp_path_factory.mktemp("wheels")
  return {name: _wheel(ROOT / TABLE.distributions[name].manifest, out) for name in LOCKSTEP}


def test_the_manifests_are_generated_from_the_table() -> None:
  paths = [str(path.relative_to(ROOT)) for path in stale(TABLE)]
  assert not paths, f"stale against distributions.toml, run `uv run scripts/distributions.py`: {paths}"


def test_every_source_file_is_in_exactly_one_wheel(wheels: dict[str, Path]) -> None:
  shipped: dict[str, list[str]] = {}
  for name, wheel in wheels.items():
    for path in _files(wheel):
      shipped.setdefault(path, []).append(name)
  twice = sorted(f"{path}: {names}" for path, names in shipped.items() if len(names) > 1)
  missing = sorted(_tree() - set(shipped))
  extra = sorted(set(shipped) - _tree())
  assert not twice and not missing and not extra, f"in two wheels: {twice}; in none: {missing}; not in src/: {extra}"
  wrong = sorted(
    f"{path} in {names[0]}, owned by {source_owner(path, TABLE)}" for path, names in shipped.items() if source_owner(path, TABLE) != names[0]
  )
  assert not wrong, f"a wheel ships a file distributions.toml gives another: {wrong}"


def test_each_wheel_declares_the_entry_points_of_its_own_modules(wheels: dict[str, Path]) -> None:
  for name, wheel in wheels.items():
    assert _entry_points(wheel) == _declared(name), name
  declared = {(group, key) for name in LOCKSTEP for group, entries in _declared(name).items() for key in entries}
  table = {(group, key) for group, entries in TABLE.entry_points.items() for key in entries} | {("console_scripts", key) for key in TABLE.scripts}
  assert declared == table, "every entry point and script of the table is declared by exactly the distribution shipping its module"


@pytest.mark.parametrize("name", LOCKSTEP)
def test_a_wheel_rebuilt_from_the_sdist_is_the_same(name: str, wheels: dict[str, Path], tmp_path: Path) -> None:
  sdist = _sdist(ROOT / TABLE.distributions[name].manifest, tmp_path / "sdist")
  with tarfile.open(sdist) as archive:
    archive.extractall(tmp_path / "unpacked", filter="data")
  (unpacked,) = (tmp_path / "unpacked").iterdir()
  rebuilt = _wheel(unpacked, tmp_path / "wheel")
  assert _files(rebuilt) == _files(wheels[name])
  assert _entry_points(rebuilt) == _entry_points(wheels[name])


@pytest.mark.parametrize("name", sorted(TABLE.distributions))
def test_the_workspace_installs_every_distribution(name: str) -> None:
  """``uv sync`` installs them all, through the root's dev group; a lockstep one with its manifest's
  entry points, which a stale install would not have."""
  installed = distribution(name)
  dist = TABLE.distributions[name]
  if dist.lockstep:
    assert installed.version == TABLE.version
    got: dict[str, dict[str, str]] = {}
    for ep in installed.entry_points:
      got.setdefault(ep.group, {})[ep.name] = ep.value
    assert got == _declared(name), f"{name} is installed with other entry points than its manifest declares; run `uv sync`"


def test_the_core_names_the_distribution_of_each_namespace() -> None:
  """``sc.<namespace>`` names the distribution to install when it is missing; ``scaly.testing``
  stays out of the ``scaly`` namespace."""
  namespaces = {path.split("/")[1] for d in TABLE.lockstep if d.name != "scaly-core" for path in d.paths} - {"testing"}
  assert sc._NAMESPACES == {name: source_owner(f"scaly/{name}/__init__.py", TABLE) for name in sorted(namespaces)}


def _installed(wheels: dict[str, Path], names: tuple[str, ...], code: str, tmp_path: Path) -> subprocess.CompletedProcess[str]:
  """Run ``code`` where only the wheels of ``names`` are installed, unpacked into one directory, and
  NumPy and SciPy: no site-packages, so neither the workspace's ``src/`` nor its metadata."""
  site = tmp_path / "site"
  for name in names:
    with zipfile.ZipFile(wheels[name]) as archive:
      archive.extractall(site)
  deps = tmp_path / "deps"
  deps.mkdir()
  for module in (numpy, scipy):
    installed = Path(module.__file__ or "").parents[1]
    for entry in (module.__name__, f"{module.__name__}.libs"):  # a Linux wheel's libraries sit beside it
      if (installed / entry).exists():
        (deps / entry).symlink_to(installed / entry)
  env = {**os.environ, "PYTHONPATH": os.pathsep.join([str(site), str(deps)])}
  prologue = f"import scaly as sc\nassert sc.__file__.startswith({str(site)!r}), sc.__file__\n"
  return subprocess.run([sys.executable, "-S", "-c", prologue + textwrap.dedent(code)], env=env, cwd=tmp_path, capture_output=True, text=True)


def test_an_install_of_core_and_numerics_names_scaly_control(wheels: dict[str, Path], tmp_path: Path) -> None:
  code = """
    from scaly import linalg as la
    assert sc.linalg is la
    try:
      sc.ocp
    except AttributeError as exc:
      assert "scaly.ocp needs scaly-control" in str(exc), exc
    else:
      raise AssertionError("sc.ocp without scaly-control")
  """
  proc = _installed(wheels, ("scaly-core", "scaly-numerics"), code, tmp_path)
  assert proc.returncode == 0, proc.stderr


def test_an_install_without_scaly_experimental_names_it_for_its_methods(wheels: dict[str, Path], tmp_path: Path) -> None:
  code = """
    assert sorted(sc.ocp.REGISTRY.installed()) == ["direct", "ilqr", "tinyadmm"], sc.ocp.REGISTRY.installed()
    for attr in ("ALTRO", "SCvx"):
      try:
        getattr(sc.ocp, attr)
      except AttributeError as exc:
        assert f"scaly.ocp.{attr} needs scaly-experimental" in str(exc), exc
      else:
        raise AssertionError(f"sc.ocp.{attr} without scaly-experimental")
    assert sc.ocp.ILQR is sc.ocp.REGISTRY.get("ilqr")
  """
  proc = _installed(wheels, ("scaly-core", "scaly-numerics", "scaly-control"), code, tmp_path)
  assert proc.returncode == 0, proc.stderr


def test_the_example_runner_knows_the_lockstep_distributions() -> None:
  """An example names the distributions it needs, and the runner marks it with the methods of those
  beside scaly's own, for CI's method job: every lockstep distribution is scaly's own."""
  assert examples.FIRST_PARTY == set(LOCKSTEP)
  plugins = sorted(d.name for d in TABLE.distributions.values() if not d.lockstep)
  requirements = examples.Requirements(None, (*LOCKSTEP, *plugins))
  expected = sorted(ep.name for name in plugins for ep in distribution(name).entry_points if ep.group == "scaly.methods")
  assert examples.methods(requirements) == expected


def test_every_test_owner_has_its_node_id_baseline() -> None:
  baselines = {path.name for path in (ROOT / "tests" / "baseline").glob("*_nodeids.txt")}
  assert baselines == {f"{owner}_nodeids.txt" for owner in owner_tests(TABLE)}
  unknown = sorted(path for paths in owner_tests(TABLE).values() for path in paths if not (ROOT / path).exists())
  assert not unknown, f"distributions.toml names tests that do not exist: {unknown}"
