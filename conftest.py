import importlib.abc
import os
import sys
from collections import defaultdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from scripts.distributions import nodeid_owner, owner_tests  # noqa: E402  (the root has to be on the path first)


class _BlockedImports(importlib.abc.MetaPathFinder):
  """``SCALY_BLOCK_IMPORTS=scaly.linalg pytest tests/ir ...`` runs tests as if the named packages
  were not installed: importing one, or anything under it, raises ``ModuleNotFoundError`` the way a
  missing distribution does. It is installed before anything imports ``scaly``."""

  def __init__(self, names: list[str]) -> None:
    self.names = names

  def find_spec(self, fullname, path, target=None):
    if any(fullname == name or fullname.startswith(f"{name}.") for name in self.names):
      raise ModuleNotFoundError(f"No module named {fullname!r} (SCALY_BLOCK_IMPORTS)", name=fullname)
    return None


if _blocked := [name for name in os.environ.get("SCALY_BLOCK_IMPORTS", "").split(",") if name]:
  sys.meta_path.insert(0, _BlockedImports(_blocked))

# One node-ID baseline per owner of tests: each distribution of distributions.toml, and the
# repository's own tests. A run that collects an owner's tests in full checks its baseline.
NODEID_BASELINES = ROOT / "tests" / "baseline"
_XDIST_NODEID_CHECKED = False


def nodeid_baseline(owner: str) -> Path:
  """The node-ID baseline of ``owner``: ``tests/baseline/<owner>_nodeids.txt``."""
  return NODEID_BASELINES / f"{owner}_nodeids.txt"


def pytest_addoption(parser: pytest.Parser) -> None:
  parser.addoption(
    "--write-nodeid-baselines",
    action="store_true",
    help="with --collect-only over the whole suite, write every owner's node-ID baseline from this collection instead of checking it",
  )


_SELECTING = ("lf", "failedfirst", "newfirst", "stepwise", "sw", "deselect", "ignore", "ignore_glob")


def _checked_owners(config) -> list[str]:
  """The owners whose tests this run collects in full: every one for the whole suite, and for the
  paths given the ones whose test paths all lie inside them. A selection (``-k``, ``-m``, ``--lf``,
  a node ID, ...) or a run from another directory collects no owner in full."""
  root = Path(config.rootpath).resolve()
  if Path.cwd().resolve() != root:
    return []
  if (
    getattr(config.option, "markexpr", "") or getattr(config.option, "keyword", "") or any(getattr(config.option, name, False) for name in _SELECTING)
  ):
    return []
  args = [arg for arg in config.invocation_params.args if not arg.startswith("-")]
  if any("::" in arg for arg in args):
    return []
  owners = owner_tests()
  given = [Path(arg).resolve() for arg in args if Path(arg).exists()]
  if not given:
    return sorted(owners)
  if not all(path.is_relative_to(root) for path in given):
    return []
  rels = [path.relative_to(root).as_posix() for path in given]

  def covered(test_path: str) -> bool:
    return any(rel == "." or test_path == rel or test_path.startswith(f"{rel}/") for rel in rels)

  return sorted(owner for owner, paths in owners.items() if paths and all(covered(path) for path in paths))


def _by_owner(nodeids) -> tuple[dict[str, list[str]], list[str]]:
  owned: dict[str, list[str]] = defaultdict(list)
  orphans = []
  for nodeid in nodeids:
    owner = nodeid_owner(nodeid)
    if owner is None:
      orphans.append(nodeid)
    else:
      owned[owner].append(nodeid)
  return owned, orphans


def _check_nodeid_baselines(config, nodeids) -> None:
  owners = _checked_owners(config)
  if not owners:
    return
  owned, orphans = _by_owner(nodeids)
  problems = []
  if orphans:
    problems.append(f"{len(orphans)} collected node IDs belong to no distribution in distributions.toml, such as {orphans[0]}")
  for owner in owners:
    path = nodeid_baseline(owner)
    if not path.is_file():
      problems.append(f"{owner}: the baseline {path.relative_to(ROOT)} is missing")
      continue
    expected = {line.strip() for line in path.read_text().splitlines() if line.strip()}
    actual = set(owned[owner])
    if actual != expected:
      problems.append(
        f"{owner}: {len(actual - expected)} collected node IDs are missing from {path.name} and {len(expected - actual)} in it are extra"
      )
  if problems:
    raise pytest.UsageError(
      "pytest node-ID baselines are stale:\n  "
      + "\n  ".join(problems)
      + "\nRegenerate them with the recipe in docs/dev/contributing.md (pytest --collect-only -q --write-nodeid-baselines)."
    )


def _write_nodeid_baselines(session) -> None:
  config = session.config
  if not getattr(config.option, "collectonly", False) or getattr(config.option, "numprocesses", None) not in (None, 0):
    raise pytest.UsageError("--write-nodeid-baselines needs --collect-only, without -n")
  if set(_checked_owners(config)) != set(owner_tests()):
    raise pytest.UsageError("--write-nodeid-baselines needs the whole suite: no selection, and no paths short of every owner's tests")
  if session.testsfailed:
    raise pytest.UsageError("--write-nodeid-baselines: the collection has errors; fix them first")
  owned, orphans = _by_owner(item.nodeid for item in session.items)
  if orphans:
    raise pytest.UsageError(f"{len(orphans)} collected node IDs belong to no distribution in distributions.toml, such as {orphans[0]}")
  for owner in owner_tests():
    nodeid_baseline(owner).write_text("".join(f"{nodeid}\n" for nodeid in sorted(owned[owner])))


def pytest_collection_finish(session) -> None:
  """Reject a stale node-ID baseline of every owner whose tests the run collects in full, or with
  ``--write-nodeid-baselines`` write them all."""
  config = session.config
  if getattr(config, "workerinput", None) is not None:
    return
  if config.getoption("write_nodeid_baselines"):
    _write_nodeid_baselines(session)
    return
  if getattr(config.option, "numprocesses", None) not in (None, 0) and not getattr(config.option, "collectonly", False):
    return
  _check_nodeid_baselines(config, (item.nodeid for item in session.items))


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids) -> None:
  """Check the full collection reported by xdist workers once."""
  global _XDIST_NODEID_CHECKED
  if _XDIST_NODEID_CHECKED or not ids:
    return
  _XDIST_NODEID_CHECKED = True
  _check_nodeid_baselines(node.config, ids)
