import importlib.abc
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


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

NODEID_BASELINE = Path(__file__).resolve().parent / "tests" / "baseline" / "pytest_nodeids.txt"
_XDIST_NODEID_CHECKED = False


def _is_full_suite(config) -> bool:
  if Path.cwd().resolve() != Path(config.rootpath).resolve():
    return False
  if any(Path(arg.split("::", 1)[0]).exists() for arg in config.invocation_params.args if not arg.startswith("-")):
    return False
  if getattr(config.option, "markexpr", "") or getattr(config.option, "keyword", ""):
    return False
  return not any(
    getattr(config.option, name, False) for name in ("lf", "failedfirst", "newfirst", "stepwise", "sw", "deselect", "ignore", "ignore_glob")
  )


def _check_nodeid_baseline(config, nodeids) -> None:
  if not _is_full_suite(config):
    return
  if not NODEID_BASELINE.is_file():
    raise pytest.UsageError(f"pytest node-ID baseline is missing: {NODEID_BASELINE}")
  expected = sorted(line.strip() for line in NODEID_BASELINE.read_text().splitlines() if line.strip())
  actual = sorted(nodeids)
  if actual == expected:
    return
  missing = sorted(set(actual) - set(expected))
  extra = sorted(set(expected) - set(actual))
  raise pytest.UsageError(
    "pytest node-ID baseline is stale: "
    f"{len(missing)} collected node IDs are missing and {len(extra)} baseline IDs are extra. "
    "Regenerate it with the safe recipe in docs/dev/contributing.md; the initial collection "
    "may exit nonzero while this baseline is stale."
  )


def pytest_collection_finish(session) -> None:
  """Reject a stale full-suite pytest node-ID baseline."""
  config = session.config
  if getattr(config, "workerinput", None) is not None:
    return
  if getattr(config.option, "numprocesses", None) not in (None, 0) and not getattr(config.option, "collectonly", False):
    return
  _check_nodeid_baseline(config, (item.nodeid for item in session.items))


@pytest.hookimpl(optionalhook=True)
def pytest_xdist_node_collection_finished(node, ids) -> None:
  """Check the full collection reported by xdist workers once."""
  global _XDIST_NODEID_CHECKED
  if _XDIST_NODEID_CHECKED or not ids:
    return
  _XDIST_NODEID_CHECKED = True
  _check_nodeid_baseline(node.config, ids)


from scaly.opt.external.paths import solver_diagnostic, solver_loadable  # noqa: E402 -- needs the path above


def pytest_collection_modifyitems(items) -> None:
  """Make `@pytest.mark.solver("piqp")` both select and skip.

  CI splits the suite with `-m solver` / `-m "not solver"`, so the marker decides which
  job a test runs in; deriving the skip from the same marker means a test cannot land in
  the solver job while silently erroring in the core one. The solver job sets
  SCALY_REQUIRE_SOLVERS=1, where a skip is itself the failure — that job exists to run
  these tests, so quietly reporting success without them is the outcome to avoid."""
  require = os.environ.get("SCALY_REQUIRE_SOLVERS") == "1"
  missing = set()
  for item in items:
    for mark in item.iter_markers(name="solver"):
      name = mark.args[0]
      if not solver_loadable(name):
        missing.add(name)
        if not require:
          item.add_marker(pytest.mark.skip(reason=solver_diagnostic(name)))
  if require and missing:
    raise pytest.UsageError(f"SCALY_REQUIRE_SOLVERS=1 but not loadable: {', '.join(sorted(missing))}\n\n{solver_diagnostic()}")
