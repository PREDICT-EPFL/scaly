"""The pytest plugin of scaly's test suites: the ``method`` marker, which selects the tests that need a method and skips them where it is missing."""

from __future__ import annotations

import importlib
import os
from functools import cache

import pytest

REQUIRE = "SCALY_REQUIRE_METHODS"
"""The environment variable that turns a missing method into an error: CI's method job sets it to 1."""


@cache
def method_available(name: str) -> tuple[bool, str]:
  """Whether the method ``name`` (``"opt.piqp"``, ``"ocp.ilqr"``) can run here, and why not: its
  entry point is installed, and an external solver's vendored library loads."""
  domain, _, short = name.partition(".")
  if not short:
    return False, f"method names read domain.name, got {name!r}"
  try:
    methods = importlib.import_module(f"scaly.{domain}").REGISTRY
  except (ModuleNotFoundError, AttributeError):
    return False, f"no method domain scaly.{domain}"
  if short not in methods.installed():
    return False, f"method {name} is not installed"
  if domain == "opt":
    from scaly.opt.external.paths import solver_diagnostic, solver_loadable, solver_paths

    if short in solver_paths(required=False).loads and not solver_loadable(short):
      return False, solver_diagnostic(short)
  return True, ""


def pytest_configure(config: pytest.Config) -> None:
  config.addinivalue_line(
    "markers",
    "method(name): needs the method `name` (as 'opt.piqp'); skipped where it is missing, an error under SCALY_REQUIRE_METHODS=1, selected by CI's method job",
  )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
  """Make ``@pytest.mark.method("opt.piqp")`` both select and skip.

  CI splits the suite with ``-m method`` and ``-m "not method"``, so the marker decides which job a
  test runs in; deriving the skip from the same marker means a test cannot land in the method job
  while silently erroring in the core one. The method job sets ``SCALY_REQUIRE_METHODS=1``, where a
  skip is itself the failure: that job exists to run these tests, so quietly reporting success
  without them is the outcome to avoid."""
  require = os.environ.get(REQUIRE) == "1"
  missing: dict[str, str] = {}
  for item in items:
    for mark in item.iter_markers(name="method"):
      ok, reason = method_available(mark.args[0])
      if not ok:
        missing[mark.args[0]] = reason
        if not require:
          item.add_marker(pytest.mark.skip(reason=reason))
  if require and missing:
    raise pytest.UsageError(f"{REQUIRE}=1 but not available: {', '.join(sorted(missing))}\n\n" + "\n\n".join(missing.values()))


__all__ = ["REQUIRE", "method_available"]
