import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))

from alloy.toolchain import solver_diagnostic, solver_loadable  # noqa: E402 -- needs the path above


def pytest_collection_modifyitems(items) -> None:
  """Make `@pytest.mark.solver("piqp")` both select and skip.

  CI splits the suite with `-m solver` / `-m "not solver"`, so the marker decides
  which job a test runs in; deriving the skip from the same marker means a test
  cannot land in the solver job while silently erroring in the core one."""
  for item in items:
    for mark in item.iter_markers(name="solver"):
      name = mark.args[0]
      if not solver_loadable(name):
        item.add_marker(pytest.mark.skip(reason=solver_diagnostic(name)))
