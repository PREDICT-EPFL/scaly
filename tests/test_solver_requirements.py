"""Check that installed-wheel CI requires only the supported solver libraries."""

from types import SimpleNamespace

import pytest

import conftest


@pytest.mark.parametrize(
  "required,missing,fails", [("1", "ipopt", True), ("piqp,sqp", "ipopt", False), ("piqp,sqp", "piqp", True), ("piqp,sqp", "sqp", True)]
)
def test_required_solver_collection(required, missing, fails, monkeypatch):
  monkeypatch.setenv("SCALY_REQUIRE_SOLVERS", required)
  monkeypatch.setattr(conftest, "solver_loadable", lambda name: name != missing)
  monkeypatch.setattr(conftest, "solver_diagnostic", lambda *args: "missing library")
  skipped = []
  item = SimpleNamespace(iter_markers=lambda **kwargs: [SimpleNamespace(args=(missing,))], add_marker=skipped.append)
  if fails:
    with pytest.raises(pytest.UsageError, match=missing):
      conftest.pytest_collection_modifyitems([item])
  else:
    conftest.pytest_collection_modifyitems([item])
    assert len(skipped) == 1
