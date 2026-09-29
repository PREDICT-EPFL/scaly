"""The public surface of ``scaly.testing``: each module's names, every callable one documented, and
the package kept out of the ``scaly`` namespace."""

from __future__ import annotations

import importlib

import pytest

import scaly as sc

MODULES = {
  "scaly.testing.plugin": ["REQUIRE", "method_available"],
  "scaly.testing.hyperdual": ["HyperDual", "lagrangian_hessian_np"],
  "scaly.testing.helpers": ["build_nlp", "build_qp", "solve_nlp", "solve_qp"],
  "scaly.testing.qp": [
    "INFINITE",
    "QP",
    "RawProblem",
    "infeasible_problems",
    "kkt_residuals",
    "make_qp",
    "maros_meszaros",
    "maros_meszaros_names",
    "mpc_qp",
    "random_qp",
    "raw",
  ],
  "scaly.testing.conformance.qp": [
    "SOLVABLE",
    "TOL",
    "UNSOLVABLE",
    "as_problem",
    "check_refuses",
    "check_solution",
    "check_solves",
    "solve",
    "stored",
    "violation",
  ],
  "scaly.testing.conformance.ocp": ["X0", "check", "problems", "reference", "solve"],
  "scaly.testing.conformance.roots": ["TOL", "check", "problems"],
}


@pytest.mark.parametrize("name", sorted(MODULES))
def test_each_modules_names_are_documented(name: str) -> None:
  module = importlib.import_module(name)
  assert module.__doc__, name
  assert sorted(module.__all__) == MODULES[name]
  for attr in module.__all__:
    value = getattr(module, attr)
    if callable(value):
      assert value.__doc__, f"{name}.{attr}"


def test_the_package_stays_out_of_the_scaly_namespace() -> None:
  assert "testing" not in sc.__all__
