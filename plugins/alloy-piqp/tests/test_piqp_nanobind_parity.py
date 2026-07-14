"""Behavioral parity between the nanobind and retained ctypes PIQP wrappers."""

from __future__ import annotations

from pathlib import Path
from typing import TypedDict

import numpy as np
import pytest

from alloy.toolchain import solver_diagnostic, solver_loadable
from alloy_piqp import _piqp_ext
from alloy_piqp._piqp import PIQPDenseSolver
from alloy_piqp._piqp_ctypes import PIQPDenseSolver as CtypesPIQPDenseSolver

need_piqp = pytest.mark.skipif(not solver_loadable("piqp"), reason=solver_diagnostic("piqpc"))


class Problem(TypedDict):
  P: np.ndarray
  c: np.ndarray
  A_eq: np.ndarray | None
  b_eq: np.ndarray | None
  G_ineq: np.ndarray | None
  l_ineq: np.ndarray | None
  u_ineq: np.ndarray | None
  x_lb: np.ndarray | None
  x_ub: np.ndarray | None


def _problem(rng: np.random.Generator, n: int, p: int, m: int, box: bool) -> Problem:
  R = rng.standard_normal((n, n))
  P = R.T @ R + np.eye(n)
  feasible = rng.standard_normal(n)
  A = rng.standard_normal((p, n)) if p else None
  G = rng.standard_normal((m, n)) if m else None
  center = G @ feasible if G is not None else None
  return {
    "P": P,
    "c": rng.standard_normal(n),
    "A_eq": A,
    "b_eq": A @ feasible if A is not None else None,
    "G_ineq": G,
    "l_ineq": center - rng.uniform(0.5, 2.0, m) if center is not None else None,
    "u_ineq": center + rng.uniform(0.5, 2.0, m) if center is not None else None,
    "x_lb": feasible - rng.uniform(0.5, 2.0, n) if box else None,
    "x_ub": feasible + rng.uniform(0.5, 2.0, n) if box else None,
  }


def _assert_same(actual, expected) -> None:
  assert actual.status == expected.status
  assert actual.iter == expected.iter
  assert actual.primal_obj == expected.primal_obj
  # Both wrappers call the same library, so copied result buffers must be bitwise identical.
  for name in ("x", "lam_eq", "lam_ineq_l", "lam_ineq_u", "lam_box_l", "lam_box_u"):
    np.testing.assert_array_equal(getattr(actual, name), getattr(expected, name))


@need_piqp
@pytest.mark.parametrize(("n", "p", "m", "box"), [(3, 0, 0, False), (3, 2, 0, True), (3, 0, 4, False), (8, 2, 4, True), (8, 0, 4, True)])
def test_nanobind_matches_ctypes(n: int, p: int, m: int, box: bool) -> None:
  rng = np.random.default_rng(1000 + 100 * n + 10 * p + m + box)
  settings = {"eps_abs": 1e-10, "eps_rel": 1e-10}
  nanobind_solver, ctypes_solver = PIQPDenseSolver(n, p, m, settings=settings), CtypesPIQPDenseSolver(n, p, m, settings=settings)
  problem = _problem(rng, n, p, m, box)
  nanobind_solver.update(**problem)
  ctypes_solver.update(**problem)
  _assert_same(nanobind_solver.solve(), ctypes_solver.solve())

  problem["c"] = problem["c"] + rng.standard_normal(n) * 0.05
  if p and problem["b_eq"] is not None:
    problem["b_eq"] = problem["b_eq"] + rng.standard_normal(p) * 0.01
  if m and problem["l_ineq"] is not None and problem["u_ineq"] is not None:
    problem["l_ineq"] = problem["l_ineq"] - 0.01
    problem["u_ineq"] = problem["u_ineq"] + 0.01
  nanobind_solver.update(**problem)
  ctypes_solver.update(**problem)
  _assert_same(nanobind_solver.solve(), ctypes_solver.solve())


def test_extension_uses_stable_abi() -> None:
  assert ".abi3." in Path(_piqp_ext.__file__).name
