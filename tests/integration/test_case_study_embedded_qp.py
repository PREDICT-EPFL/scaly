"""The embedded-QP case study's path: a generated PIQP whose QP Hessian is a parameter.

`examples/case_studies/embedded_qp` solves qoco-benchmarks' oscillating-masses MPC with the state and
input weights `q`, `r` as parameters beside the initial state, so the generated solver builds `P` from
its parameters on every call (the path behind CS-12's dense workspace). `test_qp_solvers_example.py`
covers generated PIQP with the initial state as the only parameter; this checks the parametric Hessian:
the generated solver against the PIQP library, iteration for iteration, on two weightings.
"""

from __future__ import annotations

import runpy
import sys
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

import scaly as sc

HERE = Path(__file__).resolve().parents[2] / "examples" / "qp_solvers"
NM, T = 2, 3
NX, NU = 2 * NM, NM


def masses() -> sc.opt.NLP:
  band = -2 * np.eye(NM) + np.eye(NM, k=1) + np.eye(NM, k=-1)
  ac = np.block([[np.zeros((NM, NM)), np.eye(NM)], [band, np.zeros((NM, NM))]])
  a = np.eye(NX) + 0.25 * ac  # explicit Euler: the study's matrix exponential is not what is under test
  b = 0.25 * np.vstack([np.zeros((NM, NM)), np.eye(NM)])

  @sc.opt.problem(
    vars=sc.G(sc.L("u", (T, NU)), sc.L("x", (T + 1, NX))), params=sc.G(sc.L("q", NX), sc.L("r", NU), sc.L("x0", NX)), name="masses_param_hessian"
  )
  def problem(variables, params):
    u, x = variables
    q, r, x0 = params
    cost = (x * x * q.reshape((1, NX))).sum() + (u * u * r.reshape((1, NU))).sum()
    dyn = (x[1:] - x[:-1] @ sc.const(a.T) - u @ sc.const(b.T)).vec()
    x_bound = np.full((T + 1, NX), 0.6)
    x_bound[T] = np.inf
    return sc.opt.ProblemSpec(
      minimize=cost,
      eq=(x[0] - x0, dyn),
      lb=(sc.const(np.full((T, NU), -0.5)), sc.const(-x_bound)),
      ub=(sc.const(np.full((T, NU), 0.5)), sc.const(x_bound)),
    )

  return problem


@pytest.fixture(scope="module")
def generated_piqp() -> Iterator[dict]:
  sys.path.insert(0, str(HERE))
  try:
    yield runpy.run_path(str(HERE / "generated_piqp.py"))
  finally:
    sys.path.remove(str(HERE))


@pytest.mark.solver("piqp")
def test_generated_piqp_with_a_parametric_hessian_matches_the_library(generated_piqp: dict) -> None:
  p = masses()
  generated = generated_piqp["solver"](p, "sparse", name="test_masses_param_hessian_generated")
  library = sc.opt.solver(p, sc.opt.PIQP(sparse=True), name="test_masses_param_hessian_library")
  zeros = p.vars.unflatten(tuple(np.zeros(s) for s in p.vars.shapes))
  x0 = np.array([0.5, -0.4, 0.3, 0.2])
  solutions = []
  for q, r in ((np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.5, 0.1])), (np.array([9.0, 0.1, 0.1, 5.0]), np.array([3.0, 7.0]))):
    params = (q, r, x0)
    x, _, _, _, status, iters, obj = generated(params)
    out = library(zeros, zeros, np.zeros(p.n_eq), np.zeros(p.n_ineq), params)
    x_lib = np.concatenate([np.ravel(v) for v in p.vars.flatten_numerical(out[0], "x")])
    stats = sc.opt.solver_stats(library)
    assert int(status) == 1 and stats.status.name == "OK"
    assert int(iters) == stats.iter
    np.testing.assert_allclose(x, x_lib, atol=1e-8 * (1 + np.abs(x_lib).max()))
    u, xs = x_lib[: T * NU].reshape(T, NU), x_lib[T * NU :].reshape(T + 1, NX)
    assert abs(float(obj) - ((xs * xs * q).sum() + (u * u * r).sum())) < 1e-8 * (1 + abs(float(obj)))
    solutions.append(x)
  assert np.abs(solutions[0] - solutions[1]).max() > 1e-3  # the weights reach the solver
