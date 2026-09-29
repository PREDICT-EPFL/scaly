"""Ruiz equilibration as generated code against the NumPy reference of PIQP's."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.opt.ipm import QPValues, ruiz, scale
from tests.opt.ipm import reference as ref
from scaly.testing.qp import infeasible_problems, maros_meszaros, maros_meszaros_names, mpc_qp
from tests.opt.ipm.problems import ipm_inputs

NAMES = maros_meszaros_names()
ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")


def _function(qp, tag: str, scale_cost: bool):
  s, values = ipm_inputs(qp)
  syms = {k: sc.sym(f"{k}", np.shape(values[k])) for k in ORDER}
  v = QPValues.preprocess(s, **syms)
  scaling = ruiz(s, v, scale_cost=scale_cost)
  scaled = scale(s, v, scaling)
  sv = scaled.values
  outs = [scaling.delta, scaling.delta_b, scaling.c, sv.P, sv.c, scaled.x_b, sv.A, sv.b, sv.G, sv.h_l, sv.h_u, sv.x_l, sv.x_u]
  names = ["d", "db", "c", "P", "cv", "xb", "A", "b", "G", "hl", "hu", "xl", "xu"]
  fn = sc.Function.from_exprs(f"ruiz_{tag}_{int(scale_cost)}", [syms[k] for k in ORDER], outs, list(ORDER), names)
  return fn, values


@pytest.mark.parametrize("scale_cost", [False, True])
@pytest.mark.parametrize("name", NAMES)
def test_scalings_match_the_reference(name: str, scale_cost: bool) -> None:
  qp = maros_meszaros(name)
  fn, values = _function(qp, name, scale_cost)
  delta, delta_b, c, P, cv, xb, A, b, G, hl, hu, xl, xu = fn(tuple(values[k] for k in ORDER))
  r = ref.Solver(qp, ref.Settings(preconditioner_scale_cost=scale_cost))
  d = r.data
  np.testing.assert_allclose(delta, r.pre.delta, rtol=1e-13)
  np.testing.assert_allclose(delta_b, r.pre.delta_b, rtol=1e-13)
  np.testing.assert_allclose(c, r.pre.c, rtol=1e-13)
  np.testing.assert_allclose(cv, d.c, rtol=1e-12, atol=1e-300)
  np.testing.assert_allclose(xb, d.x_b_scaling, rtol=1e-13)
  s, _ = ipm_inputs(qp)
  # The scaled data, entry by entry: the matrices on their patterns, the finite bounds.
  np.testing.assert_allclose(P, d.P.toarray()[s.P_rows, s.P_cols], rtol=1e-12, atol=1e-300)
  np.testing.assert_allclose(A, d.A.toarray()[s.A_rows, s.A_cols], rtol=1e-12, atol=1e-300)
  np.testing.assert_allclose(G, d.G.toarray()[s.G_rows, s.G_cols], rtol=1e-12, atol=1e-300)
  np.testing.assert_allclose(b, d.b, rtol=1e-12, atol=1e-300)
  for got, want in ((hl, d.h_l), (hu, d.h_u)):
    finite = np.abs(want) < 1e29
    np.testing.assert_allclose(got[finite], want[finite], rtol=1e-12, atol=1e-300)
  np.testing.assert_allclose(xl, d.x_l[: d.n_x_l], rtol=1e-12, atol=1e-300)
  np.testing.assert_allclose(xu, d.x_u[: d.n_x_u], rtol=1e-12, atol=1e-300)


@pytest.mark.parametrize("name", sorted(infeasible_problems()))
def test_free_rows_and_absent_bounds(name: str) -> None:
  qp, _ = infeasible_problems()[name]
  fn, values = _function(qp, name, False)
  delta, delta_b, c, *_ = fn(tuple(values[k] for k in ORDER))
  r = ref.Solver(qp)
  np.testing.assert_allclose(delta, r.pre.delta, rtol=1e-13)
  np.testing.assert_allclose(delta_b, r.pre.delta_b, rtol=1e-13)


def test_mpc() -> None:
  qp = mpc_qp(12, 4, 20)
  fn, values = _function(qp, "mpc", False)
  delta, *_ = fn(tuple(values[k] for k in ORDER))
  np.testing.assert_allclose(delta, ref.Solver(qp).pre.delta, rtol=1e-13)
