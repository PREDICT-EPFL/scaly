"""The KKT system as generated code against the reference's, for both backends."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.solvers.ipm import KKT, Iterate, QPValues, ruiz, scale
from tests.ipm import reference as ref
from tests.ipm.problems import infeasible_problems, ipm_inputs, maros_meszaros, mpc_qp, random_qp

ORDER = ("P", "c", "A", "b", "G", "h_l", "h_u", "x_l", "x_u")
PROBLEMS = ["HS21", "HS35MOD", "QAFIRO", "DUAL1", "CVXQP1_S", "QPCBLEND", "DPKLO1", "GENHS28", "QSC205", "PRIMALC1"]


def _random_state(d: ref.Data, rng: np.random.Generator) -> tuple[ref.Variables, ref.Variables]:
  """A positive iterate in the reference's layout, and a right-hand side."""
  n, p, m = d.n, d.p, d.m
  it, rhs = ref.Variables.zeros(n, p, m), ref.Variables.zeros(n, p, m)
  for v in (it, rhs):
    v.x, v.y = rng.standard_normal(n), rng.standard_normal(p)
  for v, scale_ in ((it, (0.5, 2.0)), (rhs, (-1.0, 1.0))):
    v.z_l[d.h_l_idx] = rng.uniform(*scale_, d.n_h_l)
    v.s_l[d.h_l_idx] = rng.uniform(*scale_, d.n_h_l)
    v.z_u[d.h_u_idx] = rng.uniform(*scale_, d.n_h_u)
    v.s_u[d.h_u_idx] = rng.uniform(*scale_, d.n_h_u)
    v.z_bl[: d.n_x_l] = rng.uniform(*scale_, d.n_x_l)
    v.s_bl[: d.n_x_l] = rng.uniform(*scale_, d.n_x_l)
    v.z_bu[: d.n_x_u] = rng.uniform(*scale_, d.n_x_u)
    v.s_bu[: d.n_x_u] = rng.uniform(*scale_, d.n_x_u)
  return it, rhs


def _compact(d: ref.Data, v: ref.Variables) -> list[np.ndarray]:
  nl, nu = d.n_x_l, d.n_x_u
  return [v.x, v.y, v.z_l, v.z_u, v.z_bl[:nl], v.z_bu[:nu], v.s_l, v.s_u, v.s_bl[:nl], v.s_bu[:nu]]


def _check(qp, backend: str, tag: str) -> None:
  s, values = ipm_inputs(qp)
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  v = QPValues.preprocess(s, **syms)
  q = scale(s, v, ruiz(s, v))
  kkt = KKT(s, q, backend, name=f"t_{tag}")
  sizes = Iterate.sizes(s)
  it_sym, rhs_sym = sc.sym("it", (sum(sizes),)), sc.sym("rhs", (sum(sizes),))
  rho, delta = sc.sym("rho", ()), sc.sym("delta", ())
  factor = kkt.factor(rho, delta, Iterate.unflat(s, it_sym))
  lhs = factor.solve(Iterate.unflat(s, rhs_sym))
  fn = sc.Function._from_exprs(
    f"kkt_{tag}_{backend}",
    [*(syms[k] for k in ORDER), it_sym, rhs_sym, rho, delta],
    [lhs.flat(), factor.ok],
    [*ORDER, "it", "rhs", "rho", "delta"],
    ["lhs", "ok"],
  )
  solver = ref.Solver(qp)
  d = solver.data
  rng = np.random.default_rng(0)
  for rho_v, delta_v in ((1e-6, 1e-4), (1e-3, 1e-2)):
    it, rhs = _random_state(d, rng)
    out = ref.Variables.zeros(d.n, d.p, d.m)
    assert solver.kkt.update_scalings_and_factor(d, solver.settings, False, rho_v, delta_v, it)
    solver.kkt.solve(d, solver.settings, rhs, out)
    got, ok = fn((*(values[k] for k in ORDER), np.concatenate(_compact(d, it)), np.concatenate(_compact(d, rhs)), np.array(rho_v), np.array(delta_v)))
    assert ok
    want = np.concatenate(_compact(d, out))
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9 * (1 + np.abs(want).max()))


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("name", PROBLEMS)
def test_solves_match_the_reference(name: str, backend: str) -> None:
  _check(maros_meszaros(name), backend, name)


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_free_rows_and_mpc(backend: str) -> None:
  qp, _ = infeasible_problems()["empty_slab"]
  _check(qp, backend, "slab")
  _check(mpc_qp(4, 2, 10), backend, "mpc")


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_two_sided_rows(backend: str) -> None:
  qp = random_qp(20, 15, 5, seed=0)
  assert np.any(np.isfinite(qp.h_l) & np.isfinite(qp.h_u))
  _check(qp, backend, "twosided")
