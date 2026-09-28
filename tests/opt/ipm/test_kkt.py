"""The KKT system as generated code against the reference's, for both backends."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly.opt.ipm import KKT, Backend, Iterate, Kernels, QPStructure, QPValues, Refinement, ScaledQP, Scaling, ruiz, scale
from tests.opt.ipm import reference as ref
from tests.opt.ipm.problems import infeasible_problems, ipm_inputs, maros_meszaros, mpc_qp, random_qp

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


CASES = ((1e-6, 1e-4, False), (1e-3, 1e-2, False), (1e-6, 1e-4, True), (1e-9, 1e-9, True))


def _check(qp, backend: Backend, tag: str, *, static_eps: float = 1e-8, cases=CASES) -> None:
  s, values = ipm_inputs(qp)
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  v = QPValues.preprocess(s, **syms)
  q = scale(s, v, ruiz(s, v))
  kkt = KKT(Kernels(s, backend, Refinement(static_eps=static_eps), name=f"t_{tag}"), q)
  sizes = Iterate.sizes(s)
  it_sym, rhs_sym = sc.sym("it", (sum(sizes),)), sc.sym("rhs", (sum(sizes),))
  rho, delta, ir = sc.sym("rho", ()), sc.sym("delta", ()), sc.sym("ir", ())
  factor = kkt.factor(rho, delta, Iterate.unflat(s, it_sym), ir=ir)
  lhs = factor.solve(Iterate.unflat(s, rhs_sym))
  fn = sc.Function.from_exprs(
    f"kkt_{tag}_{backend}",
    [*(syms[k] for k in ORDER), it_sym, rhs_sym, rho, delta, ir],
    [lhs.flat(), factor.ok, factor.ir],
    [*ORDER, "it", "rhs", "rho", "delta", "ir"],
    ["lhs", "ok", "ir_out"],
  )
  solver = ref.Solver(qp, ref.Settings(iterative_refinement_static_regularization_eps=static_eps))
  d = solver.data
  rng = np.random.default_rng(0)
  for rho_v, delta_v, refine in cases:
    it, rhs = _random_state(d, rng)
    out = ref.Variables.zeros(d.n, d.p, d.m)
    assert solver.kkt.update_scalings_and_factor(d, solver.settings, refine, rho_v, delta_v, it)
    solver.kkt.solve(d, solver.settings, rhs, out)
    args = (np.concatenate(_compact(d, it)), np.concatenate(_compact(d, rhs)), np.array(rho_v), np.array(delta_v), np.array(float(refine)))
    got, ok, ir_out = fn((*(values[k] for k in ORDER), *args))
    assert ok and ir_out == float(refine)
    want = np.concatenate(_compact(d, out))
    np.testing.assert_allclose(got, want, rtol=1e-7, atol=1e-9 * (1 + np.abs(want).max()))


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("name", PROBLEMS)
def test_solves_match_the_reference(name: str, backend: Backend) -> None:
  _check(maros_meszaros(name), backend, name)


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_free_rows_and_mpc(backend: Backend) -> None:
  qp, _ = infeasible_problems()["empty_slab"]
  _check(qp, backend, "slab")
  _check(mpc_qp(4, 2, 10), backend, "mpc")


@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_two_sided_rows(backend: Backend) -> None:
  qp = random_qp(20, 15, 5, seed=0)
  assert np.any(np.isfinite(qp.h_l) & np.isfinite(qp.h_u))
  _check(qp, backend, "twosided")


@pytest.mark.parametrize("backend", ["dense", "sparse"])
@pytest.mark.parametrize("name", ["QAFIRO", "CVXQP1_S"])
def test_refinement_stops_when_it_slows(name: str, backend: Backend) -> None:
  """A static regularization ten times delta makes each refinement step gain only about 10%:
  PIQP keeps the first refined solution and stops, far from the converged one."""
  _check(maros_meszaros(name), backend, f"slow_{name}", static_eps=1e-3, cases=((1e-6, 1e-4, True),))


def _noise_pivot(p22: float, ir: float, retries: int = 10) -> dict[str, float]:
  """The dense factorization's retry loop on P = [[4, 2], [2, p22]] alone, rho = 0 and no static
  regularization: a condensed matrix whose second Cholesky pivot is p22 - 1, exactly."""
  P = np.array([[4.0, 2.0], [2.0, p22]])
  s = QPStructure.from_patterns(
    P, np.zeros((0, 2)), np.zeros((0, 2)), h_l=np.zeros(0), h_u=np.zeros(0), x_l=np.full(2, -np.inf), x_u=np.full(2, np.inf)
  )
  pv = sc.sym("pv", s.P_rows.size)
  empty = sc.const(np.zeros(0))
  values = QPValues(P=pv, c=sc.const(np.zeros(2)), A=empty, b=empty, G=empty, h_l=empty, h_u=empty, x_l=empty, x_u=empty)
  unit = Scaling(sc.const(np.ones(2)), sc.const(np.ones(2)), sc.const(1.0))
  kkt = KKT(
    Kernels(s, "dense", Refinement(static_eps=0.0, static_rel=0.0, max_factor_retires=retries), name=f"noise_{int(ir)}_{p22 > 1}_{retries}"),
    ScaledQP(values, sc.const(np.ones(2)), unit),
  )
  it = Iterate.unflat(s, sc.const(np.zeros(sum(Iterate.sizes(s)))))
  factor = kkt.factor(0.0, 1e-4, it, ir=ir)
  keys = ["ok", "ir", "delta", "retries"]
  fn = sc.Function.from_exprs(f"noise_{int(ir)}_{p22 > 1}_{retries}", [pv], [factor.head[k] for k in keys], ["pv"], keys)
  return dict(zip(keys, (float(v) for v in fn(P[s.P_rows, s.P_cols])), strict=True))


def test_a_noise_pivot_turns_refinement_on_and_nothing_more() -> None:
  """The dense backend's second pivot is eps against a diagonal entry of 1 + eps: positive, so
  PIQP's own test passes, but without a digit left. With refinement off it counts as a failure
  and turns refinement on; with refinement on it is accepted, with no retry. A pivot of exactly
  zero fails PIQP's test too, and without regularization no retry helps."""
  eps = float(np.finfo(np.float64).eps)
  assert _noise_pivot(1.0 + eps, 0.0) == {"ok": 1.0, "ir": 1.0, "delta": 1e-4, "retries": 0.0}
  assert _noise_pivot(1.0 + eps, 1.0) == {"ok": 1.0, "ir": 1.0, "delta": 1e-4, "retries": 0.0}
  zero = _noise_pivot(1.0, 0.0)
  assert zero["ok"] == 0.0 and zero["retries"] == 10.0

  # With no retries allowed a failure still turns refinement on once, as PIQP's loop would (its
  # settings refuse 0 retries, the generated solver takes it); with refinement already on, nothing.
  assert _noise_pivot(1.0, 0.0, retries=0) == {"ok": 0.0, "ir": 1.0, "delta": 1e-4, "retries": 0.0}
  assert _noise_pivot(1.0, 1.0, retries=0) == {"ok": 0.0, "ir": 1.0, "delta": 1e-4, "retries": 0.0}


@pytest.mark.parametrize("tolerance", [0.0, 1e2])
@pytest.mark.parametrize("backend", ["dense", "sparse"])
def test_a_solve_without_refinement_is_the_plain_solve(backend: Backend, tolerance: float) -> None:
  """Refinement off, the solve is the kernel's solve to the last bit: its gate stays shut even where
  refinement would change the answer (tolerance zero: every residual is above it). Refinement on,
  a residual already below the tolerance takes no step: the plain solve again."""
  qp = maros_meszaros("QAFIRO")
  s, values = ipm_inputs(qp)
  syms = {k: sc.sym(k, np.shape(values[k])) for k in ORDER}
  v = QPValues.preprocess(s, **syms)
  q = scale(s, v, ruiz(s, v))
  kern = Kernels(s, backend, Refinement(eps_abs=tolerance, eps_rel=0.0), name=f"plain_{backend}_{int(tolerance)}")
  kkt = KKT(kern, q)
  sizes = Iterate.sizes(s)
  it_sym, rhs_sym, ir = sc.sym("it", (sum(sizes),)), sc.sym("r", (kern.size,)), sc.sym("ir", ())
  factor = kkt.factor(1e-6, 1e-4, Iterate.unflat(s, it_sym), ir=0.0)
  plain = kern._solve.symbolic_call((factor.record, kkt.data, rhs_sym))
  gated = kern.solve(factor.record, kkt.data, rhs_sym, ir > 0.5)
  fn = sc.Function.from_exprs(
    f"plain_{backend}_{int(tolerance)}",
    [*(syms[k] for k in ORDER), it_sym, rhs_sym, ir],
    [plain, gated],
    [*ORDER, "it", "r", "ir"],
    ["plain", "gated"],
  )
  d = ref.Solver(qp, ref.Settings()).data
  it, _ = _random_state(d, np.random.default_rng(1))
  args = (*(values[k] for k in ORDER), np.concatenate(_compact(d, it)), np.random.default_rng(2).standard_normal(kern.size))
  off, off_gated = fn((*args, np.array(0.0)))
  np.testing.assert_array_equal(off_gated, off)
  on, on_gated = fn((*args, np.array(1.0)))
  np.testing.assert_array_equal(on, off)
  if tolerance:
    np.testing.assert_array_equal(on_gated, on)  # the gate opens, and no step is taken
  else:
    assert not np.array_equal(on_gated, on)  # refined: the gate does open
