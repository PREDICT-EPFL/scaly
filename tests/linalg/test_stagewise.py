"""``linalg.stagewise``: the Riccati factorization and solve of a stage-structured LQ problem against the
dense KKT solve, its implicit derivatives against the complex-step derivative of that dense solve, and
its gains against the TinyMPC example's Riccati cache."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
from scaly.linalg.stagewise import Riccati

TINYMPC = Path(__file__).resolve().parents[2] / "examples" / "ocp" / "tinympc"
sys.path.insert(0, str(TINYMPC))  # the example's modules import each other by name

import problem as tp  # noqa: E402  # ty: ignore[unresolved-import]
import problems as tps  # noqa: E402  # ty: ignore[unresolved-import]

NX, NU = 3, 2
NAMES = ("A", "B", "Q", "R", "S", "QN", "x0", "q", "r", "c", "qN")


def _spd(rng: np.random.Generator, n: int, stack: int | None) -> np.ndarray:
  m = rng.normal(size=(n, n) if stack is None else (stack, n, n))
  return m @ np.swapaxes(m, -1, -2) + n * np.eye(n)


def _data(n: int, shared: bool, seed: int = 0) -> dict[str, np.ndarray]:
  """A random convex problem; ``shared`` gives one matrix for every stage."""
  rng = np.random.default_rng(seed)
  k = None if shared else n
  lead = () if shared else (n,)
  return {
    "A": 0.5 * rng.normal(size=(*lead, NX, NX)),
    "B": rng.normal(size=(*lead, NX, NU)),
    "Q": _spd(rng, NX, k),
    "R": _spd(rng, NU, k),
    "S": 0.1 * rng.normal(size=(*lead, NU, NX)),
    "QN": _spd(rng, NX, None),
    "x0": rng.normal(size=NX),
    "q": rng.normal(size=(n, NX)),
    "r": rng.normal(size=(n, NU)),
    "c": rng.normal(size=(n, NX)),
    "qN": rng.normal(size=NX),
  }


def _stages(m: np.ndarray, n: int) -> np.ndarray:
  return m if m.ndim == 3 else np.broadcast_to(m, (n, *m.shape))


def _dense(d: dict[str, np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """The KKT system assembled whole and solved by NumPy (complex data too): variables
  ``x_0, u_0, ..., x_N`` then multipliers ``lam_0 .. lam_N``; ``Q``, ``R`` and ``QN`` by their symmetric parts."""
  a, b, s = (_stages(d[key], n) for key in "ABS")
  q_mat, r_mat = (_stages(0.5 * (d[key] + np.swapaxes(d[key], -1, -2)), n) for key in "QR")
  qn = 0.5 * (d["QN"] + d["QN"].T)
  nz, nl = (n + 1) * NX + n * NU, (n + 1) * NX
  m = np.zeros((nz + nl, nz + nl), dtype=np.result_type(*d.values()))
  rhs = np.zeros(nz + nl, dtype=m.dtype)
  xi, ui, li = (lambda k: k * (NX + NU)), (lambda k: k * (NX + NU) + NX), (lambda k: nz + k * NX)
  for k in range(n):
    xs, us, ls = slice(xi(k), xi(k) + NX), slice(ui(k), ui(k) + NU), slice(li(k + 1), li(k + 1) + NX)
    m[xs, xs], m[us, us], m[us, xs], m[xs, us] = q_mat[k], r_mat[k], s[k], s[k].T
    rhs[xs], rhs[us] = -d["q"][k], -d["r"][k]
    m[ls, xs], m[ls, us], m[ls, xi(k + 1) : xi(k + 1) + NX] = a[k], b[k], -np.eye(NX)
    rhs[ls] = -d["c"][k]
  last = slice(xi(n), xi(n) + NX)
  m[last, last], rhs[last] = qn, -d["qN"]
  m[li(0) : li(0) + NX, 0:NX], rhs[li(0) : li(0) + NX] = -np.eye(NX), -d["x0"]
  m[:nz, nz:] = m[nz:, :nz].T
  z = np.linalg.solve(m, rhs)
  x = np.stack([z[xi(k) : xi(k) + NX] for k in range(n + 1)])
  u = np.stack([z[ui(k) : ui(k) + NU] for k in range(n)])
  return x, u, z[nz:].reshape(n + 1, NX)


def _solve(parts: dict[str, sc.Expr], n: int, with_s: bool = True) -> tuple[Riccati, sc.Expr, sc.Expr, sc.Expr]:
  fac = Riccati(parts["A"], parts["B"], parts["Q"], parts["R"], parts["QN"], S=parts["S"] if with_s else None, N=n)
  return (fac, *fac.solve(parts["x0"], parts["q"], parts["r"], parts["c"], parts["qN"]))


def _packed(d: dict[str, np.ndarray]) -> tuple[sc.Expr, dict[str, sc.Expr], np.ndarray]:
  """Every datum sliced from one vector ``theta``, so one Jacobian covers them all."""
  theta0 = np.concatenate([np.ravel(d[key]) for key in NAMES])
  theta = sc.sym("theta", theta0.size)
  parts, offset = {}, 0
  for key in NAMES:
    size = d[key].size
    parts[key] = theta[offset : offset + size].reshape(d[key].shape)
    offset += size
  return theta, parts, theta0


def _unpacked(t: np.ndarray, d: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
  out, offset = {}, 0
  for key in NAMES:
    out[key] = t[offset : offset + d[key].size].reshape(d[key].shape)
    offset += d[key].size
  return out


@pytest.mark.parametrize("shared", [False, True], ids=["per_stage", "shared"])
@pytest.mark.parametrize("n", [1, 5])
@pytest.mark.parametrize("with_s", [True, False], ids=["S", "no_S"])
def test_the_solve_matches_the_dense_kkt_solve(shared: bool, n: int, with_s: bool) -> None:
  d = _data(n, shared)
  if not with_s:
    d["S"] = np.zeros_like(d["S"])
  syms = {key: sc.sym(key, value.shape) for key, value in d.items()}
  _, x, u, lam = _solve(syms, n, with_s)
  fn = sc.Function.from_exprs(f"sw_solve_{int(shared)}_{n}_{int(with_s)}", list(syms.values()), [x, u, lam], list(syms), ["x", "u", "lam"])
  got = fn._flat_numerical_call(*d.values())
  for g, w in zip(got, _dense(d, n), strict=True):
    np.testing.assert_allclose(g, w, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize("shared", [False, True], ids=["per_stage", "shared"])
def test_first_derivatives_match_the_complex_step_of_the_dense_solve(shared: bool) -> None:
  """Forward (``jacobian``) and reverse (``gradient``) mode, with respect to every matrix and linear
  term at once: the implicit rules, a shared matrix's cotangent summed over the stages."""
  n = 4
  d = _data(n, shared, seed=1)
  theta, parts, theta0 = _packed(d)
  _, x, u, lam = _solve(parts, n)
  z = sc.concat([x.reshape((x.size,)), u.reshape((u.size,)), lam.reshape((lam.size,))])
  w = np.random.default_rng(2).normal(size=z.size)
  fn = sc.Function.from_exprs(
    f"sw_jac_{int(shared)}", [theta], [sc.jacobian(z, theta), sc.gradient((z * sc.const(w)).sum(), theta)], ["t"], ["J", "g"]
  )
  jac, grad = fn._flat_numerical_call(theta0)
  h = 1e-30

  def dense(t: np.ndarray) -> np.ndarray:
    return np.concatenate([v.ravel() for v in _dense(_unpacked(t, d), n)])

  ref = np.stack([dense(theta0 + 1j * h * e).imag / h for e in np.eye(theta0.size)], axis=1)
  np.testing.assert_allclose(jac, ref, rtol=1e-11, atol=1e-11 * np.abs(ref).max())
  np.testing.assert_allclose(grad, ref.T @ w, rtol=1e-11, atol=1e-11 * np.abs(ref.T @ w).max())


def test_second_derivatives_go_through_the_rules_too() -> None:
  """The rules solve through a solve with rules of its own: the Hessian of a cost of the solution
  against central differences of its gradient."""
  n = 3
  d = _data(n, False, seed=3)
  theta, parts, theta0 = _packed(d)
  _, x, u, lam = _solve(parts, n)
  cost = sc.sumsqr(x) + (u * u * u).sum() + (lam * x).sum()
  fn = sc.Function.from_exprs("sw_hess", [theta], [sc.gradient(cost, theta), sc.hessian(cost, theta)], ["t"], ["g", "H"])
  _, hess = fn._flat_numerical_call(theta0)
  step = 1e-6
  fd = np.stack(
    [(fn._flat_numerical_call(theta0 + step * e)[0] - fn._flat_numerical_call(theta0 - step * e)[0]) / (2 * step) for e in np.eye(theta0.size)],
    axis=1,
  )
  np.testing.assert_allclose(hess, fd, rtol=1e-5, atol=1e-5 * np.abs(fd).max())


def test_the_gains_and_cost_to_go_match_a_numpy_recursion() -> None:
  n = 6
  d = _data(n, False, seed=4)
  syms = {key: sc.sym(key, d[key].shape) for key in ("A", "B", "Q", "R", "S", "QN")}
  fac = Riccati(syms["A"], syms["B"], syms["Q"], syms["R"], syms["QN"], S=syms["S"])
  fn = sc.Function.from_exprs("sw_gains", list(syms.values()), [fac.gains, fac.cost_to_go], list(syms), ["K", "P"])
  gains, costs = fn._flat_numerical_call(*(d[key] for key in syms))
  p = d["QN"]
  for k in reversed(range(n)):
    a, b = d["A"][k], d["B"][k]
    gain = -np.linalg.solve(d["R"][k] + b.T @ p @ b, d["S"][k] + b.T @ p @ a)
    p = d["Q"][k] + a.T @ p @ a + (d["S"][k] + b.T @ p @ a).T @ gain
    np.testing.assert_allclose(gains[k], gain, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(costs[k], p, rtol=1e-11, atol=1e-12)
  np.testing.assert_allclose(costs[n], d["QN"], rtol=1e-15)


def _tiny_steps(p: tp.Problem) -> int:
  """How many steps the library's recursion takes before ``K`` moves by less than 1e-5 (``tp.cache``)."""
  q1, r1, a, b = np.diag(p.Q + p.rho), np.diag(p.R + p.rho), p.A, p.B
  k_prev, p_next = np.zeros((p.nu, p.nx)), p.rho * np.eye(p.nx)
  for step in range(1, 1001):
    kinf = np.linalg.inv(r1 + b.T @ p_next @ b) @ b.T @ p_next @ a
    if np.abs(kinf - k_prev).max() < 1e-5:
      return step
    k_prev, p_next = kinf, q1 + a.T @ p_next @ (a - b @ kinf)
  return 1000


@pytest.mark.parametrize(
  "scenario",
  [lambda: tps.random_mpc(6, 3, 8), lambda: tps.safety_filter(4, 10), lambda: tps.rocket_landing(8)],
  ids=["random_mpc", "safety_filter", "rocket_landing"],
)
def test_the_tinympc_riccati_cache(scenario) -> None:
  """The example's cache is the recursion from ``P = rho I`` run until the gain settles: the first
  stage of a factorization that many stages long, on the penalized weights, is the same gain and
  cost-to-go (the example's gain has the opposite sign)."""
  p = scenario().problem
  cache = tp.cache(p)
  n = _tiny_steps(p)
  fac = Riccati(p.A, p.B, np.diag(p.Q + p.rho), np.diag(p.R + p.rho), p.rho * np.eye(p.nx), N=n)
  fn = sc.Function.from_exprs(f"sw_tiny_{p.nx}_{p.nu}_{n}", [], [fac.gains[0], fac.cost_to_go[0]], [], ["K0", "P0"])
  k0, p0 = fn._flat_numerical_call()
  np.testing.assert_allclose(-k0, cache.K, rtol=1e-9, atol=1e-9 * np.abs(cache.K).max())
  np.testing.assert_allclose(p0, 0.5 * (cache.P + cache.P.T), rtol=1e-9, atol=1e-9 * np.abs(cache.P).max())


def test_a_shared_matrix_is_the_same_matrix_at_every_stage() -> None:
  n = 4
  shared = _data(n, True, seed=5)
  per_stage = {key: (np.broadcast_to(v, (n, *v.shape)).copy() if key in "ABQRS" else v) for key, v in shared.items()}
  outs = []
  for tag, d in (("shared", shared), ("stacked", per_stage)):
    syms = {key: sc.sym(key, value.shape) for key, value in d.items()}
    _, x, u, lam = _solve(syms, n)
    cost = sc.sumsqr(x) + sc.sumsqr(u)
    fn = sc.Function.from_exprs(f"sw_same_{tag}", list(syms.values()), [x, u, lam, sc.gradient(cost, syms["A"])], list(syms), ["x", "u", "lam", "gA"])
    outs.append(fn._flat_numerical_call(*d.values()))
  for a, b in zip(outs[0][:3], outs[1][:3], strict=True):
    np.testing.assert_allclose(a, b, rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(outs[0][3], outs[1][3].sum(axis=0), rtol=1e-12, atol=1e-12)  # a shared matrix's cotangent sums the stages'


def test_the_jacobian_pattern_includes_the_weights() -> None:
  """The solve reads ``Q`` only through the factorization, whose derivative the rules carry: the
  pattern says so, or a sparse Jacobian would drop it."""
  n = 3
  d = _data(n, True)
  syms = {key: sc.sym(key, value.shape) for key, value in d.items()}
  _, x, _, _ = _solve(syms, n)
  pattern = sc.jacobian_sparsity(x.reshape((x.size,)), syms["Q"])
  assert len(pattern.rows) == x.size * d["Q"].size


def test_malformed_problems_are_refused() -> None:
  a, b, q, r = sc.sym("a", (3, 3)), sc.sym("b", (3, 2)), sc.sym("q", (3, 3)), sc.sym("r", (2, 2))
  with pytest.raises(ValueError, match="N must be given"):
    Riccati(a, b, q, r, q)
  with pytest.raises(ValueError, match="disagree on N"):
    Riccati(sc.sym("as", (4, 3, 3)), sc.sym("bs", (5, 3, 2)), q, r, q)
  with pytest.raises(ValueError, match=r"R must have shape \(2, 2\) or \(4, 2, 2\)"):
    Riccati(a, b, q, sc.sym("r3", (3, 3)), q, N=4)
  fac = Riccati(a, b, q, r, q, N=4)
  with pytest.raises(ValueError, match=r"q must have shape \(3,\) or \(4, 3\)"):
    fac.solve(sc.sym("x0", 3), q=sc.sym("qq", (5, 3)))
