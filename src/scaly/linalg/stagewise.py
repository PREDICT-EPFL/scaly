"""Stage-structured linear-quadratic problems: the Riccati factorization of their KKT system, its solve, and the implicit derivative of the solution.

The problem over ``N`` stages is

    minimize    sum_k 1/2 x_k' Q_k x_k + u_k' S_k x_k + 1/2 u_k' R_k u_k + q_k' x_k + r_k' u_k
                + 1/2 x_N' Q_N x_N + q_N' x_N
    subject to  x_{k+1} = A_k x_k + B_k u_k + c_k,   x_0 given,

and its KKT system is block tridiagonal, one block per stage. ``Riccati`` factors it by the backward
recursion on the cost-to-go ``P_k`` (the matrices), and ``Riccati.solve`` takes the linear terms and
the initial state: a backward pass for the affine terms, then a forward rollout of ``u_k = K_k x_k +
k_k``. The multipliers ``lam_k = P_k x_k + p_k`` of the dynamics (and of the initial state, ``lam_0``)
come with the solution.
"""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from ..function.model import ConcreteFunction
from ..function.sugar import custom_derivative, scan, vmap
from ..ir.expr import Expr, as_expr, concat, gather
from .dense import cho_solve, cholesky

__all__ = ["Riccati"]

_NAMES = itertools.count()


def _sym(m: Expr) -> Expr:
  return 0.5 * (m + m.T)


def _reversed(flat: Expr, n: int, size: int) -> Expr:
  """The ``n`` rows of ``size`` values in ``flat``, last first: a backward scan's stacked output in stage order."""
  rows = (np.arange(n)[::-1, None] * size + np.arange(size)[None, :]).reshape(-1)
  return gather(flat, rows)


def _call(fn: ConcreteFunction, *args: Expr) -> tuple[Expr, ...]:
  out = fn.symbolic_call(tuple(args))
  return out if isinstance(out, tuple) else (out,)


class Riccati:
  """The Riccati factorization of a stage-structured linear-quadratic problem's KKT system.

  ``A`` (``nx x nx``), ``B`` (``nx x nu``), ``Q`` (``nx x nx``), ``R`` (``nu x nu``) and ``S``
  (``nu x nx``, the cross term, zero when omitted) are each one matrix shared by every stage or a
  stack of ``N`` matrices, one per stage; ``QN`` is the terminal weight. ``N`` is read from the
  stacks, and has to be given when every matrix is shared. ``Q``, ``R`` and ``QN`` enter through their
  symmetric parts. Each stage's ``R_k + B_k' P_{k+1} B_k`` must be positive definite, as it is when
  the problem is convex in the controls; it is factored by the generated ``cholesky``.

  Building the factorization runs the backward recursion, one ``scan`` over the stages:
  ``H_uu = R + B' P B``, ``H_ux = S + B' P A``, ``K = -H_uu^{-1} H_ux`` and
  ``P <- Q + A' P A + H_ux' K``, from ``P_N = Q_N``. The factorization is differentiable through that
  loop; ``solve`` differentiates implicitly and never needs to.
  """

  def __init__(self, A: Any, B: Any, Q: Any, R: Any, QN: Any, *, S: Any = None, N: int | None = None, name: str | None = None) -> None:
    B = as_expr(B)
    if len(B.shape) not in (2, 3):
      raise ValueError(f"B must be one nx x nu matrix or a stack of them, got shape {B.shape}")
    nx, nu = B.shape[-2], B.shape[-1]
    per_stage = {"A": (nx, nx), "B": (nx, nu), "Q": (nx, nx), "R": (nu, nu), "S": (nu, nx)}
    given = {"A": A, "B": B, "Q": Q, "R": R, **({} if S is None else {"S": S})}
    data = {key: as_expr(value) for key, value in given.items()}
    stacked = {key: value.shape[0] for key, value in data.items() if len(value.shape) == 3}
    horizons = set(stacked.values()) | ({int(N)} if N is not None else set())
    if len(horizons) != 1:
      raise ValueError(f"the stages disagree on N: {stacked}, N={N}" if horizons else "N must be given when every matrix is shared by all stages")
    self.N = horizons.pop()
    if self.N < 1:
      raise ValueError(f"N must be at least 1, got {self.N}")
    for key, value in data.items():
      if value.shape not in (per_stage[key], (self.N, *per_stage[key])):
        raise ValueError(f"{key} must have shape {per_stage[key]} or {(self.N, *per_stage[key])}, got {value.shape}")
    self._QN = as_expr(QN)
    if self._QN.shape != (nx, nx):
      raise ValueError(f"QN must have shape {(nx, nx)}, got {self._QN.shape}")
    self.nx, self.nu = nx, nu
    self.name = name or f"riccati{next(_NAMES)}"
    self._shared = {key: len(value.shape) == 2 for key, value in data.items()}
    self._data = {key: value.reshape((value.size,)) for key, value in data.items()}  # flat, per stage or shared
    self._shapes = per_stage
    self._solvers: dict[int, ConcreteFunction] = {}
    self._factor()

  # --- the factorization --------------------------------------------------------------------------

  def _slice(self, key: str, backward: bool) -> tuple[Expr, int, int]:
    """``data[key]`` as a scan or map input: the shared matrix every step, or stage ``k`` at step ``k``
    (at step ``N - 1 - k`` when ``backward``)."""
    size = int(np.prod(self._shapes[key]))
    if self._shared[key]:
      return self._data[key], 0, 0
    return (self._data[key], (self.N - 1) * size, -size) if backward else (self._data[key], 0, size)

  def _factor(self) -> None:
    nx, nu, N = self.nx, self.nu, self.N
    p_flat = Expr.sym("P", (nx * nx,))
    mats = {key: Expr.sym(key, self._shapes[key]) for key in self._data}
    a, b = mats["A"], mats["B"]
    p = p_flat.reshape((nx, nx))
    pa, pb = p @ a, p @ b
    huu = _sym(mats["R"]) + b.T @ pb
    hux = b.T @ pa
    if "S" in mats:
      hux = hux + mats["S"]
    chol = cholesky(huu)
    gain = -cho_solve(chol, hux)
    p_new = _sym(_sym(mats["Q"]) + a.T @ pa + hux.T @ gain)
    names = ["P", *mats]
    step = ConcreteFunction.from_exprs(
      f"{self.name}_factor_step", [p_flat, *mats.values()], [p_new.reshape((nx * nx,)), gain, chol, p_new], names, ["P_prev", "K", "L", "P_out"]
    )
    terminal = _sym(self._QN)
    _, gains, chols, costs = scan(step, terminal.reshape((nx * nx,)), [self._slice(key, True) for key in mats], length=N)
    self._K = _reversed(gains, N, nu * nx)
    self._L = _reversed(chols, N, nu * nu)
    self._P = concat([_reversed(costs, N, nx * nx), terminal.reshape((nx * nx,))])

  @property
  def gains(self) -> Expr:
    """The feedback gains ``K_k``, ``(N, nu, nx)`` in stage order: ``u_k = K_k x_k + k_k``."""
    return self._K.reshape((self.N, self.nu, self.nx))

  @property
  def cost_to_go(self) -> Expr:
    """The cost-to-go matrices ``P_0 .. P_N``, ``(N + 1, nx, nx)``; ``P_N`` is ``Q_N``'s symmetric part."""
    return self._P.reshape((self.N + 1, self.nx, self.nx))

  # --- the solve ----------------------------------------------------------------------------------

  def _inputs(self) -> list[tuple[str, int]]:
    """The solve Function's inputs, flat: the factorization, the matrices, then the linear terms."""
    nx, nu, N = self.nx, self.nu, self.N
    mats = [(key, (1 if self._shared[key] else N) * int(np.prod(self._shapes[key]))) for key in self._data]
    return [
      ("K", N * nu * nx),
      ("L", N * nu * nu),
      ("P", (N + 1) * nx * nx),
      *mats,
      ("QN", nx * nx),
      ("x0", nx),
      ("q", N * nx),
      ("r", N * nu),
      ("c", N * nx),
      ("qN", nx),
    ]

  def _raw_solve(self, v: dict[str, Expr]) -> tuple[Expr, Expr, Expr]:
    """The backward pass for the affine terms and the forward rollout, from the flat inputs ``v``."""
    nx, nu, N = self.nx, self.nu, self.N
    sl = {key: (v[key], *self._slice(key, True)[1:]) for key in self._data}
    p1 = Expr.sym("p", (nx,))
    a, b, k, chol, pn = Expr.sym("A", (nx, nx)), Expr.sym("B", (nx, nu)), Expr.sym("K", (nu, nx)), Expr.sym("L", (nu, nu)), Expr.sym("Pn", (nx, nx))
    q, r, c = Expr.sym("q", (nx,)), Expr.sym("r", (nu,)), Expr.sym("c", (nx,))
    w = pn @ c + p1
    hu = r + b.T @ w
    ff = -cho_solve(chol, hu)
    p0 = q + a.T @ w + k.T @ hu
    back = ConcreteFunction.from_exprs(
      f"{self.name}_solve_back",
      [p1, a, b, k, chol, pn, q, r, c],
      [p0, ff, p0],
      ["p", "A", "B", "K", "L", "Pn", "q", "r", "c"],
      ["p_prev", "k", "p_out"],
    )
    rev = [
      sl["A"],
      sl["B"],
      (v["K"], (N - 1) * nu * nx, -nu * nx),
      (v["L"], (N - 1) * nu * nu, -nu * nu),
      (v["P"], N * nx * nx, -nx * nx),
      (v["q"], (N - 1) * nx, -nx),
      (v["r"], (N - 1) * nu, -nu),
      (v["c"], (N - 1) * nx, -nx),
    ]
    _, ffs, ps = scan(back, v["qN"], rev, length=N)
    x, pk, lk = Expr.sym("x", (nx,)), Expr.sym("Pk", (nx, nx)), Expr.sym("pk", (nx,))
    kk = Expr.sym("kk", (nu,))
    u = k @ x + kk
    fwd = ConcreteFunction.from_exprs(
      f"{self.name}_solve_forward",
      [x, a, b, k, kk, c, pk, lk],
      [a @ x + b @ u + c, x, u, pk @ x + lk],
      ["x", "A", "B", "K", "k", "c", "Pk", "pk"],
      ["x_next", "x_out", "u", "lam"],
    )
    fw = {key: (v[key], *self._slice(key, False)[1:]) for key in ("A", "B")}
    xs_in = [fw["A"], fw["B"], (v["K"], 0, nu * nx), (ffs, (N - 1) * nu, -nu), (v["c"], 0, nx), (v["P"], 0, nx * nx), (ps, (N - 1) * nx, -nx)]
    x_last, xs, us, lams = scan(fwd, v["x0"], xs_in, length=N)
    lam_last = v["P"][N * nx * nx :].reshape((nx, nx)) @ x_last + v["qN"]
    return concat([xs, x_last]), us, concat([lams, lam_last])

  def _solver(self) -> ConcreteFunction:
    """The solve as a Function of ``_inputs()`` with the implicit derivative rules, two levels deep
    as ``SparseLDL.solve``'s: the rules solve with the same factorization through a solve that has
    rules of its own, so second derivatives are implicit too."""
    if not self._solvers:
      syms = {key: Expr.sym(key, (size,)) for key, size in self._inputs()}
      base = ConcreteFunction.from_exprs(f"{self.name}_solve", list(syms.values()), list(self._raw_solve(syms)), list(syms), ["x", "u", "lam"])
      inner = base
      pattern = self._sparsity()
      for level in (1, 2):
        inner = custom_derivative(base, jvp=self._jvp_rule(inner, level), vjp=self._vjp_rule(inner, level), sparsity=pattern)
      self._solvers[0] = inner
    return self._solvers[0]

  def _sparsity(self) -> Any:
    """Every output on every input but the factorization, whose derivative the rules take to be zero
    (the matrices carry it). The body's own pattern would miss ``Q``, ``R``, ``S`` and ``QN``, which
    it reads only through the factorization."""
    sizes = [size for _, size in self._inputs()]
    outs = [(self.N + 1) * self.nx, self.N * self.nu, (self.N + 1) * self.nx]

    def pattern(output: int, k: int) -> Any:
      return None if k < 3 else np.ones((outs[output], sizes[k]), dtype=bool)

    return pattern

  def _stage_map(self, fn: ConcreteFunction, specs: list[tuple[Expr, int, int]], size: int) -> Expr:
    return vmap(fn, self.N, specs).reshape((self.N, size))

  def _jvp_rule(self, inner: ConcreteFunction, level: int) -> ConcreteFunction:
    """``dz = M^{-1} (-F_theta dtheta)``: one more solve with the same factorization, whose linear
    terms and initial state are the tangents of the data acting on the solution."""
    nx, nu, N = self.nx, self.nu, self.N
    names = [key for key, _ in self._inputs()]
    v = {key: Expr.sym(key, (size,)) for key, size in self._inputs()}
    dv = {key: Expr.sym(f"d{key}", (size,)) for key, size in self._inputs()}
    x, u, lam = _call(inner, *v.values())
    mats = {key: Expr.sym(f"d{key}", self._shapes[key]) for key in self._data}
    xk, uk, lk1 = Expr.sym("x", (nx,)), Expr.sym("u", (nu,)), Expr.sym("lam1", (nx,))
    da, db = mats["A"], mats["B"]
    ds = mats.get("S")
    qe = _sym(mats["Q"]) @ xk + da.T @ lk1
    re = _sym(mats["R"]) @ uk + db.T @ lk1
    if ds is not None:
      qe, re = qe + ds.T @ uk, re + ds @ xk
    ce = da @ xk + db @ uk
    stage = ConcreteFunction.from_exprs(
      f"{self.name}_stage_rhs{level}", [*mats.values(), xk, uk, lk1], [concat([qe, re, ce])], [*[f"d{k}" for k in mats], "x", "u", "lam1"], ["rhs"]
    )
    specs = [(dv[key], *self._slice(key, False)[1:]) for key in self._data] + [(x, 0, nx), (u, 0, nu), (lam, nx, nx)]
    rhs = self._stage_map(stage, specs, 2 * nx + nu)
    q = dv["q"] + rhs[:, :nx].reshape((N * nx,))
    r = dv["r"] + rhs[:, nx : nx + nu].reshape((N * nu,))
    c = dv["c"] + rhs[:, nx + nu :].reshape((N * nx,))
    qn = dv["qN"] + _sym(dv["QN"].reshape((nx, nx))) @ x[N * nx :]
    rest = [v[key] for key in names[: names.index("x0")]]
    outs = _call(inner, *rest, dv["x0"], q, r, c, qn)
    return ConcreteFunction.from_exprs(
      f"{self.name}_solve_jvp{level}", [*v.values(), *dv.values()], list(outs), [*names, *(f"d{k}" for k in names)], ["dx", "du", "dlam"]
    )

  def _vjp_rule(self, inner: ConcreteFunction, level: int) -> ConcreteFunction:
    """The adjoint of the KKT system is the system itself: ``w = M^{-1} zbar`` is one more solve with
    the same factorization, and ``thetabar = -F_theta' w``, outer products of ``w`` with the solution,
    summed over the stages for a matrix they share."""
    nx, nu, N = self.nx, self.nu, self.N
    names = [key for key, _ in self._inputs()]
    v = {key: Expr.sym(key, (size,)) for key, size in self._inputs()}
    x, u, lam = Expr.sym("xo", ((N + 1) * nx,)), Expr.sym("uo", (N * nu,)), Expr.sym("lamo", ((N + 1) * nx,))
    xb, ub, lb = Expr.sym("xbar", ((N + 1) * nx,)), Expr.sym("ubar", (N * nu,)), Expr.sym("lambar", ((N + 1) * nx,))
    rest = [v[key] for key in names[: names.index("x0")]]
    wx, wu, wl = _call(inner, *rest, -lb[:nx], -xb[: N * nx], -ub, -lb[nx:], -xb[N * nx :])
    xk, uk, lk1 = Expr.sym("x", (nx,)), Expr.sym("u", (nu,)), Expr.sym("lam1", (nx,))
    wxk, wuk, wlk1 = Expr.sym("wx", (nx,)), Expr.sym("wu", (nu,)), Expr.sym("wl1", (nx,))

    def outer(a: Expr, b: Expr) -> Expr:
      return a.reshape((a.size, 1)) @ b.reshape((1, b.size))

    parts = {
      "A": -(outer(lk1, wxk) + outer(wlk1, xk)),
      "B": -(outer(lk1, wuk) + outer(wlk1, uk)),
      "Q": -0.5 * (outer(wxk, xk) + outer(xk, wxk)),
      "R": -0.5 * (outer(wuk, uk) + outer(uk, wuk)),
      "S": -(outer(wuk, xk) + outer(uk, wxk)),
    }
    keys = list(self._data)
    sizes = [int(np.prod(self._shapes[key])) for key in keys]
    stage = ConcreteFunction.from_exprs(
      f"{self.name}_stage_bars{level}",
      [xk, uk, lk1, wxk, wuk, wlk1],
      [concat([parts[key].reshape((size,)) for key, size in zip(keys, sizes, strict=True)])],
      ["x", "u", "lam1", "wx", "wu", "wl1"],
      ["bars"],
    )
    bars = self._stage_map(stage, [(x, 0, nx), (u, 0, nu), (lam, nx, nx), (wx, 0, nx), (wu, 0, nu), (wl, nx, nx)], sum(sizes))
    out: dict[str, Expr] = {key: Expr.const(np.zeros(size)) for key, size in self._inputs()[:3]}
    offset = 0
    for key, size in zip(keys, sizes, strict=True):
      block = bars[:, offset : offset + size]
      offset += size
      out[key] = (Expr.const(np.ones((1, N))) @ block).reshape((size,)) if self._shared[key] else block.reshape((N * size,))
    x_last, w_last = x[N * nx :], wx[N * nx :]
    out["QN"] = (-0.5 * (outer(w_last, x_last) + outer(x_last, w_last))).reshape((nx * nx,))
    out.update(x0=-wl[:nx], q=-wx[: N * nx], r=-wu, c=-wl[nx:], qN=-w_last)
    return ConcreteFunction.from_exprs(
      f"{self.name}_solve_vjp{level}",
      [*v.values(), x, u, lam, xb, ub, lb],
      [out[key] for key in names],
      [*names, "xo", "uo", "lamo", "xbar", "ubar", "lambar"],
      [f"{key}bar" for key in names],
    )

  def _linear(self, value: Any, size: int, what: str) -> Expr:
    """A linear term as ``N`` stacked rows: zeros when omitted, a shared vector repeated."""
    if value is None:
      return Expr.const(np.zeros(self.N * size))
    value = as_expr(value)
    if value.shape == (size,):
      return gather(value, np.tile(np.arange(size), self.N))
    if value.shape == (self.N, size):
      return value.reshape((self.N * size,))
    raise ValueError(f"{what} must have shape {(size,)} or {(self.N, size)}, got {value.shape}")

  def solve(self, x0: Any, q: Any = None, r: Any = None, c: Any = None, qN: Any = None) -> tuple[Expr, Expr, Expr]:
    """The solution ``(x, u, lam)`` for the initial state ``x0`` and the linear terms: ``x`` the
    states ``x_0 .. x_N`` (``(N + 1, nx)``), ``u`` the controls (``(N, nu)``), and ``lam`` the
    multipliers of the initial state and of each stage's dynamics (``(N + 1, nx)``), the gradient
    of the optimal cost-to-go at each state. ``q``, ``r`` and ``c`` are one vector for every stage
    or ``N`` rows of them, ``qN`` the terminal linear term; each is zero when omitted.

    Differentiable in every matrix and linear term by the implicit rule on the KKT system: a
    tangent or a cotangent is one more solve with the same factorization, and never differentiates
    the recursion."""
    nx, nu, N = self.nx, self.nu, self.N
    x0 = as_expr(x0)
    if x0.shape != (nx,):
      raise ValueError(f"x0 must have shape {(nx,)}, got {x0.shape}")
    terminal = Expr.const(np.zeros(nx)) if qN is None else as_expr(qN)
    if terminal.shape != (nx,):
      raise ValueError(f"qN must have shape {(nx,)}, got {terminal.shape}")
    args = [self._K, self._L, self._P, *self._data.values(), self._QN.reshape((nx * nx,)), x0]
    args += [self._linear(q, nx, "q"), self._linear(r, nu, "r"), self._linear(c, nx, "c"), terminal]
    x, u, lam = _call(self._solver(), *args)
    return x.reshape((N + 1, nx)), u.reshape((N, nu)), lam.reshape((N + 1, nx))
