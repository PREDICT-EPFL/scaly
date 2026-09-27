"""Implicit Runge-Kutta steps: Newton on the stage equations, derivatives by the implicit function theorem."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np

from ..ad.derivatives import jacobian
from ..ad.forward import jvp
from ..ad.reverse import vjp
from ..function.model import ConcreteFunction, Function
from ..function.sugar import custom_derivative, while_loop
from ..ir.expr import Expr, concat, lu, norm_inf, stack, substitute
from ..ir.types import dtypes
from ..linalg.dense import lu_solve, solve
from .explicit import increment
from .model import discrete_map, model_rhs
from .tableau import Tableau, tableau

__all__ = ["implicit"]


def implicit(
  f: Function[Any, Any, Any, Any],
  method: str | Tableau = "radau_iia",
  *,
  stages: int | None = None,
  dt: float | None,
  steps: int = 1,
  newton_iters: int = 3,
  tol: float | None = None,
  max_iter: int = 20,
  newton: Literal["simplified", "full"] = "simplified",
  name: str | None = None,
) -> Any:
  """The discrete map ``F(x, ...) -> xnext`` of an implicit Runge-Kutta method over an interval ``dt``.

  Args:
    f: the model, as for ``explicit``: the state first, its derivative the one output.
    method: a family from ``FAMILIES`` (``"gauss_legendre"``, ``"radau_iia"``, ``"lobatto_iiia"``,
      ``"lobatto_iiic"``) with ``stages``; a named method (``"backward_euler"``,
      ``"implicit_midpoint"``, ``"trapezoidal"``, ``"sdirk2"``, ``"sdirk3"``); or an implicit
      ``Tableau``.
    stages: the number of stages of a family.
    dt: the interval, folded into the generated code; ``None`` appends a ``dt`` parameter instead.
    steps: equal substeps per interval, unrolled up to ``UNROLL_STEPS`` and looped beyond.
    newton_iters: Newton iterations per step when ``tol`` is ``None``: a fixed amount of work, the
      usual choice for control, where every call must finish in the same time.
    tol: iterate while the stage residual exceeds ``tol * (1 + |K|)`` in the max norm, at most
      ``max_iter`` times, in a ``while_loop``.
    max_iter: the bound on the iterations with ``tol``.
    newton: ``"simplified"`` factors ``I - h A (x) J(x)`` once per step, ``J`` the model's Jacobian at
      the start of the step; ``"full"`` refactors the exact stage Jacobian at every iteration.
    name: the Function's name, by default ``{f.name}_{method}``.

  Every stage starts from ``f(x)``. A diagonally implicit method (``sdirk2``, ``sdirk3``,
  ``backward_euler``) solves its stages one after another, each a system of the state's size. Any
  other couples all ``s`` stages; simplified Newton then splits the coupled system by the
  eigenvalues of ``A`` into one system of the state's size per real eigenvalue and one of twice it
  per complex pair, as RADAU5 does, and full Newton factors it whole with ``linalg.lu``.

  The derivative does not go through the iterations. At the stages ``K`` the step found, the
  implicit function theorem gives ``dK = -G_K^{-1} (G_x dx + G_u du + ...)`` for the stage
  equations ``G = 0``, one factorization of ``G_K`` for every direction, and reverse mode is one
  transposed solve. Second derivatives are implicit too, which is what an NLP solver's Lagrangian
  Hessian needs. All of them are exact at the ``K`` found, so their accuracy is the solve's.
  """
  tab = tableau(method, stages)
  if tab.explicit:
    raise ValueError(f"{tab.name} is an explicit method; use sc.integrators.explicit")
  if newton not in ("simplified", "full"):
    raise ValueError(f"newton must be 'simplified' or 'full', got {newton!r}")
  if tol is None and (int(newton_iters) != newton_iters or newton_iters < 1):
    raise ValueError(f"newton_iters must be a positive integer, got {newton_iters}")
  if tol is not None and (not tol > 0 or int(max_iter) != max_iter or max_iter < 1):
    raise ValueError(f"tol must be positive and max_iter a positive integer, got tol={tol}, max_iter={max_iter}")
  solver = _Newton(int(newton_iters), tol, int(max_iter), newton == "full")
  return discrete_map(f, tab.name, dt, steps, name, lambda model, fname, h: _Stages(tab, model, fname, h, solver).step)


type Residual = Callable[[Expr, Sequence[Expr]], Expr]
"""``(z, params) -> residual``, every quantity besides ``z`` read from ``params``."""


class _Linear:
  """How a Newton step solves with its matrix: ``factor(z, params)`` gives the factors, and
  ``solve(factors, r)`` applies the inverse; by default one ``lu`` of ``matrix(z, params)``."""

  def __init__(self, matrix: Residual | None = None) -> None:
    self.matrix = matrix

  def factor(self, z: Expr, params: Sequence[Expr]) -> list[Expr]:
    assert self.matrix is not None
    return [lu(self.matrix(z, params))]

  def solve(self, factors: Sequence[Expr], r: Expr) -> Expr:
    return lu_solve(factors[0], r)


class _Newton:
  """How a system of stage equations is solved: a fixed number of iterations, or to a tolerance."""

  def __init__(self, iters: int, tol: float | None, max_iter: int, full: bool) -> None:
    self.iters, self.tol, self.max_iter, self.full = iters, tol, max_iter, full

  def run(self, name: str, residual: Residual, linear: _Linear, z0: Expr, params: Sequence[Expr]) -> Expr:
    """``z`` with ``residual(z, params) = 0`` from ``z0``, factoring with ``linear``: once at ``z0``,
    or at every iterate when ``full``. With a tolerance the iterations are a ``while_loop`` whose
    body and condition are Functions over ``params`` (and the factors), named from ``name``; the
    carry holds the iterate and its residual, so each iteration evaluates the residual once."""
    if self.tol is None:
      z, factors = z0, None if self.full else linear.factor(z0, params)
      for _ in range(self.iters):
        z = z - linear.solve(linear.factor(z, params) if factors is None else factors, residual(z, params))
      return z
    m, tol = z0.size, self.tol
    held = [*params, *([] if self.full else linear.factor(z0, params))]
    syms = [Expr.sym(f"q{i}", p.shape, dtype=p.type.dtype) for i, p in enumerate(held)]
    fixed, carry = syms[: len(params)], Expr.sym("zr", (2 * m,))
    z, r = carry[:m], carry[m:]
    znew = z - linear.solve(linear.factor(z, fixed) if self.full else syms[len(params) :], r)
    labels = ["zr", *(f"q{i}" for i in range(len(syms)))]
    body = ConcreteFunction._from_exprs(f"{name}_newton", [carry, *syms], [concat([znew, residual(znew, fixed)])], labels, ["zr_next"])
    cond = ConcreteFunction._from_exprs(f"{name}_unconverged", [carry, *syms], [norm_inf(r) > tol * (1.0 + norm_inf(z))], labels, ["go"])
    out, _ = while_loop(cond, body, concat([z0, residual(z0, params)]), max_iter=self.max_iter, params=held)
    return out[:m]


class _EigenSplit(_Linear):
  """Simplified Newton's matrix ``I - h A (x) J`` split by a real block diagonalization of the stage
  coefficients, ``A = T L T^{-1}`` (Butcher 1976, as RADAU5): in ``W = (T^{-1} (x) I) K`` the system
  is block diagonal, ``I - h g J`` for each real eigenvalue ``g`` and
  ``[[I - h al J, -h be J], [h be J, I - h al J]]`` for each pair ``al +- i be``. Radau IIA with three
  stages factors one system of the state's size and one of twice it, not one of three times it: a
  third of the work, and orders small enough to be straight-line code. A zero eigenvalue (Lobatto
  IIIA's explicit stage) needs no factor."""

  def __init__(
    self, split: tuple[np.ndarray, np.ndarray, list[tuple[float, ...]]], jacobian_at: Residual, h: Callable[[Sequence[Expr]], Expr | float], n: int
  ) -> None:
    super().__init__()
    self.t, self.t_inv, self.blocks = split
    self.jacobian_at, self.h, self.n = jacobian_at, h, n

  def factor(self, z: Expr, params: Sequence[Expr]) -> list[Expr]:
    j, h = self.jacobian_at(z, params), self.h(params)
    eye, factors = Expr.const(np.eye(self.n)), []
    for block in self.blocks:
      if len(block) == 1:
        if block[0] != 0.0:
          factors.append(lu(eye - (h * block[0]) * j))
        continue
      diagonal, off = eye - (h * block[0]) * j, (h * block[1]) * j
      factors.append(lu(concat([concat([diagonal, -off], axis=1), concat([off, diagonal], axis=1)], axis=0)))
    return factors

  def solve(self, factors: Sequence[Expr], r: Expr) -> Expr:
    s, n = self.t.shape[0], self.n
    rw = Expr.const(self.t_inv) @ r.reshape((s, n))
    rows, row, it = [], 0, iter(factors)
    for block in self.blocks:
      if len(block) == 1:
        rows.append(rw[row] if block[0] == 0.0 else lu_solve(next(it), rw[row]))
        row += 1
        continue
      pair = lu_solve(next(it), concat([rw[row], rw[row + 1]]))
      rows += [pair[:n], pair[n:]]
      row += 2
    return (Expr.const(self.t) @ stack(rows)).reshape((s * n,))


def eigen_split(a: np.ndarray) -> tuple[np.ndarray, np.ndarray, list[tuple[float, ...]]] | None:
  """``(T, T^{-1}, blocks)`` with ``a = T L T^{-1}``, ``L`` real block diagonal: a block is ``(g,)``
  for a real eigenvalue or ``(al, be)`` for the pair ``al +- i be`` (``be > 0``), which takes two
  columns of ``T``, the eigenvector's real and imaginary parts.

  ``None`` when ``T`` is ill-conditioned (above ``1e6``), and Newton then factors the whole system:
  the split amplifies rounding by ``T``'s condition number, which stays below ``3e3`` for every
  family up to seven stages, while a defective ``a`` has no eigenvector basis at all and the one
  computed for it is nearly singular (``3e8`` for a 2x2 Jordan block)."""
  values, vectors = np.linalg.eig(a)
  if not np.all(np.isfinite(vectors)) or np.linalg.cond(vectors) > 1e6:
    return None
  columns, blocks = [], []
  for value, vector in zip(values, vectors.T, strict=True):
    if abs(value.imag) < 1e-12:
      columns.append(vector.real / np.abs(vector.real).max())
      blocks.append((float(value.real),))
    elif value.imag > 0:
      columns += [vector.real, vector.imag]
      blocks.append((float(value.real), float(value.imag)))
  t = np.stack(columns, axis=1)
  return t, np.linalg.inv(t), blocks


class _Stages:
  """One method prepared for one model: the stage equations, how Newton solves them, and the
  Function that carries their implicit derivative, built once for every step of the map.

  A step's quantities travel as one list, ``[x, *others, h]`` (``h`` only when it is an input): the
  rules and the loop bodies are Functions over them, and read them back with ``unpack``."""

  def __init__(self, tab: Tableau, model: ConcreteFunction[Any, Any, Any, Any], name: str, h: float | None, solver: _Newton) -> None:
    self.tab, self.model, self.name, self.h, self.solver = tab, model, name, h, solver
    self.n, self.s, self.others = model.inputs[0].size, tab.stages, len(model.inputs) - 1
    self.stage_fn = self._stage_function()

  def unpack(self, held: Sequence[Expr]) -> tuple[Expr, tuple[Expr, ...], Expr | float]:
    x, others = held[0], tuple(held[1 : 1 + self.others])
    return x, others, held[1 + self.others] if self.h is None else self.h

  def split(self, k: Expr) -> list[Expr]:
    return [k[i * self.n : (i + 1) * self.n] for i in range(self.s)]

  def residual(self, k: Expr, held: Sequence[Expr]) -> Expr:
    """``G(K) = K - f(x + h (A (x) I) K)``, stage by stage."""
    x, others, h = self.unpack(held)
    rhs, ks = model_rhs(self.model, others), self.split(k)
    return concat([ks[i] - rhs(increment(x, self.tab.a[i], ks, h)) for i in range(self.s)])

  def stage_matrix(self, k: Expr, held: Sequence[Expr]) -> Expr:
    """``G_K = I - h [a_ij J_i]``, ``J_i`` the model's Jacobian at stage ``i``'s state: one Jacobian
    of the model per stage, not one of the whole residual in ``K``."""
    x, others, h = self.unpack(held)
    rhs, ks = model_rhs(self.model, others), self.split(k)
    states = [increment(x, self.tab.a[i], ks, h) for i in range(self.s)]
    return _block_matrix(self.tab.a, [jacobian(rhs(y), y) for y in states], h)

  def step(self, x: Expr, others: tuple[Expr, ...], h: Expr | float) -> Expr:
    held = [x, *others, *([h] if isinstance(h, Expr) else [])]
    fx = model_rhs(self.model, others)(x)
    kstar = self._dirk(held, fx) if self.tab.diagonally_implicit else self._coupled(held, fx)
    k = self.stage_fn.symbolic_call(tuple([*held, kstar]))
    return increment(x, self.tab.b, self.split(k[0] if isinstance(k, tuple) else k), h)

  def _coupled(self, held: list[Expr], fx: Expr) -> Expr:
    """All ``s`` stages as one system of ``s n`` equations: split by the eigenvalues of ``A`` for
    simplified Newton, factored whole for full Newton, whose stage Jacobians differ."""
    a = self.tab.a

    def start_jacobian(_k: Expr, params: Sequence[Expr]) -> Expr:
      x, others, _ = self.unpack(params)
      return jacobian(model_rhs(self.model, others)(x), x)

    split = None if self.solver.full else eigen_split(a)
    if split is not None:
      linear: _Linear = _EigenSplit(split, start_jacobian, lambda params: self.unpack(params)[2], self.n)
    elif self.solver.full:
      linear = _Linear(self.stage_matrix)
    else:
      linear = _Linear(lambda k, params: _block_matrix(a, [start_jacobian(k, params)] * self.s, self.unpack(params)[2]))
    return self.solver.run(f"{self.name}_stages", self.residual, linear, concat([fx] * self.s), held)

  def _dirk(self, held: list[Expr], fx: Expr) -> Expr:
    """One stage after another: stage ``i`` solves ``z = f(x + h sum_{j<i} a_ij k_j + h a_ii z)``, a
    system of the state's size, the stages before it held as one more parameter."""
    a, ks = self.tab.a, []
    for i in range(self.s):

      def state(z: Expr, params: Sequence[Expr], i: int = i) -> tuple[Expr, tuple[Expr, ...], Expr | float, Expr]:
        x, others, h = self.unpack(params)
        before = increment(x, a[i, :i], self.split(params[-1])[:i], h) if i else x
        return before, others, h, increment(before, [a[i, i]], [z], h)

      def residual(z: Expr, params: Sequence[Expr], i: int = i) -> Expr:
        _, others, _, y = state(z, params, i)
        return z - model_rhs(self.model, others)(y)

      def matrix(z: Expr, params: Sequence[Expr], i: int = i) -> Expr:
        _, others, h, y = state(z, params, i)
        at = y if self.solver.full else self.unpack(params)[0]
        return _block_matrix(a[i : i + 1, i : i + 1], [jacobian(model_rhs(self.model, others)(at), at)], h)

      earlier = [concat([*ks, *([fx] * (self.s - i))])] if i else []
      ks.append(self.solver.run(f"{self.name}_stage{i}", residual, _Linear(matrix), fx, [*held, *earlier]))
    return concat(ks)

  def _stage_function(self) -> ConcreteFunction[Any, Any, Any, Any]:
    """``(x, others..., [h], kstar) -> K``, the identity on ``kstar``, with rules from the implicit
    function theorem at ``G(K; x, others, h) = 0``. The rules never read ``kstar``'s tangent and give
    it a zero cotangent, so neither mode differentiates the Newton iterations that produced it.

    Two levels, as ``SparseLDL.solve``: the level-2 rules find the stages through the level-1
    Function, so a second derivative applies the level-1 rules and is exact; only a third would
    reach the iterations, whose ``lu`` refuses it. Every solve is ``solve(..., assume="gen")``."""
    m = self.n * self.s
    kinds = [(e.shape, e.type.dtype) for e in self.model.inputs] + ([((), dtypes.float64)] if self.h is None else [])
    names = ["x", *(f"a{i}" for i in range(self.others)), *(["h"] if self.h is None else [])]

    def symbols(prefix: str = "") -> list[Expr]:
      return [Expr.sym(f"{prefix}{nm}", shape, dtype=dtype) for nm, (shape, dtype) in zip(names, kinds, strict=True)]

    held, kstar = symbols(), Expr.sym("kstar", (m,))
    base = ConcreteFunction._from_exprs(f"{self.name}_k", [*held, kstar], [kstar + 0.0], [*names, "kstar"], ["k"])

    def jvp_rule(inner: ConcreteFunction[Any, Any, Any, Any], level: int) -> ConcreteFunction[Any, Any, Any, Any]:
      held, kstar, tangents, dkstar = symbols(), Expr.sym("kstar", (m,)), symbols("d"), Expr.sym("dkstar", (m,))
      k = inner.symbolic_call(tuple([*held, kstar]))
      k = k[0] if isinstance(k, tuple) else k
      # G's partial derivatives hold K fixed: differentiate at a free K, then put the stages back. At
      # level 2, k depends on the inputs through the level-1 rules, and differentiating through it
      # would give the total derivative of G(K(x), x), which is zero.
      free = Expr.sym("kfree", (m,))
      g = self.residual(free, held)
      push = substitute(sum((jvp(g, p, t) for p, t in zip(held, tangents, strict=True)), Expr.const(np.zeros(m))), {free: k})
      dk = -solve(self.stage_matrix(k, held), push, assume="gen")
      labels = [*names, "kstar", *(f"d{nm}" for nm in names), "dkstar"]
      return ConcreteFunction._from_exprs(f"{self.name}_k_jvp{level}", [*held, kstar, *tangents, dkstar], [dk], labels, ["dk"])

    # The reverse rule reads the stages from the Function's output, whose derivative is the Function's
    # own, so one rule serves both levels.
    held, kstar, k, kbar = symbols(), Expr.sym("kstar", (m,)), Expr.sym("ko", (m,)), Expr.sym("kbar", (m,))
    lam = solve(self.stage_matrix(k, held).T, kbar, assume="gen")
    grads = [-g for g in vjp([self.residual(k, held)], held, [lam])]
    labels, outs = [*names, "kstar", "ko", "kbar"], [*grads, Expr.const(np.zeros(m))]
    vjp_rule = ConcreteFunction._from_exprs(f"{self.name}_k_vjp", [*held, kstar, k, kbar], outs, labels, [f"{nm}bar" for nm in (*names, "kstar")])
    inner = base
    for level in (1, 2):
      inner = custom_derivative(base, jvp=jvp_rule(inner, level), vjp=vjp_rule)
    return inner


def _block_matrix(a: np.ndarray, jacobians: list[Expr], h: Expr | float) -> Expr:
  """``I - h [a_ij J_i]`` for the stage coefficients ``a``, one Jacobian per block row."""
  n, s = jacobians[0].shape[0], a.shape[0]
  zero = Expr.const(np.zeros((n, n)))
  rows = [concat([float(a[i, j]) * jacobians[i] if a[i, j] != 0 else zero for j in range(s)], axis=1) for i in range(s)]
  blocks = concat(rows, axis=0) if s > 1 else rows[0]
  return Expr.const(np.eye(n * s)) - h * blocks
