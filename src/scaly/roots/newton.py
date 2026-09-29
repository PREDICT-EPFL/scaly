"""``Newton`` on a square system and ``NewtonBisection`` on one unknown in a bracket, as generated loops, and the ``Linear`` solves a Newton step makes."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar

from ..ad.forward import jvp
from ..ad.sparse import sparse_jacobian
from ..function.method import Support
from ..function.model import ConcreteFunction
from ..function.sugar import while_loop
from ..ir.expr import Expr, concat, equal, isfinite, maximum, minimum, norm_inf, stack, substitute, where
from ..linalg.dense import cho_solve, cholesky, lu, lu_solve
from ..linalg.sparse import SparseMatrix
from ..linalg.sparse_factor import SparseLDL
from .implicit import Linear, Residual, jacobian_at
from .method import METHOD_API, Info, status
from .problem import Root

LINEAR = ("lu", "cholesky", "sparse_ldl")
"""The linear solves ``Newton`` makes by name: dense LU, dense Cholesky for a symmetric positive
definite Jacobian, and ``linalg.SparseLDL`` on the compact sparse Jacobian for a symmetric one."""


@dataclass(frozen=True)
class _Dense:
  """The dense Jacobian of ``residual``, factored by LU or, ``cholesky``, by Cholesky."""

  residual: Residual
  cholesky: bool = False

  def factor(self, z: Expr, params: Sequence[Expr]) -> list[Expr]:
    j = jacobian_at(self.residual, z, params)
    return [cholesky(j) if self.cholesky else lu(j)]

  def solve(self, factors: Sequence[Expr], r: Expr) -> Expr:
    return cho_solve(factors[0], r) if self.cholesky else lu_solve(factors[0], r)


@dataclass(frozen=True)
class _SparseLDL:
  """The compact sparse Jacobian of ``residual``, symmetric, factored by ``linalg.SparseLDL``."""

  residual: Residual
  name: str

  def factor(self, z: Expr, params: Sequence[Expr]) -> SparseLDL:
    return SparseLDL(SparseMatrix.from_sparse_jacobian(sparse_jacobian(self.residual(z, params), z)), name=self.name)

  def solve(self, factors: SparseLDL, r: Expr) -> Expr:
    return factors.solve(r)


def _positive_int(value: Any, what: str) -> None:
  if int(value) != value or value < 1:
    raise ValueError(f"{what} must be a positive integer, got {value!r}")


@dataclass(frozen=True)
class Newton:
  """Newton's method for ``F(z; p) = 0``: ``z <- z - F_z^{-1} F(z)`` from the warm start.

  ``tol`` stops the iteration once ``|F|_inf <= tol + rtol |z|_inf``, at most ``max_iter`` times, in a
  ``while_loop``; ``tol=None`` takes exactly ``max_iter`` iterations as straight-line code, a fixed
  amount of work, the usual choice for control. A relative ``rtol`` suits residuals measured in the
  unknowns' units (an implicit integrator's stages take ``rtol = tol``); it is 0 by default, since a
  bounded residual would otherwise pass the test as an iterate diverges. ``linear`` names the solve (``LINEAR``), and
  ``simplified`` factors once, at the warm start, instead of at every iterate. ``max_step`` caps
  each step's largest entry, a damped Newton for a start far from the root.

  The solution's derivative is the implicit function theorem's at the root found, never the
  iterations'. The status is ``OK`` when the test passes (with no ``tol``, when the residual is
  finite), ``MAX_ITER`` when the bound stopped it, ``NUMERICS`` for a residual that is not finite."""

  name: ClassVar[str] = "roots.newton"
  problem: ClassVar[type] = Root
  api: ClassVar[int] = METHOD_API

  tol: float | None = 1e-10
  max_iter: int = 50
  linear: str = "lu"
  rtol: float = 0.0
  simplified: bool = False
  max_step: float | None = None

  def __post_init__(self) -> None:
    if self.linear not in LINEAR:
      raise ValueError(f"linear must be one of {LINEAR}, got {self.linear!r}")
    if self.tol is not None and not self.tol > 0:
      raise ValueError(f"tol must be positive or None, got {self.tol}")
    if not self.rtol >= 0:
      raise ValueError(f"rtol must be nonnegative, got {self.rtol}")
    _positive_int(self.max_iter, "max_iter")
    if self.max_step is not None and not self.max_step > 0:
      raise ValueError(f"max_step must be positive or None, got {self.max_step}")
    if self.linear == "sparse_ldl" and self.simplified and self.tol is not None:
      raise ValueError("simplified Newton with a tol passes its factors into the loop, which a SparseLDL's cannot be; use tol=None or full Newton")

  def supports(self, problem: Any) -> Support:
    if not isinstance(problem, Root):
      return Support((f"{type(problem).__name__} is not a Root",))
    if problem.lb is not None or problem.ub is not None:
      return Support(("its unknowns are bounded, which Newton would ignore; NewtonBisection brackets one unknown",))
    return Support()

  def build(self, problem: Root[Any, Any, Any, Any], *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    # A symmetric solve serves the implicit derivative as well: a sparse problem's stays sparse.
    ift = self.linear_solve(problem.residual, f"{name}_root") if self.linear != "lu" else None
    return problem.function(lambda p, z0, params, fname: self.iterate(p.residual, z0, params, name=fname), name=name, linear=ift)

  def linear_solve(self, residual: Residual, name: str) -> Linear:
    """The ``Linear`` that ``linear`` names, for ``residual``."""
    if self.linear == "sparse_ldl":
      return _SparseLDL(residual, f"{name}_ldl")
    return _Dense(residual, cholesky=self.linear == "cholesky")

  def _threshold(self, z: Expr) -> Expr | float:
    """``tol + rtol |z|_inf``, written ``tol (1 + |z|_inf)`` when the two are equal."""
    assert self.tol is not None
    if self.rtol == self.tol:
      return self.tol * (1.0 + norm_inf(z))
    return self.tol if self.rtol == 0.0 else self.tol + self.rtol * norm_inf(z)

  def _damped(self, step: Expr) -> Expr:
    return step if self.max_step is None else step * minimum(1.0, self.max_step / norm_inf(step))

  def iterate(self, residual: Residual, z0: Expr, params: Sequence[Expr], *, name: str, linear: Linear | None = None) -> tuple[Expr, Info]:
    """``z`` with ``residual(z, params) = 0`` from ``z0``, and its ``Info``, as expressions; ``linear``
    replaces the solve ``linear`` names (an integrator's structured stage matrix, say). With a
    tolerance the iterations are a ``while_loop`` whose body and condition are Functions over
    ``params`` (and the factors), named from ``name``; the carry holds the iterate and its residual,
    so each iteration evaluates the residual once. An ``Info`` the caller does not use costs nothing."""
    solver = linear if linear is not None else self.linear_solve(residual, name)
    params = list(params)
    if self.tol is None:
      z, factors = z0, solver.factor(z0, params) if self.simplified else None
      for _ in range(self.max_iter):
        z = z - self._damped(solver.solve(solver.factor(z, params) if factors is None else factors, residual(z, params)))
      r = norm_inf(residual(z, params))
      return z, Info(status=status(isfinite(r), z, r), iter=Expr.const(float(self.max_iter)), residual=r)
    m = z0.size
    held = [*params, *(solver.factor(z0, params) if self.simplified else [])]
    syms = [Expr.sym(f"q{i}", p.shape, dtype=p.type.dtype) for i, p in enumerate(held)]
    fixed, carry = syms[: len(params)], Expr.sym("zr", (2 * m,))
    z, r = carry[:m], carry[m:]
    znew = z - self._damped(solver.solve(syms[len(params) :] if self.simplified else solver.factor(z, fixed), r))
    labels = ["zr", *(f"q{i}" for i in range(len(syms)))]
    body = ConcreteFunction.from_exprs(f"{name}_newton", [carry, *syms], [concat([znew, residual(znew, fixed)])], labels, ["zr_next"])
    cond = ConcreteFunction.from_exprs(f"{name}_unconverged", [carry, *syms], [norm_inf(r) > self._threshold(z)], labels, ["go"])
    out, n_iter = while_loop(cond, body, concat([z0, residual(z0, params)]), max_iter=self.max_iter, params=held)
    z, res = out[:m], norm_inf(out[m:])
    return z, Info(status=status(res <= self._threshold(z), z, res), iter=n_iter, residual=res)


def _slope(residual: Callable[[Expr, Sequence[Expr]], Expr]) -> Callable[[Expr, Sequence[Expr]], Expr]:
  def slope(x: Expr, params: Sequence[Expr]) -> Expr:
    free = Expr.sym("xfree", ())
    return substitute(jvp(residual(free, params), free, Expr.const(1.0)), {free: x})

  return slope


@dataclass(frozen=True)
class NewtonBisection:
  """Newton's method on one unknown in a bracket ``[lb, ub]`` where the residual changes sign,
  increasing through its root (``increasing=False`` for a decreasing one). Each iteration shrinks the
  bracket by the residual's sign and takes the Newton step when the slope is positive and the step
  stays inside, the bracket's midpoint otherwise, so it never leaves the bracket and never stalls.

  It stops once a step moves the unknown by no more than ``tol`` times the larger of its magnitude
  and the bracket's first width, once the bracket closes to that, or at a zero residual or a Newton
  step too small to move the unknown, at most ``max_iter`` times. Both tests are relative, so neither the residual's scale nor the unknown's
  changes when it stops. The problem's ``RootSpec`` gives the bracket."""

  name: ClassVar[str] = "roots.newton_bisection"
  problem: ClassVar[type] = Root
  api: ClassVar[int] = METHOD_API

  tol: float = 1e-14
  max_iter: int = 60
  increasing: bool = True

  def __post_init__(self) -> None:
    if not self.tol > 0:
      raise ValueError(f"tol must be positive, got {self.tol}")
    _positive_int(self.max_iter, "max_iter")

  def supports(self, problem: Any) -> Support:
    if not isinstance(problem, Root):
      return Support((f"{type(problem).__name__} is not a Root",))
    return Support.unless(
      more_than_one_unknown=problem.n != 1, no_lower_bound_to_bracket_it=problem.lb is None, no_upper_bound_to_bracket_it=problem.ub is None
    )

  def build(self, problem: Root[Any, Any, Any, Any], *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    def iterate(p: Root[Any, Any, Any, Any], z0: Expr, params: list[Expr], fname: str) -> tuple[Expr, Info]:
      lo, hi = p.bounds(params)
      assert lo is not None and hi is not None
      x, info = self.iterate(lambda x, q: p.residual(x.reshape((1,)), q)[0], z0[0], lo[0], hi[0], params, name=fname)
      return x.reshape((1,)), info

    return problem.function(iterate, name=name)

  def iterate(
    self,
    residual: Callable[[Expr, Sequence[Expr]], Expr],
    x0: Expr,
    lo: Expr,
    hi: Expr,
    params: Sequence[Expr],
    *,
    name: str,
    names: Sequence[str] | None = None,
    slope: Callable[[Expr, Sequence[Expr]], Expr] | None = None,
  ) -> tuple[Expr, Info]:
    """The root ``x`` of the scalar ``residual(x, params)`` in ``[lo, hi]`` from ``x0`` inside it, and
    its ``Info``, as expressions. ``slope(x, params)`` is the residual's derivative (by default
    forward mode); ``names`` label ``params`` in the loop's condition (``{name}_go``) and step
    (``{name}_step``), whose carry is ``(x, lo, hi, last step, first width)``."""
    params = list(params)
    labels = list(names) if names is not None else [f"p{i}" for i in range(len(params))]
    sign = 1.0 if self.increasing else -1.0
    dres = slope if slope is not None else _slope(residual)
    carry, syms = Expr.sym("c", (5,)), [Expr.sym(nm, p.shape, dtype=p.type.dtype) for nm, p in zip(labels, params, strict=True)]

    def going(c: Expr) -> Expr:
      x, lo, hi, last, scale = c[0], c[1], c[2], c[3], c[4]
      small = self.tol * maximum(x.abs(), scale)
      return (last.abs() > small) & (hi - lo > small)

    x, lo_c, hi_c, scale = carry[0], carry[1], carry[2], carry[4]
    r, d = residual(x, syms), dres(x, syms)
    if sign < 0:
      r, d = -r, -d
    lo_n, hi_n = where(r < 0.0, x, lo_c), where(r > 0.0, x, hi_c)
    step = x - r / d
    inside = (d > 0.0) & (step > lo_n) & (step < hi_n)
    # A step too small to move x (below half an ulp) is converged, though x is then an end of the
    # bracket and the step not strictly inside it: bisecting would throw the root away.
    nxt = where(equal(r, 0.0) | equal(step, x), x, where(inside, step, 0.5 * (lo_n + hi_n)))
    body = ConcreteFunction.from_exprs(f"{name}_step", [carry, *syms], [stack([nxt, lo_n, hi_n, nxt - x, scale])], ["c", *labels], ["c_next"])
    cond = ConcreteFunction.from_exprs(f"{name}_go", [carry, *syms], [going(carry)], ["c", *labels], ["go"])
    out, n_iter = while_loop(cond, body, stack([x0, lo, hi, hi - lo, hi - lo]), max_iter=self.max_iter, params=params)
    res = residual(out[0], params).abs()
    return out[0], Info(status=status(~going(out), out[0], res), iter=n_iter, residual=res)


__all__ = ["LINEAR", "Linear", "Newton", "NewtonBisection"]
