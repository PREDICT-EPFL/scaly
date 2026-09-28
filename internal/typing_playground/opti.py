"""Problems and solvers: a ``Problem`` of expressions over declared trees, and ``solver`` as a plain ``Function``.

The real ``solver`` builds the descriptor and ``SOLVER_CALL`` nodes as ``solvers/nlp.py`` and
``solvers/qp.py`` do; here its body returns its inputs. The QP proof must be written on
``ad/sparsity.py``'s ``_jac_mask`` over the real gradient and Hessian expressions, not on the
``degree`` counter that stands in for it here.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from .expr import Buffer, Expr
from .function import Function
from .trees import G, L, Tree, leaves


@dataclass(frozen=True)
class Bounded:
  """``lo <= expr <= hi``; ``None`` is an open side. ``name`` labels the group in diagnostics."""

  expr: Expr
  lo: Expr | float | None = None
  hi: Expr | float | None = None
  name: str | None = None


def bounded(expr: Expr, lo: Expr | float | None = None, hi: Expr | float | None = None, *, name: str | None = None) -> Bounded:
  if lo is None and hi is None:
    raise ValueError("bounded needs at least one of lo / hi")
  return Bounded(expr, lo, hi, name)


@dataclass(frozen=True)
class ProblemSpec[SymbolicVars]:
  """What a problem body returns. Every field is an expression over the declared vars and params.
  ``lb``/``ub`` have the variables' structure (a leaf may be a constant ``Expr``)."""

  minimize: Expr
  eq: tuple[Expr, ...] = ()
  ineq: tuple[Bounded, ...] = ()
  lb: SymbolicVars | None = None
  ub: SymbolicVars | None = None


@dataclass(frozen=True)
class Problem[SV, NV, SP, NP]:
  """A backend-free problem: the spec plus the two trees it was traced over."""

  name: str
  spec: ProblemSpec[SV]
  vars: Tree[SV, NV]
  params: Tree[SP, NP]

  @property
  def n_eq(self) -> int:
    return sum(e.size for e in self.spec.eq)

  @property
  def n_ineq(self) -> int:
    return sum(b.expr.size for b in self.spec.ineq)


def problem[SV, NV, SP, NP](
  *, vars: Tree[SV, NV], params: Tree[SP, NP], name: str | None = None
) -> Callable[[Callable[[SV, SP], ProblemSpec[SV]]], Problem[SV, NV, SP, NP]]:
  def decorate(fn: Callable[[SV, SP], ProblemSpec[SV]]) -> Problem[SV, NV, SP, NP]:
    spec = fn(vars.symbols(degree=1), params.symbols(degree=0))
    if spec.minimize.shape != ():
      raise TypeError(f"cost must be scalar, got shape {spec.minimize.shape}")
    for e in spec.eq:
      if len(e.shape) > 1:
        raise TypeError(f"equality constraints must be rank-1, got shape {e.shape}")
    for b in spec.ineq:
      if len(b.expr.shape) > 1:
        raise TypeError(f"inequality constraints must be rank-1, got shape {b.expr.shape}")
    for side, bound in (("lb", spec.lb), ("ub", spec.ub)):
      if bound is not None and len(leaves(bound)) != vars.size:
        raise TypeError(f"{side} must have the variables' structure {vars.names}")
    return Problem(name or getattr(fn, "__name__", "problem"), spec, vars, params)

  return decorate


BACKENDS: dict[str, tuple[Literal["qp", "nlp"], Literal["lower", "upper"]]] = {
  "piqp": ("qp", "upper"),
  "ipopt": ("nlp", "lower"),
  "sqp": ("nlp", "upper"),
}


class NotQuadratic(ValueError):
  pass


def _prove_quadratic(p: Problem[Any, Any, Any, Any]) -> None:
  """The QP gate: cost quadratic, constraints affine in the variables. Structural, so conservative."""
  if p.spec.minimize.degree is None or p.spec.minimize.degree > 2:
    raise NotQuadratic(f"{p.name}: cost has degree {p.spec.minimize.degree} in the variables, a QP backend needs at most 2")
  for i, e in enumerate(p.spec.eq):
    if e.degree is None or e.degree > 1:
      raise NotQuadratic(f"{p.name}: eq[{i}] has degree {e.degree}, a QP backend needs affine constraints")
  for b in p.spec.ineq:
    if b.expr.degree is None or b.expr.degree > 1:
      raise NotQuadratic(f"{p.name}: ineq {b.name or '?'} has degree {b.expr.degree}, a QP backend needs affine constraints")


def solver[SV, NV, SP, NP](
  p: Problem[SV, NV, SP, NP], backend: str, /, *, name: str | None = None, options: dict[str, Any] | None = None
) -> Function[tuple[SV, SV, Expr, Expr, SP], tuple[NV, NV, Buffer, Buffer, NP], tuple[SV, SV, Expr, Expr], tuple[NV, NV, Buffer, Buffer]]:
  """Build a solver for ``p``: a plain ``Function`` over ``(vars_init, lam_box0, lam_eq0, lam_ineq0, params)``.
  Multipliers are always present, size 0 when the category is absent, so the signature never
  depends on the problem; box multipliers have the variables' structure."""
  if backend not in BACKENDS:
    raise ValueError(f"unknown backend {backend!r}; available {sorted(BACKENDS)}")
  kind, _triangle = BACKENDS[backend]
  if kind == "qp":
    _prove_quadratic(p)
  lam_box = p.vars.relabel("lam:")
  inputs = G(p.vars, lam_box, L("lam_eq", p.n_eq), L("lam_ineq", p.n_ineq), p.params)
  outputs = G(p.vars, lam_box, L("lam_eq", p.n_eq), L("lam_ineq", p.n_ineq))

  def body(args: tuple[SV, SV, Expr, Expr, SP]) -> tuple[SV, SV, Expr, Expr]:
    vars_init, lam_box0, lam_eq0, lam_ineq0, _params = args
    return vars_init, lam_box0, lam_eq0, lam_ineq0

  return Function(name or f"{p.name}_{backend}", body, inputs, outputs)


type QPData[T] = tuple[tuple[T, T], tuple[T, T], tuple[T, T, T]]


def qp_problem(n: int, n_eq: int, n_ineq: int) -> Problem[Expr, Buffer, QPData[Expr], QPData[Buffer]]:
  """The typed data form: ``min 0.5 x'Px + c'x  s.t.  A x = b,  g_lb <= G x <= g_ub``. Every block is a
  parameter, so a consumer updates the QP data per call; an absent block has size 0."""

  @problem(
    vars=L("x", n),
    params=G(G(L("P", (n, n)), L("c", n)), G(L("A", (n_eq, n)), L("b", n_eq)), G(L("G", (n_ineq, n)), L("g_lb", n_ineq), L("g_ub", n_ineq))),
    name="qp",
  )
  def qp(x: Expr, params: QPData[Expr]) -> ProblemSpec[Expr]:
    (P, c), (A, b), (Gm, g_lb, g_ub) = params
    return ProblemSpec(minimize=0.5 * (x @ P @ x) + c @ x, eq=(A @ x - b,), ineq=(bounded(Gm @ x, g_lb, g_ub, name="g"),))

  return qp
