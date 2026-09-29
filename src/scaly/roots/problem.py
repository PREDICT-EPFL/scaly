"""``Root`` and ``LeastSquares``: equations over typed unknowns and parameters, and the Function that solves one with a method's iteration and the implicit derivative."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, overload

import numpy as np

from ..ad.derivatives import gradient
from ..function.model import ConcreteFunction
from ..function.tree import G, Tree, param_list
from ..ir.expr import Expr, ExprOp, as_expr, concat, substitute, sumsqr, topo
from ..ir.types import TensorType
from .implicit import Linear, Residual, custom_root
from .method import METHOD_API, Info

type Iterate = Callable[[Any, Expr, list[Expr], str], tuple[Expr, Info]]
"""A method's iteration: ``(problem, z0, params, name) -> (z, info)`` over the flat unknowns."""


@dataclass(frozen=True, slots=True)
class RootSpec[SymbolicVars]:
  """What a ``root`` body returns when the unknowns are bounded: the ``residual`` (an expression or a
  tuple of them, flattened and joined), and ``lb`` and ``ub`` with the unknowns' structure, each an
  expression of the parameters or ``None``. A bracketing method needs both."""

  residual: Expr | tuple[Expr, ...]
  lb: SymbolicVars | None = None
  ub: SymbolicVars | None = None


@dataclass(frozen=True)
class _Equations[SymbolicVars, NumericalVars, SymbolicParams, NumericalParams]:
  """Equations in the unknowns ``vars`` and the parameters ``params``, the residual one vector."""

  method_api: ClassVar[int] = METHOD_API

  name: str
  residual_expr: Expr
  vars: Tree[SymbolicVars, NumericalVars]
  params: Tree[SymbolicParams, NumericalParams] | None
  lb: Expr | None
  ub: Expr | None
  _var_symbols: tuple[Expr, ...]
  _param_symbols: tuple[Expr, ...]

  @property
  def n(self) -> int:
    """The number of unknowns, every leaf flattened."""
    return sum(e.size for e in self._var_symbols)

  @property
  def m(self) -> int:
    """The number of equations, or residuals."""
    return self.residual_expr.size

  def _swap(self, z: Expr, params: Sequence[Expr]) -> dict[Expr, Expr]:
    offsets = np.cumsum([0, *(e.size for e in self._var_symbols)])
    swap = {v: z[a:b].reshape(v.shape) for v, a, b in zip(self._var_symbols, offsets[:-1], offsets[1:], strict=True)}
    return {**swap, **dict(zip(self._param_symbols, params, strict=True))}

  def residual(self, z: Expr, params: Sequence[Expr]) -> Expr:
    """The residual at the flat unknowns ``z`` and the parameters' leaves ``params``."""
    return substitute(self.residual_expr, self._swap(z, params))

  def bounds(self, params: Sequence[Expr]) -> tuple[Expr | None, Expr | None]:
    """The flat bounds on the unknowns at the parameters' leaves, ``None`` for a side not given."""
    swap = dict(zip(self._param_symbols, params, strict=True))
    return tuple(None if b is None else substitute(b, swap) for b in (self.lb, self.ub))  # ty: ignore[invalid-return-type]

  def _implicit(self) -> tuple[Residual, str]:
    return self.residual, "root"

  def function(self, iterate: Iterate, *, name: str, linear: Linear | None = None) -> ConcreteFunction[Any, Any, Any, Any]:
    """The Function that solves these equations with ``iterate``: the unknowns' warm start and the
    parameters in, the solution and an ``Info`` out. The solution's derivative is the implicit
    function theorem's at the point found (``custom_root``, solving with ``linear`` when the method
    gives a symmetric one), never the iterations'; the warm start has none."""
    n = self.n
    names = self.params.names if self.params is not None else ()
    residual, kind = self._implicit()
    root = custom_root(
      residual, Expr.sym("z", (n,)), list(self._param_symbols), name=f"{name}_{kind}", names=[f"p{i}" for i in range(len(names))], linear=linear
    )
    warm = self.vars.with_types(tuple(TensorType(e.shape, e.type.dtype, e.type.sparsity, diff=False) for e in self._var_symbols))
    shapes, sizes = [e.shape for e in self._var_symbols], [e.size for e in self._var_symbols]
    offsets = np.cumsum([0, *sizes])

    def body(z0_tree: Any, *rest: Any) -> Any:
      leaves = warm.flatten_symbolic(z0_tree, f"{name} warm start")
      z0 = concat([leaf.reshape((leaf.size,)) for leaf in leaves]) if len(leaves) > 1 else leaves[0].reshape((n,))
      params = list(self.params.flatten_symbolic(rest[0], f"{name} parameters")) if self.params is not None else []
      zstar, info = iterate(self, z0, params, name)
      out = root.symbolic_call(tuple([*params, zstar]) if params else zstar)
      z = out[0] if isinstance(out, tuple) else out
      solution = self.vars.unflatten(tuple(z[a:b].reshape(s) for a, b, s in zip(offsets[:-1], offsets[1:], shapes, strict=True)))
      return solution, info

    inputs = param_list(warm, self.params) if self.params is not None else param_list(warm)
    return ConcreteFunction(name, body, inputs, G(self.vars, Info.tree()))


@dataclass(frozen=True)
class Root[SymbolicVars, NumericalVars, SymbolicParams, NumericalParams](_Equations[SymbolicVars, NumericalVars, SymbolicParams, NumericalParams]):
  """``F(z; p) = 0``: as many equations as unknowns, from ``sc.roots.root``. A solver's solution is
  differentiable in the parameters by the implicit function theorem, ``dz = -F_z^{-1} F_p dp``,
  wherever ``F_z`` is nonsingular."""


@dataclass(frozen=True)
class LeastSquares[SymbolicVars, NumericalVars, SymbolicParams, NumericalParams](
  _Equations[SymbolicVars, NumericalVars, SymbolicParams, NumericalParams]
):
  """``minimize 1/2 |r(z; p)|^2`` over the unknowns, at least as many residuals as unknowns, from
  ``sc.roots.least_squares``. A solver's solution is differentiable in the parameters through the
  stationarity condition ``J^T r = 0``, by the implicit function theorem with the full Hessian of
  ``1/2 |r|^2``, wherever it is nonsingular."""

  def stationarity(self, z: Expr, params: Sequence[Expr]) -> Expr:
    """The gradient ``J^T r`` of ``1/2 |r|^2`` at the flat unknowns, zero at a local minimum."""
    return gradient(0.5 * sumsqr(self.residual(z, params)), z).reshape((z.size,))

  def _implicit(self) -> tuple[Residual, str]:
    return self.stationarity, "stationary"


def _trace(
  fn: Callable[..., Any], vars: Tree[Any, Any], params: Tree[Any, Any] | None, name: str | None, *, bounds: bool
) -> tuple[str, Expr, Tree[Any, Any], Tree[Any, Any] | None, Expr | None, Expr | None, tuple[Expr, ...], tuple[Expr, ...]]:
  problem_name = name or getattr(fn, "__name__", "equations")
  symbolic_vars = vars.symbols(diff=True)
  var_exprs = vars.flatten_symbolic(symbolic_vars, f"{problem_name} unknowns")
  resolved_vars = vars.with_types(tuple(e.type for e in var_exprs))
  param_exprs: tuple[Expr, ...] = ()
  resolved_params = None
  if params is not None:
    symbolic_params = params.symbols(diff=True)
    param_exprs = params.flatten_symbolic(symbolic_params, f"{problem_name} parameters")
    resolved_params = params.with_types(tuple(e.type for e in param_exprs))
    out = fn(symbolic_vars, symbolic_params)
  else:
    out = fn(symbolic_vars)
  spec = out if isinstance(out, RootSpec) else RootSpec(out)
  if not bounds and (spec.lb is not None or spec.ub is not None):
    raise TypeError(f"{problem_name}: least squares takes no bounds")
  parts = spec.residual if isinstance(spec.residual, tuple) else (spec.residual,)
  residual = (
    concat([as_expr(e).reshape((as_expr(e).size,)) for e in parts]) if len(parts) > 1 else as_expr(parts[0]).reshape((as_expr(parts[0]).size,))
  )
  sides: list[Expr | None] = []
  for side, value in (("lb", spec.lb), ("ub", spec.ub)):
    if value is None:
      sides.append(None)
      continue
    leaves = vars.flatten_symbolic(value, f"{problem_name} {side}", allow_scalar=True)
    sides.append(
      concat([(as_expr(leaf) + Expr.const(np.zeros(t.shape))).reshape((t.size,)) for leaf, t in zip(leaves, resolved_vars.types, strict=True)])
    )
  declared = {e.id for e in (*var_exprs, *param_exprs)}
  exprs = [residual, *(s for s in sides if s is not None)]
  undeclared = sorted({node.name or f"%{node.id}" for node in topo(exprs) if node.op == ExprOp.INPUT and node.id not in declared})
  if undeclared:
    raise ValueError(f"{problem_name!r} reads symbols it does not declare: {undeclared}; declare them as params")
  if any(s is not None and any(node.id in {v.id for v in var_exprs} for node in topo([s])) for s in sides):
    raise ValueError(f"{problem_name!r}: the bounds must not depend on the unknowns")
  return problem_name, residual, resolved_vars, resolved_params, sides[0], sides[1], var_exprs, param_exprs


@overload
def root[SV, NV, SP, NP](
  *, vars: Tree[SV, NV], params: Tree[SP, NP], name: str | None = None
) -> Callable[[Callable[[SV, SP], Any]], Root[SV, NV, SP, NP]]: ...


@overload
def root[SV, NV](
  *, vars: Tree[SV, NV], params: None = None, name: str | None = None
) -> Callable[[Callable[[SV], Any]], Root[SV, NV, None, None]]: ...


def root(
  *, vars: Tree[Any, Any], params: Tree[Any, Any] | None = None, name: str | None = None
) -> Callable[[Callable[..., Any]], Root[Any, Any, Any, Any]]:
  """Trace a system of equations ``F(z; p) = 0`` over the declared unknowns ``vars`` and parameters
  ``params`` (or none). The body returns the residual, an expression or a tuple of them, or a
  ``RootSpec`` that also bounds the unknowns; the residuals, flattened, must be as many as the
  unknowns. Everything the body reads besides the two must be a constant."""

  def decorate(fn: Callable[..., Any]) -> Root[Any, Any, Any, Any]:
    traced = _trace(fn, vars, params, name, bounds=True)
    problem = Root(*traced)
    if problem.m != problem.n:
      raise ValueError(f"{problem.name!r} has {problem.m} equations in {problem.n} unknowns; a root needs as many of each")
    return problem

  return decorate


@overload
def least_squares[SV, NV, SP, NP](
  *, vars: Tree[SV, NV], params: Tree[SP, NP], name: str | None = None
) -> Callable[[Callable[[SV, SP], Any]], LeastSquares[SV, NV, SP, NP]]: ...


@overload
def least_squares[SV, NV](
  *, vars: Tree[SV, NV], params: None = None, name: str | None = None
) -> Callable[[Callable[[SV], Any]], LeastSquares[SV, NV, None, None]]: ...


def least_squares(
  *, vars: Tree[Any, Any], params: Tree[Any, Any] | None = None, name: str | None = None
) -> Callable[[Callable[..., Any]], LeastSquares[Any, Any, Any, Any]]:
  """Trace ``minimize 1/2 |r(z; p)|^2`` over the declared unknowns and parameters: the body returns
  the residual ``r``, an expression or a tuple of them, at least as many entries as unknowns."""

  def decorate(fn: Callable[..., Any]) -> LeastSquares[Any, Any, Any, Any]:
    problem = LeastSquares(*_trace(fn, vars, params, name, bounds=False))
    if problem.m < problem.n:
      raise ValueError(f"{problem.name!r} has {problem.m} residuals in {problem.n} unknowns; least squares needs at least as many residuals")
    return problem

  return decorate


__all__ = ["LeastSquares", "Root", "RootSpec", "least_squares", "root"]
