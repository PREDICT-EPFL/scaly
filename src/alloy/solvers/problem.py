"""Typed backend-free optimal-control problem declarations."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast, overload

from ..function.tree import Tree, flat_tree
from ..ir.expr import Expr, as_expr, substitute
from ..ir.types import TensorType
from ._oracle import collect_free_inputs


@dataclass(frozen=True, slots=True)
class Bounded:
  """A named constraint group with an optional lower and upper bound."""

  expr: Expr
  lo: Expr | None = None
  hi: Expr | None = None
  name: str | None = None


def bounded(expr: Expr, lo: Any = None, hi: Any = None, *, name: str | None = None) -> Bounded:
  """Declare a lower- and/or upper-bounded inequality group."""
  if lo is None and hi is None:
    raise ValueError("bounded needs at least one of lo / hi")
  return Bounded(as_expr(expr), None if lo is None else as_expr(lo), None if hi is None else as_expr(hi), name)


@dataclass(frozen=True, slots=True)
class ProblemSpec[SymbolicVars]:
  """The objective, constraint groups, and box bounds returned by a problem body."""

  minimize: Expr
  eq: tuple[Expr, ...] = ()
  ineq: tuple[Bounded, ...] = ()
  lb: SymbolicVars | None = None
  ub: SymbolicVars | None = None


@dataclass(frozen=True)
class Problem[SymbolicVars, NumericalVars, SymbolicParams, NumericalParams]:
  """A traced backend-free problem and its declared variable and parameter trees."""

  name: str
  spec: ProblemSpec[SymbolicVars]
  vars: Tree[SymbolicVars, NumericalVars]
  params: Tree[SymbolicParams, NumericalParams]
  _var_symbols: tuple[Expr, ...] = field(repr=False)
  _param_symbols: tuple[Expr, ...] = field(repr=False)
  _cache: dict[str, Any] = field(default_factory=dict, repr=False, compare=False)

  @property
  def n_eq(self) -> int:
    """The total number of scalar equality constraints."""
    return sum(expr.size for expr in self.spec.eq)

  @property
  def n_ineq(self) -> int:
    """The total number of scalar bounded inequality constraints."""
    return sum(group.expr.size for group in self.spec.ineq)


def _normalize_spec[SV](spec: ProblemSpec[SV], vars: Tree[SV, Any]) -> ProblemSpec[SV]:
  if not isinstance(spec, ProblemSpec):
    raise TypeError(f"problem body must return ProblemSpec, got {type(spec).__name__}")
  minimize = as_expr(spec.minimize)
  if minimize.size != 1:
    raise TypeError(f"cost must be scalar, got shape {minimize.shape}")
  if minimize.shape:
    minimize = minimize.reshape(())

  eq = tuple(as_expr(expr) for expr in spec.eq)
  for expr in eq:
    if len(expr.shape) > 1:
      raise TypeError(f"equality constraints must be scalar or rank-1, got shape {expr.shape}")

  ineq = tuple(bounded(group.expr, group.lo, group.hi, name=group.name) for group in spec.ineq)
  for group in ineq:
    if len(group.expr.shape) > 1:
      raise TypeError(f"inequality constraints must be scalar or rank-1, got shape {group.expr.shape}")

  for side, value in (("lb", spec.lb), ("ub", spec.ub)):
    if value is not None:
      try:
        vars.flatten_symbolic(value, f"problem {side}")
      except ValueError as exc:
        raise TypeError(str(exc)) from exc

  return ProblemSpec(minimize, eq, ineq, spec.lb, spec.ub)


def _spec_exprs[SV](spec: ProblemSpec[SV], vars: Tree[SV, Any]) -> tuple[Expr, ...]:
  exprs: list[Expr] = [spec.minimize, *spec.eq]
  for group in spec.ineq:
    exprs.append(group.expr)
    if group.lo is not None:
      exprs.append(group.lo)
    if group.hi is not None:
      exprs.append(group.hi)
  if spec.lb is not None:
    exprs.extend(vars.flatten_symbolic(spec.lb, "problem lb"))
  if spec.ub is not None:
    exprs.extend(vars.flatten_symbolic(spec.ub, "problem ub"))
  return tuple(exprs)


def _substitute_spec[SV](spec: ProblemSpec[SV], vars: Tree[SV, Any], replacements: dict[Expr, Expr]) -> ProblemSpec[SV]:
  def sub(expr: Expr | None) -> Expr | None:
    return None if expr is None else substitute(expr, replacements)

  ineq = tuple(Bounded(cast(Expr, sub(group.expr)), sub(group.lo), sub(group.hi), group.name) for group in spec.ineq)
  lb = spec.lb
  if lb is not None:
    lb = cast(SV, vars.unflatten(tuple(sub(expr) for expr in vars.flatten_symbolic(lb, "problem lb"))))
  ub = spec.ub
  if ub is not None:
    ub = cast(SV, vars.unflatten(tuple(sub(expr) for expr in vars.flatten_symbolic(ub, "problem ub"))))
  return ProblemSpec(
    cast(Expr, sub(spec.minimize)),
    tuple(cast(Expr, sub(expr)) for expr in spec.eq),
    ineq,
    lb,
    ub,
  )


@overload
def problem[SV, NV, SP, NP](
  *, vars: Tree[SV, NV], params: Tree[SP, NP], name: str | None = None
) -> Callable[[Callable[[SV, SP], ProblemSpec[SV]]], Problem[SV, NV, SP, NP]]: ...


@overload
def problem[SV, NV](
  *, vars: Tree[SV, NV], params: None = None, name: str | None = None
) -> Callable[[Callable[[SV], ProblemSpec[SV]]], Problem[SV, NV, Any, Any]]: ...


def problem(
  *, vars: Tree[Any, Any], params: Tree[Any, Any] | None = None, name: str | None = None
) -> Callable[[Callable[..., ProblemSpec[Any]]], Problem[Any, Any, Any, Any]]:
  """Trace a backend-free problem over declared variables and parameters."""

  def decorate(fn: Callable[..., ProblemSpec[Any]]) -> Problem[Any, Any, Any, Any]:
    problem_name = name or getattr(fn, "__name__", "problem")
    symbolic_vars = vars.symbols(diff=True)
    var_exprs = vars.flatten_symbolic(symbolic_vars, f"{problem_name} variables")
    resolved_vars = vars.with_types(tuple(expr.type for expr in var_exprs))

    if params is not None:
      symbolic_params = params.symbols(diff=False)
      param_exprs = params.flatten_symbolic(symbolic_params, f"{problem_name} parameters")
      resolved_params = params.with_types(tuple(expr.type for expr in param_exprs))
      spec = _normalize_spec(fn(symbolic_vars, symbolic_params), vars)
      declared = {expr.id for expr in (*var_exprs, *param_exprs)}
      undeclared = [expr.name or f"%{expr.id}" for expr in collect_free_inputs(_spec_exprs(spec, vars)) if expr.id not in declared]
      if undeclared:
        raise ValueError(f"problem {problem_name!r} has undeclared symbolic inputs: {undeclared}")
      return Problem(problem_name, spec, resolved_vars, resolved_params, var_exprs, param_exprs)

    spec = _normalize_spec(fn(symbolic_vars), vars)
    free = tuple(expr for expr in collect_free_inputs(_spec_exprs(spec, vars)) if expr.id not in {var.id for var in var_exprs})
    if any(expr.name is None for expr in free):
      raise ValueError(f"problem {problem_name!r} has unnamed inferred parameters")
    inferred_params = flat_tree(
      cast(tuple[str, ...], tuple(expr.name for expr in free)),
      tuple(TensorType(expr.shape, expr.type.dtype, expr.type.sparsity, diff=False) for expr in free),
    )
    param_exprs = inferred_params.flatten_symbolic(inferred_params.symbols(diff=False), f"{problem_name} parameters")
    replacements = dict(zip(free, param_exprs, strict=True))
    return Problem(problem_name, _substitute_spec(spec, vars, replacements), resolved_vars, inferred_params, var_exprs, param_exprs)

  return decorate
