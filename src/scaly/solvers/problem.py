"""Typed backend-free optimal-control problem declarations and their stacked NLP form."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, cast, overload

import numpy as np

from ..ad.sparse import SparseJacobian, Triangle, sparse_hessian
from ..function.api import gradient, sparse_jacobian
from ..function.concrete import ConcreteFunction
from ..function.tree import Tree, append_parameter, arg, flat_tree, parameter_list
from ..function.tree import group as tree_group
from ..ir.expr import Expr, ExprOp, as_expr, check_prints_reach, concat, recording_prints, substitute
from ..ir.types import SparsityPattern, TensorType
from ..passes.expr import simplify_cse_fixpoint
from ._oracle import collect_free_inputs


NO_LB: Expr = Expr.const(float("-inf"))
"""A scalar expression that leaves one variable block unbounded below."""

NO_UB: Expr = Expr.const(float("inf"))
"""A scalar expression that leaves one variable block unbounded above."""


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
  """The objective, constraint groups, and box bounds returned by a problem body.

  ``lb`` and ``ub`` have the variables’ tree structure. Scalar expression leaves broadcast over
  their variable blocks. Use ``NO_LB`` or ``NO_UB`` for an open side of one leaf.
  """

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

  @property
  def n_eq(self) -> int:
    """The total number of scalar equality constraints."""
    return sum(expr.size for expr in self.spec.eq)

  @property
  def n_ineq(self) -> int:
    """The total number of scalar bounded inequality constraints."""
    return sum(group.expr.size for group in self.spec.ineq)

  @cached_property
  def _nlp(self) -> _NlpForm:
    return _NlpForm(self)


def _concat_vec(exprs: tuple[Expr, ...]) -> Expr | None:
  if not exprs:
    return None
  vectors = tuple(expr if len(expr.shape) == 1 else expr.vec() for expr in exprs)
  return vectors[0] if len(vectors) == 1 else concat(vectors)


def _bound(expr: Expr | None, shape: tuple[int, ...], fill: float) -> Expr:
  if expr is None:
    return Expr.const(np.full(shape, fill))
  if expr.shape == shape:
    return expr
  if expr.shape == ():
    return Expr.const(np.zeros(shape)) + expr
  raise TypeError(f"bound has shape {expr.shape}, expected scalar or {shape}")


@dataclass(frozen=True, eq=False)
class _NlpForm:
  """A problem with its variable blocks stacked into one vector ``x``, the form NLP and QP solvers take.

  Each part is derived when first read, so a QP never builds the Lagrangian Hessian.
  """

  problem: Problem[Any, Any, Any, Any]
  _hessians: dict[Triangle, ConcreteFunction] = field(init=False, default_factory=dict)

  @cached_property
  def var_sizes(self) -> tuple[int, ...]:
    return tuple(expr.size for expr in self.problem._var_symbols)

  @cached_property
  def x_name(self) -> str:
    name = "_".join(self.problem.vars.names)
    while name in self.problem.params.names:
      name = "_" + name
    return name

  @cached_property
  def x(self) -> Expr:
    return Expr(ExprOp.INPUT, type=TensorType((sum(self.var_sizes),)), name=self.x_name)

  @cached_property
  def _replacements(self) -> dict[Expr, Expr]:
    replacements: dict[Expr, Expr] = {}
    offset = 0
    for original, size in zip(self.problem._var_symbols, self.var_sizes, strict=True):
      block = self.x if offset == 0 and size == self.x.size else self.x[offset : offset + size]
      replacements[original] = block if original.shape == (size,) else block.reshape(original.shape)
      offset += size
    return replacements

  def _stacked(self, expr: Expr) -> Expr:
    return substitute(expr, self._replacements)

  @cached_property
  def f(self) -> Expr:
    return self._stacked(self.problem.spec.minimize)

  @cached_property
  def equalities(self) -> tuple[Expr, ...]:
    return tuple(self._stacked(expr) for expr in self.problem.spec.eq)

  @cached_property
  def inequalities(self) -> tuple[Expr, ...]:
    return tuple(self._stacked(inequality.expr) for inequality in self.problem.spec.ineq)

  @cached_property
  def h(self) -> Expr | None:
    return _concat_vec(tuple(expr for expr in self.equalities if expr.size))

  @cached_property
  def g_ineq(self) -> Expr | None:
    return _concat_vec(tuple(expr for expr in self.inequalities if expr.size))

  @cached_property
  def g(self) -> Expr | None:
    return _concat_vec(tuple(expr for expr in (self.h, self.g_ineq) if expr is not None))

  @cached_property
  def base(self) -> ConcreteFunction:
    problem, f, g = self.problem, self.f, self.g
    input_tree = parameter_list((arg(self.x_name, self.x.type), problem.params))
    output_tree: Tree[Any, Any] = arg("f", f.type) if g is None else tree_group(arg("f", f.type), arg("g", g.type))
    outputs = (f,) if g is None else (f, g)
    return ConcreteFunction.build(f"{problem.name}_base", input_tree, (self.x, *problem._param_symbols), output_tree, outputs)

  @cached_property
  def grad(self) -> ConcreteFunction:
    return gradient(self.base, "f", self.x_name, name=f"{self.problem.name}_grad").instantiate()

  @cached_property
  def jac(self) -> ConcreteFunction | None:
    return None if self.g is None else sparse_jacobian(self.base, "g", self.x_name, name=f"{self.problem.name}_jac").instantiate()

  @cached_property
  def jac_sparsity(self) -> SparsityPattern:
    if self.jac is None:
      return SparsityPattern.empty((0, self.x.size))
    sparsity = self.jac.output_sparsities[0]
    assert sparsity is not None
    return sparsity

  @cached_property
  def _multipliers(self) -> tuple[Tree[Any, Any], tuple[Expr, ...]]:
    tree = self.base.output_tree.relabel("lam:")
    return tree, tuple(tree.flatten_symbolic(tree.symbols(), f"{self.problem.name} multipliers"))

  @cached_property
  def hess_full(self) -> SparseJacobian:
    """The full sparse Hessian of the Lagrangian ``lam:f * f + lam:g . g``."""
    multipliers = self._multipliers[1]
    lagrangian = multipliers[0] * self.f
    if self.g is not None:
      lagrangian = lagrangian + (multipliers[1] * self.g).sum()
    return sparse_hessian(lagrangian, self.x)

  def hess(self, triangle: Triangle) -> ConcreteFunction:
    """The Lagrangian Hessian oracle that stores ``triangle`` of the matrix."""
    if triangle not in self._hessians:
      tree, multipliers = self._multipliers
      hess = self.hess_full.triangle(triangle)
      self._hessians[triangle] = ConcreteFunction.build(
        f"{self.problem.name}_hess_{triangle}",
        append_parameter(self.base.input_tree, tree),
        (self.x, *self.problem._param_symbols, *multipliers),
        arg(f"sphess_gamma_{self.x_name}_{self.x_name}", hess.values.type),
        (hess.values,),
        output_sparsities=(hess.sparsity,),
        output_coloring_widths=(hess.coloring_width,),
      )
    return self._hessians[triangle]

  @cached_property
  def cost_hessian(self) -> SparseJacobian:
    return sparse_hessian(simplify_cse_fixpoint(self.f), self.x)

  @cached_property
  def bounds(self) -> ConcreteFunction:
    """The variable bounds and, when there are inequalities, their bounds, from the parameters."""
    problem = self.problem
    lower_ineq: list[Expr] = []
    upper_ineq: list[Expr] = []
    for inequality, expr in zip(problem.spec.ineq, self.inequalities, strict=True):
      if not expr.size:
        continue
      lower_ineq.append(_bound(None if inequality.lo is None else self._stacked(inequality.lo), expr.shape, -np.inf).vec())
      upper_ineq.append(_bound(None if inequality.hi is None else self._stacked(inequality.hi), expr.shape, np.inf).vec())
    variable_bounds: list[Expr] = []
    for side, declared, fill in (("lb", problem.spec.lb, -np.inf), ("ub", problem.spec.ub, np.inf)):
      if declared is None:
        variable_bounds.append(Expr.const(np.full(self.x.size, fill)))
        continue
      leaves = problem.vars.flatten_symbolic(declared, f"{problem.name} {side}")
      stacked = _concat_vec(
        tuple(_bound(self._stacked(expr), original.shape, fill).vec() for expr, original in zip(leaves, problem._var_symbols, strict=True))
      )
      assert stacked is not None
      variable_bounds.append(stacked)
    if self.g_ineq is None:
      outputs, names = tuple(variable_bounds), ("x_lb", "x_ub")
    else:
      l_ineq, u_ineq = _concat_vec(tuple(lower_ineq)), _concat_vec(tuple(upper_ineq))
      assert l_ineq is not None and u_ineq is not None
      outputs, names = (*variable_bounds, l_ineq, u_ineq), ("x_lb", "x_ub", "l_ineq", "u_ineq")
    return ConcreteFunction._from_exprs(f"{problem.name}_bounds", problem._param_symbols, outputs, problem.params.names, names)


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

  bounds: list[SV | None] = []
  for side, value in (("lb", spec.lb), ("ub", spec.ub)):
    if value is None:
      bounds.append(None)
      continue
    try:
      leaves = vars.flatten_symbolic(value, f"problem {side}", allow_scalar=True)
    except ValueError as exc:
      raise TypeError(str(exc)) from exc
    bounds.append(
      cast(
        SV,
        vars.unflatten(
          tuple(
            expr if expr.shape == type_.shape else Expr.const(np.zeros(type_.shape)) + expr for expr, type_ in zip(leaves, vars.types, strict=True)
          )
        ),
      )
    )

  return ProblemSpec(minimize, eq, ineq, bounds[0], bounds[1])


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
      with recording_prints() as prints:
        spec = _normalize_spec(fn(symbolic_vars, symbolic_params), vars)
      check_prints_reach(prints, _spec_exprs(spec, vars), f"problem {problem_name!r}")
      declared = {expr.id for expr in (*var_exprs, *param_exprs)}
      undeclared = [expr.name or f"%{expr.id}" for expr in collect_free_inputs(_spec_exprs(spec, vars)) if expr.id not in declared]
      if undeclared:
        raise ValueError(f"problem {problem_name!r} has undeclared symbolic inputs: {undeclared}")
      return Problem(problem_name, spec, resolved_vars, resolved_params, var_exprs, param_exprs)

    with recording_prints() as prints:
      spec = _normalize_spec(fn(symbolic_vars), vars)
    check_prints_reach(prints, _spec_exprs(spec, vars), f"problem {problem_name!r}")
    free = tuple(expr for expr in collect_free_inputs(_spec_exprs(spec, vars)) if expr.id not in {var.id for var in var_exprs})
    if any(expr.name is None for expr in free):
      raise ValueError(f"problem {problem_name!r} has unnamed inferred parameters")
    inferred_params = flat_tree(
      cast(tuple[str, ...], tuple(expr.name for expr in free)),
      tuple(TensorType(expr.shape, expr.type.dtype, diff=False) for expr in free),
    )
    param_exprs = inferred_params.flatten_symbolic(inferred_params.symbols(diff=False), f"{problem_name} parameters")
    replacements = dict(zip(free, param_exprs, strict=True))
    return Problem(problem_name, _substitute_spec(spec, vars, replacements), resolved_vars, inferred_params, var_exprs, param_exprs)

  return decorate
