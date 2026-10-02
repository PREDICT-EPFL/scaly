"""Build NLP solver descriptors from typed backend-free problems."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ..ad.sparse import SparseJacobian, sparse_hessian
from ..function.concrete import ConcreteFunction
from ..function.model import Function
from ..function.api import gradient, sparse_jacobian
from ..function.tree import group, arg, Tree, parameter_list, append_parameter
from ..ir.expr import Expr, ExprOp, concat, substitute
from ..ir.types import SparsityPattern, TensorType
from .model import SolverDescriptor, descriptor_function
from .problem import Problem
from .registry import NlpSolverBackend


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


def _lowered(problem: Problem[Any, Any, Any, Any]) -> dict[str, Any]:
  cached = problem._cache.get("nlp")
  if cached is not None:
    return cast(dict[str, Any], cached)

  var_sizes = tuple(expr.size for expr in problem._var_symbols)
  n = sum(var_sizes)
  x_name = "_".join(problem.vars.names)
  while x_name in problem.params.names:
    x_name = "_" + x_name
  x = Expr(ExprOp.INPUT, type=TensorType((n,)), name=x_name)

  replacements: dict[Expr, Expr] = {}
  offset = 0
  for original, size in zip(problem._var_symbols, var_sizes, strict=True):
    block = x if offset == 0 and size == n else x[offset : offset + size]
    replacements[original] = block if original.shape == (size,) else block.reshape(original.shape)
    offset += size

  f = substitute(problem.spec.minimize, replacements)
  equalities = tuple(substitute(expr, replacements) for expr in problem.spec.eq)
  inequalities = tuple(substitute(group.expr, replacements) for group in problem.spec.ineq)
  h = _concat_vec(tuple(expr for expr in equalities if expr.size))
  g_ineq = _concat_vec(tuple(expr for expr in inequalities if expr.size))
  constraints = tuple(expr for expr in (h, g_ineq) if expr is not None)
  g = _concat_vec(constraints)

  lower_ineq: list[Expr] = []
  upper_ineq: list[Expr] = []
  for inequality, expr in zip(problem.spec.ineq, inequalities, strict=True):
    if not expr.size:
      continue
    lower_ineq.append(_bound(None if inequality.lo is None else substitute(inequality.lo, replacements), expr.shape, -np.inf).vec())
    upper_ineq.append(_bound(None if inequality.hi is None else substitute(inequality.hi, replacements), expr.shape, np.inf).vec())
  l_ineq = _concat_vec(tuple(lower_ineq))
  u_ineq = _concat_vec(tuple(upper_ineq))

  if problem.spec.lb is None:
    x_lb = Expr.const(np.full(n, -np.inf))
  else:
    leaves = problem.vars.flatten_symbolic(problem.spec.lb, f"{problem.name} lb")
    x_lb = _concat_vec(
      tuple(
        _bound(substitute(expr, replacements), original.shape, -np.inf).vec() for expr, original in zip(leaves, problem._var_symbols, strict=True)
      )
    )
    assert x_lb is not None
  if problem.spec.ub is None:
    x_ub = Expr.const(np.full(n, np.inf))
  else:
    leaves = problem.vars.flatten_symbolic(problem.spec.ub, f"{problem.name} ub")
    x_ub = _concat_vec(
      tuple(_bound(substitute(expr, replacements), original.shape, np.inf).vec() for expr, original in zip(leaves, problem._var_symbols, strict=True))
    )
    assert x_ub is not None

  x_tree = arg(x_name, x.type)
  base_input_tree = parameter_list((x_tree, problem.params))
  if g is None:
    base_output_tree: Tree[Any, Any] = arg("f", f.type)
    base_outputs = (f,)
  else:
    base_output_tree = group(arg("f", f.type), arg("g", g.type))
    base_outputs = (f, g)
  base = ConcreteFunction._from_exprs(
    f"{problem.name}_base",
    (x, *problem._param_symbols),
    base_outputs,
    base_input_tree.names,
    base_output_tree.names,
  )._with_trees(base_input_tree, base_output_tree)

  grad = gradient(base, "f", x_name, name=f"{problem.name}_grad").instantiate()
  if g is None:
    jac = None
    jac_sparsity = SparsityPattern.empty((0, n))
  else:
    jac = sparse_jacobian(base, "g", x_name, name=f"{problem.name}_jac").instantiate()
    jac_sparsity = jac.output_sparsities[0]
    assert jac_sparsity is not None

  multiplier_tree = base.output_tree.relabel("lam:")
  multiplier_exprs = list(multiplier_tree.flatten_symbolic(multiplier_tree.symbols(), f"{problem.name} multipliers"))
  lagrangian = multiplier_exprs[0] * f
  if g is not None:
    lagrangian = lagrangian + (multiplier_exprs[1] * g).sum()
  hess_full = sparse_hessian(lagrangian, x)
  assert isinstance(hess_full, SparseJacobian)

  if g_ineq is None:
    bound_outputs = (x_lb, x_ub)
    bound_names = ("x_lb", "x_ub")
  else:
    assert l_ineq is not None and u_ineq is not None
    bound_outputs = (x_lb, x_ub, l_ineq, u_ineq)
    bound_names = ("x_lb", "x_ub", "l_ineq", "u_ineq")
  bounds = ConcreteFunction._from_exprs(
    f"{problem.name}_bounds",
    problem._param_symbols,
    bound_outputs,
    problem.params.names,
    bound_names,
  )

  cached = {
    "x": x,
    "var_sizes": var_sizes,
    "f": f,
    "h": h,
    "g_ineq": g_ineq,
    "equalities": equalities,
    "inequalities": inequalities,
    "base": base,
    "grad": grad,
    "jac": jac,
    "jac_sparsity": jac_sparsity,
    "hess_full": hess_full,
    "hess_inputs": (x, *problem._param_symbols, *multiplier_exprs),
    "hess_input_tree": append_parameter(base.input_tree, multiplier_tree),
    "bounds": bounds,
  }
  problem._cache["nlp"] = cached
  return cached


def build_nlp[SV, NV, SP, NP](
  problem: Problem[SV, NV, SP, NP],
  backend: NlpSolverBackend,
  *,
  name: str,
  options: dict[str, str | int | float] | None,
) -> Function[
  tuple[SV, SV, Expr, Expr, SP],
  tuple[NV, NV, np.ndarray, np.ndarray, NP],
  tuple[SV, SV, Expr, Expr],
  tuple[NV, NV, np.ndarray, np.ndarray],
]:
  """Build a typed plain ConcreteFunction around an NLP plugin descriptor."""
  cached = _lowered(problem)
  x = cast(Expr, cached["x"])
  triangle = backend.hess_triangle
  hess_key = f"hess:{triangle}"
  hess_fn = cast(ConcreteFunction | None, problem._cache.get(hess_key))
  if hess_fn is None:
    hess_full = cast(SparseJacobian, cached["hess_full"])
    hess = hess_full.triangle(triangle)
    hess_name = f"sphess_gamma_{x.name}_{x.name}"
    hess_fn = ConcreteFunction._from_exprs(
      f"{problem.name}_hess_{triangle}",
      cast(tuple[Expr, ...], cached["hess_inputs"]),
      (hess.values,),
      cast(Tree[Any, Any], cached["hess_input_tree"]).names,
      (hess_name,),
      (hess.sparsity,),
      output_coloring_widths=(hess.coloring_width,),
    )._with_trees(cast(Tree[Any, Any], cached["hess_input_tree"]), arg(hess_name, hess.values.type))
    problem._cache[hess_key] = hess_fn
  hess_sparsity = hess_fn.output_sparsities[0]
  assert hess_sparsity is not None

  solver_vars = problem.vars.with_types(tuple(TensorType(expr.shape, expr.type.dtype, diff=False) for expr in problem._var_symbols))
  input_tree = group(
    solver_vars,
    solver_vars.relabel("lam:"),
    arg("lam_eq", TensorType((problem.n_eq,), diff=False)),
    arg("lam_ineq", TensorType((problem.n_ineq,), diff=False)),
    problem.params,
  )
  output_tree = group(
    solver_vars,
    solver_vars.relabel("lam:"),
    arg("lam_eq", TensorType((problem.n_eq,), diff=False)),
    arg("lam_ineq", TensorType((problem.n_ineq,), diff=False)),
  )
  input_signature = tuple(zip(input_tree.names, input_tree.shapes, strict=True))
  output_signature = tuple(zip(output_tree.names, output_tree.shapes, strict=True))
  resolved_options: dict[str, str | int | float] = {"print_level": 0}
  if options:
    resolved_options.update(options)

  descriptor = SolverDescriptor(
    name=name,
    backend=backend.name,
    n=sum(cast(tuple[int, ...], cached["var_sizes"])),
    n_eq=problem.n_eq,
    n_ineq=problem.n_ineq,
    input_signature=input_signature,
    output_signature=output_signature,
    param_names=problem.params.names,
    n_var_blocks=problem.vars.size,
    base=cast(ConcreteFunction, cached["base"]),
    grad=cast(ConcreteFunction, cached["grad"]),
    jac=cast(ConcreteFunction | None, cached["jac"]),
    hess=hess_fn,
    bounds=cast(ConcreteFunction, cached["bounds"]),
    jac_sparsity=cast(SparsityPattern, cached["jac_sparsity"]),
    hess_sparsity=hess_sparsity,
    options=tuple(sorted(resolved_options.items())),
  )
  return cast(Any, descriptor_function(descriptor, input_tree, output_tree))
