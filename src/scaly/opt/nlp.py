"""Build NLP solver descriptors from typed backend-free problems."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from ..ad.sparse import SparseJacobian, sparse_hessian
from ..function import ConcreteFunction
from ..function.api import gradient, sparse_jacobian
from ..function.tree import G, L, Tree, param_list
from ..ir.expr import Expr, ExprOp, concat, substitute
from ..ir.types import SparsityType, TensorType
from .external.model import SolverDescriptor, descriptor_function
from .problem import NLP
from .method import Info

if TYPE_CHECKING:
  from .external.method import External


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


def _lowered(problem: NLP[Any, Any, Any, Any]) -> dict[str, Any]:
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
  for group, expr in zip(problem.spec.ineq, inequalities, strict=True):
    if not expr.size:
      continue
    lower_ineq.append(_bound(None if group.lo is None else substitute(group.lo, replacements), expr.shape, -np.inf).vec())
    upper_ineq.append(_bound(None if group.hi is None else substitute(group.hi, replacements), expr.shape, np.inf).vec())
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

  x_tree = L(x_name, x.type)
  base_input_tree = G(x_tree, problem.params)
  if g is None:
    base_output_tree: Tree[Any, Any] = L("f", f.type)
    base_outputs = (f,)
  else:
    base_output_tree = G(L("f", f.type), L("g", g.type))
    base_outputs = (f, g)
  base = ConcreteFunction.from_exprs(
    f"{problem.name}_base",
    (x, *problem._param_symbols),
    base_outputs,
    base_input_tree.names,
    base_output_tree.names,
  )._with_trees(param_list(base_input_tree), base_output_tree)

  grad = gradient(base, "f", x_name, name=f"{problem.name}_grad")
  if g is None:
    jac = None
    jac_sparsity = SparsityType.empty((0, n))
  else:
    jac = sparse_jacobian(base, "g", x_name, name=f"{problem.name}_jac")
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
  bounds = ConcreteFunction.from_exprs(
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
    "hess_input_tree": G(base_input_tree, multiplier_tree),
    "bounds": bounds,
  }
  problem._cache["nlp"] = cached
  return cached


@dataclass(frozen=True)
class NLPOracles:
  """The oracles of a problem in nonlinear normal form, as ``nlp_oracles`` builds them:

      minimize f(x, p)  subject to  h(x, p) = 0,  l_ineq <= g_ineq(x, p) <= u_ineq,  x_lb <= x <= x_ub

  over the variables flattened in declaration order (``x``), with ``g = [h; g_ineq]``: ``n_eq`` rows
  of equalities, then ``n_ineq`` of inequalities. ``base`` maps ``(x, params)`` to ``f`` (and ``g``),
  ``grad`` to the gradient of ``f``, ``jac`` to the sparse Jacobian of ``g`` (``None`` without
  constraints; its pattern ``jac_sparsity``), and ``bounds`` maps the parameters to ``x_lb, x_ub``
  (and ``l_ineq, u_ineq``). ``hess`` is the sparse Hessian, both triangles, of the Lagrangian
  ``lam_f f + lam_g' g`` over ``hess_inputs``: ``x``, the parameters, ``lam_f`` and ``lam_g``."""

  x: Expr
  n_eq: int
  n_ineq: int
  base: ConcreteFunction[Any, Any, Any, Any]
  grad: ConcreteFunction[Any, Any, Any, Any]
  jac: ConcreteFunction[Any, Any, Any, Any] | None
  jac_sparsity: SparsityType
  hess: SparseJacobian
  hess_inputs: tuple[Expr, ...]
  bounds: ConcreteFunction[Any, Any, Any, Any]


def nlp_oracles(problem: NLP[Any, Any, Any, Any]) -> NLPOracles:
  """The nonlinear normal form of ``problem`` (``NLPOracles``), built once and kept with the problem."""
  cached = _lowered(problem)
  return NLPOracles(
    x=cast(Expr, cached["x"]),
    n_eq=problem.n_eq,
    n_ineq=problem.n_ineq,
    base=cast(ConcreteFunction, cached["base"]),
    grad=cast(ConcreteFunction, cached["grad"]),
    jac=cast(ConcreteFunction | None, cached["jac"]),
    jac_sparsity=cast(SparsityType, cached["jac_sparsity"]),
    hess=cast(SparseJacobian, cached["hess_full"]),
    hess_inputs=cast(tuple[Expr, ...], cached["hess_inputs"]),
    bounds=cast(ConcreteFunction, cached["bounds"]),
  )


def build_nlp[SV, NV, SP, NP](
  problem: NLP[SV, NV, SP, NP],
  method: External,
  *,
  name: str,
  options: dict[str, str | int | float] | None,
) -> ConcreteFunction[
  [SV, SV, Expr, Expr, SP],
  [NV, NV, np.ndarray, np.ndarray, NP],
  tuple[SV, SV, Expr, Expr, Info],
  tuple[NV, NV, np.ndarray, np.ndarray, Info],
]:
  """Build a typed plain Function around an NLP plugin descriptor."""
  cached = _lowered(problem)
  x = cast(Expr, cached["x"])
  triangle = method.hess_triangle
  hess_key = f"hess:{triangle}"
  hess_fn = cast(ConcreteFunction | None, problem._cache.get(hess_key))
  if hess_fn is None:
    hess_full = cast(SparseJacobian, cached["hess_full"])
    hess = hess_full.triangle(triangle)
    hess_name = f"sphess_gamma_{x.name}_{x.name}"
    hess_fn = ConcreteFunction.from_exprs(
      f"{problem.name}_hess_{triangle}",
      cast(tuple[Expr, ...], cached["hess_inputs"]),
      (hess.values,),
      cast(Tree[Any, Any], cached["hess_input_tree"]).names,
      (hess_name,),
      (hess.sparsity,),
      output_coloring_widths=(hess.coloring_width,),
    )._with_trees(param_list(cast(Tree[Any, Any], cached["hess_input_tree"])), L(hess_name, hess.values.type))
    problem._cache[hess_key] = hess_fn
  hess_sparsity = hess_fn.output_sparsities[0]
  assert hess_sparsity is not None

  solver_vars = problem.vars.with_types(
    tuple(TensorType(expr.shape, expr.type.dtype, expr.type.sparsity, diff=False) for expr in problem._var_symbols)
  )
  input_tree = param_list(
    solver_vars,
    solver_vars.relabel("lam:"),
    L("lam_eq", TensorType((problem.n_eq,), diff=False)),
    L("lam_ineq", TensorType((problem.n_ineq,), diff=False)),
    problem.params,
  )
  output_tree = G(
    solver_vars,
    solver_vars.relabel("lam:"),
    L("lam_eq", TensorType((problem.n_eq,), diff=False)),
    L("lam_ineq", TensorType((problem.n_ineq,), diff=False)),
  )
  input_signature = tuple(zip(input_tree.names, input_tree.shapes, strict=True))
  output_signature = tuple(zip(output_tree.names, output_tree.shapes, strict=True))
  resolved_options: dict[str, str | int | float] = {"print_level": 0, "sb": "yes"}
  if options:
    resolved_options.update(options)

  descriptor = SolverDescriptor(
    name=name,
    backend=method.backend,
    method=method,
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
    jac_sparsity=cast(SparsityType, cached["jac_sparsity"]),
    hess_sparsity=hess_sparsity,
    options=tuple(sorted(resolved_options.items())),
  )
  return cast(Any, descriptor_function(descriptor, input_tree, output_tree))
