"""Prove and extract typed quadratic problems for QP solver plugins."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

import numpy as np

from ..ad.derivatives import jacobian
from ..ad.sparse import SparseJacobian, sparse_hessian, sparse_jacobian
from ..ad.sparsity import _jac_mask
from ..function import ConcreteFunction
from ..function.tree import G, L, Tree, param_list
from ..ir.expr import Expr, ExprOp, callees_of, concat, substitute, topo
from ..ir.types import SparsityType, TensorType
from ..passes.expr import simplify_cse_fixpoint
from .model import SolverDescriptor, descriptor_function
from .nlp import _lowered
from .problem import Problem, ProblemSpec, bounded, problem
from .registry import SolverBackend

type QPData[T] = tuple[tuple[T, T], tuple[T, T], tuple[T, T, T]]


class NotQuadratic(ValueError):
  """A problem rejected because a QP oracle depends nonlinearly on its variables."""


def _qp_matrix_sparsity(mat: Expr, params: Sequence[Expr], probe: np.ndarray, *, triu: bool = False) -> SparsityType:
  """Return the structural matrix pattern in compressed sparse column order."""
  nrow, ncol = mat.shape
  vec = mat.vec()
  keep = np.asarray(probe, dtype=np.float64).reshape(-1) != 0.0
  for param in params:
    keep |= np.asarray(_jac_mask(vec, param, {}).sum(axis=1)).reshape(-1) != 0
  rows, cols = np.divmod(np.flatnonzero(keep), ncol)
  if triu:
    upper = rows <= cols
    rows, cols = rows[upper], cols[upper]
  if rows.size == 0:
    rows, cols = np.array([0]), np.array([0])
  order = np.lexsort((rows, cols))
  return SparsityType((nrow, ncol), tuple(int(row) for row in rows[order]), tuple(int(col) for col in cols[order]))


def _gathered(mat: Expr, sparsity: SparsityType) -> Expr:
  """Return compact matrix values in the pattern's compressed sparse column order."""
  flat = np.asarray(sparsity.rows, dtype=np.int64) * mat.shape[1] + np.asarray(sparsity.cols, dtype=np.int64)
  return mat.vec().gather(flat)


def _reaches_solver_call(exprs: Sequence[Expr]) -> bool:
  """Return whether an expression reaches a solver, including through Function calls."""
  seen: set[int] = set()

  def visit(targets: Sequence[Expr]) -> bool:
    for node in topo(targets):
      if node.op == ExprOp.SOLVER_CALL:
        return True
      for callee in callees_of(node):
        if id(callee) not in seen:
          seen.add(id(callee))
          if visit(callee.outputs):
            return True
    return False

  return visit(exprs)


def _prove_variable_independent_bounds(problem: Problem[Any, Any, Any, Any]) -> None:
  for side, bound in (("lb", problem.spec.lb), ("ub", problem.spec.ub)):
    if bound is None:
      continue
    for name, expr in zip(problem.vars.names, problem.vars.flatten_symbolic(bound, f"{problem.name} {side}"), strict=True):
      if _reaches_solver_call((expr,)):
        raise NotQuadratic(f"{problem.name}: cannot prove {side} for {name!r} independent through a nested solver")
      if any(_jac_mask(expr, variable, {}).nnz for variable in problem._var_symbols):
        raise NotQuadratic(f"{problem.name}: {side} for {name!r} depends on the variables")
  for index, group in enumerate(problem.spec.ineq):
    label = group.name or str(index)
    for side, bound in (("lower bound", group.lo), ("upper bound", group.hi)):
      if bound is not None:
        if _reaches_solver_call((bound,)):
          raise NotQuadratic(f"{problem.name}: cannot prove ineq {label} {side} independent through a nested solver")
        if any(_jac_mask(bound, variable, {}).nnz for variable in problem._var_symbols):
          raise NotQuadratic(f"{problem.name}: ineq {label} {side} depends on the variables")


def _prove_quadratic(problem: Problem[Any, Any, Any, Any], cached: dict[str, Any]) -> None:
  """Prove that the cost is quadratic and every constraint is affine in the variables."""
  x = cast(Expr, cached["x"])
  proof_targets = (cast(Expr, cached["f"]), *cast(tuple[Expr, ...], cached["equalities"]), *cast(tuple[Expr, ...], cached["inequalities"]))
  if _reaches_solver_call(proof_targets):
    raise NotQuadratic(f"{problem.name}: cannot prove QP structure through a nested solver")
  hessian = sparse_hessian(simplify_cse_fixpoint(cast(Expr, cached["f"])), x)
  cached["qp_hessian"] = hessian
  if _jac_mask(hessian.values, x, {}).nnz:
    raise NotQuadratic(f"{problem.name}: cost is not quadratic in the variables")

  def prove_affine(expr: Expr, label: str) -> None:
    derivative = simplify_cse_fixpoint(jacobian(expr, x))
    if _jac_mask(derivative, x, {}).nnz:
      raise NotQuadratic(f"{problem.name}: {label} is not affine in the variables")

  for index, expr in enumerate(cast(tuple[Expr, ...], cached["equalities"])):
    prove_affine(expr, f"eq[{index}]")
  for index, (group, expr) in enumerate(zip(problem.spec.ineq, cast(tuple[Expr, ...], cached["inequalities"]), strict=True)):
    prove_affine(expr, f"ineq {group.name or index}")


def _bound(expr: Expr | None, shape: tuple[int, ...], fill: float) -> Expr:
  if expr is None:
    return Expr.const(np.full(shape, fill))
  if expr.shape == shape:
    return expr
  if expr.shape == ():
    return Expr.const(np.zeros(shape)) + expr
  raise TypeError(f"bound has shape {expr.shape}, expected scalar or {shape}")


def _concat_vectors(exprs: tuple[Expr, ...]) -> Expr:
  if not exprs:
    return Expr.const(np.zeros(0))
  vectors = tuple(expr.vec() for expr in exprs)
  return vectors[0] if len(vectors) == 1 else concat(vectors)


def _qp_data(problem: Problem[Any, Any, Any, Any], cached: dict[str, Any]) -> tuple[Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr, Expr]:
  x = cast(Expr, cached["x"])
  n = x.size
  zero = Expr.const(np.zeros(x.shape))
  replacements = {x: zero}

  hessian = cast(SparseJacobian, cached["qp_hessian"])
  gradient = cast(ConcreteFunction, cached["grad"])
  P = simplify_cse_fixpoint(substitute(hessian.to_dense(), replacements))
  c = simplify_cse_fixpoint(substitute(gradient.outputs[0], replacements))

  h = cast(Expr | None, cached["h"])
  if h is None:
    A = Expr.const(np.zeros((0, n)))
    b = Expr.const(np.zeros(0))
  else:
    A = simplify_cse_fixpoint(substitute(sparse_jacobian(h, x).to_dense(), replacements))
    b = simplify_cse_fixpoint(-substitute(h, replacements))

  g = cast(Expr | None, cached["g_ineq"])
  if g is None:
    G_mat = Expr.const(np.zeros((0, n)))
    g_lb = Expr.const(np.zeros(0))
    g_ub = Expr.const(np.zeros(0))
  else:
    G_mat = simplify_cse_fixpoint(substitute(sparse_jacobian(g, x).to_dense(), replacements))
    g_lb = Expr.const(np.zeros(g.shape))
    g_ub = Expr.const(np.zeros(g.shape))

  variable_bounds: list[Expr] = []
  for side, declared, fill in (("lb", problem.spec.lb, -np.inf), ("ub", problem.spec.ub, np.inf)):
    if declared is None:
      variable_bounds.append(Expr.const(np.full(n, fill)))
    else:
      leaves = problem.vars.flatten_symbolic(declared, f"{problem.name} {side}")
      variable_bounds.append(
        _concat_vectors(tuple(_bound(expr, variable.shape, fill) for expr, variable in zip(leaves, problem._var_symbols, strict=True)))
      )
  x_lb, x_ub = variable_bounds
  if g is not None:
    lower = _concat_vectors(
      tuple(
        _bound(group.lo, expr.shape, -np.inf) for group, expr in zip(problem.spec.ineq, cast(tuple[Expr, ...], cached["inequalities"]), strict=True)
      )
    )
    upper = _concat_vectors(
      tuple(
        _bound(group.hi, expr.shape, np.inf) for group, expr in zip(problem.spec.ineq, cast(tuple[Expr, ...], cached["inequalities"]), strict=True)
      )
    )
    offset = simplify_cse_fixpoint(substitute(g, replacements))
    g_lb = simplify_cse_fixpoint(lower - offset)
    g_ub = simplify_cse_fixpoint(upper - offset)
  return P, c, A, b, G_mat, g_lb, g_ub, x_lb, x_ub


def build_qp[SV, NV, SP, NP](
  problem: Problem[SV, NV, SP, NP],
  backend: SolverBackend,
  *,
  name: str,
  options: dict[str, Any] | None,
) -> ConcreteFunction[
  [SV, SV, Expr, Expr, SP],
  [NV, NV, np.ndarray, np.ndarray, NP],
  tuple[SV, SV, Expr, Expr],
  tuple[NV, NV, np.ndarray, np.ndarray],
]:
  """Build a typed QP solver after proving and extracting the problem's matrix data."""
  _prove_variable_independent_bounds(problem)
  cached = _lowered(problem)
  _prove_quadratic(problem, cached)
  P, c, A, b, G_mat, g_lb, g_ub, x_lb, x_ub = _qp_data(problem, cached)
  n, n_eq, n_ineq = P.shape[0], A.shape[0], G_mat.shape[0]

  resolved_options = dict(options or {})
  sparse_option = resolved_options.pop("sparse", False)
  if not isinstance(sparse_option, bool):
    raise TypeError("PIQP option 'sparse' must be a bool")
  sparse = sparse_option
  params = problem._param_symbols

  P_sp = A_sp = G_sp = None
  if sparse:
    matrices = (P, A, G_mat)
    probe = ConcreteFunction._from_exprs(
      f"{name}_pattern_probe",
      params,
      tuple(matrix.vec() for matrix in matrices),
      problem.params.names,
      ("P", "A", "G"),
    )
    rng = np.random.default_rng(0)
    sample = probe.input_tree.unflatten(tuple(rng.standard_normal(param.shape) for param in params))
    values = probe.numerical_call(*sample)
    P_sp = _qp_matrix_sparsity(P, params, values[0], triu=True)
    if n_eq:
      A_sp = _qp_matrix_sparsity(A, params, values[1])
    if n_ineq:
      G_sp = _qp_matrix_sparsity(G_mat, params, values[2])

  oracle_outputs: list[Expr] = [_gathered(P, P_sp) if P_sp is not None else P.vec(), c]
  oracle_names = ["P", "c"]
  if n_eq:
    oracle_outputs.extend((_gathered(A, A_sp) if A_sp is not None else A.vec(), b))
    oracle_names.extend(("A_eq", "b_eq"))
  if n_ineq:
    oracle_outputs.extend((_gathered(G_mat, G_sp) if G_sp is not None else G_mat.vec(), g_lb, g_ub))
    oracle_names.extend(("G_ineq", "l_ineq", "u_ineq"))
  oracle_outputs.extend((x_lb, x_ub))
  oracle_names.extend(("x_lb", "x_ub"))
  oracle = ConcreteFunction._from_exprs(
    f"{name}_oracle", params, oracle_outputs, problem.params.names, tuple(f"qp:{output}" for output in oracle_names)
  )

  solver_vars = problem.vars.with_types(
    tuple(TensorType(expr.shape, expr.type.dtype, expr.type.sparsity, diff=False) for expr in problem._var_symbols)
  )
  input_tree = param_list(
    solver_vars,
    solver_vars.relabel("lam:"),
    L("lam_eq", TensorType((n_eq,), diff=False)),
    L("lam_ineq", TensorType((n_ineq,), diff=False)),
    problem.params,
  )
  output_tree: Tree[Any, Any] = G(
    solver_vars,
    solver_vars.relabel("lam:"),
    L("lam_eq", TensorType((n_eq,), diff=False)),
    L("lam_ineq", TensorType((n_ineq,), diff=False)),
  )
  descriptor = SolverDescriptor(
    name=name,
    backend=backend.name,
    n=n,
    n_eq=n_eq,
    n_ineq=n_ineq,
    input_signature=tuple(zip(input_tree.names, input_tree.shapes, strict=True)),
    output_signature=tuple(zip(output_tree.names, output_tree.shapes, strict=True)),
    param_names=problem.params.names,
    n_var_blocks=problem.vars.size,
    oracle=oracle,
    options=tuple(sorted({"verbose": 0, **resolved_options}.items())),
    oracle_output_names=tuple(oracle_names),
    sparse=sparse,
    P_sparsity=P_sp,
    A_sparsity=A_sp,
    G_sparsity=G_sp,
  )
  return cast(Any, descriptor_function(descriptor, input_tree, output_tree))


def qp_problem(n: int, n_eq: int, n_ineq: int) -> Problem[Expr, np.ndarray, QPData[Expr], QPData[np.ndarray]]:
  """Return the typed matrix-data form of a quadratic problem."""

  @problem(
    vars=L("x", n),
    params=G(
      G(L("P", (n, n)), L("c", n)),
      G(L("A", (n_eq, n)), L("b", n_eq)),
      G(L("G", (n_ineq, n)), L("g_lb", n_ineq), L("g_ub", n_ineq)),
    ),
    name="qp",
  )
  def qp(x: Expr, params: QPData[Expr]) -> ProblemSpec[Expr]:
    (P, c), (A, b), (G_mat, g_lb, g_ub) = params
    return ProblemSpec(
      minimize=0.5 * (x @ P @ x) + c @ x,
      eq=(A @ x - b,),
      ineq=(bounded(G_mat @ x, g_lb, g_ub, name="g"),),
    )

  return qp
