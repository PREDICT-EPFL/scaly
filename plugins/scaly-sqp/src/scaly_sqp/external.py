"""Construct an SQP solver descriptor over already-generated C-ABI oracles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from scaly.function import Function
from scaly.function.tree import group, arg, flat_tree
from scaly.ir.types import SparsityPattern, TensorType
from scaly.solvers.model import ExternalOracle, SolverDescriptor, descriptor_function


def external_nlp(
  *,
  name: str,
  n: int,
  n_eq: int,
  n_ineq: int,
  params: Sequence[tuple[str, tuple[int, ...]]],
  source: str,
  raw_symbols: Mapping[str, str],
  jac_sparsity: SparsityPattern,
  hess_sparsity: SparsityPattern,
  options: Mapping[str, str | int | float] | None = None,
) -> Function:
  """Build the typed NLP solve interface around foreign flat-buffer oracles."""
  total_constraints = n_eq + n_ineq
  required = {"base", "grad", "hess", "bounds"} | ({"jac"} if total_constraints else set())
  missing = sorted(required - set(raw_symbols))
  if missing:
    raise ValueError(f"external NLP is missing raw symbols: {missing}")
  if jac_sparsity.shape != (total_constraints, n):
    raise ValueError(f"external NLP Jacobian sparsity has shape {jac_sparsity.shape}, expected {(total_constraints, n)}")
  if hess_sparsity.shape != (n, n):
    raise ValueError(f"external NLP Hessian sparsity has shape {hess_sparsity.shape}, expected {(n, n)}")

  param_signature = tuple(params)
  param_names = tuple(param_name for param_name, _ in param_signature)

  def oracle(key: str, inputs: tuple[tuple[str, tuple[int, ...]], ...], outputs: tuple[tuple[str, tuple[int, ...]], ...]) -> ExternalOracle:
    return ExternalOracle(f"{name}_{key}", raw_symbols[key], source, inputs, outputs)

  x_and_params = (("x", (n,)), *param_signature)
  base_outputs = (("f", ()), *((("g", (total_constraints,)),) if total_constraints else ()))
  base = oracle("base", x_and_params, base_outputs)
  grad = oracle("grad", x_and_params, (("grad_f", (n,)),))
  jac = oracle("jac", x_and_params, (("jac_g", (jac_sparsity.nnz,)),)) if total_constraints else None
  hess_inputs = (("x", (n,)), *param_signature, ("lam_f", ()), *((("lam_g", (total_constraints,)),) if total_constraints else ()))
  hess = oracle("hess", hess_inputs, (("hess_lag", (hess_sparsity.nnz,)),))
  bounds_outputs = (("x_lb", (n,)), ("x_ub", (n,)), *((("l_ineq", (n_ineq,)), ("u_ineq", (n_ineq,))) if n_ineq else ()))
  bounds = oracle("bounds", param_signature, bounds_outputs)

  param_tree = flat_tree(param_names, tuple(TensorType(shape, diff=False) for _, shape in param_signature))
  input_tree = group(
    arg("x", TensorType((n,), diff=False)),
    arg("lam:x", TensorType((n,), diff=False)),
    arg("lam_eq", TensorType((n_eq,), diff=False)),
    arg("lam_ineq", TensorType((n_ineq,), diff=False)),
    param_tree,
  )
  output_tree = group(
    arg("x", TensorType((n,), diff=False)),
    arg("lam:x", TensorType((n,), diff=False)),
    arg("lam_eq", TensorType((n_eq,), diff=False)),
    arg("lam_ineq", TensorType((n_ineq,), diff=False)),
  )
  resolved_options: dict[str, Any] = {"max_iter": 50, "tol": 1e-6}
  if options:
    resolved_options.update(options)
  descriptor = SolverDescriptor(
    name=name,
    backend="sqp",
    n=n,
    n_eq=n_eq,
    n_ineq=n_ineq,
    input_signature=tuple(zip(input_tree.names, input_tree.shapes, strict=True)),
    output_signature=tuple(zip(output_tree.names, output_tree.shapes, strict=True)),
    param_names=param_names,
    n_var_blocks=1,
    base=base,
    grad=grad,
    jac=jac,
    hess=hess,
    bounds=bounds,
    jac_sparsity=jac_sparsity,
    hess_sparsity=hess_sparsity,
    options=tuple(sorted(resolved_options.items())),
  )
  return descriptor_function(descriptor, input_tree, output_tree)
