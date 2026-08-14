"""Construct an SQP solver descriptor over already-generated C-ABI oracles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from alloy.solvers.solver_function import ExternalOracle, SolverDescriptor, SolverFunction
from alloy.types import SparsityType


def external_nlp(
  *,
  name: str,
  n: int,
  n_eq: int,
  n_ineq: int,
  params: Sequence[tuple[str, tuple[int, ...]]],
  source: str,
  raw_symbols: Mapping[str, str],
  jac_sparsity: SparsityType,
  hess_sparsity: SparsityType,
  options: Mapping[str, str | int | float] | None = None,
) -> SolverFunction:
  """Build the standard NLP solve interface around foreign flat-buffer oracles.

  Required symbols are ``base``, ``grad``, ``jac``, ``hess``, and ``bounds``.
  Their signatures exactly mirror the corresponding functions produced by
  :func:`alloy.nlp`; ``source`` defines those symbols in the generated solver
  translation unit.
  """
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

  def oracle(key: str, inputs: tuple[tuple[str, tuple[int, ...]], ...], outputs: tuple[tuple[str, tuple[int, ...]], ...]):
    return ExternalOracle(
      name=f"{name}_{key}",
      raw_symbol=raw_symbols[key],
      source=source,
      input_signature=inputs,
      output_signature=outputs,
    )

  x_and_params = (("x", (n,)), *param_signature)
  base_outputs = (("f", ()), *((("g", (total_constraints,)),) if total_constraints else ()))
  base = oracle("base", x_and_params, base_outputs)
  grad = oracle("grad", x_and_params, (("grad_f", (n,)),))
  jac = oracle("jac", x_and_params, (("jac_g", (jac_sparsity.nnz,)),)) if total_constraints else None
  hess_inputs = (("x", (n,)), ("lam_f", ()), *((("lam_g", (total_constraints,)),) if total_constraints else ()), *param_signature)
  hess = oracle("hess", hess_inputs, (("hess_lag", (hess_sparsity.nnz,)),))
  bounds_outputs = (("x_lb", (n,)), ("x_ub", (n,)), *((("l_ineq", (n_ineq,)), ("u_ineq", (n_ineq,))) if n_ineq else ()))
  bounds = oracle("bounds", param_signature, bounds_outputs)
  input_signature = (
    ("x0", (n,)),
    ("lam_eq0", (n_eq,)),
    ("lam_ineq0", (n_ineq,)),
    ("lam_box0", (n,)),
    *param_signature,
  )
  output_signature = (
    ("x", (n,)),
    ("f", ()),
    ("h_eq", (n_eq,)),
    ("g_ineq", (n_ineq,)),
    ("lam_eq", (n_eq,)),
    ("lam_ineq", (n_ineq,)),
    ("lam_box", (n,)),
  )
  resolved_options: dict[str, Any] = {"max_iter": 50, "tol": 1e-6}
  if options:
    resolved_options.update(options)
  return SolverFunction(
    SolverDescriptor(
      name=name,
      backend="sqp",
      n=n,
      n_eq=n_eq,
      n_ineq=n_ineq,
      input_signature=input_signature,
      output_signature=output_signature,
      param_names=param_names,
      base=base,
      grad=grad,
      jac=jac,
      hess=hess,
      bounds=bounds,
      jac_sparsity=jac_sparsity,
      hess_sparsity=hess_sparsity,
      hess_lower_mask=tuple(r >= c for r, c in zip(hess_sparsity.rows, hess_sparsity.cols, strict=True)),
      options=tuple(sorted(resolved_options.items())),
    )
  )
