"""Build NLP solver descriptors from typed backend-free problems."""

from __future__ import annotations

from typing import Any, cast

import numpy as np

from ..function.model import Function
from ..function.tree import group, arg
from ..ir.expr import Expr
from ..ir.types import TensorType
from .model import SolverDescriptor, descriptor_function, solver_options
from .problem import Problem
from .registry import NlpSolverBackend
from ._stage import stage_oracles


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
  """Build a typed plain Function around an NLP plugin descriptor."""
  compile_options, resolved_options = backend.prepare_options(options or {})
  nlp = problem._nlp
  staged = stage_oracles(problem) if getattr(backend, "stage_oracles", False) and problem.stage_functions is not None else None
  hess_fn = nlp.hess(backend.hess_triangle) if staged is None else None
  hess_sparsity = hess_fn.output_sparsities[0] if hess_fn is not None else None

  solver_vars = problem.vars
  input_tree = group(
    solver_vars,
    solver_vars.relabel("lam:"),
    arg("lam_eq", TensorType((problem.n_eq,))),
    arg("lam_ineq", TensorType((problem.n_ineq,))),
    problem.params,
  )
  output_tree = group(
    solver_vars,
    solver_vars.relabel("lam:"),
    arg("lam_eq", TensorType((problem.n_eq,))),
    arg("lam_ineq", TensorType((problem.n_ineq,))),
  )
  input_signature = tuple(zip(input_tree.names, input_tree.shapes, strict=True))
  output_signature = tuple(zip(output_tree.names, output_tree.shapes, strict=True))
  descriptor = SolverDescriptor(
    name=name,
    backend=backend.name,
    n=nlp.x.size,
    n_eq=problem.n_eq,
    n_ineq=problem.n_ineq,
    input_signature=input_signature,
    output_signature=output_signature,
    param_names=problem.params.names,
    n_var_blocks=problem.vars.size,
    base=nlp.base,
    grad=staged.grad if staged is not None else nlp.grad,
    jac=nlp.jac if staged is None else None,
    hess=hess_fn,
    bounds=nlp.bounds,
    jac_sparsity=nlp.jac_sparsity if staged is None else None,
    hess_sparsity=hess_sparsity,
    _stage_layout=problem.stage_layout,
    stage_oracles=staged,
    compile_options=tuple(sorted(compile_options.items())),
    runtime_options=solver_options(resolved_options),
  )
  return cast(Any, descriptor_function(descriptor, input_tree, output_tree))
