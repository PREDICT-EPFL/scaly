"""The curated public surface: re-exports only, no definitions of its own (``docs/how_it_works/architecture.md``)."""

from typing import Any

from .codegen.abi import C_API_SIGNATURE, BufferType, c_api_signature
from .ad import gradient as expr_gradient
from .ad import hessian as expr_hessian
from .ad import jacobian as expr_jacobian
from .ad import jvp, jvp_many, vjp, vjp_many
from .function.factory import adj, fwd, grad, hess, jac, sphess, spjac
from .function.sugar import map_
from .function.api import (
  adjoint,
  forward,
  function,
  gradient,
  hessian,
  jacobian,
  lagrangian_hessian,
  sparse_lagrangian_hessian,
  sphessian,
  spjacobian,
)
from .ir.text import render_expr_assembly, render_program_assembly
from .ir.expr import (
  COMMON_OPS,
  Expr,
  ExprOp,
  OP_INFO,
  atan2,
  concat,
  dot,
  format_expr,
  gather,
  maximum,
  minimum,
  norm_2,
  scatter,
  split,
  stack,
  sumsqr,
  vec,
)
from .function import DerivSpec, Function, Port
from .ir.match import Pattern, PatternMatcher, rewrite
from .passes.expr import cse, cse_many, simplify
from .solvers import ALLOY_SOLVER_STATS_VERSION, AlloySolveStatus, SolverFunction, SolverStats, SolverStatus, nlp, qp
from .ir.expr_spec import spec_expr, spec_expr_shared, verify_expr
from .ir.spec import Rule, Spec, VerifyError
from .ad.sparse import SparseJacobian, sparse_hessian, sparse_jacobian, sparse_jacobian_colored, sparse_jacobian_reference
from .ad.sparsity import color_groups, column_coloring, jacobian_sparsity
from .ir.types import BACKEND_SUPPORT, BackendSupport, DeviceSpec, DType, ScalarType, SparsityType, TensorType, as_dtype, backend_supports, dtypes

sym = Expr.sym
const = Expr.const
scan = map_


def __getattr__(name: str) -> Any:
  # Graph JSON is viz, and reaching it runs ``alloy/viz/__init__.py``, which pulls in the whole
  # backend and arms the render observer. Deferring that to the first use of these two names
  # keeps a plain ``import alloy`` free of it.
  if name in ("expr_graph", "program_graph"):
    from .viz import graph

    return getattr(graph, name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "BACKEND_SUPPORT",
  "ALLOY_SOLVER_STATS_VERSION",
  "AlloySolveStatus",
  "BackendSupport",
  "BufferType",
  "C_API_SIGNATURE",
  "COMMON_OPS",
  "DType",
  "DeviceSpec",
  "OP_INFO",
  "Expr",
  "DerivSpec",
  "Function",
  "ExprOp",
  "Pattern",
  "PatternMatcher",
  "Port",
  "ScalarType",
  "SolverFunction",
  "SolverStats",
  "SolverStatus",
  "Spec",
  "SparseJacobian",
  "SparsityType",
  "TensorType",
  "VerifyError",
  "Rule",
  "adj",
  "adjoint",
  "as_dtype",
  "atan2",
  "backend_supports",
  "c_api_signature",
  "dtypes",
  "color_groups",
  "column_coloring",
  "concat",
  "const",
  "cse",
  "cse_many",
  "dot",
  "expr_gradient",
  "expr_graph",
  "expr_hessian",
  "expr_jacobian",
  "format_expr",
  "forward",
  "fwd",
  "function",
  "gather",
  "grad",
  "gradient",
  "hess",
  "hessian",
  "jac",
  "jacobian",
  "jacobian_sparsity",
  "jvp",
  "jvp_many",
  "vjp",
  "vjp_many",
  "lagrangian_hessian",
  "map_",
  "scan",
  "maximum",
  "minimum",
  "nlp",
  "norm_2",
  "program_graph",
  "qp",
  "render_expr_assembly",
  "render_program_assembly",
  "rewrite",
  "scatter",
  "sparse_hessian",
  "sparse_lagrangian_hessian",
  "simplify",
  "sparse_jacobian",
  "sparse_jacobian_colored",
  "sparse_jacobian_reference",
  "spec_expr",
  "spec_expr_shared",
  "sphess",
  "sphessian",
  "spjac",
  "spjacobian",
  "split",
  "stack",
  "sumsqr",
  "vec",
  "sym",
  "verify_expr",
]
