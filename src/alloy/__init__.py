from .abi import C_API_SIGNATURE, BufferType, c_api_signature
from .ad import gradient as expr_gradient
from .ad import hessian as expr_hessian
from .ad import jacobian as expr_jacobian
from .ad import jvp, jvp_many, vjp, vjp_many
from .api import adjoint, forward, function, gradient, hessian, jacobian, lagrangian_hessian, sparse_lagrangian_hessian, sphessian, spjacobian
from .expr import Expr, atan2, concat, dot, format_expr, gather, map_, maximum, minimum, norm_2, scatter, split, stack, sumsqr, vec
from .function import Function, Port
from .ops import COMMON_OPS, OP_INFO, Ops
from .rewrite import Pattern, PatternMatcher, cse, cse_many, rewrite, simplify
from .solvers import SolverFunction, SolverStatus, nlp, qp
from .spec import Spec, VerifyError, VerifyRule, spec_semantic, spec_semantic_shared, verify_expr
from .sparsity import (
  SparseJacobian,
  color_groups,
  column_coloring,
  jacobian_sparsity,
  sparse_hessian,
  sparse_jacobian,
  sparse_jacobian_colored,
  sparse_jacobian_reference,
)
from .tape import Instruction, Tape, TapeRegion, WorkspacePlan, WorkspaceSlot, format_tape, linearize
from .types import BACKEND_SUPPORT, BackendSupport, DeviceSpec, DType, ScalarType, SparsityType, TensorType, as_dtype, backend_supports, dtypes

sym = Expr.sym
const = Expr.const
# ``al.scan`` is a transitional alias for ``al.map_``. Today both build an
# ``Ops.MAP`` node, since current workloads only need independent repeated calls.
# A true ``Ops.SCAN`` constructor for dependent recurrences will land when a
# workload demands carry semantics (see roadmap Phase 3). The alias is kept so
# existing callers don't churn for what is mostly a naming choice.
scan = map_

__all__ = [
  "BACKEND_SUPPORT",
  "BackendSupport",
  "BufferType",
  "C_API_SIGNATURE",
  "COMMON_OPS",
  "DType",
  "DeviceSpec",
  "OP_INFO",
  "Expr",
  "Function",
  "Instruction",
  "Ops",
  "Pattern",
  "PatternMatcher",
  "Port",
  "ScalarType",
  "SolverFunction",
  "SolverStatus",
  "Spec",
  "SparseJacobian",
  "SparsityType",
  "Tape",
  "TapeRegion",
  "TensorType",
  "VerifyError",
  "VerifyRule",
  "WorkspacePlan",
  "WorkspaceSlot",
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
  "expr_hessian",
  "expr_jacobian",
  "format_expr",
  "format_tape",
  "forward",
  "function",
  "gather",
  "gradient",
  "hessian",
  "jacobian",
  "jacobian_sparsity",
  "jvp",
  "jvp_many",
  "vjp",
  "vjp_many",
  "lagrangian_hessian",
  "linearize",
  "map_",
  "scan",
  "maximum",
  "minimum",
  "nlp",
  "norm_2",
  "qp",
  "rewrite",
  "scatter",
  "sparse_hessian",
  "sparse_lagrangian_hessian",
  "simplify",
  "sparse_jacobian",
  "sparse_jacobian_colored",
  "sparse_jacobian_reference",
  "spec_semantic",
  "spec_semantic_shared",
  "sphessian",
  "spjacobian",
  "split",
  "stack",
  "sumsqr",
  "vec",
  "sym",
  "verify_expr",
]
