"""Derivative construction: forward and reverse mode, whole derivatives, and sparse variants."""

# linear_combination holds no AD and lives in ir/expr.py; re-exported to keep its public address.
from ..ir.expr import linear_combination
from .derivatives import basis, finite_difference, gradient, hessian, jacobian
from .forward import _jvp_many_structural, _jvp_many_unrolled, jvp, jvp_many
from .reverse import vjp, vjp_many
from .sparse import SparseJacobian, sparse_hessian, sparse_jacobian, sparse_jacobian_colored, sparse_jacobian_reference
from .sparsity import color_groups, column_coloring, jacobian_sparsity

__all__ = [
  "_jvp_many_structural",
  "_jvp_many_unrolled",
  "SparseJacobian",
  "basis",
  "color_groups",
  "column_coloring",
  "finite_difference",
  "gradient",
  "hessian",
  "jacobian",
  "jacobian_sparsity",
  "jvp",
  "jvp_many",
  "linear_combination",
  "sparse_hessian",
  "sparse_jacobian",
  "sparse_jacobian_colored",
  "sparse_jacobian_reference",
  "vjp",
  "vjp_many",
]
