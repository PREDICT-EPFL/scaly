"""The curated public surface: re-exports only, no definitions of its own (``docs/dev/codebase.md``)."""

import importlib
import pkgutil
from types import ModuleType
from typing import TYPE_CHECKING, Any

from .codegen.abi import C_API_SIGNATURE, BufferType, c_api_signature
from .ad import jvp, jvp_many, vjp, vjp_many
from .function.sugar import custom_derivative, scan, vmap, while_loop
from .function.api import (
  adjoint,
  forward,
  function,
  gradient,
  hessian,
  jacobian,
  lagrangian_hessian,
  sparse_hessian,
  sparse_jacobian,
  sparse_lagrangian_hessian,
)
from .ir.text import render_expr_assembly, render_program_assembly
from .ir.expr import (
  COMMON_OPS,
  Expr,
  ExprOp,
  atan2,
  cast,
  concat,
  copysign,
  dot,
  equal,
  format_expr,
  gather,
  greater,
  greater_equal,
  index_add,
  index_set,
  isfinite,
  less,
  less_equal,
  logical_and,
  logical_not,
  logical_or,
  maximum,
  minimum,
  norm_1,
  norm_2,
  norm_inf,
  not_equal,
  put,
  put_add,
  reduce_max,
  reduce_min,
  scatter,
  segment_max,
  segment_min,
  segment_sum,
  split,
  stack,
  sumsqr,
  take,
  vec,
  where,
)
from .function import ConcreteFunction, Function, G, L, NotConcrete, factory
from .function.method import Status
from .utils.options import Options, get_options, options, set_options
from .ir.match import Pattern, PatternMatcher, rewrite
from .passes.expr import cse, cse_many, simplify
from .ir.expr_spec import spec_expr, spec_expr_shared, verify_expr
from .ir.spec import Rule, Spec, VerifyError
from .ad.sparse import SparseJacobian, sparse_jacobian_colored, sparse_jacobian_reference
from .ad.sparsity import color_groups, column_coloring, jacobian_sparsity, star_coloring
from .ir.types import BACKEND_SUPPORT, BackendSupport, DeviceSpec, DType, ScalarType, SparsityType, TensorType, as_dtype, backend_supports, dtypes

sym = Expr.sym
const = Expr.const


# ``scaly`` is one import namespace that several distributions install into. They normally share one
# directory; where they do not (a ``--target`` install beside another, a checkout on ``PYTHONPATH``
# beside installed wheels), extending the package's path merges the directories, so ``scaly.linalg``
# is found wherever scaly-numerics put it. The core's own modules, imported above, sit beside this file.
__path__ = pkgutil.extend_path(__path__, __name__)

# The namespaces of the other distributions, which ``import scaly`` leaves unloaded so that it is the
# compiler alone: ``sc.linalg`` imports ``scaly.linalg`` the first time it is read, and names the
# distribution to install when it is missing. ``tests/test_distributions.py`` holds this table to
# ``distributions.toml``.
_NAMESPACES = {
  "export": "scaly-tools",
  "geometry": "scaly-experimental",
  "integrators": "scaly-numerics",
  "interp": "scaly-numerics",
  "linalg": "scaly-numerics",
  "nn": "scaly-experimental",
  "ocp": "scaly-control",
  "opt": "scaly-numerics",
  "roots": "scaly-numerics",
  "sets": "scaly-control",
  "viz": "scaly-tools",
}

if TYPE_CHECKING:
  from . import export as export
  from . import geometry as geometry
  from . import integrators as integrators
  from . import interp as interp
  from . import linalg as linalg
  from . import nn as nn
  from . import ocp as ocp
  from . import opt as opt
  from . import roots as roots
  from . import sets as sets
  from . import viz as viz


def _namespace(name: str) -> ModuleType:
  try:
    return importlib.import_module(f"{__name__}.{name}")
  except ModuleNotFoundError as exc:
    if exc.name != f"{__name__}.{name}":
      raise  # the namespace is there and something it imports is not
    distribution = _NAMESPACES[name]
    raise AttributeError(f"{__name__}.{name} needs {distribution}, which is not installed (uv add {distribution})") from None


def __getattr__(name: str) -> Any:
  if name in _NAMESPACES:
    return _namespace(name)
  # Graph JSON is viz, and reaching it runs ``scaly/viz/__init__.py``, which pulls in the whole
  # backend and arms the render observer. Deferring that to the first use of these two names
  # keeps a plain ``import scaly`` free of it.
  if name in ("expr_graph", "program_graph"):
    _namespace("viz")
    return getattr(importlib.import_module(f"{__name__}.viz.graph"), name)
  raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
  "BACKEND_SUPPORT",
  "Status",
  "BackendSupport",
  "BufferType",
  "C_API_SIGNATURE",
  "COMMON_OPS",
  "DType",
  "DeviceSpec",
  "Expr",
  "ConcreteFunction",
  "Function",
  "G",
  "L",
  "NotConcrete",
  "ExprOp",
  "Pattern",
  "PatternMatcher",
  "ScalarType",
  "Spec",
  "SparseJacobian",
  "SparsityType",
  "TensorType",
  "VerifyError",
  "Rule",
  "adjoint",
  "as_dtype",
  "atan2",
  "backend_supports",
  "c_api_signature",
  "dtypes",
  "color_groups",
  "column_coloring",
  "star_coloring",
  "concat",
  "const",
  "cse",
  "cse_many",
  "dot",
  "expr_graph",
  "format_expr",
  "forward",
  "factory",
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
  "vmap",
  "scan",
  "while_loop",
  "custom_derivative",
  "maximum",
  "minimum",
  "cast",
  "copysign",
  "equal",
  "greater",
  "greater_equal",
  "index_add",
  "index_set",
  "isfinite",
  "less",
  "less_equal",
  "logical_and",
  "logical_not",
  "logical_or",
  "not_equal",
  "put",
  "put_add",
  "where",
  "norm_1",
  "norm_2",
  "norm_inf",
  "reduce_max",
  "reduce_min",
  "Options",
  "get_options",
  "options",
  "set_options",
  "program_graph",
  "render_expr_assembly",
  "render_program_assembly",
  "rewrite",
  "scatter",
  "segment_max",
  "segment_min",
  "segment_sum",
  "sparse_hessian",
  "sparse_lagrangian_hessian",
  "simplify",
  "sparse_jacobian",
  "sparse_jacobian_colored",
  "sparse_jacobian_reference",
  "spec_expr",
  "spec_expr_shared",
  "split",
  "stack",
  "sumsqr",
  "take",
  "vec",
  "sym",
  "verify_expr",
]
