"""The extension API: what a package outside the compiler uses to add expression ops and their rules
(with the derivative and sparsity helpers those rules share), lowerings, program passes, option
namespaces, Functions with extern bodies, output adapters, and Functions built from expressions.

Everything here is defined elsewhere in ``scaly`` and collected for extension authors. A package
checks the version it was written against at import: ``require_ext_api(1, "my-package")``.
"""

from __future__ import annotations

from .ad.forward import JVPManyUnsupported, is_zero_const, zeros_many
from .ad.reverse import NoAdjoint
from .ad.sparsity import empty_mask, incidence, mask_compose, mask_or
from .codegen.adapter import Adapter, EntryHook, HeaderSpec, available_adapters, get_adapter, register_adapter
from .codegen.jit import load_library
from .function.extern import BuildRequirements, ExternCallee, ExternRenderCtx, ExternSource, ExternState, extern_function, extern_functions
from .function.model import ConcreteFunction, Function
from .function.tree import SymbolicValue
from .ir import program
from .ir.expr import Expr, ExprOp, OpDef, define_rules, define_traits, has_trait, op_def, register_op, registered_ops, registry_version
from .ir.spec import Rule
from .passes.lowering import LowerCtx, LoweringError, PositionRanges, Positions, lowers
from .passes.program import insert_after, insert_before, pipeline
from .utils.ext_api import EXT_API_VERSION, require_ext_api
from .utils.options import OptionNamespace, register_option_namespace

from_exprs = Function.from_exprs
"""A Function over expressions already built (``Function.from_exprs``)."""

__all__ = [
  "EXT_API_VERSION",
  "Adapter",
  "BuildRequirements",
  "ConcreteFunction",
  "EntryHook",
  "Expr",
  "ExprOp",
  "ExternCallee",
  "ExternRenderCtx",
  "ExternSource",
  "ExternState",
  "Function",
  "HeaderSpec",
  "JVPManyUnsupported",
  "LowerCtx",
  "LoweringError",
  "NoAdjoint",
  "OpDef",
  "OptionNamespace",
  "PositionRanges",
  "Positions",
  "Rule",
  "SymbolicValue",
  "available_adapters",
  "define_rules",
  "define_traits",
  "empty_mask",
  "extern_function",
  "extern_functions",
  "from_exprs",
  "get_adapter",
  "has_trait",
  "incidence",
  "insert_after",
  "insert_before",
  "is_zero_const",
  "load_library",
  "lowers",
  "mask_compose",
  "mask_or",
  "op_def",
  "pipeline",
  "program",
  "register_adapter",
  "register_op",
  "register_option_namespace",
  "registered_ops",
  "registry_version",
  "require_ext_api",
  "zeros_many",
]
