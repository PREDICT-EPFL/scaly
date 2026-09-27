"""Expression dialect vocabulary: ``ExprOp``, ``OP_INFO``, ``Expr``, interning, builders, topo.

The verify rules are ``ir/expr_spec.py`` and the one builder that needs a ``Function`` — ``vmap``
— is ``function/sugar.py``. The printers are ``ir/text.py``, with one exception: ``format_expr``
stays here because ``Expr.debug`` calls it, and moving it would make ``ir/expr.py`` import
``ir/text.py``, which imports ``ir/expr.py``.
"""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from ..utils.options import get_options
from .types import DType, Lowering, TensorType, as_dtype, as_shape, broadcast_shape, dtypes


class ExprOp(StrEnum):
  """The expression dialect's operation set.

  A ``StrEnum``, so an op is a proper enum value and still prints and serializes as its name.
  ``OP_INFO`` carries the arity, the NumPy evaluation rule and the differentiability of each.
  """

  INPUT = "input"
  CONST = "const"
  NEG = "neg"
  SIN = "sin"
  COS = "cos"
  TAN = "tan"
  ASIN = "asin"
  ACOS = "acos"
  ATAN = "atan"
  SINH = "sinh"
  COSH = "cosh"
  TANH = "tanh"
  ERF = "erf"
  EXP = "exp"
  LOG = "log"
  SQRT = "sqrt"
  ABS = "abs"
  FLOOR = "floor"
  CEIL = "ceil"
  ADD = "add"
  SUB = "sub"
  MUL = "mul"
  DIV = "div"
  POW = "pow"
  ATAN2 = "atan2"
  MINIMUM = "minimum"
  MAXIMUM = "maximum"
  COPYSIGN = "copysign"
  LT = "lt"
  LE = "le"
  EQ = "eq"
  NE = "ne"
  AND = "and"
  OR = "or"
  NOT = "not"
  ISFINITE = "isfinite"
  SELECT = "select"
  CAST = "cast"
  SUM = "sum"
  MAX = "max"
  MIN = "min"
  SEGMENT_MAX = "segment_max"
  SEGMENT_MIN = "segment_min"
  INDEX_ADD = "index_add"
  INDEX_SET = "index_set"
  TAKE = "take"
  PUT_ADD = "put_add"
  PUT = "put"
  RAGGED_ADD = "ragged_add"
  RAGGED_DOT = "ragged_dot"
  RESHAPE = "reshape"
  TRANSPOSE = "transpose"
  SLICE = "slice"
  GATHER = "gather"
  SCATTER = "scatter"
  STACK = "stack"
  CONCAT = "concat"
  MATMUL = "matmul"
  CHOLESKY = "cholesky"
  LDL = "ldl"
  SPARSE_LDL = "sparse_ldl"
  SPARSE_LDL_SOLVE = "sparse_ldl_solve"
  TRISOLVE = "trisolve"
  CALL = "call"
  VMAP = "vmap"
  SCAN = "scan"
  WHILE = "while"
  SOLVER_CALL = "solver_call"


COMMON_ELEMENTWISE_UNARY = {
  ExprOp.NEG,
  ExprOp.SIN,
  ExprOp.COS,
  ExprOp.TAN,
  ExprOp.ASIN,
  ExprOp.ACOS,
  ExprOp.ATAN,
  ExprOp.SINH,
  ExprOp.COSH,
  ExprOp.TANH,
  ExprOp.ERF,
  ExprOp.EXP,
  ExprOp.LOG,
  ExprOp.SQRT,
  ExprOp.ABS,
  ExprOp.FLOOR,
  ExprOp.CEIL,
}

COMMON_ELEMENTWISE_BINARY = {
  ExprOp.ADD,
  ExprOp.SUB,
  ExprOp.MUL,
  ExprOp.DIV,
  ExprOp.POW,
  ExprOp.ATAN2,
  ExprOp.MINIMUM,
  ExprOp.MAXIMUM,
  ExprOp.COPYSIGN,
}

# Ops whose result is ``bool``: comparisons of two same-dtype operands, logic on bools, and the
# finiteness test. None of them is differentiable; a derivative flows only through ``SELECT``.
COMPARE_OPS = {ExprOp.LT, ExprOp.LE, ExprOp.EQ, ExprOp.NE}
LOGICAL_OPS = {ExprOp.AND, ExprOp.OR, ExprOp.NOT}
PREDICATE_OPS = COMPARE_OPS | LOGICAL_OPS | {ExprOp.ISFINITE}

COMMON_CONTROL = PREDICATE_OPS | {ExprOp.SELECT, ExprOp.CAST}

COMMON_STRUCTURAL = {
  ExprOp.INPUT,
  ExprOp.CONST,
  ExprOp.SUM,
  ExprOp.MAX,
  ExprOp.MIN,
  ExprOp.SEGMENT_MAX,
  ExprOp.SEGMENT_MIN,
  ExprOp.INDEX_ADD,
  ExprOp.INDEX_SET,
  ExprOp.TAKE,
  ExprOp.PUT_ADD,
  ExprOp.PUT,
  ExprOp.RAGGED_ADD,
  ExprOp.RAGGED_DOT,
  ExprOp.RESHAPE,
  ExprOp.TRANSPOSE,
  ExprOp.SLICE,
  ExprOp.GATHER,
  ExprOp.SCATTER,
  ExprOp.STACK,
  ExprOp.CONCAT,
  ExprOp.MATMUL,
  ExprOp.CHOLESKY,
  ExprOp.LDL,
  ExprOp.SPARSE_LDL,
  ExprOp.SPARSE_LDL_SOLVE,
  ExprOp.TRISOLVE,
  ExprOp.CALL,
  ExprOp.VMAP,
  ExprOp.SCAN,
  ExprOp.WHILE,
  ExprOp.SOLVER_CALL,
}

# Ops whose ``callee`` attr names a ``Function`` the graph runs (a ``while`` also runs its ``cond``);
# ``callees_of`` lists them per node.
CALLEE_OPS = {ExprOp.CALL, ExprOp.VMAP, ExprOp.SCAN, ExprOp.WHILE}

# Deliberately not in the MVP set: expm1/log1p (nice but low priority), splines/interpolants
# (important but require carefully specified extrapolation, knots, derivatives, and codegen tables),
# matrix exponentials/decompositions, and solver/control-flow ops.
COMMON_OPS = COMMON_STRUCTURAL | COMMON_ELEMENTWISE_UNARY | COMMON_ELEMENTWISE_BINARY | COMMON_CONTROL


@dataclass(frozen=True, slots=True)
class OpInfo:
  op: ExprOp
  arity: int | None
  numpy: Callable[..., np.ndarray | np.generic] | None = None
  differentiable: bool = True

  @property
  def name(self) -> str:
    return self.op.value


OP_INFO: dict[ExprOp, OpInfo] = {
  ExprOp.INPUT: OpInfo(ExprOp.INPUT, 0, None),
  ExprOp.CONST: OpInfo(ExprOp.CONST, 0, None, False),
  ExprOp.NEG: OpInfo(ExprOp.NEG, 1, np.negative),
  ExprOp.SIN: OpInfo(ExprOp.SIN, 1, np.sin),
  ExprOp.COS: OpInfo(ExprOp.COS, 1, np.cos),
  ExprOp.TAN: OpInfo(ExprOp.TAN, 1, np.tan),
  ExprOp.ASIN: OpInfo(ExprOp.ASIN, 1, np.arcsin),
  ExprOp.ACOS: OpInfo(ExprOp.ACOS, 1, np.arccos),
  ExprOp.ATAN: OpInfo(ExprOp.ATAN, 1, np.arctan),
  ExprOp.SINH: OpInfo(ExprOp.SINH, 1, np.sinh),
  ExprOp.COSH: OpInfo(ExprOp.COSH, 1, np.cosh),
  ExprOp.TANH: OpInfo(ExprOp.TANH, 1, np.tanh),
  # NumPy has no erf; frompyfunc keeps constant folding vectorized without adding SciPy.
  ExprOp.ERF: OpInfo(ExprOp.ERF, 1, lambda x: np.asarray(np.frompyfunc(math.erf, 1, 1)(x), dtype=np.float64)),
  ExprOp.EXP: OpInfo(ExprOp.EXP, 1, np.exp),
  ExprOp.LOG: OpInfo(ExprOp.LOG, 1, np.log),
  ExprOp.SQRT: OpInfo(ExprOp.SQRT, 1, np.sqrt),
  ExprOp.ABS: OpInfo(ExprOp.ABS, 1, np.abs),
  ExprOp.FLOOR: OpInfo(ExprOp.FLOOR, 1, np.floor, False),
  ExprOp.CEIL: OpInfo(ExprOp.CEIL, 1, np.ceil, False),
  ExprOp.ADD: OpInfo(ExprOp.ADD, 2, np.add),
  ExprOp.SUB: OpInfo(ExprOp.SUB, 2, np.subtract),
  ExprOp.MUL: OpInfo(ExprOp.MUL, 2, np.multiply),
  ExprOp.DIV: OpInfo(ExprOp.DIV, 2, np.divide),
  ExprOp.POW: OpInfo(ExprOp.POW, 2, np.power),
  ExprOp.ATAN2: OpInfo(ExprOp.ATAN2, 2, np.arctan2),
  ExprOp.MINIMUM: OpInfo(ExprOp.MINIMUM, 2, np.minimum),
  ExprOp.MAXIMUM: OpInfo(ExprOp.MAXIMUM, 2, np.maximum),
  ExprOp.COPYSIGN: OpInfo(ExprOp.COPYSIGN, 2, np.copysign),
  ExprOp.LT: OpInfo(ExprOp.LT, 2, np.less, False),
  ExprOp.LE: OpInfo(ExprOp.LE, 2, np.less_equal, False),
  ExprOp.EQ: OpInfo(ExprOp.EQ, 2, np.equal, False),
  ExprOp.NE: OpInfo(ExprOp.NE, 2, np.not_equal, False),
  ExprOp.AND: OpInfo(ExprOp.AND, 2, np.logical_and, False),
  ExprOp.OR: OpInfo(ExprOp.OR, 2, np.logical_or, False),
  ExprOp.NOT: OpInfo(ExprOp.NOT, 1, np.logical_not, False),
  ExprOp.ISFINITE: OpInfo(ExprOp.ISFINITE, 1, np.isfinite, False),
  ExprOp.SELECT: OpInfo(ExprOp.SELECT, 3, np.where),
  ExprOp.CAST: OpInfo(ExprOp.CAST, 1, None),
  ExprOp.SUM: OpInfo(ExprOp.SUM, 1, np.sum),
  ExprOp.MAX: OpInfo(ExprOp.MAX, 1, np.max),
  ExprOp.MIN: OpInfo(ExprOp.MIN, 1, np.min),
  ExprOp.SEGMENT_MAX: OpInfo(ExprOp.SEGMENT_MAX, 1, None),
  ExprOp.SEGMENT_MIN: OpInfo(ExprOp.SEGMENT_MIN, 1, None),
  ExprOp.INDEX_ADD: OpInfo(ExprOp.INDEX_ADD, 2, None),
  ExprOp.INDEX_SET: OpInfo(ExprOp.INDEX_SET, 2, None),
  ExprOp.TAKE: OpInfo(ExprOp.TAKE, 2, None),
  ExprOp.PUT_ADD: OpInfo(ExprOp.PUT_ADD, 3, None),
  ExprOp.PUT: OpInfo(ExprOp.PUT, 3, None),
  ExprOp.RAGGED_ADD: OpInfo(ExprOp.RAGGED_ADD, 5, None),
  ExprOp.RAGGED_DOT: OpInfo(ExprOp.RAGGED_DOT, 4, None),
  ExprOp.RESHAPE: OpInfo(ExprOp.RESHAPE, 1, np.reshape),
  ExprOp.TRANSPOSE: OpInfo(ExprOp.TRANSPOSE, 1, np.transpose),
  ExprOp.SLICE: OpInfo(ExprOp.SLICE, 1, None),
  ExprOp.GATHER: OpInfo(ExprOp.GATHER, 1, None),
  ExprOp.SCATTER: OpInfo(ExprOp.SCATTER, 1, None),
  ExprOp.STACK: OpInfo(ExprOp.STACK, None, np.stack),
  ExprOp.CONCAT: OpInfo(ExprOp.CONCAT, None, np.concatenate),
  ExprOp.MATMUL: OpInfo(ExprOp.MATMUL, 2, np.matmul),
  ExprOp.CHOLESKY: OpInfo(ExprOp.CHOLESKY, 1, None),
  ExprOp.LDL: OpInfo(ExprOp.LDL, 1, None),
  ExprOp.SPARSE_LDL: OpInfo(ExprOp.SPARSE_LDL, 1, None),
  ExprOp.SPARSE_LDL_SOLVE: OpInfo(ExprOp.SPARSE_LDL_SOLVE, 2, None),
  ExprOp.TRISOLVE: OpInfo(ExprOp.TRISOLVE, 2, None),
  ExprOp.CALL: OpInfo(ExprOp.CALL, None, None),
  ExprOp.VMAP: OpInfo(ExprOp.VMAP, None, None),
  ExprOp.SCAN: OpInfo(ExprOp.SCAN, None, None),
  ExprOp.WHILE: OpInfo(ExprOp.WHILE, None, None),
  ExprOp.SOLVER_CALL: OpInfo(ExprOp.SOLVER_CALL, None, None, differentiable=False),
}


def _asarray(value: Any, *, dtype: DType | str | None = None) -> np.ndarray:
  np_dtype = as_dtype(dtype).numpy() if dtype is not None else np.float64
  return np.asarray(value, dtype=np_dtype)


# Construction-time hash-consing cache. Two ``Expr(...)`` constructions with the same
# (op, args-by-identity, type, name, value-bytes, attrs, lowering) collapse to a single
# Python object — so structural equality becomes identity, ``is`` works as a fast equality
# check, and any pass that builds new graphs gets CSE for free.
#
# WeakValueDictionary lets nodes be garbage-collected when no live reference remains; the
# cache shrinks automatically. Arg refs in cache keys are weakrefs, not raw ``id(...)``
# integers, so CPython id reuse cannot alias a new subgraph to a still-cached old node.
_NODE_CACHE: weakref.WeakValueDictionary[tuple[Any, ...], "Expr"] = weakref.WeakValueDictionary()


def _intern_key(
  op: ExprOp | str,
  args: tuple["Expr", ...],
  type_: "TensorType",
  name: str | None,
  value: np.ndarray | None,
  attrs: dict[str, Any],
  lowering: Lowering,
) -> tuple[Any, ...]:
  op_norm = op if isinstance(op, ExprOp) else ExprOp(op)
  args_key = tuple(weakref.ref(a) for a in args)
  value_key = None if value is None else (value.shape, str(value.dtype), value.tobytes())
  return (op_norm.value, args_key, type_, name, value_key, _attrs_key(attrs), lowering)


@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False)
class Expr:
  op: ExprOp | str
  args: tuple[Expr, ...] = ()
  type: TensorType = field(default_factory=TensorType)
  name: str | None = None
  value: np.ndarray | None = None
  attrs: dict[str, Any] = field(default_factory=dict)
  lowering: Lowering = "auto"
  # Frozen → safe to cache. Populated lazily by structural_key on first call.
  _key_cache: tuple[Any, ...] | None = field(default=None, init=False, repr=False, compare=False)
  _initialized: bool = field(default=False, init=False, repr=False, compare=False)

  __array_priority__ = 1000

  @property
  def id(self) -> int:
    # Interned: two structurally-equal Expr's are the same Python object, so Python's
    # ``id()`` doubles as the structural identity. Stable for the object's lifetime; the
    # WeakValueDictionary releases the slot when GC reclaims the node, so id reuse is
    # safe (any memo dict still holding the node also holds it alive).
    return id(self)

  def __new__(
    cls,
    op: ExprOp | str | None = None,
    args: tuple["Expr", ...] = (),
    type: TensorType | None = None,
    name: str | None = None,
    value: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
    lowering: Lowering = "auto",
  ) -> "Expr":
    if op is None:  # callers like ``copy.copy`` / pickling instantiate w/o args
      return object.__new__(cls)
    type_eff = type if type is not None else TensorType()
    attrs_eff = attrs if attrs is not None else {}
    key = _intern_key(op, args, type_eff, name, value, attrs_eff, lowering)
    cached = _NODE_CACHE.get(key)
    if cached is not None:
      return cached
    instance = object.__new__(cls)
    _NODE_CACHE[key] = instance
    return instance

  def __post_init__(self) -> None:
    # On cache hit, dataclass __init__ re-ran with the same args; everything it set was
    # idempotent, so we only need to mark ``_initialized`` on the first construction.
    if self._initialized:
      return
    if not isinstance(self.op, ExprOp):
      object.__setattr__(self, "op", ExprOp(self.op))
    object.__setattr__(self, "_initialized", True)

  @staticmethod
  def sym(
    name: str,
    shape: int | tuple[int, ...] | None = None,
    *,
    dtype: DType | str = dtypes.float64,
    diff: bool = True,
    lowering: Lowering = "auto",
  ) -> Expr:
    """Create a named symbolic input — the leaf every graph is built from.

    Args:
      name: the input's name, which becomes its name on any ``Function`` declaring it.
      shape: an ``int`` for a rank-1 shape, a tuple as given, or ``None`` for a scalar.
      dtype: the element type; ``float64`` unless you say otherwise.
      diff: whether derivatives with respect to this input are meaningful. Setting it to
        ``False`` tells AD the input is a constant parameter, so terms through it vanish.
    """
    return Expr(ExprOp.INPUT, type=TensorType(as_shape(shape), dtype=as_dtype(dtype), diff=diff), name=name, lowering=lowering)

  @staticmethod
  def const(value: Any, *, dtype: DType | str | None = None, lowering: Lowering = "auto") -> Expr:
    """Wrap an array or scalar as a constant node.

    Constants are never differentiable: a derivative with respect to one is structurally zero.
    """
    dtype_eff = as_dtype(dtype) if dtype is not None else dtypes.float64
    arr = _asarray(value, dtype=dtype_eff)
    return Expr(ExprOp.CONST, type=TensorType(tuple(arr.shape), dtype=dtype_eff, diff=False), value=arr, lowering=lowering)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.type.shape

  @property
  def size(self) -> int:
    return self.type.size

  def structural_key(self) -> tuple[Any, ...]:
    cached = self._key_cache
    if cached is not None:
      return cached
    value_key = None if self.value is None else (self.value.shape, str(self.value.dtype), self.value.tobytes())
    key = (
      ExprOp(self.op).value,
      self.name,
      self.type.shape,
      self.type.dtype,
      self.type.diff,
      self.lowering,
      _attrs_key(self.attrs),
      value_key,
      tuple(arg.structural_key() for arg in self.args),
    )
    object.__setattr__(self, "_key_cache", key)
    return key

  def structural_hash(self) -> int:
    return hash(self.structural_key())

  def structurally_equal(self, other: Expr) -> bool:
    return isinstance(other, Expr) and self.structural_key() == other.structural_key()

  def debug(self) -> str:
    return format_expr(self)

  def with_lowering(self, lowering: Lowering) -> Expr:
    return Expr(self.op, self.args, self.type, self.name, self.value, self.attrs, lowering)

  def scalar(self) -> Expr:
    """Request scalarized code: expand eligible float64 procedures into scalar calculations, overriding automatic size limits."""
    return self.with_lowering("scalar")

  def block(self) -> Expr:
    """Keep the containing procedure in loopy form, retaining buffers and loops during scalarization."""
    return self.with_lowering("block")

  def opaque(self) -> Expr:
    """Prevent scalar expansion of the containing procedure, retaining its loopy form."""
    return self.with_lowering("opaque")

  def reshape(self, shape: int | tuple[int, ...]) -> Expr:
    shape = as_shape(shape)
    if np.prod(shape, dtype=int) != self.size:
      raise ValueError(f"cannot reshape {self.shape} with {self.size} entries to {shape}")
    return Expr(
      ExprOp.RESHAPE, (self,), TensorType(shape, dtype=self.type.dtype, diff=self.type.diff), attrs={"shape": shape}, lowering=self.lowering
    )

  def vec(self) -> Expr:
    return self.reshape((self.size,))

  def transpose(self, axes: tuple[int, ...] | None = None) -> Expr:
    axes = tuple(reversed(range(len(self.shape)))) if axes is None else tuple(axes)
    if sorted(axes) != list(range(len(self.shape))):
      raise ValueError(f"transpose axes {axes} are not a permutation for shape {self.shape}")
    return Expr(
      ExprOp.TRANSPOSE,
      (self,),
      TensorType(tuple(self.shape[i] for i in axes), dtype=self.type.dtype, diff=self.type.diff),
      attrs={"axes": axes},
      lowering=self.lowering,
    )

  @property
  def T(self) -> Expr:
    return self.transpose()

  def __getitem__(self, index: Any) -> Expr:
    index = _normalize_index(index, self.shape)
    shape = tuple(np.empty(self.shape)[index].shape)
    return Expr(ExprOp.SLICE, (self,), TensorType(shape, dtype=self.type.dtype, diff=self.type.diff), attrs={"index": index}, lowering=self.lowering)

  def gather(self, indices: Any) -> Expr:
    return gather(self, indices)

  def sum(self) -> Expr:
    return Expr(ExprOp.SUM, (self,), TensorType((), dtype=self.type.dtype, diff=self.type.diff), lowering=self.lowering)

  def max(self) -> Expr:
    return reduce_max(self)

  def min(self) -> Expr:
    return reduce_min(self)

  def dot(self, other: Any) -> Expr:
    return dot(self, other)

  def sumsqr(self) -> Expr:
    return sumsqr(self)

  def norm_2(self) -> Expr:
    return norm_2(self)

  def sin(self) -> Expr:
    return unary(ExprOp.SIN, self)

  def cos(self) -> Expr:
    return unary(ExprOp.COS, self)

  def tan(self) -> Expr:
    return unary(ExprOp.TAN, self)

  def asin(self) -> Expr:
    return unary(ExprOp.ASIN, self)

  def acos(self) -> Expr:
    return unary(ExprOp.ACOS, self)

  def atan(self) -> Expr:
    return unary(ExprOp.ATAN, self)

  def atan2(self, other: Any) -> Expr:
    return atan2(self, other)

  def sinh(self) -> Expr:
    return unary(ExprOp.SINH, self)

  def cosh(self) -> Expr:
    return unary(ExprOp.COSH, self)

  def tanh(self) -> Expr:
    return unary(ExprOp.TANH, self)

  def erf(self) -> Expr:
    return unary(ExprOp.ERF, self)

  def exp(self) -> Expr:
    return unary(ExprOp.EXP, self)

  def log(self) -> Expr:
    return unary(ExprOp.LOG, self)

  def sqrt(self) -> Expr:
    return unary(ExprOp.SQRT, self)

  def abs(self) -> Expr:
    return unary(ExprOp.ABS, self)

  def floor(self) -> Expr:
    return unary(ExprOp.FLOOR, self)

  def ceil(self) -> Expr:
    return unary(ExprOp.CEIL, self)

  def minimum(self, other: Any) -> Expr:
    return minimum(self, other)

  def maximum(self, other: Any) -> Expr:
    return maximum(self, other)

  def copysign(self, other: Any) -> Expr:
    return copysign(self, other)

  def isfinite(self) -> Expr:
    return isfinite(self)

  def cast(self, dtype: DType | str) -> Expr:
    return cast(self, dtype)

  def __bool__(self) -> bool:
    # A comparison builds a ``bool`` expression, not a Python truth value; branching on one would
    # silently take the same path for every input. ``sc.where`` is the data-dependent choice.
    raise TypeError("an Expr has no Python truth value; use sc.where for a data-dependent choice")

  def __lt__(self, other: Any) -> Expr:
    return less(self, other)

  def __le__(self, other: Any) -> Expr:
    return less_equal(self, other)

  def __gt__(self, other: Any) -> Expr:
    return greater(self, other)

  def __ge__(self, other: Any) -> Expr:
    return greater_equal(self, other)

  def __and__(self, other: Any) -> Expr:
    return logical_and(self, other)

  def __rand__(self, other: Any) -> Expr:
    return logical_and(other, self)

  def __or__(self, other: Any) -> Expr:
    return logical_or(self, other)

  def __ror__(self, other: Any) -> Expr:
    return logical_or(other, self)

  def __invert__(self) -> Expr:
    return logical_not(self)

  def __neg__(self) -> Expr:
    return unary(ExprOp.NEG, self)

  @staticmethod
  def _defers(other: Any) -> bool:
    """Whether ``other`` opts out of being converted, NumPy's way (``__array_ufunc__ = None``), so its
    reflected operator runs instead: a ``SparseMatrix`` on the right of ``@``, typically."""
    return not isinstance(other, Expr) and getattr(type(other), "__array_ufunc__", False) is None

  def _operand(self, other: Any) -> Expr:
    """``other`` as an operand of ``+``, ``-`` or ``*`` beside this expression: a Python integer next
    to an integer expression is an integer constant, so index arithmetic needs no casts."""
    if not isinstance(other, Expr) and self.type.dtype.is_integer and isinstance(other, (int, np.integer)) and not isinstance(other, bool):
      return Expr.const(other, dtype=self.type.dtype)
    return as_expr(other)

  def __add__(self, other: Any) -> Expr:
    if self._defers(other):
      return NotImplemented
    return binary(ExprOp.ADD, self, self._operand(other))

  def __radd__(self, other: Any) -> Expr:
    return binary(ExprOp.ADD, self._operand(other), self)

  def __sub__(self, other: Any) -> Expr:
    if self._defers(other):
      return NotImplemented
    return binary(ExprOp.SUB, self, self._operand(other))

  def __rsub__(self, other: Any) -> Expr:
    return binary(ExprOp.SUB, self._operand(other), self)

  def __mul__(self, other: Any) -> Expr:
    if self._defers(other):
      return NotImplemented
    return binary(ExprOp.MUL, self, self._operand(other))

  def __rmul__(self, other: Any) -> Expr:
    return binary(ExprOp.MUL, self._operand(other), self)

  def __truediv__(self, other: Any) -> Expr:
    if self._defers(other):
      return NotImplemented
    return binary(ExprOp.DIV, self, as_expr(other))

  def __rtruediv__(self, other: Any) -> Expr:
    return binary(ExprOp.DIV, as_expr(other), self)

  def __pow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, self, as_expr(other))

  def __rpow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, as_expr(other), self)

  def __matmul__(self, other: Any) -> Expr:
    if self._defers(other):
      return NotImplemented
    return matmul(self, as_expr(other))

  def __rmatmul__(self, other: Any) -> Expr:
    return matmul(as_expr(other), self)

  def inputs(self) -> dict[str, Expr]:
    ret: dict[str, Expr] = {}
    for e in topo([self]):
      if e.op == ExprOp.INPUT and e.name is not None:
        ret[e.name] = e
    return ret


def _attrs_key(attrs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
  def key(v: Any) -> Any:
    if isinstance(v, Expr):
      return v.structural_key()
    if hasattr(v, "structural_key") and callable(getattr(v, "structural_key")):
      return v.structural_key()
    if hasattr(v, "name") and hasattr(v, "input_names") and hasattr(v, "output_names"):
      return ("Function", weakref.ref(v))
    if isinstance(v, np.ndarray):
      return ("ndarray", v.shape, str(v.dtype), v.tobytes())
    if isinstance(v, dict):
      return tuple((k, key(x)) for k, x in sorted(v.items()))
    if isinstance(v, slice):
      return ("slice", v.start, v.stop, v.step)
    if isinstance(v, (tuple, list)):
      return tuple(key(x) for x in v)
    return v

  return tuple((k, key(v)) for k, v in sorted(attrs.items()))


def callees_of(node: Expr) -> tuple[Any, ...]:
  """The Functions ``node`` runs: its callee for a call, a map or a loop (and a while loop's
  condition), and none otherwise."""
  if node.op not in CALLEE_OPS:
    return ()
  return (node.attrs["callee"], node.attrs["cond"]) if node.op == ExprOp.WHILE else (node.attrs["callee"],)


def as_expr(x: Any) -> Expr:
  return x if isinstance(x, Expr) else Expr.const(x)


def common_lowering(*exprs: Expr) -> Lowering:
  explicit = {e.lowering for e in exprs if e.lowering != "auto"}
  if len(explicit) == 1:
    return next(iter(explicit))
  if len(explicit) > 1:
    return "auto"
  return "auto"


def diff_any(*exprs: Expr) -> bool:
  return any(e.type.diff for e in exprs)


def op_diff(op: ExprOp | str, *exprs: Expr) -> bool:
  return OP_INFO[ExprOp(op)].differentiable and diff_any(*exprs)


def promote_dtype(*exprs: Expr) -> DType:
  if not exprs:
    return dtypes.float64
  first = exprs[0].type.dtype
  for e in exprs[1:]:
    if e.type.dtype != first:
      raise TypeError(f"mixed-dtype operation not yet supported: {first} vs {e.type.dtype}; insert an explicit cast")
  return first


def unary(op: ExprOp | str, x: Expr) -> Expr:
  return Expr(op, (x,), TensorType(x.shape, dtype=x.type.dtype, diff=op_diff(op, x)), lowering=x.lowering)


def binary(op: ExprOp | str, x: Expr, y: Expr) -> Expr:
  return Expr(
    op,
    (x, y),
    TensorType(broadcast_shape(x.shape, y.shape), dtype=promote_dtype(x, y), diff=op_diff(op, x, y)),
    lowering=common_lowering(x, y),
  )


def atan2(y: Any, x: Any) -> Expr:
  """Two-argument arctangent, elementwise: the angle of the point ``(x, y)``."""
  return binary(ExprOp.ATAN2, as_expr(y), as_expr(x))


def minimum(x: Any, y: Any) -> Expr:
  """Elementwise minimum. Its derivative at a tie follows ``sc.options(nonsmooth=...)``."""
  return binary(ExprOp.MINIMUM, *_operands(x, y))


def maximum(x: Any, y: Any) -> Expr:
  """Elementwise maximum. Its derivative at a tie follows ``sc.options(nonsmooth=...)``."""
  return binary(ExprOp.MAXIMUM, *_operands(x, y))


def _reduce(op: ExprOp, x: Any) -> Expr:
  x = as_expr(x)
  if x.size == 0:
    raise ValueError(f"{op.value} of an empty expression has no value")
  return Expr(op, (x,), TensorType((), dtype=x.type.dtype, diff=x.type.diff), lowering=x.lowering)


def reduce_max(x: Any) -> Expr:
  """Largest entry, as a scalar; NaN if any entry is NaN. Ties follow ``sc.options(nonsmooth=...)``."""
  return _reduce(ExprOp.MAX, x)


def reduce_min(x: Any) -> Expr:
  """Smallest entry, as a scalar; NaN if any entry is NaN. Ties follow ``sc.options(nonsmooth=...)``."""
  return _reduce(ExprOp.MIN, x)


def norm_inf(x: Any) -> Expr:
  """Largest absolute entry, as a scalar."""
  return reduce_max(as_expr(x).abs())


def norm_1(x: Any) -> Expr:
  """Sum of absolute entries, as a scalar."""
  return as_expr(x).abs().sum()


def copysign(x: Any, y: Any) -> Expr:
  """Elementwise magnitude of ``x`` with the sign of ``y``, as C's ``copysign``."""
  return binary(ExprOp.COPYSIGN, as_expr(x), as_expr(y))


def _as_bool(x: Any) -> Expr:
  e = x if isinstance(x, Expr) else Expr.const(x, dtype=dtypes.bool_)
  if not e.type.dtype.is_bool:
    raise TypeError(f"logical operation needs bool operands, got {e.type.dtype}; compare first or use sc.cast")
  return e


def _predicate(op: ExprOp, *xs: Expr) -> Expr:
  shape: tuple[int, ...] = ()
  for x in xs:
    shape = broadcast_shape(shape, x.shape)
  return Expr(op, xs, TensorType(shape, dtype=dtypes.bool_, diff=False), lowering=common_lowering(*xs))


def _compare(op: ExprOp, x: Any, y: Any) -> Expr:
  x, y = _operands(x, y)
  promote_dtype(x, y)
  return _predicate(op, x, y)


def _operands(x: Any, y: Any) -> tuple[Expr, Expr]:
  """Two operands where a Python number takes the dtype of the ``Expr`` beside it."""
  if isinstance(x, Expr) and not isinstance(y, Expr):
    return x, Expr.const(y, dtype=x.type.dtype)
  if isinstance(y, Expr) and not isinstance(x, Expr):
    return Expr.const(x, dtype=y.type.dtype), y
  return as_expr(x), as_expr(y)


def less(x: Any, y: Any) -> Expr:
  """Elementwise ``x < y`` as a ``bool`` expression. Comparisons with NaN are false."""
  return _compare(ExprOp.LT, x, y)


def less_equal(x: Any, y: Any) -> Expr:
  """Elementwise ``x <= y`` as a ``bool`` expression."""
  return _compare(ExprOp.LE, x, y)


def greater(x: Any, y: Any) -> Expr:
  """Elementwise ``x > y``, built as ``y < x``."""
  return _compare(ExprOp.LT, y, x)


def greater_equal(x: Any, y: Any) -> Expr:
  """Elementwise ``x >= y``, built as ``y <= x``."""
  return _compare(ExprOp.LE, y, x)


def equal(x: Any, y: Any) -> Expr:
  """Elementwise ``x == y`` as a ``bool`` expression. ``==`` on two ``Expr`` stays identity, which
  hash-consing relies on, so value equality is spelled out."""
  return _compare(ExprOp.EQ, x, y)


def not_equal(x: Any, y: Any) -> Expr:
  """Elementwise ``x != y`` as a ``bool`` expression. True whenever either side is NaN."""
  return _compare(ExprOp.NE, x, y)


def logical_and(x: Any, y: Any) -> Expr:
  """Elementwise conjunction of two ``bool`` expressions."""
  return _predicate(ExprOp.AND, _as_bool(x), _as_bool(y))


def logical_or(x: Any, y: Any) -> Expr:
  """Elementwise disjunction of two ``bool`` expressions."""
  return _predicate(ExprOp.OR, _as_bool(x), _as_bool(y))


def logical_not(x: Any) -> Expr:
  """Elementwise negation of a ``bool`` expression."""
  return _predicate(ExprOp.NOT, _as_bool(x))


def isfinite(x: Any) -> Expr:
  """Elementwise test that ``x`` is neither infinite nor NaN, as a ``bool`` expression."""
  x = as_expr(x)
  if not x.type.dtype.is_floating:
    raise TypeError(f"isfinite needs a floating operand, got {x.type.dtype}")
  return _predicate(ExprOp.ISFINITE, x)


def where(cond: Any, x: Any, y: Any) -> Expr:
  """Elementwise choice: ``x`` where ``cond`` is true, ``y`` elsewhere, with NumPy broadcasting.

  Both branches are always evaluated, so a branch may produce inf or NaN where it is not chosen
  without affecting the result. The derivative flows through the chosen branch only.
  """
  if isinstance(cond, bool):
    # ``k == 0`` on an expression is Python identity and yields a bool, which would select one
    # branch for good; a constant condition is never what a caller of ``where`` means.
    raise TypeError("where needs a bool expression as its condition, got a Python bool; compare with sc.equal, sc.less, ... (== is identity)")
  c = _as_bool(cond)
  a, b = _operands(x, y)
  dtype = promote_dtype(a, b)
  shape = broadcast_shape(c.shape, broadcast_shape(a.shape, b.shape))
  return Expr(ExprOp.SELECT, (c, a, b), TensorType(shape, dtype=dtype, diff=diff_any(a, b)), lowering=common_lowering(c, a, b))


def cast(x: Any, dtype: DType | str) -> Expr:
  """Convert ``x`` to ``dtype``. To ``bool`` it means ``x != 0``; from ``bool`` it gives 0 or 1.

  A cast between floating types keeps the derivative; any other cast has none.
  """
  x, target = as_expr(x), as_dtype(dtype)
  if x.type.dtype == target:
    return x
  if target.is_bool:
    return not_equal(x, 0)
  diff = x.type.diff and x.type.dtype.is_floating and target.is_floating
  return Expr(ExprOp.CAST, (x,), TensorType(x.shape, dtype=target, diff=diff), lowering=x.lowering)


def _normalize_index(index: Any, shape: tuple[int, ...]) -> tuple[Any, ...]:
  if not isinstance(index, tuple):
    index = (index,)
  if Ellipsis in index:
    ellipsis_pos = index.index(Ellipsis)
    fill = len(shape) - (len(index) - 1)
    if fill < 0:
      raise IndexError(f"too many indices for shape {shape}")
    index = index[:ellipsis_pos] + (slice(None),) * fill + index[ellipsis_pos + 1 :]
  if len(index) > len(shape):
    raise IndexError(f"too many indices for shape {shape}")
  index = index + (slice(None),) * (len(shape) - len(index))
  for item in index:
    if not isinstance(item, int | slice):
      raise TypeError(f"unsupported index component {item!r}")
  return index


def matmul(x: Expr, y: Expr) -> Expr:
  if len(x.shape) == 1 and len(y.shape) == 1:
    if x.shape[0] != y.shape[0]:
      raise ValueError(f"cannot matmul shapes {x.shape} and {y.shape}")
    shape: tuple[int, ...] = ()
  elif len(x.shape) == 2 and len(y.shape) == 1:
    if x.shape[1] != y.shape[0]:
      raise ValueError(f"cannot matmul shapes {x.shape} and {y.shape}")
    shape = (x.shape[0],)
  elif len(x.shape) == 1 and len(y.shape) == 2:
    if x.shape[0] != y.shape[0]:
      raise ValueError(f"cannot matmul shapes {x.shape} and {y.shape}")
    shape = (y.shape[1],)
  elif len(x.shape) == 2 and len(y.shape) == 2:
    if x.shape[1] != y.shape[0]:
      raise ValueError(f"cannot matmul shapes {x.shape} and {y.shape}")
    shape = (x.shape[0], y.shape[1])
  else:
    raise NotImplementedError(f"matmul shape inference for {x.shape} @ {y.shape}")
  return Expr(ExprOp.MATMUL, (x, y), TensorType(shape, dtype=promote_dtype(x, y), diff=diff_any(x, y)), lowering=common_lowering(x, y))


def _square(a: Any, what: str) -> Expr:
  a = as_expr(a)
  if len(a.shape) != 2 or a.shape[0] != a.shape[1]:
    raise ValueError(f"{what} needs a square matrix, got shape {a.shape}")
  if not a.type.dtype.is_floating:
    raise TypeError(f"{what} needs a floating-point matrix, got {a.type.dtype}")
  return a


def _unroll_attr(n: int) -> dict[str, bool]:
  """Whether a dense factorization or solve of order ``n`` becomes straight-line code, decided when
  the graph is built (``sc.options(dense_unroll=...)``), so the choice is part of the graph."""
  return {"unroll": n <= get_options().dense_unroll}


def cholesky(a: Any) -> Expr:
  """The lower Cholesky factor ``L`` of a symmetric positive definite matrix, ``A = L L^T``.

  Only the lower triangle of ``a`` is read; the upper triangle of the result is zero. No check is
  made: a matrix that is not positive definite gives NaN (a square root of a negative number).
  Differentiable, reading the derivative of the lower triangle as that of a symmetric matrix.
  """
  a = _square(a, "cholesky")
  return Expr(ExprOp.CHOLESKY, (a,), TensorType(a.shape, dtype=a.type.dtype, diff=a.type.diff), attrs=_unroll_attr(a.shape[0]), lowering=a.lowering)


def ldl(a: Any) -> Expr:
  """``A = L D L^T`` without pivoting, packed in one matrix: ``L`` (unit lower) below the diagonal,
  ``D`` on it, zeros above. For quasi-definite matrices (positive and negative definite diagonal
  blocks), where every leading pivot is nonzero; a zero pivot gives inf or NaN. Only the lower
  triangle of ``a`` is read."""
  a = _square(a, "ldl")
  return Expr(ExprOp.LDL, (a,), TensorType(a.shape, dtype=a.type.dtype, diff=a.type.diff), attrs=_unroll_attr(a.shape[0]), lowering=a.lowering)


SPARSE_LDL_TABLES = ("a_ptr", "a_rows", "a_src", "l_ptr", "l_rows", "r_cols", "r_pos", "ck_ptr", "ck_q", "ck_width", "ck_len")
"""The analysis tables a ``sparse_ldl_factor`` node carries, as ``linalg.symbolic`` names them (``a_src`` is
its ``a_source``; the ``ck_*`` tables are ``SymbolicLDL.chunks``)."""

SPARSE_LDL_MAX_WIDTH = 8
"""The most columns one chunk of a ``sparse_ldl_factor`` update covers."""


def _pointers_ok(ptr: np.ndarray, size: int) -> bool:
  """``ptr`` a pointer array into ``size`` entries: from 0, not decreasing, ending at ``size``."""
  return bool(ptr.size and ptr[0] == 0 and ptr[-1] == size and np.all(np.diff(ptr) >= 0))


def _in_range(values: np.ndarray, stop: int) -> bool:
  return bool(values.size == 0 or (values.min() >= 0 and values.max() < stop))


def _check_l_pattern(a: dict[str, np.ndarray], n: int, what: str) -> None:
  """The pattern of ``L``: every row inside the matrix and every column's rows below it, sorted."""
  l_ptr, l_rows = a["l_ptr"], a["l_rows"]
  if not _pointers_ok(l_ptr, l_rows.size) or not _in_range(l_rows, n):
    raise ValueError(f"{what}: l_ptr and l_rows do not describe the columns of an order-{n} L")
  col = np.repeat(np.arange(n), np.diff(l_ptr))
  if np.any(l_rows <= col) or np.any((np.diff(l_rows) <= 0) & (col[1:] == col[:-1])):
    raise ValueError(f"{what}: each column of L needs its rows below the diagonal, sorted")


def sparse_ldl_factor(values: Any, tables: dict[str, Any]) -> Expr:
  """The ``L D L^T`` factor of a symmetric matrix with a fixed sparsity pattern, without pivoting, as
  one vector ``[L below the diagonal, CSC | D]`` of the permuted matrix.

  ``values`` holds the matrix entries, and ``tables`` the analysis ``linalg.symbolic`` made of their
  pattern (``SPARSE_LDL_TABLES``): column ``j`` of the permuted lower triangle is
  ``values[a_src[p]]`` at rows ``a_rows[p]``, ``a_ptr[j] <= p < a_ptr[j + 1]``; column ``j`` of ``L``
  has rows ``l_rows[l_ptr[j]:l_ptr[j + 1]]``; row ``j`` of ``L`` lists its columns ``r_cols`` and the
  positions ``r_pos`` of its entries, cut into chunks of consecutive entries whose columns have the
  same rows from ``j`` down (``ck_*``: per column a range of chunks, each its first entry, width and
  number of rows). The generated code is a left-looking factorization that updates a work column
  from each chunk in one pass, keeping the update order of one column at a time, so the result is
  the same as ``SparseLDL(schedule="scan")``'s, bit for bit but for the sign of a zero or a NaN.
  The builder checks every table the generated code indexes; nothing is checked at run time: a zero
  pivot gives inf or NaN, as for ``ldl``. The derivative is not implemented: ``SparseLDL.solve``
  differentiates implicitly and never needs it, and ``SparseLDL(schedule="scan")`` differentiates
  the factorization through its loops."""
  values = as_expr(values)
  if len(values.shape) != 1 or not values.type.dtype.is_floating:
    raise ValueError(f"sparse_ldl_factor needs a floating-point vector of matrix entries, got {values.type.dtype}{values.shape}")
  missing = [k for k in SPARSE_LDL_TABLES if k not in tables]
  if missing:
    raise ValueError(f"sparse_ldl_factor needs the tables {missing}")
  attrs = {k: np.ascontiguousarray(np.asarray(tables[k], dtype=np.int64).reshape(-1)) for k in SPARSE_LDL_TABLES}
  n = attrs["a_ptr"].size - 1
  if n < 0 or any(attrs[k].size != n + 1 for k in ("l_ptr", "ck_ptr")):
    raise ValueError("sparse_ldl_factor tables disagree on the order of the matrix")
  if attrs["a_src"].size and (attrs["a_src"].min() < 0 or attrs["a_src"].max() >= values.size):
    raise ValueError(f"sparse_ldl_factor reads entries outside its {values.size} values")
  if attrs["ck_width"].size and (attrs["ck_width"].min() < 1 or attrs["ck_width"].max() > SPARSE_LDL_MAX_WIDTH):
    raise ValueError(f"sparse_ldl_factor chunks cover 1 to {SPARSE_LDL_MAX_WIDTH} columns")
  # Everything the generated code indexes stays inside its table: nothing is checked at run time.
  nnz_l, a = attrs["l_rows"].size, attrs
  if not _pointers_ok(a["a_ptr"], a["a_rows"].size) or a["a_src"].size != a["a_rows"].size or not _in_range(a["a_rows"], n):
    raise ValueError("sparse_ldl_factor: a_ptr, a_rows and a_src do not describe the columns of the matrix")
  _check_l_pattern(a, n, "sparse_ldl_factor")
  if a["r_cols"].size != nnz_l or a["r_pos"].size != nnz_l or not _in_range(a["r_cols"], n) or not _in_range(a["r_pos"], nnz_l):
    raise ValueError("sparse_ldl_factor: r_cols and r_pos need one entry per entry of L")
  chunks = a["ck_q"].size
  if not _pointers_ok(a["ck_ptr"], chunks) or a["ck_width"].size != chunks or a["ck_len"].size != chunks or not _in_range(a["ck_q"], nnz_l):
    raise ValueError("sparse_ldl_factor: the chunk tables disagree")
  last = a["ck_q"] + a["ck_width"]
  if np.any(last > nnz_l) or np.any(a["ck_len"] < 0):
    raise ValueError("sparse_ldl_factor: a chunk runs past the entries of L")
  for k in range(SPARSE_LDL_MAX_WIDTH):
    live = a["ck_width"] > k
    if np.any(a["r_pos"][a["ck_q"][live] + k] + a["ck_len"][live] > nnz_l):
      raise ValueError("sparse_ldl_factor: a chunk's rows run past the entries of L")
  size = attrs["l_rows"].size + n
  return Expr(
    ExprOp.SPARSE_LDL, (values,), TensorType((size,), dtype=values.type.dtype, diff=values.type.diff), attrs=attrs, lowering=values.lowering
  )


SPARSE_LDL_SOLVE_TABLES = ("perm", "l_ptr", "l_rows", "sn_first", "sn_width")
"""The tables a ``sparse_ldl_solve`` node carries: the ordering (``perm[new] = old``), the pattern of
``L`` and its chunks of columns (``SymbolicLDL.solve_chunks``)."""


def sparse_ldl_solve(factor: Any, b: Any, tables: dict[str, Any]) -> Expr:
  """``K^{-1} b`` from the ``[L | D]`` of ``sparse_ldl_factor`` (or of ``SparseLDL``), for the
  analysis in ``tables`` (``SPARSE_LDL_SOLVE_TABLES``).

  ``b`` permuted, the unit lower sweep, then the diagonal and the transposed sweep, and the result
  permuted back as each unknown is found. The forward sweep takes consecutive columns that form a
  chain, each column's rows the next column followed by that column's rows (a supernode), in chunks
  of up to ``SPARSE_LDL_MAX_WIDTH`` (``sn_first``, ``sn_width``): the chain's own rows column by
  column, then each shared row once for the whole chunk, the sum in a register. Every entry sees
  its updates in the order of one column at a time, and the transposed sweep sums as the ``scan``
  schedule does, so the result is that schedule's, bit for bit but for the sign of a zero or a NaN.
  Linear in ``b``, with that derivative; the derivative in the factor is not implemented
  (``SparseLDL.solve`` differentiates implicitly and never needs it)."""
  factor, b = as_expr(factor), as_expr(b)
  missing = [k for k in SPARSE_LDL_SOLVE_TABLES if k not in tables]
  if missing:
    raise ValueError(f"sparse_ldl_solve needs the tables {missing}")
  attrs = {k: np.ascontiguousarray(np.asarray(tables[k], dtype=np.int64).reshape(-1)) for k in SPARSE_LDL_SOLVE_TABLES}
  n = attrs["perm"].size
  if attrs["l_ptr"].size != n + 1 or factor.shape != (attrs["l_rows"].size + n,) or b.shape != (n,):
    raise ValueError(
      f"sparse_ldl_solve of order {n} needs a factor of {attrs['l_rows'].size + n} and a right-hand side of {n}, got {factor.shape} and {b.shape}"
    )
  first, width = attrs["sn_first"], attrs["sn_width"]
  if width.size != first.size or int(width.sum()) != n or not np.array_equal(first, np.cumsum(width) - width):
    raise ValueError("sparse_ldl_solve chunks must cover every column once, in order")
  if width.size and (width.min() < 1 or width.max() > SPARSE_LDL_MAX_WIDTH):
    raise ValueError(f"sparse_ldl_solve chunks cover 1 to {SPARSE_LDL_MAX_WIDTH} columns")
  if not np.array_equal(np.sort(attrs["perm"]), np.arange(n)):
    raise ValueError("sparse_ldl_solve: perm must be a permutation of the columns")
  _check_l_pattern(attrs, n, "sparse_ldl_solve")
  l_ptr, l_rows = attrs["l_ptr"], attrs["l_rows"]
  for f, w in zip(first.tolist(), width.tolist(), strict=True):
    for c in range(f, f + w - 1):  # a chain: each column's rows the next column, then that column's rows
      rows = l_rows[l_ptr[c] : l_ptr[c + 1]]
      if rows.size == 0 or rows[0] != c + 1 or not np.array_equal(rows[1:], l_rows[l_ptr[c + 1] : l_ptr[c + 2]]):
        raise ValueError(f"sparse_ldl_solve: columns {f}..{f + w - 1} are not a chain")
  return Expr(
    ExprOp.SPARSE_LDL_SOLVE,
    (factor, b),
    TensorType((n,), dtype=promote_dtype(factor, b), diff=diff_any(factor, b)),
    attrs=attrs,
    lowering=common_lowering(factor, b),
  )


def solve_triangular(t: Any, b: Any, *, lower: bool = True, trans: bool = False, unit_diagonal: bool = False) -> Expr:
  """``X`` with ``op(T) X = B``, ``op(T) = T`` or ``T^T``, for a triangular ``T``; ``B`` a vector or a
  matrix of right-hand sides. Only the triangle named by ``lower`` is read, and its diagonal only
  when ``unit_diagonal`` is false."""
  t = _square(t, "solve_triangular")
  b = as_expr(b)
  if len(b.shape) not in (1, 2) or b.shape[0] != t.shape[0]:
    raise ValueError(f"solve_triangular with a {t.shape} matrix needs a right-hand side of {t.shape[0]} rows, got shape {b.shape}")
  return Expr(
    ExprOp.TRISOLVE,
    (t, b),
    TensorType(b.shape, dtype=promote_dtype(t, b), diff=diff_any(t, b)),
    attrs={"lower": bool(lower), "trans": bool(trans), "unit": bool(unit_diagonal), **_unroll_attr(t.shape[0])},
    lowering=common_lowering(t, b),
  )


def dot(x: Any, y: Any) -> Expr:
  """Scalar product of two expressions with the same number of entries, whatever their shapes."""
  x, y = as_expr(x), as_expr(y)
  if x.size != y.size:
    raise ValueError(f"dot size mismatch: {x.shape} has {x.size} entries, {y.shape} has {y.size}")
  return (x.vec() * y.vec()).sum()


def sumsqr(x: Any) -> Expr:
  """Sum of squares of every entry: ``dot(x, x)``, a scalar."""
  x = as_expr(x)
  return dot(x, x)


def norm_2(x: Any) -> Expr:
  """Euclidean norm of every entry, as a scalar."""
  return sumsqr(x).sqrt()


def linear_combination(terms: dict[str, Expr], multipliers: dict[str, Expr], names: list[str]) -> Expr:
  out: Expr | None = None
  for name in names:
    term = dot(multipliers[name], terms[name])
    out = term if out is None else out + term
  return as_expr(0.0) if out is None else out


def _index_array(indices: Any, upper_bound: int) -> np.ndarray:
  idx = np.asarray(indices, dtype=np.int64)
  if idx.size and (idx.min() < 0 or idx.max() >= upper_bound):
    raise IndexError(f"indices must be in [0, {upper_bound})")
  return idx


def gather(x: Any, indices: Any) -> Expr:
  """Read ``x`` at flat ``indices``. The result takes the shape of ``indices``."""
  x = as_expr(x)
  idx = _index_array(indices, x.size)
  return Expr(ExprOp.GATHER, (x,), TensorType(tuple(idx.shape), dtype=x.type.dtype, diff=x.type.diff), attrs={"indices": idx}, lowering=x.lowering)


def scatter(values: Any, indices: Any, shape: int | tuple[int, ...]) -> Expr:
  """Place ``values`` at flat ``indices`` in a zero tensor of ``shape``.

  Repeated indices accumulate. ``indices`` must have as many entries as ``values``.
  """
  values = as_expr(values)
  shape = as_shape(shape)
  idx = _index_array(indices, int(np.prod(shape, dtype=int)))
  if idx.size != values.size:
    raise ValueError(f"scatter has {idx.size} indices but values shape {values.shape} has {values.size} entries")
  return Expr(
    ExprOp.SCATTER, (values,), TensorType(shape, dtype=values.type.dtype, diff=values.type.diff), attrs={"indices": idx}, lowering=values.lowering
  )


def index_add(base: Any, indices: Any, values: Any) -> Expr:
  """``base`` with ``values`` added at flat ``indices`` (repeated indices accumulate).

  The same value as ``base + scatter(values, indices, base.shape)``, kept as one update so that a
  loop whose carry is changed only this way can update the carry in place: see ``sc.scan``.
  """
  return _index_update(ExprOp.INDEX_ADD, base, indices, values)


def index_set(base: Any, indices: Any, values: Any) -> Expr:
  """``base`` with the entries at flat ``indices`` replaced by ``values``; the indices must be distinct."""
  return _index_update(ExprOp.INDEX_SET, base, indices, values)


def _index_update(op: ExprOp, base: Any, indices: Any, values: Any) -> Expr:
  base, values = _operands(base, values)
  idx = _index_array(np.asarray(indices).reshape(-1), base.size)
  if idx.size != values.size:
    raise ValueError(f"{op.value} has {idx.size} indices for {values.size} values")
  if op == ExprOp.INDEX_SET and np.unique(idx).size != idx.size:
    raise ValueError("index_set indices must be distinct")
  promote_dtype(base, values)
  return Expr(
    op,
    (base, values.reshape((values.size,))),
    TensorType(base.shape, dtype=base.type.dtype, diff=diff_any(base, values)),
    attrs={"indices": idx},
    lowering=common_lowering(base, values),
  )


# Ops that address memory through an index computed at run time.
RUNTIME_INDEX_OPS = frozenset(
  {ExprOp.TAKE, ExprOp.PUT_ADD, ExprOp.PUT, ExprOp.RAGGED_ADD, ExprOp.RAGGED_DOT, ExprOp.SPARSE_LDL, ExprOp.SPARSE_LDL_SOLVE}
)


def _runtime_indices(indices: Any, op: str) -> Expr:
  idx = as_expr(indices) if isinstance(indices, Expr) else Expr.const(np.asarray(indices, dtype=np.int64).reshape(-1), dtype=dtypes.int64)
  if idx.type.dtype != dtypes.int64 or len(idx.shape) != 1:
    raise TypeError(f"{op} needs a rank-1 int64 index vector, got {idx.type.dtype}{idx.shape}")
  return idx


def take(x: Any, indices: Any, *, fill: float = 0.0, in_range: bool = False) -> Expr:
  """``out[..., j] = x[..., indices[j]]``, with ``indices`` an ``int64`` vector known only at run time.

  The last axis of ``x`` is indexed; leading axes are kept, so ``x`` of shape ``(..., n)`` and
  ``L`` indices give shape ``(..., L)``. An index outside ``[0, n)`` reads ``fill``: padding a
  ragged index list with ``n`` (or ``-1``) is the intended use, and no index reads out of bounds.
  Differentiable in ``x``; the adjoint is ``put_add``. Unlike ``gather``, whose indices are fixed
  when the graph is built, the indices here may be any ``int64`` expression: a slice of a table
  selected by a loop's step number, typically.

  ``in_range=True`` promises every index is inside ``[0, n)``: the generated code then reads without
  a check, and an index outside reads whatever memory it reaches. For library code whose padded
  tables point at real entries.
  """
  x = as_expr(x)
  idx = _runtime_indices(indices, "take")
  if not x.shape:
    raise ValueError("take needs an array to index, got a scalar")
  shape = (*x.shape[:-1], idx.size)
  return Expr(
    ExprOp.TAKE,
    (x, idx),
    TensorType(shape, dtype=x.type.dtype, diff=x.type.diff),
    attrs={"fill": float(fill), **({"in_range": True} if in_range else {})},
    lowering=common_lowering(x, idx),
  )


def put_add(base: Any, indices: Any, values: Any, *, in_range: bool = False) -> Expr:
  """``base`` with ``values[..., j]`` added at ``[..., indices[j]]``; repeated indices accumulate.

  The run-time-index counterpart of ``index_add``, on the last axis of ``base``: ``base`` has shape
  ``(..., n)`` and ``values`` shape ``(..., L)`` for ``L`` indices. An index outside ``[0, n)``
  drops its value (each such lane writes a scratch slot of its own, so padded lanes never form a
  chain of updates to one address). ``in_range=True`` promises every index is inside ``[0, n)`` and
  drops the check (see ``take``).
  """
  return _put(ExprOp.PUT_ADD, base, indices, values, in_range)


def put(base: Any, indices: Any, values: Any, *, in_range: bool = False) -> Expr:
  """``base`` with the entries at ``[..., indices[j]]`` replaced by ``values[..., j]``.

  With repeated indices inside ``[0, n)`` the last write wins, and only its value gets a derivative.
  An index outside ``[0, n)`` drops its value; ``in_range=True`` promises there is none.
  """
  return _put(ExprOp.PUT, base, indices, values, in_range)


def _put(op: ExprOp, base: Any, indices: Any, values: Any, in_range: bool = False) -> Expr:
  base, values = _operands(base, values)
  idx = _runtime_indices(indices, op.value)
  if not base.shape:
    raise ValueError(f"{op.value} needs an array to update, got a scalar")
  expected = (*base.shape[:-1], idx.size)
  if values.shape != expected:
    raise ValueError(f"{op.value} into {base.shape} with {idx.size} indices needs values of shape {expected}, got {values.shape}")
  promote_dtype(base, values)
  return Expr(
    op,
    (base, idx, values),
    TensorType(base.shape, dtype=base.type.dtype, diff=diff_any(base, values)),
    attrs={"in_range": True} if in_range else {},
    lowering=common_lowering(base, idx, values),
  )


def _ragged_map(table: Any, what: str) -> np.ndarray | None:
  if table is None:
    return None
  arr = np.asarray(table, dtype=np.int64).reshape(-1)
  if arr.size and arr.min() < 0:
    raise ValueError(f"{what} entries must be non-negative")
  return arr


def _ragged_bounds(lo: Any, hi: Any) -> tuple[Expr, Expr]:
  lo, hi = as_expr(lo), as_expr(hi)
  for e, what in ((lo, "lo"), (hi, "hi")):
    if e.type.dtype != dtypes.int64 or len(e.shape) != 1:
      raise TypeError(f"ragged {what} must be a rank-1 int64 vector, got {e.type.dtype}{e.shape}")
  if lo.shape != hi.shape:
    raise ValueError(f"ragged lo and hi must have one entry per group, got {lo.shape} and {hi.shape}")
  return lo, hi


def ragged_add(base: Any, src: Any, lo: Any, hi: Any, scale: Any, *, dst_map: Any = None, src_map: Any = None) -> Expr:
  """``base`` plus, for every group ``g`` and every ``p`` with ``lo[g] <= p < hi[g]``,
  ``src[src_map[p]] * scale[g]`` added at ``dst_map[p]`` (``None`` maps are the identity).

  Ranges of run-time length: the inner loop of a sparse update such as a left-looking column
  update (``w[rows[p]] -= L[p] * s_k`` over a contiguous run of a column). ``lo``/``hi`` are ``int64``
  vectors, ``scale`` a float vector, all with one entry per group; a group with ``lo == hi`` does
  nothing. The maps are fixed tables. Nothing is checked at run time: every ``p`` must lie inside
  the maps and every mapped index inside ``base`` and ``src``. Library code builds the ranges from
  its own tables.
  """
  base, src, scale = as_expr(base), as_expr(src), as_expr(scale)
  lo, hi = _ragged_bounds(lo, hi)
  if len(base.shape) != 1 or len(src.shape) != 1 or scale.shape != lo.shape:
    raise ValueError(f"ragged_add needs vectors and one scale per group, got base {base.shape}, src {src.shape}, scale {scale.shape}")
  promote_dtype(base, src, scale)
  attrs = {"dst_map": _ragged_map(dst_map, "dst_map"), "src_map": _ragged_map(src_map, "src_map")}
  return Expr(
    ExprOp.RAGGED_ADD,
    (base, src, lo, hi, scale),
    TensorType(base.shape, dtype=base.type.dtype, diff=diff_any(base, src, scale)),
    attrs=attrs,
    lowering=common_lowering(base, src, lo, hi, scale),
  )


def ragged_dot(a: Any, b: Any, lo: Any, hi: Any, *, a_map: Any = None, b_map: Any = None) -> Expr:
  """One dot product per group: ``out[g] = sum over lo[g] <= p < hi[g] of a[a_map[p]] * b[b_map[p]]``
  (``None`` maps are the identity). The counterpart of ``ragged_add``, with the same unchecked
  contract; ``ragged_add``'s derivative with respect to its scale is one of these."""
  a, b = as_expr(a), as_expr(b)
  lo, hi = _ragged_bounds(lo, hi)
  if len(a.shape) != 1 or len(b.shape) != 1:
    raise ValueError(f"ragged_dot needs vectors, got {a.shape} and {b.shape}")
  dtype = promote_dtype(a, b)
  return Expr(
    ExprOp.RAGGED_DOT,
    (a, b, lo, hi),
    TensorType(lo.shape, dtype=dtype, diff=diff_any(a, b)),
    attrs={"a_map": _ragged_map(a_map, "a_map"), "b_map": _ragged_map(b_map, "b_map")},
    lowering=common_lowering(a, b, lo, hi),
  )


def segment_sum(values: Any, segment_ids: Any, num_segments: int) -> Expr:
  """Sum ``values`` into ``num_segments`` bins: entry ``k`` adds into bin ``segment_ids[k]``.

  The ids are fixed when the graph is built, which is what lets code generation pick an
  implementation for this exact pattern. Empty bins are zero. The same as ``scatter`` into a
  vector, spelled the way sparse kernels read.
  """
  values = as_expr(values)
  return scatter(values.reshape((values.size,)), np.asarray(segment_ids).reshape(-1), (int(num_segments),))


def _segment_extremum(op: ExprOp, values: Any, segment_ids: Any, num_segments: int, fill: float | None) -> Expr:
  values = as_expr(values)
  ids = _index_array(np.asarray(segment_ids).reshape(-1), int(num_segments))
  if ids.size != values.size:
    raise ValueError(f"{op.value} has {ids.size} segment ids for {values.size} values")
  if fill is None:
    fill = -math.inf if op == ExprOp.SEGMENT_MAX else math.inf
  return Expr(
    op,
    (values.reshape((values.size,)),),
    TensorType((int(num_segments),), dtype=values.type.dtype, diff=values.type.diff),
    attrs={"indices": ids, "fill": float(fill)},
    lowering=values.lowering,
  )


def segment_max(values: Any, segment_ids: Any, num_segments: int, *, fill: float | None = None) -> Expr:
  """Largest value in each of ``num_segments`` bins; ``fill`` (default ``-inf``) where a bin is empty.

  NaN propagates within its bin. Ties follow ``sc.options(nonsmooth=...)``.
  """
  return _segment_extremum(ExprOp.SEGMENT_MAX, values, segment_ids, num_segments, fill)


def segment_min(values: Any, segment_ids: Any, num_segments: int, *, fill: float | None = None) -> Expr:
  """Smallest value in each of ``num_segments`` bins; ``fill`` (default ``inf``) where a bin is empty."""
  return _segment_extremum(ExprOp.SEGMENT_MIN, values, segment_ids, num_segments, fill)


def split(x: Any, sections: int | Iterable[int], *, axis: int = 0) -> tuple[Expr, ...]:
  """Split ``x`` along ``axis``, into ``sections`` equal parts or at the given boundaries."""
  x = as_expr(x)
  axis = axis if axis >= 0 else axis + len(x.shape)
  if axis < 0 or axis >= len(x.shape):
    raise ValueError(f"split axis {axis} out of bounds for shape {x.shape}")
  if isinstance(sections, int):
    if sections <= 0 or x.shape[axis] % sections != 0:
      raise ValueError(f"cannot split axis of length {x.shape[axis]} into {sections} equal sections")
    sizes = (x.shape[axis] // sections,) * sections
  else:
    sizes = tuple(int(s) for s in sections)
    if any(s < 0 for s in sizes):
      raise ValueError(f"split sizes {sizes} cannot contain negative entries")
    if sum(sizes) != x.shape[axis]:
      raise ValueError(f"split sizes {sizes} do not sum to axis length {x.shape[axis]}")
  ret: list[Expr] = []
  start = 0
  for size in sizes:
    index = [slice(None)] * len(x.shape)
    index[axis] = slice(start, start + size)
    ret.append(x[tuple(index)])
    start += size
  return tuple(ret)


def stack(xs: Iterable[Any], *, axis: int = 0) -> Expr:
  """Join equally shaped expressions along a new axis, adding one to the rank."""
  exprs = tuple(as_expr(x) for x in xs)
  if not exprs:
    raise ValueError("stack requires at least one expression")
  base = exprs[0].shape
  if any(e.shape != base for e in exprs):
    raise ValueError("stack arguments must have identical shape")
  axis = axis if axis >= 0 else axis + len(base) + 1
  if axis < 0 or axis > len(base):
    raise ValueError(f"stack axis {axis} out of bounds for shape {base}")
  shape = base[:axis] + (len(exprs),) + base[axis:]
  return Expr(
    ExprOp.STACK, exprs, TensorType(shape, dtype=promote_dtype(*exprs), diff=diff_any(*exprs)), attrs={"axis": axis}, lowering=common_lowering(*exprs)
  )


def concat(xs: Iterable[Any], *, axis: int = 0) -> Expr:
  """Join expressions along an existing ``axis``. All other dimensions must agree."""
  exprs = tuple(as_expr(x) for x in xs)
  if not exprs:
    raise ValueError("concat requires at least one expression")
  base = exprs[0].shape
  axis = axis if axis >= 0 else axis + len(base)
  if axis < 0 or axis >= len(base):
    raise ValueError(f"concat axis {axis} out of bounds for shape {base}")
  for e in exprs:
    if len(e.shape) != len(base) or any(a != b for i, (a, b) in enumerate(zip(e.shape, base, strict=True)) if i != axis):
      raise ValueError(f"cannot concat shapes {[x.shape for x in exprs]} along axis {axis}")
  shape = base[:axis] + (sum(e.shape[axis] for e in exprs),) + base[axis + 1 :]
  return Expr(
    ExprOp.CONCAT,
    exprs,
    TensorType(shape, dtype=promote_dtype(*exprs), diff=diff_any(*exprs)),
    attrs={"axis": axis},
    lowering=common_lowering(*exprs),
  )


def vec(x: Any) -> Expr:
  """Flatten ``x`` to rank 1 in row-major order."""
  return as_expr(x).vec()


def zeros_like(x: Expr) -> Expr:
  return Expr.const(np.zeros(x.shape, dtype=np.float64), lowering=x.lowering)


def format_expr(outputs: Expr | Iterable[Expr]) -> str:
  """Render a graph as a topological dump with stable ``%0``, ``%1``, ... names.

  For the stable assembly form meant for diffs and tests, use ``render_expr_assembly``.
  """
  outs = (outputs,) if isinstance(outputs, Expr) else tuple(outputs)
  nodes = topo(outs)
  loc = {e.id: i for i, e in enumerate(nodes)}
  lines: list[str] = []
  for i, e in enumerate(nodes):
    lhs = f"%{i}"
    if e.op == ExprOp.INPUT:
      rhs = f"input {e.name}"
    elif e.op == ExprOp.CONST:
      assert e.value is not None
      rhs = f"const {np.array2string(e.value, threshold=6)}"
    elif e.op == ExprOp.CALL:
      callee = e.attrs["callee"]
      rhs = f"call {callee.name}[{e.attrs['output']}]({', '.join(f'%{loc[a.id]}' for a in e.args)})"
    elif e.op == ExprOp.VMAP:
      callee = e.attrs["callee"]
      length = e.attrs["length"]
      slice_size = e.attrs["slice_size"]
      bindings = ", ".join(f"%{loc[a.id]}[{s}::{st}]" for a, s, st in zip(e.args, e.attrs["starts"], e.attrs["strides"], strict=True))
      rhs = f"vmap[{length}x{slice_size}] {callee.name}[{e.attrs['output']}]({bindings})"
    elif e.op == ExprOp.SCAN:
      callee = e.attrs["callee"]
      starts, strides = e.attrs["starts"], e.attrs["strides"]
      bindings = ", ".join([f"%{loc[e.args[0].id]}", *(f"%{loc[a.id]}[{s}::{st}]" for a, s, st in zip(e.args[1:], starts, strides, strict=True))])
      rhs = f"scan[{e.attrs['length']}] {callee.name}[{e.attrs['output']}]({bindings})"
    elif e.op == ExprOp.WHILE:
      bindings = ", ".join(f"%{loc[a.id]}" for a in e.args)
      index = " index" if e.attrs.get("index") else ""
      rhs = f"while[{e.attrs['max_iter']}{index}] {e.attrs['cond'].name} {e.attrs['callee'].name}[{e.attrs['output']}]({bindings})"
    else:
      rhs = f"{ExprOp(e.op).value}({', '.join(f'%{loc[a.id]}' for a in e.args)})"
    lines.append(f"{lhs} = {rhs} : {e.type.dtype}{e.shape}")
  lines.append("outputs " + ", ".join(f"%{loc[e.id]}" for e in outs))
  return "\n".join(lines)


def substitute(expr: Expr, replacements: Mapping[Expr, Expr]) -> Expr:
  """Replace expressions simultaneously and rebuild only their changed ancestors."""
  for source, replacement in replacements.items():
    if source.shape != replacement.shape or source.type.dtype != replacement.type.dtype:
      raise ValueError(
        f"cannot substitute {source.name or source.op} {source.type.dtype}{source.shape} with "
        f"{replacement.name or replacement.op} {replacement.type.dtype}{replacement.shape}"
      )
  rebuilt = {source.id: replacement for source, replacement in replacements.items()}
  for node in topo((expr,)):
    if node.id in rebuilt:
      continue
    args = tuple(rebuilt[arg.id] for arg in node.args)
    rebuilt[node.id] = (
      node
      if all(before is after for before, after in zip(node.args, args, strict=True))
      else Expr(node.op, args, node.type, node.name, node.value, dict(node.attrs), node.lowering)
    )
  return rebuilt[expr.id]


def independent(exprs: Iterable[Expr], wrts: Iterable[Expr]) -> tuple[tuple[Expr, ...], tuple[Expr, ...], dict[Expr, Expr]]:
  """Make every ``wrt`` an input before differentiating: ``(exprs, wrts, back)``.

  A derivative treats each ``wrt`` as an independent variable. One that is not an input, such as a
  slice of the carry in a loop body, is replaced in ``exprs`` by a stand-in input; ``substitute(d,
  back)`` puts it back into a derivative ``d``. Differentiating at the original node instead would
  lose every dependence that a pass folds past it (``x[:3][1]`` into ``x[1]``), silently.
  """
  exprs, wrts = tuple(exprs), tuple(wrts)
  stand_ins = {w: Expr(ExprOp.INPUT, type=TensorType(w.shape, w.type.dtype, diff=True), name=f"wrt%{w.id}") for w in wrts if w.op != ExprOp.INPUT}
  if not stand_ins:
    return exprs, wrts, {}
  rebuilt = tuple(substitute(e, stand_ins) for e in exprs)
  return rebuilt, tuple(stand_ins.get(w, w) for w in wrts), {v: k for k, v in stand_ins.items()}


def topo(outputs: Iterable[Expr]) -> list[Expr]:
  seen: set[int] = set()
  ret: list[Expr] = []
  stack = [(out, False) for out in reversed(tuple(outputs))]
  while stack:
    expr, visited = stack.pop()
    if expr.id in seen:
      continue
    if visited:
      seen.add(expr.id)
      ret.append(expr)
      continue
    stack.append((expr, True))
    stack.extend((arg, False) for arg in reversed(expr.args) if arg.id not in seen)
  return ret
