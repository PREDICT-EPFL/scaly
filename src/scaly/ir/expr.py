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
  RESHAPE = "reshape"
  TRANSPOSE = "transpose"
  SLICE = "slice"
  GATHER = "gather"
  SCATTER = "scatter"
  STACK = "stack"
  CONCAT = "concat"
  MATMUL = "matmul"
  CALL = "call"
  VMAP = "vmap"
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
  ExprOp.RESHAPE,
  ExprOp.TRANSPOSE,
  ExprOp.SLICE,
  ExprOp.GATHER,
  ExprOp.SCATTER,
  ExprOp.STACK,
  ExprOp.CONCAT,
  ExprOp.MATMUL,
  ExprOp.CALL,
  ExprOp.VMAP,
  ExprOp.SOLVER_CALL,
}

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
  ExprOp.RESHAPE: OpInfo(ExprOp.RESHAPE, 1, np.reshape),
  ExprOp.TRANSPOSE: OpInfo(ExprOp.TRANSPOSE, 1, np.transpose),
  ExprOp.SLICE: OpInfo(ExprOp.SLICE, 1, None),
  ExprOp.GATHER: OpInfo(ExprOp.GATHER, 1, None),
  ExprOp.SCATTER: OpInfo(ExprOp.SCATTER, 1, None),
  ExprOp.STACK: OpInfo(ExprOp.STACK, None, np.stack),
  ExprOp.CONCAT: OpInfo(ExprOp.CONCAT, None, np.concatenate),
  ExprOp.MATMUL: OpInfo(ExprOp.MATMUL, 2, np.matmul),
  ExprOp.CALL: OpInfo(ExprOp.CALL, None, None),
  ExprOp.VMAP: OpInfo(ExprOp.VMAP, None, None),
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

  def __add__(self, other: Any) -> Expr:
    return binary(ExprOp.ADD, self, as_expr(other))

  def __radd__(self, other: Any) -> Expr:
    return binary(ExprOp.ADD, as_expr(other), self)

  def __sub__(self, other: Any) -> Expr:
    return binary(ExprOp.SUB, self, as_expr(other))

  def __rsub__(self, other: Any) -> Expr:
    return binary(ExprOp.SUB, as_expr(other), self)

  def __mul__(self, other: Any) -> Expr:
    return binary(ExprOp.MUL, self, as_expr(other))

  def __rmul__(self, other: Any) -> Expr:
    return binary(ExprOp.MUL, as_expr(other), self)

  def __truediv__(self, other: Any) -> Expr:
    return binary(ExprOp.DIV, self, as_expr(other))

  def __rtruediv__(self, other: Any) -> Expr:
    return binary(ExprOp.DIV, as_expr(other), self)

  def __pow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, self, as_expr(other))

  def __rpow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, as_expr(other), self)

  def __matmul__(self, other: Any) -> Expr:
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
  flat = idx.reshape(-1)
  if len(set(flat.tolist())) != flat.size:
    raise ValueError("scatter indices must be unique")
  return Expr(
    ExprOp.SCATTER, (values,), TensorType(shape, dtype=values.type.dtype, diff=values.type.diff), attrs={"indices": idx}, lowering=values.lowering
  )


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
