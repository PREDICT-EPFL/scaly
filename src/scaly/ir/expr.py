"""Expression dialect vocabulary: ``ExprOp``, ``OP_INFO``, ``Expr``, interning, builders, topo.

The verify rules are ``ir/expr_spec.py`` and ``function/sugar.py`` owns ``vmap``, the builder that
needs a ``Function``. The assembly printers are ``ir/text.py``. ``format_expr`` stays here with
helpers for type text because ``Expr.debug`` calls it, and moving it would make ``ir/expr.py``
import ``ir/text.py``, which imports ``ir/expr.py``.
"""

from __future__ import annotations

import hashlib
import string
import struct
import weakref
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generator, Iterable, Literal, Mapping

import numpy as np

from .support import Support, compute_support, reduce_entries, slice_entries, transpose_entries, _constant_condition
from .types import (
  CanonicalPattern,
  SparsityPattern,
  DType,
  Lowering,
  TensorType,
  _array_data,
  as_dtype,
  as_shape,
  broadcast_shape,
  dtypes,
  frozen,
  stored_indices,
)


class ExprOp(StrEnum):
  """The expression dialect's operation set.

  A ``StrEnum``, so an op is a proper enum value and still prints and serializes as its name.
  ``OP_INFO`` carries the arity and the differentiability of each.
  """

  INPUT = "input"
  CONST = "const"
  PACK = "pack"
  VALUES = "values"
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
  FLOORDIV = "floordiv"
  MOD = "mod"
  REDUCE = "reduce"
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
  EXTERN = "extern"
  LOOP = "loop"
  SOLVER_CALL = "solver_call"
  PRINT = "print"
  STOP_GRADIENT = "stop_gradient"


CALLEE_OPS = {ExprOp.CALL, ExprOp.VMAP, ExprOp.LOOP, ExprOp.EXTERN}


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
  ExprOp.NOT,
  ExprOp.ISFINITE,
  ExprOp.CAST,
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
  ExprOp.LT,
  ExprOp.LE,
  ExprOp.EQ,
  ExprOp.NE,
  ExprOp.AND,
  ExprOp.OR,
  ExprOp.FLOORDIV,
  ExprOp.MOD,
}

COMMON_ELEMENTWISE_TERNARY = {ExprOp.SELECT}

COMMON_STRUCTURAL = {
  ExprOp.INPUT,
  ExprOp.CONST,
  ExprOp.PACK,
  ExprOp.VALUES,
  ExprOp.REDUCE,
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
  ExprOp.EXTERN,
  ExprOp.LOOP,
  ExprOp.SOLVER_CALL,
  ExprOp.PRINT,
  ExprOp.STOP_GRADIENT,
}

# Deliberately not in the MVP set: expm1/log1p (nice but low priority), splines/interpolants
# (important but require carefully specified extrapolation, knots, derivatives, and codegen tables),
# matrix exponentials/decompositions, and solver/control-flow ops.
COMMON_OPS = COMMON_STRUCTURAL | COMMON_ELEMENTWISE_UNARY | COMMON_ELEMENTWISE_BINARY | COMMON_ELEMENTWISE_TERNARY


@dataclass(frozen=True, slots=True)
class OpInfo:
  """What the compiler knows about one ``ExprOp``: its arity (``None`` for variadic) and whether it
  has derivative rules. ``OP_INFO`` holds one per op.

  ``pattern_rule`` gives the stored pattern of an elementwise op or ``MATMUL`` with a sparse
  operand. An elementwise op without one refuses a sparse operand. Movement ops and reductions
  compute the pattern from the stored coordinates instead. ``support_rule`` names how ``Expr.support`` follows the op, ``None`` when
  any entry may be nonzero.
  """

  op: ExprOp
  arity: int | None
  differentiable: bool = True
  pattern_rule: Literal["preserve", "union", "intersection", "matmul"] | None = None
  support_rule: (
    Literal[
      "const",
      "pack",
      "values",
      "preserve",
      "union",
      "select",
      "numerator",
      "reduce",
      "transpose",
      "slice",
      "gather",
      "scatter",
      "stack",
      "concat",
      "call",
      "vmap",
      "loop",
    ]
    | None
  ) = None

  @property
  def name(self) -> str:
    return self.op.value


OP_INFO: dict[ExprOp, OpInfo] = {
  ExprOp.INPUT: OpInfo(ExprOp.INPUT, 0),
  ExprOp.CONST: OpInfo(ExprOp.CONST, 0, False, support_rule="const"),
  ExprOp.PACK: OpInfo(ExprOp.PACK, 1, support_rule="pack"),
  ExprOp.VALUES: OpInfo(ExprOp.VALUES, 1, support_rule="values"),
  ExprOp.NEG: OpInfo(ExprOp.NEG, 1, pattern_rule="preserve"),
  ExprOp.SIN: OpInfo(ExprOp.SIN, 1, pattern_rule="preserve"),
  ExprOp.COS: OpInfo(ExprOp.COS, 1),
  ExprOp.TAN: OpInfo(ExprOp.TAN, 1, pattern_rule="preserve"),
  ExprOp.ASIN: OpInfo(ExprOp.ASIN, 1, pattern_rule="preserve"),
  ExprOp.ACOS: OpInfo(ExprOp.ACOS, 1),
  ExprOp.ATAN: OpInfo(ExprOp.ATAN, 1, pattern_rule="preserve"),
  ExprOp.SINH: OpInfo(ExprOp.SINH, 1, pattern_rule="preserve"),
  ExprOp.COSH: OpInfo(ExprOp.COSH, 1),
  ExprOp.TANH: OpInfo(ExprOp.TANH, 1, pattern_rule="preserve"),
  ExprOp.ERF: OpInfo(ExprOp.ERF, 1, pattern_rule="preserve"),
  ExprOp.EXP: OpInfo(ExprOp.EXP, 1),
  ExprOp.LOG: OpInfo(ExprOp.LOG, 1),
  ExprOp.SQRT: OpInfo(ExprOp.SQRT, 1, pattern_rule="preserve"),
  ExprOp.ABS: OpInfo(ExprOp.ABS, 1, pattern_rule="preserve"),
  ExprOp.FLOOR: OpInfo(ExprOp.FLOOR, 1, False, pattern_rule="preserve"),
  ExprOp.CEIL: OpInfo(ExprOp.CEIL, 1, False, pattern_rule="preserve"),
  ExprOp.ADD: OpInfo(ExprOp.ADD, 2, pattern_rule="union"),
  ExprOp.SUB: OpInfo(ExprOp.SUB, 2, pattern_rule="union"),
  ExprOp.MUL: OpInfo(ExprOp.MUL, 2, pattern_rule="intersection"),
  ExprOp.DIV: OpInfo(ExprOp.DIV, 2, support_rule="numerator"),
  ExprOp.POW: OpInfo(ExprOp.POW, 2),
  ExprOp.ATAN2: OpInfo(ExprOp.ATAN2, 2),
  ExprOp.MINIMUM: OpInfo(ExprOp.MINIMUM, 2, False, support_rule="union"),
  ExprOp.MAXIMUM: OpInfo(ExprOp.MAXIMUM, 2, False, support_rule="union"),
  # Each predicate or logic op that is false wherever both operands are zero follows their union;
  # the others (x <= x, x == x, not 0, isfinite(0)) are true at zero and have no support rule.
  ExprOp.LT: OpInfo(ExprOp.LT, 2, False, support_rule="union"),
  ExprOp.LE: OpInfo(ExprOp.LE, 2, False),
  ExprOp.EQ: OpInfo(ExprOp.EQ, 2, False),
  ExprOp.NE: OpInfo(ExprOp.NE, 2, False, support_rule="union"),
  ExprOp.AND: OpInfo(ExprOp.AND, 2, False, support_rule="union"),
  ExprOp.OR: OpInfo(ExprOp.OR, 2, False, support_rule="union"),
  ExprOp.NOT: OpInfo(ExprOp.NOT, 1, False),
  ExprOp.ISFINITE: OpInfo(ExprOp.ISFINITE, 1, False),
  ExprOp.SELECT: OpInfo(ExprOp.SELECT, 3, pattern_rule="union", support_rule="select"),
  ExprOp.CAST: OpInfo(ExprOp.CAST, 1, support_rule="preserve"),
  # A zero divisor gives 0, so the dividend's zeros stay.
  ExprOp.FLOORDIV: OpInfo(ExprOp.FLOORDIV, 2, False, support_rule="numerator"),
  ExprOp.MOD: OpInfo(ExprOp.MOD, 2, False, support_rule="numerator"),
  ExprOp.REDUCE: OpInfo(ExprOp.REDUCE, 1, support_rule="reduce"),
  ExprOp.RESHAPE: OpInfo(ExprOp.RESHAPE, 1, support_rule="preserve"),
  ExprOp.TRANSPOSE: OpInfo(ExprOp.TRANSPOSE, 1, support_rule="transpose"),
  ExprOp.SLICE: OpInfo(ExprOp.SLICE, 1, support_rule="slice"),
  ExprOp.GATHER: OpInfo(ExprOp.GATHER, 1, support_rule="gather"),
  ExprOp.SCATTER: OpInfo(ExprOp.SCATTER, 1, support_rule="scatter"),
  ExprOp.STACK: OpInfo(ExprOp.STACK, None, support_rule="stack"),
  ExprOp.CONCAT: OpInfo(ExprOp.CONCAT, None, support_rule="concat"),
  ExprOp.MATMUL: OpInfo(ExprOp.MATMUL, 2, pattern_rule="matmul"),
  ExprOp.CALL: OpInfo(ExprOp.CALL, None, support_rule="call"),
  ExprOp.VMAP: OpInfo(ExprOp.VMAP, None, support_rule="vmap"),
  ExprOp.EXTERN: OpInfo(ExprOp.EXTERN, None),
  ExprOp.LOOP: OpInfo(ExprOp.LOOP, None, support_rule="loop"),
  ExprOp.SOLVER_CALL: OpInfo(ExprOp.SOLVER_CALL, None, False),
  ExprOp.PRINT: OpInfo(ExprOp.PRINT, None, support_rule="preserve"),
  ExprOp.STOP_GRADIENT: OpInfo(ExprOp.STOP_GRADIENT, 1, pattern_rule="preserve"),
}


def _broadcast_pattern(type_: TensorType, shape: tuple[int, int]) -> CanonicalPattern:
  pattern = type_.pattern
  if pattern is None:
    return CanonicalPattern(shape, tuple(r for r in range(shape[0]) for _ in range(shape[1])), tuple(range(shape[1])) * shape[0])
  if pattern.shape == shape:
    return pattern
  if broadcast_shape(pattern.shape, shape) != shape:
    raise ValueError(f"cannot broadcast pattern {pattern.shape} to {shape}")
  coordinates = [
    (r, c)
    for row, col in zip(pattern.rows, pattern.cols, strict=True)
    for r in (range(shape[0]) if pattern.shape[0] == 1 else (row,))
    for c in (range(shape[1]) if pattern.shape[1] == 1 else (col,))
  ]
  return CanonicalPattern(shape, tuple(r for r, _ in coordinates), tuple(c for _, c in coordinates))


def _movement_entries(op: ExprOp | str, types: tuple[TensorType, ...], shape: tuple[int, ...], attrs: Mapping[str, Any]) -> np.ndarray:
  chunks = []
  offset = 0
  for i, type_ in enumerate(types):
    entries = stored_indices(type_)
    if op == ExprOp.TRANSPOSE:
      entries = transpose_entries(entries, type_.shape, attrs["axes"])
    elif op == ExprOp.SLICE:
      entries, _ = slice_entries(entries, type_.shape, attrs["index"])
    elif op == ExprOp.GATHER:
      entries = np.flatnonzero(np.isin(attrs["indices"].reshape(-1), entries))
    elif op == ExprOp.SCATTER:
      entries = attrs["indices"].reshape(-1)[entries]
    elif op in (ExprOp.STACK, ExprOp.CONCAT):
      axis = attrs["axis"]
      coords = list(np.unravel_index(entries, type_.shape))
      if op == ExprOp.STACK:
        coords.insert(axis, np.full(len(entries), i))
      else:
        coords[axis] += offset
        offset += type_.shape[axis]
      entries = np.ravel_multi_index(tuple(coords), shape)
    chunks.append(entries)
  return np.concatenate(chunks)


def stored_pattern(
  op: ExprOp | str, types: tuple[TensorType, ...], shape: tuple[int, ...], attrs: Mapping[str, Any] | None = None
) -> CanonicalPattern | None:
  """Apply the fixed storage rule, also usable for support restricted to sparse operands."""
  if op == ExprOp.SELECT:
    if types[0].pattern is not None and not any(t.pattern is not None for t in types[1:]):
      raise ValueError("select has no stored-pattern rule for a sparse condition with dense branches; use to_dense() explicitly")
    types = types[1:]
  if op == ExprOp.MATMUL and attrs is not None and (sampled := attrs.get("sampled")) is not None:
    if not isinstance(sampled, CanonicalPattern) or sampled.shape != shape or any(t.ndim != 2 or t.pattern is None for t in types):
      raise ValueError("sampled MATMUL requires two sparse matrices and a matching canonical output pattern")
    return sampled
  if not any(t.pattern is not None for t in types):
    return None
  if op == ExprOp.REDUCE:
    assert attrs is not None
    if len(shape) < 2 or (attrs["kind"] != "add" and any(types[0].shape[axis] == 0 for axis in attrs["axes"])):
      return None
    if all(axis < types[0].ndim - 2 for axis in attrs["axes"]) and all(types[0].shape[axis] for axis in attrs["axes"]):
      return types[0].pattern
    entries = reduce_entries(stored_indices(types[0]), types[0].shape, attrs["axes"])
    width = shape[-2] * shape[-1]
    local = np.unique(entries % width) if width else np.empty(0, dtype=np.int64)
    return CanonicalPattern((shape[-2], shape[-1]), tuple(map(int, local // shape[-1])), tuple(map(int, local % shape[-1])))
  if op == ExprOp.RESHAPE and shape == types[0].shape:
    return types[0].pattern
  if op == ExprOp.TRANSPOSE and attrs is not None and attrs["axes"] == tuple(range(types[0].ndim)):
    return types[0].pattern
  if op in (ExprOp.RESHAPE, ExprOp.TRANSPOSE, ExprOp.SLICE, ExprOp.GATHER, ExprOp.SCATTER, ExprOp.STACK, ExprOp.CONCAT):
    if len(shape) < 2:
      return None
    entries = _movement_entries(op, types, shape, attrs or {})
    width = shape[-2] * shape[-1]
    local = np.unique(entries % width) if width else np.empty(0, dtype=np.int64)
    return CanonicalPattern((shape[-2], shape[-1]), tuple(map(int, local // shape[-1])), tuple(map(int, local % shape[-1])))
  rule = OP_INFO[ExprOp(op)].pattern_rule
  if rule == "matmul" and len(shape) < 2:
    return None
  if rule is None or len(shape) < 2:
    raise ValueError(f"{op} has no stored-pattern rule for sparse operands with output shape {shape}; use to_dense() explicitly")
  if rule == "preserve":
    return types[0].pattern
  matrix_shape = (shape[-2], shape[-1])
  if rule == "intersection" and any(t.pattern is None for t in types):
    return _broadcast_pattern(next(t for t in types if t.pattern is not None), matrix_shape)
  if rule == "matmul":
    if any(t.ndim != 2 for t in types):
      raise ValueError(f"{op} has no stored-pattern rule for these ranks; use to_dense() explicitly")
    x, y = types
    if x.pattern is None or y.pattern is None:
      rows = range(x.shape[0]) if x.pattern is None else sorted(set(x.pattern.rows))
      cols = range(y.shape[1]) if y.pattern is None else sorted(set(y.pattern.cols))
      return CanonicalPattern(matrix_shape, tuple(r for r in rows for _ in cols), tuple(c for _ in rows for c in cols))
    return _broadcast_pattern(x, (x.shape[0], x.shape[1])).matmul(_broadcast_pattern(y, (y.shape[0], y.shape[1])))
  a, b = (_broadcast_pattern(t, matrix_shape) for t in types)
  return a.union(b) if rule == "union" else a.intersection(b)


def _pattern_text(pattern: CanonicalPattern | None) -> str:
  return "" if pattern is None else f" nnz={pattern.nnz} pattern={pattern.digest[:12]}"


def _type_text(type_: TensorType) -> str:
  return f"{type_.dtype}{type_.shape}{_pattern_text(type_.pattern)}"


def check_stored_pattern(op: ExprOp | str, args: tuple[Expr, ...], type_: TensorType, attrs: Mapping[str, Any] | None = None) -> None:
  """Reject an unsupported sparse operation or an output pattern that disagrees with its rule."""
  if op == ExprOp.INPUT:
    return
  if op == ExprOp.PACK:
    if type_.pattern is None or args[0].type.pattern is not None or args[0].shape != type_.storage_shape:
      raise ValueError("PACK requires dense stored values matching the sparse storage shape")
    return
  if op == ExprOp.VALUES:
    if args[0].type.pattern is None or type_.pattern is not None or type_.shape != args[0].type.storage_shape:
      raise ValueError("VALUES requires a sparse operand and its dense storage shape")
    return
  if op in CALLEE_OPS:
    return  # The verifier checks the callee signature.
  expected = stored_pattern(op, tuple(arg.type for arg in args), type_.shape, attrs)
  if type_.pattern is not expected:
    raise ValueError(
      f"{op} output pattern disagrees with its stored-pattern rule:"
      f" got{_pattern_text(type_.pattern) or ' dense'}, expected{_pattern_text(expected) or ' dense'}"
    )


def _asarray(value: Any, *, dtype: DType | str | None = None) -> np.ndarray:
  np_dtype = as_dtype(dtype).numpy() if dtype is not None else np.float64
  if np.issubdtype(np_dtype, np.floating):
    return np.asarray(value).astype(np_dtype, copy=False)
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
  attrs: Mapping[str, Any],
  lowering: Lowering,
) -> tuple[Any, ...]:
  op_norm = op if isinstance(op, ExprOp) else ExprOp(op)
  args_key = tuple(weakref.ref(a) for a in args)
  value_key = None if value is None else _array_key(value)
  return (op_norm.value, args_key, type_, name, value_key, _attrs_key(attrs), lowering)


# ``init=False``: ``__new__`` is the whole constructor. A generated ``__init__`` would run again on
# the node an interning hit returns, which Functions already hold.
@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class Expr:
  """One node of the expression graph: an op, its argument nodes and the resulting ``TensorType``.

  Nodes are immutable and interned, so building the same expression twice returns the same
  object. Create leaves with ``sym`` and ``const``, then combine them with the operators, methods
  and builders below.
  """

  op: ExprOp | str
  args: tuple[Expr, ...] = ()
  type: TensorType = field(default_factory=TensorType)
  name: str | None = None
  value: np.ndarray | None = None
  attrs: Mapping[str, Any] = field(default_factory=dict)
  lowering: Lowering = "auto"
  # Frozen → safe to cache. Populated lazily by structural_key on first call.
  _key_cache: tuple[Any, ...] | None = field(default=None, init=False, repr=False, compare=False)
  _value_key: tuple[Any, ...] | None = field(default=None, init=False, repr=False, compare=False)
  _attrs_key_cache: tuple[tuple[str, Any], ...] = field(default=(), init=False, repr=False, compare=False)
  _support_cache: tuple[Support] | None = field(default=None, init=False, repr=False, compare=False)
  _custom_derivative_cache: bool | None = field(default=None, init=False, repr=False, compare=False)

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
    attrs: Mapping[str, Any] | None = None,
    lowering: Lowering = "auto",
  ) -> "Expr":
    if op is None:  # callers like ``copy.copy`` / pickling instantiate w/o args
      return object.__new__(cls)
    type_eff = type if type is not None else TensorType()
    if type_eff.pattern is not None or any(arg.type.pattern is not None for arg in args):
      check_stored_pattern(op, args, type_eff, attrs)
    attrs_eff = attrs if attrs is not None else {}
    if value is not None:
      data = _array_data(value)
      bits = data.view(np.dtype(f"u{value.dtype.itemsize}"))
      if bits.size and np.all(bits == bits.flat[0]):
        value = np.broadcast_to(data.flat[:1].reshape(()), value.shape)
    key = _intern_key(op, args, type_eff, name, value, attrs_eff, lowering)
    cached = _NODE_CACHE.get(key)
    if cached is not None:
      return cached
    instance = object.__new__(cls)
    put = object.__setattr__
    put(instance, "op", ExprOp(op))
    put(instance, "args", tuple(args))
    put(instance, "type", type_eff)
    put(instance, "name", name)
    put(instance, "value", None if value is None else frozen(value))
    put(instance, "attrs", frozen(attrs_eff))
    put(instance, "lowering", lowering)
    put(instance, "_key_cache", None)
    put(instance, "_value_key", key[4])
    put(instance, "_attrs_key_cache", key[5])
    put(instance, "_support_cache", None)
    put(instance, "_custom_derivative_cache", None)
    _NODE_CACHE[key] = instance
    return instance

  @staticmethod
  def sym(
    name: str,
    shape: int | tuple[int, ...] | None = None,
    *,
    dtype: DType | str = dtypes.float64,
    diff: bool = True,
    lowering: Lowering = "auto",
  ) -> Expr:
    """Create a named symbolic input, the leaf every graph is built from.

    Args:
      name: the input's name, which becomes its name on any ``Function`` declaring it.
      shape: an ``int`` for a rank-1 shape, a tuple as given, or ``None`` for a scalar.
      dtype: the element type, ``float64`` unless you say otherwise.
      diff: whether derivatives with respect to this input are meaningful. Setting it to
        ``False`` tells AD the input is a constant parameter, so terms through it vanish.
    """
    return Expr(ExprOp.INPUT, type=TensorType(as_shape(shape), dtype=as_dtype(dtype), diff=diff), name=name, lowering=lowering)

  @staticmethod
  def const(
    value: Any, *, dtype: DType | str | None = None, lowering: Lowering = "auto", pattern: CanonicalPattern | SparsityPattern | None = None
  ) -> Expr:
    """Wrap an array or scalar as a constant node.

    A NumPy array or scalar keeps its dtype, and anything else becomes ``float64``, unless
    ``dtype`` says otherwise. Constants are never differentiable: a derivative with respect to one
    is structurally zero.

    A SciPy sparse array becomes a sparse constant with its pattern. Duplicate entries are summed
    and explicitly stored zeros stay stored. ``pattern`` instead makes ``value`` the stored values
    of a sparse constant, in the pattern's coordinate order.
    """
    from scipy import sparse

    if sparse.issparse(value):
      if pattern is not None:
        raise ValueError("a SciPy sparse constant already declares its pattern")
      coo = sparse.coo_array(value, copy=True)
      if len(coo.shape) < 2:
        raise ValueError("a sparse constant must have at least two axes")
      coo.sum_duplicates()
      coordinates = np.unique(np.stack(coo.coords[-2:]), axis=1)
      canonical = CanonicalPattern(coo.shape[-2:], tuple(map(int, coordinates[0])), tuple(map(int, coordinates[1])))
      order = np.lexsort(tuple(reversed(coo.coords)))
      type_ = TensorType(coo.shape, pattern=canonical)
      logical = np.ravel_multi_index(coo.coords, coo.shape)
      if not np.array_equal(logical[order], stored_indices(type_)):
        raise ValueError("batched sparse constants require a shared stored pattern")
      return pack(Expr.const(coo.data[order].reshape(type_.storage_shape), dtype=dtype, lowering=lowering), canonical)
    if pattern is not None:
      return pack(Expr.const(value, dtype=dtype, lowering=lowering), pattern)
    if dtype is None:
      dtype = value.dtype.name if isinstance(value, np.ndarray | np.generic) else dtypes.float64
    dtype_eff = as_dtype(dtype)
    arr = _asarray(value, dtype=dtype_eff)
    return Expr(ExprOp.CONST, type=TensorType(tuple(arr.shape), dtype=dtype_eff, diff=False), value=arr, lowering=lowering)

  @property
  def shape(self) -> tuple[int, ...]:
    return self.type.shape

  @property
  def size(self) -> int:
    return self.type.size

  @property
  def support(self) -> Support:
    """Flat row-major indices of the entries that may be nonzero, or ``None`` when any may be.

    Every other entry is zero for all finite input values. A product with a constant zero counts
    as zero. The result is a set or a sorted array of indices, computed on first access and cached.
    """
    return compute_support(self, OP_INFO)

  def structural_key(self) -> tuple[Any, ...]:
    stack: list[tuple[Expr, bool]] = [(self, False)]
    while stack:
      node, visited = stack.pop()
      if node._key_cache is not None:
        continue
      if not visited:
        stack.append((node, True))
        stack.extend((arg, False) for arg in reversed(node.args) if arg._key_cache is None)
        continue
      value_key = node._value_key
      key = (
        ExprOp(node.op).value,
        node.name,
        node.type.shape,
        node.type.dtype,
        node.type.diff,
        node.lowering,
        node._attrs_key_cache,
        value_key,
        tuple(arg._key_cache for arg in node.args),
      )
      if node.type.pattern is not None:
        key = (*key, ("pattern", node.type.pattern.digest))
      object.__setattr__(node, "_key_cache", key)
    assert self._key_cache is not None
    return self._key_cache

  def structural_hash(self) -> int:
    return hash(self.structural_key())

  def structurally_equal(self, other: Expr) -> bool:
    return isinstance(other, Expr) and self.structural_key() == other.structural_key()

  def debug(self) -> str:
    return format_expr(self)

  def with_lowering(self, lowering: Lowering) -> Expr:
    return Expr(self.op, self.args, self.type, self.name, self.value, self.attrs, lowering)

  def scalar(self) -> Expr:
    """Return this expression with a hint to expand its procedure into straight-line scalar code.

    The hint overrides the automatic size limits and applies to the entry point too. Procedures
    expand only when scalar substitution preserves each store's rounding and conversion. A
    ``block`` hint in the same function wins. See the lowering page of *How it works* for the policy.
    """
    return self.with_lowering("scalar")

  def block(self) -> Expr:
    """Return this expression with a hint to keep its procedure as loops over buffers.

    The procedure is not expanded into scalar code and survives as its own C procedure. Every other
    optimization still runs on it.
    """
    return self.with_lowering("block")

  def opaque(self) -> Expr:
    """Prevent scalar expansion of the containing procedure, retaining its loopy form."""
    return self.with_lowering("opaque")

  def reshape(self, shape: int | tuple[int, ...]) -> Expr:
    shape = as_shape(shape)
    if np.prod(shape, dtype=int) != self.size:
      raise ValueError(f"cannot reshape {self.shape} with {self.size} entries to {shape}")
    return Expr(
      ExprOp.RESHAPE,
      (self,),
      TensorType(shape, dtype=self.type.dtype, diff=self.type.diff, pattern=stored_pattern(ExprOp.RESHAPE, (self.type,), shape)),
      attrs={"shape": shape},
      lowering=self.lowering,
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
      TensorType(
        tuple(self.shape[i] for i in axes),
        dtype=self.type.dtype,
        diff=self.type.diff,
        pattern=stored_pattern(ExprOp.TRANSPOSE, (self.type,), tuple(self.shape[i] for i in axes), {"axes": axes}),
      ),
      attrs={"axes": axes},
      lowering=self.lowering,
    )

  @property
  def T(self) -> Expr:
    return self.transpose()

  def __getitem__(self, index: Any) -> Expr:
    index = _normalize_index(index, self.shape)
    shape = tuple(np.broadcast_to(np.empty(()), self.shape)[index].shape)
    return Expr(
      ExprOp.SLICE,
      (self,),
      TensorType(shape, dtype=self.type.dtype, diff=self.type.diff, pattern=stored_pattern(ExprOp.SLICE, (self.type,), shape, {"index": index})),
      attrs={"index": index},
      lowering=self.lowering,
    )

  def gather(self, indices: Any) -> Expr:
    return gather(self, indices)

  def values(self) -> Expr:
    """Return the stored values in row-major coordinate order, as ``sc.values`` does."""
    return values(self)

  def to_dense(self) -> Expr:
    """Place stored values at their coordinates, filling other entries with zero."""
    return to_dense(self)

  def sum(self, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False) -> Expr:
    """Sum over the chosen axes, or every axis when ``axis`` is ``None``.

    Negative axes count from the end. An empty tuple leaves every entry unchanged.
    ``keepdims`` retains the reduced axes with extent one. Empty domains sum to zero.
    The result keeps the operand's dtype.
    """
    if not (self.type.dtype.is_floating or self.type.dtype.is_integer):
      raise TypeError(f"sum takes floating values or integers, got {self.type.dtype}; convert with sc.cast")
    result = _reduce(self, "add", axis)
    return result.reshape(tuple(1 if i in result.attrs["axes"] else n for i, n in enumerate(self.shape))) if keepdims else result

  def mean(self, axis: int | tuple[int, ...] | None = None, *, keepdims: bool = False) -> Expr:
    """Average over the chosen axes, counting structural zeros in sparse inputs.

    Axes and ``keepdims`` follow ``sum``. An empty reduction domain returns NaN.
    """
    result = _reduce(self, "add", axis)
    count = int(np.prod([self.shape[i] for i in result.attrs["axes"]], dtype=np.int64))
    if keepdims:
      result = result.reshape(tuple(1 if i in result.attrs["axes"] else n for i, n in enumerate(self.shape)))
    if count == 0:
      return _uniform(np.nan, result.shape, dtype=self.type.dtype, lowering=self.lowering)
    if result.type.pattern is not None:
      return pack(values(result) / count, result.type.pattern, shape=result.shape)
    return result / count

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
    """Elementwise absolute value, retaining the operand's dtype."""
    return unary(ExprOp.ABS, self)

  def floor(self) -> Expr:
    return unary(ExprOp.FLOOR, self)

  def ceil(self) -> Expr:
    return unary(ExprOp.CEIL, self)

  def isfinite(self) -> Expr:
    return isfinite(self)

  def cast(self, dtype: DType | str) -> Expr:
    return cast(self, dtype)

  def minimum(self, other: Any) -> Expr:
    return minimum(self, other)

  def maximum(self, other: Any) -> Expr:
    return maximum(self, other)

  def __neg__(self) -> Expr:
    return unary(ExprOp.NEG, self)

  def __add__(self, other: Any) -> Expr:
    return binary(ExprOp.ADD, self, _coerce(other, self))

  def __radd__(self, other: Any) -> Expr:
    return binary(ExprOp.ADD, _coerce(other, self), self)

  def __sub__(self, other: Any) -> Expr:
    return binary(ExprOp.SUB, self, _coerce(other, self))

  def __rsub__(self, other: Any) -> Expr:
    return binary(ExprOp.SUB, _coerce(other, self), self)

  def __mul__(self, other: Any) -> Expr:
    return binary(ExprOp.MUL, self, _coerce(other, self))

  def __rmul__(self, other: Any) -> Expr:
    return binary(ExprOp.MUL, _coerce(other, self), self)

  def __truediv__(self, other: Any) -> Expr:
    return binary(ExprOp.DIV, self, _coerce(other, self))

  def __rtruediv__(self, other: Any) -> Expr:
    return binary(ExprOp.DIV, _coerce(other, self), self)

  def __pow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, self, _coerce(other, self))

  def __rpow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, _coerce(other, self), self)

  def __floordiv__(self, other: Any) -> Expr:
    return binary(ExprOp.FLOORDIV, self, _coerce(other, self))

  def __rfloordiv__(self, other: Any) -> Expr:
    return binary(ExprOp.FLOORDIV, _coerce(other, self), self)

  def __mod__(self, other: Any) -> Expr:
    return binary(ExprOp.MOD, self, _coerce(other, self))

  def __rmod__(self, other: Any) -> Expr:
    return binary(ExprOp.MOD, _coerce(other, self), self)

  # ``==`` stays identity, which interning relies on; ``sc.equal`` compares values.
  def __lt__(self, other: Any) -> Expr:
    return binary(ExprOp.LT, self, _coerce(other, self))

  def __le__(self, other: Any) -> Expr:
    return binary(ExprOp.LE, self, _coerce(other, self))

  def __gt__(self, other: Any) -> Expr:
    return binary(ExprOp.LT, _coerce(other, self), self)

  def __ge__(self, other: Any) -> Expr:
    return binary(ExprOp.LE, _coerce(other, self), self)

  def __and__(self, other: Any) -> Expr:
    return binary(ExprOp.AND, self, _condition(other))

  def __rand__(self, other: Any) -> Expr:
    return binary(ExprOp.AND, _condition(other), self)

  def __or__(self, other: Any) -> Expr:
    return binary(ExprOp.OR, self, _condition(other))

  def __ror__(self, other: Any) -> Expr:
    return binary(ExprOp.OR, _condition(other), self)

  def __invert__(self) -> Expr:
    return unary(ExprOp.NOT, self)

  def __matmul__(self, other: Any) -> Expr:
    return matmul(self, _coerce(other, self))

  def __rmatmul__(self, other: Any) -> Expr:
    return matmul(_coerce(other, self), self)

  def __bool__(self) -> bool:
    raise TypeError("an Expr has no truth value: a Python branch on one would take the same path for every input; use sc.where to choose a value")

  def inputs(self) -> dict[str, Expr]:
    ret: dict[str, Expr] = {}
    for e in topo([self]):
      if e.op == ExprOp.INPUT and e.name is not None:
        ret[e.name] = e
    return ret


def _array_key(value: np.ndarray) -> tuple[Any, ...]:
  compact = _array_data(value)
  bits = compact.view(np.dtype(f"u{value.dtype.itemsize}"))
  if bits.size and np.all(bits == bits.flat[0]):
    return (value.shape, value.dtype.str, bits.flat[0].tobytes())
  for axis, extent in enumerate(compact.shape):
    if extent > 1:
      index = tuple(slice(0, 1) if i == axis else slice(None) for i in range(compact.ndim))
      if np.all(bits == bits[index]):
        compact, bits = compact[index], bits[index]
  data = compact.tobytes()
  return (value.shape, value.dtype.str, compact.shape, hashlib.blake2b(data, digest_size=32).digest(), data)


def _uniform(value: Any, shape: tuple[int, ...], *, dtype: DType = dtypes.float64, lowering: Lowering = "auto") -> Expr:
  return Expr.const(np.broadcast_to(np.asarray(value, dtype=dtype.numpy()), shape), dtype=dtype, lowering=lowering)


def _attrs_key(attrs: Mapping[str, Any]) -> tuple[tuple[str, Any], ...]:
  def key(v: Any) -> Any:
    # ``==`` merges what generated code tells apart: ``-0.0`` with ``0.0``, and ``1`` with ``True``
    # and ``1.0``. A float keys by its bits, which also lets a NaN match itself, and a flag by its kind.
    if isinstance(v, (bool, np.bool_)):
      return (bool, bool(v))
    if isinstance(v, (float, np.floating)):
      return (float, struct.pack("<d", v))
    if isinstance(v, Expr):
      return v.structural_key()
    if hasattr(v, "structural_key") and callable(getattr(v, "structural_key")):
      return v.structural_key()
    if hasattr(v, "name") and hasattr(v, "input_names") and hasattr(v, "output_names"):
      return ("Function", weakref.ref(v))
    if isinstance(v, np.ndarray):
      return ("ndarray", _array_key(v))
    if isinstance(v, Mapping):
      return tuple((k, key(x)) for k, x in sorted(v.items()))
    if isinstance(v, slice):
      return ("slice", v.start, v.stop, v.step)
    if isinstance(v, (tuple, list)):
      return tuple(key(x) for x in v)
    return v

  return tuple((k, key(v)) for k, v in sorted(attrs.items()))


def callees_of(node: Expr) -> tuple[Any, ...]:
  """The concrete functions invoked by ``node``, or an empty tuple for other operations."""
  if node.op == ExprOp.LOOP and node.attrs["cond"] is not None:
    return node.attrs["callee"], node.attrs["cond"]
  return (node.attrs["callee"],) if node.op in CALLEE_OPS else ()


def invocation_key(node: Expr) -> tuple[Any, ...]:
  """Group a call operation's results by operation, callee identity, argument identity and invocation attributes."""
  # VMAP's slice size describes the selected result, not the invocation.
  attrs = {k: v for k, v in node.attrs.items() if k not in ("callee", "output", "slice_size")}
  return node.op, tuple(id(callee) for callee in callees_of(node)), tuple(arg.id for arg in node.args), _attrs_key(attrs)


def as_expr(x: Any) -> Expr:
  return x if isinstance(x, Expr) else Expr.const(x)


_WEAK_KINDS = {"b": 0, "i": 1, "u": 1, "f": 2}


def _weak_kind(x: Any) -> int:
  # Read from the Python values: NumPy infers ``object`` for an int beyond int64 and ``float64`` for ``[]``.
  if isinstance(x, list | tuple):
    return max(map(_weak_kind, x), default=0)
  if isinstance(x, bool):
    return 0
  if isinstance(x, int):
    return 1
  if isinstance(x, float):
    return 2
  return _WEAK_KINDS.get(np.asarray(x).dtype.kind, len(_WEAK_KINDS))


def _coerce(x: Any, like: Expr | None) -> Expr:
  """``x`` as an operand beside ``like``. A Python value is weak and takes ``like``'s dtype; an
  ``Expr`` or a NumPy value is strong and keeps its own, which ``elementwise_dtype`` then checks."""
  if isinstance(x, Expr | np.ndarray | np.generic) or like is None:
    return as_expr(x)
  dtype = like.type.dtype
  if _weak_kind(x) > _WEAK_KINDS[dtype.numpy().kind]:
    raise TypeError(f"cannot combine {type(x).__name__} {x!r} with an expression of dtype {dtype}")
  return Expr.const(x, dtype=dtype)


def _operands(*xs: Any) -> tuple[Expr, ...]:
  """Coerce every operand beside the first ``Expr`` among them, or else the first NumPy value."""
  strong = [x for x in xs if isinstance(x, Expr | np.ndarray | np.generic)]
  like = next((x for x in strong if isinstance(x, Expr)), as_expr(strong[0]) if strong else None)
  return tuple(_coerce(x, like) for x in xs)


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
      raise TypeError(f"mixed dtypes {first} and {e.type.dtype}; convert one with sc.cast")
  return first


_PREDICATES = {ExprOp.LT, ExprOp.LE, ExprOp.EQ, ExprOp.NE}
_LOGIC = {ExprOp.AND, ExprOp.OR, ExprOp.NOT}
# What C computes exactly on integers; the libm functions and / would round through a double.
_INTEGER_OPS = _PREDICATES | {
  ExprOp.NEG,
  ExprOp.ABS,
  ExprOp.ADD,
  ExprOp.SUB,
  ExprOp.MUL,
  ExprOp.MINIMUM,
  ExprOp.MAXIMUM,
  ExprOp.FLOORDIV,
  ExprOp.MOD,
  ExprOp.SELECT,
}


def elementwise_dtype(op: ExprOp | str, *operands: DType) -> DType:
  """The dtype of an elementwise ``op`` on operands of these dtypes, or a ``TypeError`` saying what
  to change. A ``CAST`` is not asked: its dtype is its target."""
  op = ExprOp(op)
  if op == ExprOp.SELECT:
    condition, *operands_ = operands
    if not condition.is_bool:
      raise TypeError(f"where takes a bool condition, got {condition}; compare first, as in x > 0")
    operands = tuple(operands_)
  got = ", ".join(map(str, operands))
  if op in _LOGIC and not all(d.is_bool for d in operands):
    raise TypeError(f"{op.value} takes bool operands, got {got}")
  if op == ExprOp.ISFINITE and not operands[0].is_floating:
    raise TypeError(f"isfinite takes a floating operand, got {got}")
  if op in {ExprOp.FLOORDIV, ExprOp.MOD} and not all(d.is_integer for d in operands):
    raise TypeError(f"{op.value} takes integer operands, got {got}")
  if op not in _INTEGER_OPS and any(d.is_integer for d in operands):
    raise TypeError(f"{'/' if op == ExprOp.DIV else op.value} takes floating operands, got {got}; convert with sc.cast")
  if op not in _PREDICATES | _LOGIC | {ExprOp.SELECT} and any(d.is_bool for d in operands):
    raise TypeError(f"{op.value} takes numbers, got {got}; convert with sc.cast")
  first = operands[0]
  for dtype in operands[1:]:
    if dtype != first:
      raise TypeError(f"mixed dtypes {first} and {dtype}; convert one with sc.cast")
  return dtypes.bool_ if op in _PREDICATES or op == ExprOp.ISFINITE else first


def unary(op: ExprOp | str, x: Expr) -> Expr:
  dtype = elementwise_dtype(op, x.type.dtype)
  return Expr(op, (x,), TensorType(x.shape, dtype=dtype, diff=op_diff(op, x), pattern=stored_pattern(op, (x.type,), x.shape)), lowering=x.lowering)


def binary(op: ExprOp | str, x: Expr, y: Expr) -> Expr:
  shape = broadcast_shape(x.shape, y.shape)
  return Expr(
    op,
    (x, y),
    TensorType(
      shape, dtype=elementwise_dtype(op, x.type.dtype, y.type.dtype), diff=op_diff(op, x, y), pattern=stored_pattern(op, (x.type, y.type), shape)
    ),
    lowering=common_lowering(x, y),
  )


PRINTABLE = (dtypes.float64, dtypes.int32, dtypes.int64, dtypes.bool_)


def print_pieces(fmt: str) -> tuple[str, ...]:
  """The literal text around each ``{}`` placeholder of a print format: one more piece than placeholders."""
  if "\0" in fmt:
    raise ValueError(f"print format {fmt!r} contains a NUL character, which would end the C string")
  pieces = [""]
  for literal, field_name, spec, conversion in string.Formatter().parse(fmt):
    pieces[-1] += literal
    if field_name is None:
      continue
    if field_name or spec or conversion:
      raise ValueError(f"print format {fmt!r} may contain only empty '{{}}' placeholders")
    pieces.append("")
  return tuple(pieces)


_PRINT_TRACE: ContextVar[list[Expr] | None] = ContextVar("print_trace", default=None)


@contextmanager
def recording_prints() -> Generator[list[Expr]]:
  """Collect the prints this thread builds inside the block, for ``check_prints_reach``."""
  prints: list[Expr] = []
  token = _PRINT_TRACE.set(prints)
  try:
    yield prints
  finally:
    _PRINT_TRACE.reset(token)


def check_prints_reach(prints: Iterable[Expr], outputs: Iterable[Expr], owner: str) -> None:
  """Raise for a print with no print of the same format among the ancestors of ``outputs``.

  Matching formats rather than nodes accepts a print that a hint, a rewrite or a substitution rebuilt.
  """
  reached = {e.attrs["format"] for e in topo(outputs) if e.op == ExprOp.PRINT}
  for node in prints:
    if node.attrs["format"] not in reached:
      raise ValueError(f"print {node.attrs['format']!r} in {owner} does not reach its outputs; use the value it returns")


def print_(fmt: str, *values: Any) -> Expr:
  """Return the first of ``values``, printing them all each time generated code computes it.

  ``fmt`` has one ``{}`` for each value, and ``{{`` and ``}}`` for literal braces. Each print ends
  its line. A ``float64`` prints with 17 significant digits, enough to read it back exactly, an
  integer in full and a ``bool`` as ``True`` or ``False``, and a tensor prints flat as ``[a, b, ...]``. The print is part of the returned expression, so a
  function's outputs have to use it. A print they do not use raises ``ValueError`` when Scaly runs
  the function body to build its expression graph.

  Args:
    fmt: the text to print, with a ``{}`` placeholder for each value.
    values: the expressions to print. The first is returned.
  """
  exprs = tuple(as_expr(value) for value in values)
  if not exprs:
    raise ValueError("print needs at least one value, the one it returns")
  placeholders = len(print_pieces(fmt)) - 1
  if placeholders != len(exprs):
    raise ValueError(f"print format {fmt!r} has {placeholders} placeholders for {len(exprs)} values")
  if any(e.type.dtype not in PRINTABLE for e in exprs):
    raise TypeError(f"print supports {', '.join(map(str, PRINTABLE))} values, got {', '.join(str(e.type.dtype) for e in exprs)}")
  node = Expr(ExprOp.PRINT, exprs, exprs[0].type, attrs={"format": fmt}, lowering=exprs[0].lowering)
  if (trace := _PRINT_TRACE.get()) is not None:
    trace.append(node)
  return node


def stop_gradient(value: Any) -> Expr:
  """Return ``value`` unchanged in evaluation and treat it as a constant in derivatives.

  Forward and reverse derivatives through this expression are zero, as is its Jacobian sparsity.
  Other uses of ``value`` still contribute their derivatives. Its generated evaluation C is unchanged.
  """
  expr = as_expr(value)
  return Expr(ExprOp.STOP_GRADIENT, (expr,), expr.type, lowering=expr.lowering)


def atan2(y: Any, x: Any) -> Expr:
  """Two-argument arctangent, elementwise: the angle of the point ``(x, y)``."""
  return binary(ExprOp.ATAN2, *_operands(y, x))


def equal(x: Any, y: Any) -> Expr:
  """Elementwise ``x == y`` as a bool expression. The ``==`` operator compares expressions by identity."""
  return binary(ExprOp.EQ, *_operands(x, y))


def not_equal(x: Any, y: Any) -> Expr:
  """Elementwise ``x != y`` as a bool expression, true wherever either side is NaN."""
  return binary(ExprOp.NE, *_operands(x, y))


def isfinite(x: Any) -> Expr:
  """Elementwise test that a floating value is neither infinite nor NaN.

  The generated C refuses to compile under ``-ffast-math``, which assumes no value is either.
  """
  return unary(ExprOp.ISFINITE, as_expr(x))


def _condition(x: Any) -> Expr:
  if not isinstance(x, Expr | np.ndarray | np.generic):
    raise TypeError(f"a condition is a bool expression, got {type(x).__name__}; == compares identity, use sc.equal")
  return as_expr(x)


def where(condition: Any, x: Any, y: Any) -> Expr:
  """Choose elementwise: ``x`` where the bool ``condition`` holds, ``y`` elsewhere.

  The three broadcast together and ``x`` and ``y`` share a dtype. The generated C computes both
  branches, so a branch may be NaN or infinite where it is not chosen, and the derivative ignores
  the branch not chosen too: ``where(x > 0, sqrt(x), 0)`` has derivative zero at ``x = 0``,
  including when a ``Function`` or ``vmap`` computes the square root. The condition has no derivative.
  Custom reverse rules receive zero cotangents on unchosen entries and must return finite results
  there. A select inside a mapped ``Function`` does not suppress undefined derivatives in the map's arguments.
  """
  condition = _condition(condition)
  x, y = _operands(x, y)
  dtype = elementwise_dtype(ExprOp.SELECT, condition.type.dtype, x.type.dtype, y.type.dtype)
  if condition.type.pattern is not None and x.type.pattern is None and y.type.pattern is None:
    condition = to_dense(condition)
  shape = broadcast_shape(condition.shape, broadcast_shape(x.shape, y.shape))
  type_ = TensorType(
    shape, dtype=dtype, diff=op_diff(ExprOp.SELECT, x, y), pattern=stored_pattern(ExprOp.SELECT, (condition.type, x.type, y.type), shape)
  )
  if type_.pattern is not None:
    chosen = _constant_condition(condition, stored_indices(type_), shape)
    if chosen is not None:
      condition = pack(Expr.const(chosen.reshape(type_.storage_shape), dtype=dtypes.bool_, lowering=condition.lowering), type_.pattern, shape=shape)
  return Expr(
    ExprOp.SELECT,
    (condition, x, y),
    type_,
    lowering=common_lowering(condition, x, y),
  )


def cast(x: Any, dtype: DType | str) -> Expr:
  """Convert ``x`` elementwise to ``dtype``, as NumPy's ``astype`` does.

  A float becomes an integer by dropping its fraction, and a float outside the integer's range,
  NaN included, has no defined result. A cast to ``bool`` is ``x != 0``, so NaN is true. A cast
  keeps its derivative only from one float dtype to another.

  ``uint8`` is not supported for expressions and raises ``ValueError``.
  """
  x, dtype = _condition(x) if isinstance(x, bool) else as_expr(x), as_dtype(dtype)
  if dtype == x.type.dtype:
    return x
  if dtype.is_bool:
    return not_equal(x, 0)
  diff = x.type.diff and x.type.dtype.is_floating and dtype.is_floating
  return Expr(ExprOp.CAST, (x,), TensorType(x.shape, dtype=dtype, diff=diff), lowering=x.lowering)


def minimum(x: Any, y: Any) -> Expr:
  """Elementwise minimum. Non-smooth, so the result is marked non-differentiable."""
  return binary(ExprOp.MINIMUM, *_operands(x, y))


def maximum(x: Any, y: Any) -> Expr:
  """Elementwise maximum. Non-smooth, so the result is marked non-differentiable."""
  return binary(ExprOp.MAXIMUM, *_operands(x, y))


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

  def integer(item: Any) -> int:
    if isinstance(item, bool | np.bool_) or not isinstance(item, int | np.integer):
      raise TypeError(f"unsupported index component {item!r}")
    return int(item)

  return tuple(
    slice(*(None if v is None else integer(v) for v in (item.start, item.stop, item.step))) if isinstance(item, slice) else integer(item)
    for item in index
  )


def matmul(x: Expr, y: Expr, *, sampled: CanonicalPattern | None = None) -> Expr:
  """Matrix product with NumPy's ``matmul`` rules.

  The last two axes of each operand multiply as matrices and leading axes broadcast.
  A vector on the left acts as one row and a vector on the right as one column,
  and that axis is removed from the result.
  """
  if not x.type.dtype.is_floating or not y.type.dtype.is_floating:
    raise TypeError(f"@ takes floating operands, got {x.type.dtype} and {y.type.dtype}; convert with sc.cast")
  if max(len(x.shape), len(y.shape)) > 2 and any(arg.type.pattern is not None for arg in (x, y)):
    raise NotImplementedError("batched sparse MATMUL requires the shared-pattern CSR kernel in #403")
  if not x.shape or not y.shape or x.shape[-1] != y.shape[-2 if len(y.shape) > 1 else -1]:
    raise ValueError(f"cannot matmul shapes {x.shape} and {y.shape}")
  shape = broadcast_shape(x.shape[:-2], y.shape[:-2])
  shape += (() if len(x.shape) == 1 else (x.shape[-2],)) + (() if len(y.shape) == 1 else (y.shape[-1],))
  attrs = {} if sampled is None else {"sampled": sampled}
  return Expr(
    ExprOp.MATMUL,
    (x, y),
    TensorType(shape, dtype=promote_dtype(x, y), diff=diff_any(x, y), pattern=stored_pattern(ExprOp.MATMUL, (x.type, y.type), shape, attrs)),
    attrs=attrs,
    lowering=common_lowering(x, y),
  )


def _reduce(x: Expr, kind: str, axis: int | tuple[int, ...] | None = None) -> Expr:
  if kind not in {"add", "max", "min"}:
    raise ValueError(f"unknown reduction kind {kind!r}")
  if axis is not None and not isinstance(axis, int | np.integer | tuple):
    raise TypeError("reduction axis must be an integer or a tuple of integers")
  axes = tuple(range(len(x.shape))) if axis is None else (axis,) if isinstance(axis, int | np.integer) else axis
  if any(isinstance(i, bool | np.bool_) or not isinstance(i, int | np.integer) for i in axes):
    raise TypeError("reduction axes must be integers")
  if any(i < -len(x.shape) or i >= len(x.shape) for i in axes):
    raise ValueError(f"reduction axis out of range for shape {x.shape}")
  axes = tuple(sorted(int(i) % len(x.shape) for i in axes))
  if len(set(axes)) != len(axes):
    raise ValueError("duplicate reduction axis")
  shape = tuple(n for i, n in enumerate(x.shape) if i not in axes)
  attrs = {"kind": kind, "axes": axes}
  return Expr(
    ExprOp.REDUCE,
    (x,),
    TensorType(shape, dtype=x.type.dtype, diff=x.type.diff, pattern=stored_pattern(ExprOp.REDUCE, (x.type,), shape, attrs)),
    attrs=attrs,
    lowering=x.lowering,
  )


def diag(x: Any, k: int = 0) -> Expr:
  """Build a diagonal matrix from a vector, or extract a matrix's diagonal.

  ``k`` selects the diagonal above the main one when positive, below when negative.
  Only rank-one and rank-two inputs are accepted, following ``numpy.diag``.
  """
  x = as_expr(x)
  if not isinstance(k, int | np.integer):
    raise TypeError("diagonal offset must be an integer")
  k = int(k)
  if len(x.shape) == 1:
    n = x.size + abs(k)
    rows = np.arange(x.size, dtype=np.int64) + max(-k, 0)
    cols = np.arange(x.size, dtype=np.int64) + max(k, 0)
    return scatter(x, rows * n + cols, (n, n))
  if len(x.shape) == 2:
    rows, cols = x.shape
    count = max(0, min(rows + min(k, 0), cols - max(k, 0)))
    i = np.arange(count, dtype=np.int64)
    return gather(x, (i + max(-k, 0)) * cols + i + max(k, 0))
  raise ValueError("diag requires a vector or a matrix")


def dot(x: Any, y: Any) -> Expr:
  """Scalar product of two expressions with the same number of entries, whatever their shapes."""
  x, y = _operands(x, y)
  if x.size != y.size:
    raise ValueError(f"dot size mismatch: {x.shape} has {x.size} entries, {y.shape} has {y.size}")
  if x.type.pattern is not None or y.type.pattern is not None:
    if x.shape != y.shape:
      raise ValueError("sparse dot operands must have the same logical shape")
    return (x * y).sum()
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
  shape = tuple(idx.shape)
  return Expr(
    ExprOp.GATHER,
    (x,),
    TensorType(shape, dtype=x.type.dtype, diff=x.type.diff, pattern=stored_pattern(ExprOp.GATHER, (x.type,), shape, {"indices": idx})),
    attrs={"indices": idx},
    lowering=x.lowering,
  )


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
    ExprOp.SCATTER,
    (values,),
    TensorType(
      shape, dtype=values.type.dtype, diff=values.type.diff, pattern=stored_pattern(ExprOp.SCATTER, (values.type,), shape, {"indices": idx})
    ),
    attrs={"indices": idx},
    lowering=values.lowering,
  )


def segment_sum(values: Any, ids: Any, n: int) -> Expr:
  """Sum flat ``values`` by segment id into a vector of length ``n``.

  Segment ids are fixed integers in ``[0, n)``. Empty segments are zero, and
  repeated ids accumulate in input order.
  """
  return scatter(values, ids, n)


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
  exprs = _operands(*xs)
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
    ExprOp.STACK,
    exprs,
    TensorType(
      shape,
      dtype=promote_dtype(*exprs),
      diff=diff_any(*exprs),
      pattern=stored_pattern(ExprOp.STACK, tuple(e.type for e in exprs), shape, {"axis": axis}),
    ),
    attrs={"axis": axis},
    lowering=common_lowering(*exprs),
  )


def concat(xs: Iterable[Any], *, axis: int = 0) -> Expr:
  """Join expressions along an existing ``axis``. All other dimensions must agree."""
  exprs = _operands(*xs)
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
    TensorType(
      shape,
      dtype=promote_dtype(*exprs),
      diff=diff_any(*exprs),
      pattern=stored_pattern(ExprOp.CONCAT, tuple(e.type for e in exprs), shape, {"axis": axis}),
    ),
    attrs={"axis": axis},
    lowering=common_lowering(*exprs),
  )


def vec(x: Any) -> Expr:
  """Flatten ``x`` to rank 1 in row-major order."""
  return as_expr(x).vec()


def _full_like(value: int | float, x: Expr) -> Expr:
  """A uniform constant with the shape, dtype, lowering hint and stored pattern of ``x``."""
  if x.type.pattern is not None:
    return pack(_uniform(value, x.type.storage_shape, dtype=x.type.dtype, lowering=x.lowering), x.type.pattern, shape=x.shape)
  return _uniform(value, x.shape, dtype=x.type.dtype, lowering=x.lowering)


def zeros_like(x: Expr) -> Expr:
  return _full_like(0, x)


def pack(value: Any, pattern: CanonicalPattern | SparsityPattern, *, shape: tuple[int, ...] | None = None) -> Expr:
  """Build a sparse tensor from its stored values, including explicit zeros.

  A ``SparsityPattern`` takes the values in its own coordinate order, a canonical pattern in
  row-major order. Leading axes of ``value`` are batch axes. ``shape`` defaults to those axes
  followed by the pattern's shape.
  """
  x = as_expr(value)
  if isinstance(pattern, SparsityPattern):
    pattern, permutation = pattern.canonicalize()
    if not x.shape or x.shape[-1] != len(permutation):
      raise ValueError("PACK values' last axis must match the pattern's stored entries")
    indices = np.arange(x.size, dtype=np.int64).reshape(x.shape)[..., permutation]
    x = gather(x, indices)
  if not isinstance(pattern, CanonicalPattern):
    raise TypeError("PACK requires a canonical pattern or SparsityPattern")
  shape = (*x.shape[:-1], *pattern.shape) if shape is None else shape
  type_ = TensorType(shape, x.type.dtype, x.type.diff, pattern)
  return Expr(ExprOp.PACK, (x,), type_, attrs={"pattern": pattern}, lowering=x.lowering)


def values(value: Any) -> Expr:
  """Return a sparse tensor's stored values, in row-major coordinate order, as a dense tensor.

  The result has the storage shape. A dense tensor is returned unchanged.
  """
  x = as_expr(value)
  if x.type.pattern is None:
    return x
  if x.op == ExprOp.PACK:
    return x.args[0]
  return Expr(ExprOp.VALUES, (x,), TensorType(x.type.storage_shape, x.type.dtype, x.type.diff), lowering=x.lowering)


def to_dense(value: Any) -> Expr:
  """Return a dense tensor with zeros outside the pattern. A dense tensor is returned unchanged."""
  x = as_expr(value)
  if x.type.pattern is None:
    return x
  return scatter(values(x).vec(), stored_indices(x.type), x.size).reshape(x.shape)


def from_dense(value: Any, pattern: CanonicalPattern | SparsityPattern) -> Expr:
  """Keep the entries of a dense tensor inside ``pattern`` as a sparse tensor and drop the others."""
  x = as_expr(value)
  if x.type.pattern is not None:
    raise ValueError("from_dense requires a dense tensor")
  canonical = pattern.canonicalize()[0] if isinstance(pattern, SparsityPattern) else pattern
  type_ = TensorType(x.shape, x.type.dtype, x.type.diff, canonical)
  return pack(gather(x, stored_indices(type_).reshape(type_.storage_shape)), canonical, shape=x.shape)


def sparsify(value: Any) -> Expr:
  """Store only the entries that ``Expr.support`` says may be nonzero.

  Batch axes share one pattern, covering the entries of every batch entry. A sparse tensor is
  returned unchanged.
  """
  x = as_expr(value)
  if x.type.pattern is not None:
    return x
  if len(x.shape) < 2:
    raise ValueError("sparsify requires at least two axes")
  matrix_size = x.shape[-2] * x.shape[-1]
  entries = range(matrix_size) if x.support is None else sorted({i % matrix_size for i in x.support})
  return from_dense(
    x, CanonicalPattern((x.shape[-2], x.shape[-1]), tuple(i // x.shape[-1] for i in entries), tuple(i % x.shape[-1] for i in entries))
  )


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
    elif e.op == ExprOp.LOOP:
      cond = e.attrs["cond"]
      n_carry = e.attrs["n_carry"]
      outers = e.args[n_carry : n_carry + len(e.attrs["starts"])]
      bindings = ", ".join(
        [f"%{loc[a.id]}" for a in e.args[:n_carry]]
        + [f"%{loc[a.id]}[{s}::{st}]" for a, s, st in zip(outers, e.attrs["starts"], e.attrs["strides"], strict=True)]
      )
      trip = f" trip=%{loc[e.args[-1].id]}" if len(e.args) > n_carry + len(outers) else ""
      head = f"loop[{e.attrs['length']}{', reverse' if e.attrs['reverse'] else ''}{', index' if e.attrs['index'] else ''}]"
      rhs = f"{head} {e.attrs['callee'].name}[{e.attrs['output']}]({bindings}){trip}" + (f" while {cond.name}" if cond is not None else "")
    else:
      rhs = f"{ExprOp(e.op).value}({', '.join(f'%{loc[a.id]}' for a in e.args)})"
    lines.append(f"{lhs} = {rhs} : {_type_text(e.type)}")
  lines.append("outputs " + ", ".join(f"%{loc[e.id]}" for e in outs))
  return "\n".join(lines)


def substitute(expr: Expr, replacements: Mapping[Expr, Expr]) -> Expr:
  """Replace expressions simultaneously and rebuild only their changed ancestors."""
  for source, replacement in replacements.items():
    if source.shape != replacement.shape or source.type.dtype != replacement.type.dtype or source.type.pattern is not replacement.type.pattern:
      raise ValueError(
        f"cannot substitute {source.name or source.op} {_type_text(source.type)} with "
        f"{replacement.name or replacement.op} {_type_text(replacement.type)}"
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
