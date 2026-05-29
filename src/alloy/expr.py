from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from collections.abc import Mapping
from typing import Any, Iterable

import numpy as np

from .ops import OP_INFO, Ops
from .types import DType, Lowering, TensorType, as_dtype, as_shape, broadcast_shape, dtypes


def _asarray(value: Any, *, dtype: DType | str | None = None) -> np.ndarray:
  np_dtype = as_dtype(dtype).numpy() if dtype is not None else np.float64
  return np.asarray(value, dtype=np_dtype)


# Construction-time hash-consing cache. Two ``Expr(...)`` constructions with the same
# (op, args-by-identity, type, name, value-bytes, attrs, lowering) collapse to a single
# Python object — so structural equality becomes identity, ``is`` works as a fast equality
# check, and any pass that builds new graphs gets CSE for free.
#
# WeakValueDictionary lets nodes be garbage-collected when no live reference remains; the
# cache shrinks automatically. No surprises around long-lived caches retaining graphs.
_NODE_CACHE: weakref.WeakValueDictionary[tuple[Any, ...], "Expr"] = weakref.WeakValueDictionary()


def _intern_key(
  op: Ops | str, args: tuple["Expr", ...], type_: "TensorType", name: str | None, value: np.ndarray | None, attrs: dict[str, Any], lowering: Lowering
) -> tuple[Any, ...]:
  op_norm = op if isinstance(op, Ops) else Ops(op)
  # Use Python ``id`` for arg refs: interning makes id-equality = structural-equality, so
  # two args with the same id are the same subgraph. Avoids walking the args' structural_key
  # at every construction.
  args_key = tuple(id(a) for a in args)
  value_key = None if value is None else (value.shape, str(value.dtype), value.tobytes())
  return (op_norm.value, args_key, type_, name, value_key, _attrs_key(attrs), lowering)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class Expr:
  op: Ops | str
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
    op: Ops | str | None = None,
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
    if not isinstance(self.op, Ops):
      object.__setattr__(self, "op", Ops(self.op))
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
    return Expr(Ops.INPUT, type=TensorType(as_shape(shape), dtype=as_dtype(dtype), diff=diff), name=name, lowering=lowering)

  @staticmethod
  def const(value: Any, *, dtype: DType | str | None = None, lowering: Lowering = "auto") -> Expr:
    dtype_eff = as_dtype(dtype) if dtype is not None else dtypes.float64
    arr = _asarray(value, dtype=dtype_eff)
    return Expr(Ops.CONST, type=TensorType(tuple(arr.shape), dtype=dtype_eff, diff=False), value=arr, lowering=lowering)

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
      Ops(self.op).value,
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
    return self.with_lowering("scalar")

  def block(self) -> Expr:
    return self.with_lowering("block")

  def opaque(self) -> Expr:
    return self.with_lowering("opaque")

  def reshape(self, shape: int | tuple[int, ...]) -> Expr:
    shape = as_shape(shape)
    if np.prod(shape, dtype=int) != self.size:
      raise ValueError(f"cannot reshape {self.shape} with {self.size} entries to {shape}")
    return Expr(Ops.RESHAPE, (self,), TensorType(shape, dtype=self.type.dtype, diff=self.type.diff), attrs={"shape": shape}, lowering=self.lowering)

  def vec(self) -> Expr:
    return self.reshape((self.size,))

  def transpose(self, axes: tuple[int, ...] | None = None) -> Expr:
    axes = tuple(reversed(range(len(self.shape)))) if axes is None else tuple(axes)
    if sorted(axes) != list(range(len(self.shape))):
      raise ValueError(f"transpose axes {axes} are not a permutation for shape {self.shape}")
    return Expr(
      Ops.TRANSPOSE,
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
    return Expr(Ops.SLICE, (self,), TensorType(shape, dtype=self.type.dtype, diff=self.type.diff), attrs={"index": index}, lowering=self.lowering)

  def gather(self, indices: Any) -> Expr:
    return gather(self, indices)

  def sum(self, axis: int | tuple[int, ...] | None = None) -> Expr:
    """Reduce along ``axis`` (or all axes if ``axis is None``).

    With ``axis is None`` this falls back to the existing scalar ``Ops.SUM`` reduction
    over every element. With an explicit axis (or tuple of axes) the result is an
    ``Ops.SUM_AXIS`` node carrying ``attrs["axes"]`` as a sorted tuple.
    """
    if axis is None:
      return Expr(Ops.SUM, (self,), TensorType((), dtype=self.type.dtype, diff=self.type.diff), lowering=self.lowering)
    axes = (int(axis),) if isinstance(axis, int) else tuple(sorted(int(a) for a in axis))
    rank = len(self.shape)
    norm: list[int] = []
    for a in axes:
      if a < 0:
        a = a + rank
      if not 0 <= a < rank:
        raise ValueError(f"sum axis {a} out of bounds for shape {self.shape}")
      norm.append(a)
    if len(set(norm)) != len(norm):
      raise ValueError(f"sum axes {axes} must be unique")
    norm.sort()
    new_shape = tuple(d for i, d in enumerate(self.shape) if i not in norm)
    return Expr(
      Ops.SUM_AXIS,
      (self,),
      TensorType(new_shape, dtype=self.type.dtype, diff=self.type.diff),
      attrs={"axes": tuple(norm)},
      lowering=self.lowering,
    )

  def dot(self, other: Any) -> Expr:
    return dot(self, other)

  def sumsqr(self) -> Expr:
    return sumsqr(self)

  def norm_2(self) -> Expr:
    return norm_2(self)

  def sin(self) -> Expr:
    return unary(Ops.SIN, self)

  def cos(self) -> Expr:
    return unary(Ops.COS, self)

  def tan(self) -> Expr:
    return unary(Ops.TAN, self)

  def asin(self) -> Expr:
    return unary(Ops.ASIN, self)

  def acos(self) -> Expr:
    return unary(Ops.ACOS, self)

  def atan(self) -> Expr:
    return unary(Ops.ATAN, self)

  def atan2(self, other: Any) -> Expr:
    return atan2(self, other)

  def sinh(self) -> Expr:
    return unary(Ops.SINH, self)

  def cosh(self) -> Expr:
    return unary(Ops.COSH, self)

  def tanh(self) -> Expr:
    return unary(Ops.TANH, self)

  def exp(self) -> Expr:
    return unary(Ops.EXP, self)

  def log(self) -> Expr:
    return unary(Ops.LOG, self)

  def sqrt(self) -> Expr:
    return unary(Ops.SQRT, self)

  def abs(self) -> Expr:
    return unary(Ops.ABS, self)

  def floor(self) -> Expr:
    return unary(Ops.FLOOR, self)

  def ceil(self) -> Expr:
    return unary(Ops.CEIL, self)

  def minimum(self, other: Any) -> Expr:
    return minimum(self, other)

  def maximum(self, other: Any) -> Expr:
    return maximum(self, other)

  def __neg__(self) -> Expr:
    return unary(Ops.NEG, self)

  def __add__(self, other: Any) -> Expr:
    return binary(Ops.ADD, self, as_expr(other))

  def __radd__(self, other: Any) -> Expr:
    return binary(Ops.ADD, as_expr(other), self)

  def __sub__(self, other: Any) -> Expr:
    return binary(Ops.SUB, self, as_expr(other))

  def __rsub__(self, other: Any) -> Expr:
    return binary(Ops.SUB, as_expr(other), self)

  def __mul__(self, other: Any) -> Expr:
    return binary(Ops.MUL, self, as_expr(other))

  def __rmul__(self, other: Any) -> Expr:
    return binary(Ops.MUL, as_expr(other), self)

  def __truediv__(self, other: Any) -> Expr:
    return binary(Ops.DIV, self, as_expr(other))

  def __rtruediv__(self, other: Any) -> Expr:
    return binary(Ops.DIV, as_expr(other), self)

  def __pow__(self, other: Any) -> Expr:
    return binary(Ops.POW, self, as_expr(other))

  def __rpow__(self, other: Any) -> Expr:
    return binary(Ops.POW, as_expr(other), self)

  def __matmul__(self, other: Any) -> Expr:
    return matmul(self, as_expr(other))

  def __rmatmul__(self, other: Any) -> Expr:
    return matmul(as_expr(other), self)

  def eval(self, env: dict[str, Any]) -> np.ndarray:
    if self.op == Ops.INPUT:
      if self.name not in env:
        raise KeyError(f"missing input {self.name!r}")
      return _asarray(env[self.name])
    if self.op == Ops.CONST:
      assert self.value is not None
      return self.value
    vals = [arg.eval(env) for arg in self.args]
    if self.op == Ops.RESHAPE:
      return vals[0].reshape(self.attrs["shape"])
    if self.op == Ops.TRANSPOSE:
      return np.transpose(vals[0], axes=self.attrs["axes"])
    if self.op == Ops.SLICE:
      return vals[0][self.attrs["index"]]
    if self.op == Ops.GATHER:
      indices = self.attrs["indices"]
      return np.take(vals[0].reshape(-1), indices).reshape(indices.shape)
    if self.op == Ops.SCATTER:
      out = np.zeros(self.shape, dtype=np.float64).reshape(-1)
      out[self.attrs["indices"].reshape(-1)] = vals[0].reshape(-1)
      return out.reshape(self.shape)
    if self.op == Ops.STACK:
      return np.stack(vals, axis=self.attrs.get("axis", 0))
    if self.op == Ops.CONCAT:
      return np.concatenate(vals, axis=self.attrs.get("axis", 0))
    if self.op == Ops.SUM:
      return np.asarray(np.sum(vals[0]), dtype=np.float64)
    if self.op == Ops.SUM_AXIS:
      return np.asarray(np.sum(vals[0], axis=tuple(self.attrs["axes"])), dtype=np.float64)
    if self.op == Ops.MATMUL:
      return vals[0] @ vals[1]
    if self.op == Ops.CALL:
      callee = self.attrs["callee"]
      return callee.eval_interpreter(*vals)[self.attrs["output"]]
    if self.op == Ops.MAP:
      return _eval_map(self, vals)
    info = OP_INFO[Ops(self.op)]
    if info.numpy is None:
      raise NotImplementedError(f"no numpy evaluator for op {self.op}")
    return _asarray(info.numpy(*vals))

  def inputs(self) -> dict[str, Expr]:
    ret: dict[str, Expr] = {}
    for e in topo([self]):
      if e.op == Ops.INPUT and e.name is not None:
        ret[e.name] = e
    return ret


def _attrs_key(attrs: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
  def key(v: Any) -> Any:
    if isinstance(v, Expr):
      return v.structural_key()
    if hasattr(v, "structural_key") and callable(getattr(v, "structural_key")):
      return v.structural_key()
    if hasattr(v, "name") and hasattr(v, "input_names") and hasattr(v, "output_names"):
      return ("Function", v.name, v.input_names, v.output_names)
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


def op_diff(op: Ops | str, *exprs: Expr) -> bool:
  return OP_INFO[Ops(op)].differentiable and diff_any(*exprs)


def promote_dtype(*exprs: Expr) -> DType:
  if not exprs:
    return dtypes.float64
  first = exprs[0].type.dtype
  for e in exprs[1:]:
    if e.type.dtype != first:
      raise TypeError(f"mixed-dtype operation not yet supported: {first} vs {e.type.dtype}; insert an explicit cast")
  return first


def unary(op: Ops | str, x: Expr) -> Expr:
  return Expr(op, (x,), TensorType(x.shape, dtype=x.type.dtype, diff=op_diff(op, x)), lowering=x.lowering)


def binary(op: Ops | str, x: Expr, y: Expr) -> Expr:
  return Expr(
    op,
    (x, y),
    TensorType(broadcast_shape(x.shape, y.shape), dtype=promote_dtype(x, y), diff=op_diff(op, x, y)),
    lowering=common_lowering(x, y),
  )


def atan2(y: Any, x: Any) -> Expr:
  return binary(Ops.ATAN2, as_expr(y), as_expr(x))


def minimum(x: Any, y: Any) -> Expr:
  return binary(Ops.MINIMUM, as_expr(x), as_expr(y))


def maximum(x: Any, y: Any) -> Expr:
  return binary(Ops.MAXIMUM, as_expr(x), as_expr(y))


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
  elif len(x.shape) == 3 and len(y.shape) == 3:
    # Batched matmul: (B,M,K) @ (B,K,N) -> (B,M,N), matching batch and contraction dims.
    if x.shape[0] != y.shape[0] or x.shape[2] != y.shape[1]:
      raise ValueError(f"cannot batch-matmul shapes {x.shape} and {y.shape}")
    shape = (x.shape[0], x.shape[1], y.shape[2])
  else:
    raise NotImplementedError(f"matmul shape inference for {x.shape} @ {y.shape}")
  return Expr(Ops.MATMUL, (x, y), TensorType(shape, dtype=promote_dtype(x, y), diff=diff_any(x, y)), lowering=common_lowering(x, y))


def dot(x: Any, y: Any) -> Expr:
  x, y = as_expr(x), as_expr(y)
  if x.size != y.size:
    raise ValueError(f"dot size mismatch: {x.shape} has {x.size} entries, {y.shape} has {y.size}")
  return (x.vec() * y.vec()).sum()


def sumsqr(x: Any) -> Expr:
  x = as_expr(x)
  return dot(x, x)


def norm_2(x: Any) -> Expr:
  return sumsqr(x).sqrt()


def _index_array(indices: Any, upper_bound: int) -> np.ndarray:
  idx = np.asarray(indices, dtype=np.int64)
  if idx.size and (idx.min() < 0 or idx.max() >= upper_bound):
    raise IndexError(f"indices must be in [0, {upper_bound})")
  return idx


def gather(x: Any, indices: Any) -> Expr:
  x = as_expr(x)
  idx = _index_array(indices, x.size)
  return Expr(Ops.GATHER, (x,), TensorType(tuple(idx.shape), dtype=x.type.dtype, diff=x.type.diff), attrs={"indices": idx}, lowering=x.lowering)


def scatter(values: Any, indices: Any, shape: int | tuple[int, ...]) -> Expr:
  values = as_expr(values)
  shape = as_shape(shape)
  idx = _index_array(indices, int(np.prod(shape, dtype=int)))
  if idx.size != values.size:
    raise ValueError(f"scatter has {idx.size} indices but values shape {values.shape} has {values.size} entries")
  flat = idx.reshape(-1)
  if len(set(flat.tolist())) != flat.size:
    raise ValueError("scatter indices must be unique")
  return Expr(
    Ops.SCATTER, (values,), TensorType(shape, dtype=values.type.dtype, diff=values.type.diff), attrs={"indices": idx}, lowering=values.lowering
  )


def split(x: Any, sections: int | Iterable[int], *, axis: int = 0) -> tuple[Expr, ...]:
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
    Ops.STACK, exprs, TensorType(shape, dtype=promote_dtype(*exprs), diff=diff_any(*exprs)), attrs={"axis": axis}, lowering=common_lowering(*exprs)
  )


def concat(xs: Iterable[Any], *, axis: int = 0) -> Expr:
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
    Ops.CONCAT, exprs, TensorType(shape, dtype=promote_dtype(*exprs), diff=diff_any(*exprs)), attrs={"axis": axis}, lowering=common_lowering(*exprs)
  )


def vec(x: Any) -> Expr:
  return as_expr(x).vec()


def map_(callee: Any, length: int, inputs: Any, output: int = 0) -> Expr:
  """Create an ``Ops.MAP`` node: ``length`` independent calls of ``callee`` whose i-th argument list is
  sliced out of outer tensors with per-input ``(start, stride)`` strides.

  ``inputs`` is either a sequence of ``(outer_tensor, start, stride)`` tuples ordered to match
  ``callee.inputs``, or a mapping from callee input name to the same tuple. The i-th iteration reads
  ``outer[start + i*stride : start + i*stride + callee.inputs[k].size]`` for callee input ``k``.
  Iterations are independent: ``stride=0`` broadcasts the same slice every iteration.

  ``MAP`` is intentionally *independent*: there is no carry between iterations. The future ``Ops.SCAN``
  op (see roadmap Phase 3) is what models dependent recurrences ``f(state_{i-1}, x_i) -> state_i``.
  Do not extend ``MAP`` with carry semantics; build ``SCAN`` instead when a workload needs it.

  This first cut requires all callee inputs and the selected output to be rank-1; the produced node
  has shape ``(length * callee.outputs[output].size,)`` and concatenates iteration outputs along that
  flat axis.
  """
  from .function import Function

  if not isinstance(callee, Function):
    raise TypeError(f"map callee must be an alloy Function, got {type(callee).__name__}")
  length = int(length)
  if length < 0:
    raise ValueError(f"map length must be non-negative, got {length}")
  if not 0 <= output < len(callee.outputs):
    raise ValueError(f"map output index {output} out of range for callee with {len(callee.outputs)} outputs")
  out_expr = callee.outputs[output]

  if isinstance(inputs, Mapping):
    extra = [n for n in inputs if n not in callee.input_names]
    if extra:
      raise ValueError(f"map inputs reference unknown callee input names {extra}; callee accepts {list(callee.input_names)}")
    missing = [n for n in callee.input_names if n not in inputs]
    if missing:
      raise ValueError(f"map inputs missing entries for callee inputs {missing}")
    specs = tuple(inputs[n] for n in callee.input_names)
  else:
    specs = tuple(inputs)
    if len(specs) != len(callee.inputs):
      raise ValueError(f"map expects {len(callee.inputs)} input specs, got {len(specs)}")
  outers: list[Expr] = []
  starts: list[int] = []
  strides: list[int] = []
  for i, spec in enumerate(specs):
    outer, start, stride = spec
    outer = as_expr(outer)
    formal = callee.inputs[i]
    if len(outer.shape) != 1:
      raise NotImplementedError(f"map currently requires rank-1 outer tensors, got {outer.shape} for input {i}")
    start = int(start)
    stride = int(stride)
    if start < 0:
      raise ValueError(f"map input {i} start must be non-negative, got {start}")
    if stride < 0:
      raise ValueError(f"map input {i} stride must be non-negative, got {stride}")
    if length > 0:
      end = start + (length - 1) * stride + formal.size
      if end > outer.size:
        raise ValueError(
          f"map input {i} reads past outer tensor of size {outer.size}: start={start}, stride={stride}, length={length}, slice_size={formal.size}"
        )
    outers.append(outer)
    starts.append(start)
    strides.append(stride)

  diff = out_expr.type.diff and any(o.type.diff for o in outers)
  return Expr(
    Ops.MAP,
    tuple(outers),
    TensorType((length * out_expr.size,), out_expr.type.dtype, diff=diff),
    attrs={
      "callee": callee,
      "output": int(output),
      "length": length,
      "starts": tuple(starts),
      "strides": tuple(strides),
      "slice_size": int(out_expr.size),
    },
    lowering=common_lowering(*outers) if outers else "auto",
  )


def _eval_map(expr: Expr, vals: list[np.ndarray]) -> np.ndarray:
  callee = expr.attrs["callee"]
  output_idx = expr.attrs["output"]
  length = expr.attrs["length"]
  starts = expr.attrs["starts"]
  strides = expr.attrs["strides"]
  slice_size = expr.attrs["slice_size"]
  out = np.empty((length * slice_size,), dtype=np.float64)
  for it in range(length):
    callee_args = [
      vals[i][starts[i] + it * strides[i] : starts[i] + it * strides[i] + callee.inputs[i].size].reshape(callee.inputs[i].shape)
      for i in range(len(callee.inputs))
    ]
    res = np.asarray(callee.eval_interpreter(*callee_args)[output_idx], dtype=np.float64)
    out[it * slice_size : (it + 1) * slice_size] = res.reshape(-1)
  return out


def zeros_like(x: Expr) -> Expr:
  return Expr.const(np.zeros(x.shape, dtype=np.float64), lowering=x.lowering)


def format_expr(outputs: Expr | Iterable[Expr]) -> str:
  outs = (outputs,) if isinstance(outputs, Expr) else tuple(outputs)
  nodes = topo(outs)
  loc = {e.id: i for i, e in enumerate(nodes)}
  lines: list[str] = []
  for i, e in enumerate(nodes):
    lhs = f"%{i}"
    if e.op == Ops.INPUT:
      rhs = f"input {e.name}"
    elif e.op == Ops.CONST:
      assert e.value is not None
      rhs = f"const {np.array2string(e.value, threshold=6)}"
    elif e.op == Ops.CALL:
      callee = e.attrs["callee"]
      rhs = f"call {callee.name}[{e.attrs['output']}]({', '.join(f'%{loc[a.id]}' for a in e.args)})"
    elif e.op == Ops.MAP:
      callee = e.attrs["callee"]
      length = e.attrs["length"]
      slice_size = e.attrs["slice_size"]
      bindings = ", ".join(f"%{loc[a.id]}[{s}::{st}]" for a, s, st in zip(e.args, e.attrs["starts"], e.attrs["strides"], strict=True))
      rhs = f"map[{length}x{slice_size}] {callee.name}[{e.attrs['output']}]({bindings})"
    else:
      rhs = f"{Ops(e.op).value}({', '.join(f'%{loc[a.id]}' for a in e.args)})"
    lines.append(f"{lhs} = {rhs} : {e.type.dtype}{e.shape}")
  lines.append("outputs " + ", ".join(f"%{loc[e.id]}" for e in outs))
  return "\n".join(lines)


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
