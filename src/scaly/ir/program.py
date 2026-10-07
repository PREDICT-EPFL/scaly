"""Program dialect vocabulary: ``ProgramOp``, ``ProgramNode``, interning, and the builders.

The verify rules are ``ir/program_spec.py`` and the printers are ``ir/text.py``.

Program IR is the lower representation that looks like
code. Where ``Expr`` (expression IR) preserves mathematical meaning, Program IR
chooses an implementation: which loops run and which buffer holds which value.

Design choices:

- A single flat ``ProgramNode`` class (frozen dataclass, hash-consed by op + args + attrs)
  with an ``op`` tag from ``ProgramOp``. Statement vs scalar-expression vs declaration
  is encoded by op tag, matching the tinygrad UOp style and keeping ``ir/match.py``'s
  PatternMatcher infrastructure reusable.
- Address spaces and device placement live on ``BUFFER`` nodes via attrs
  (``address_space`` ∈ ``global`` / ``local`` / ``private`` / ``constant``;
  ``device`` is a ``DeviceSpec``).
- Range kinds borrow from tinygrad's ``AxisType``: ``SERIAL``, ``VECTOR``,
  ``GLOBAL``, ``REDUCE``, ``UNROLL``. The kind is a verifier-checked attribute of
  ``RANGE`` and ``FOR``, not a separate op.

Per the roadmap, Program IR is a real language: every construct must verify and every malformed
node must produce a clear diagnostic. The pretty-printer that has to show enough to debug
schedule decisions is ``ir/text.py``.
"""

from __future__ import annotations

import struct
import weakref
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any

import numpy as np

from .types import DType, DeviceSpec, dtypes


class ProgramOp(StrEnum):
  """Program IR opcode set. Statement-shaped ops sit alongside scalar/index ops."""

  # Declarations
  PROGRAM = "program"
  PROC = "proc"
  BUFFER = "buffer"
  VIEW = "view"

  # Statement-level
  BLOCK = "block"
  FOR = "for"
  RANGE = "range"
  ASSIGN = "assign"
  STORE = "store"
  STORE_PAIR = "store_pair"
  CALL = "call"

  # Scalar/index-level
  CONST_INT = "const_int"
  CONST_FLOAT = "const_float"
  VAR = "var"
  LOAD = "load"
  ADD = "add"
  SUB = "sub"
  MUL = "mul"
  DIV = "div"
  MOD = "mod"
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
  POW = "pow"
  ATAN2 = "atan2"
  MINIMUM = "minimum"
  MAXIMUM = "maximum"


class RangeKind(StrEnum):
  """Loop binding kinds borrowed from tinygrad's ``AxisType``.

  Backends use these to choose between vectorized or unrolled emission
  (``VECTOR``/``UNROLL``) or to lower reductions (``REDUCE``).
  """

  SERIAL = "serial"
  VECTOR = "vector"
  GLOBAL = "global"
  REDUCE = "reduce"
  UNROLL = "unroll"


ADDRESS_SPACES = frozenset({"global", "local", "private", "constant"})


SCALAR_OPS: frozenset[ProgramOp] = frozenset(
  {
    ProgramOp.CONST_INT,
    ProgramOp.CONST_FLOAT,
    ProgramOp.VAR,
    ProgramOp.LOAD,
    ProgramOp.ADD,
    ProgramOp.SUB,
    ProgramOp.MUL,
    ProgramOp.DIV,
    ProgramOp.MOD,
    ProgramOp.NEG,
    ProgramOp.SIN,
    ProgramOp.COS,
    ProgramOp.TAN,
    ProgramOp.ASIN,
    ProgramOp.ACOS,
    ProgramOp.ATAN,
    ProgramOp.SINH,
    ProgramOp.COSH,
    ProgramOp.TANH,
    ProgramOp.ERF,
    ProgramOp.EXP,
    ProgramOp.LOG,
    ProgramOp.SQRT,
    ProgramOp.ABS,
    ProgramOp.FLOOR,
    ProgramOp.CEIL,
    ProgramOp.POW,
    ProgramOp.ATAN2,
    ProgramOp.MINIMUM,
    ProgramOp.MAXIMUM,
  }
)


# Scalar unary/binary ProgramOp that render as a C function call (libm), keyed for the
# pretty-printer and the C renderer. NEG/ADD/SUB/MUL/DIV/MOD render as operators.
UNARY_FN_OPS: frozenset[ProgramOp] = frozenset(
  {
    ProgramOp.SIN,
    ProgramOp.COS,
    ProgramOp.TAN,
    ProgramOp.ASIN,
    ProgramOp.ACOS,
    ProgramOp.ATAN,
    ProgramOp.SINH,
    ProgramOp.COSH,
    ProgramOp.TANH,
    ProgramOp.ERF,
    ProgramOp.EXP,
    ProgramOp.LOG,
    ProgramOp.SQRT,
    ProgramOp.ABS,
    ProgramOp.FLOOR,
    ProgramOp.CEIL,
  }
)
BINARY_FN_OPS: frozenset[ProgramOp] = frozenset({ProgramOp.POW, ProgramOp.ATAN2, ProgramOp.MINIMUM, ProgramOp.MAXIMUM})


# Hash-cons Program IR nodes the same way ``Expr`` is hash-consed: structurally-
# equal nodes collapse to the same Python object so identity == equality. Child refs in
# cache keys are weakrefs, not raw ``id(...)`` integers, so CPython id reuse cannot alias
# a new Program IR subgraph to a still-cached old node.
_PROGRAM_NODE_CACHE: weakref.WeakValueDictionary[tuple[Any, ...], "ProgramNode"] = weakref.WeakValueDictionary()


def _attr_key(v: Any) -> Any:
  # ``==`` merges what C tells apart: ``-0.0`` with ``0.0``, and ``1`` with ``True`` and ``1.0``. A
  # float keys by its bits, which also lets a NaN match itself, and a flag by its kind.
  if isinstance(v, (bool, np.bool_)):
    return (bool, bool(v))
  if isinstance(v, (float, np.floating)):
    return (float, struct.pack("<d", v))
  if isinstance(v, Mapping):
    return tuple((k, _attr_key(x)) for k, x in sorted(v.items()))
  if isinstance(v, (tuple, list)):
    if v and all(type(x) is float for x in v):  # a constant table packs at once, much faster than value by value
      return (tuple, struct.pack(f"<{len(v)}d", *v))
    return tuple(map(_attr_key, v))
  return v


def _attrs_key(attrs: Mapping[str, Any]) -> tuple[Any, ...]:
  return tuple((k, _attr_key(v)) for k, v in sorted(attrs.items()))


# ``init=False``: ``__new__`` is the whole constructor. A generated ``__init__`` would run again on
# the node an interning hit returns, which programs already hold.
@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class ProgramNode:
  """A flat Program IR node.

  ``op`` selects the variant (statement-shaped, declaration-shaped, or scalar-
  shaped). ``args`` holds child nodes; ``attrs`` holds non-node metadata; the
  ``dtype`` field only matters for scalar-typed nodes (``CONST_*``, ``LOAD``,
  ``ADD``, ``MUL``, ...).
  """

  op: ProgramOp
  args: tuple["ProgramNode", ...] = ()
  attrs: Mapping[str, Any] = field(default_factory=dict)
  dtype: DType = dtypes.float64

  def __new__(
    cls,
    op: ProgramOp,
    args: tuple["ProgramNode", ...] = (),
    attrs: Mapping[str, Any] | None = None,
    dtype: DType = dtypes.float64,
  ) -> "ProgramNode":
    attrs = attrs or {}
    key = (op.value, tuple(weakref.ref(a) for a in args), _attrs_key(attrs), dtype.name)
    cached = _PROGRAM_NODE_CACHE.get(key)
    if cached is not None:
      return cached
    instance = object.__new__(cls)
    object.__setattr__(instance, "op", op)
    object.__setattr__(instance, "args", tuple(args))
    object.__setattr__(instance, "attrs", MappingProxyType(dict(attrs)))
    object.__setattr__(instance, "dtype", dtype)
    _PROGRAM_NODE_CACHE[key] = instance
    return instance

  @property
  def id(self) -> int:
    return id(self)

  def __repr__(self) -> str:  # pragma: no cover - cosmetic
    attr_str = "" if not self.attrs else f" {self.attrs}"
    return f"ProgramNode({self.op.value}, {len(self.args)} args{attr_str})"


def walk_program(root: ProgramNode) -> Iterator[ProgramNode]:
  """Yield ``root`` and its descendants once by identity in depth-first order."""
  seen: set[int] = set()
  stack = [root]
  while stack:
    node = stack.pop()
    if id(node) in seen:
      continue
    seen.add(id(node))
    yield node
    stack.extend(node.args)


# ---------------------------------------------------------------------------
# Builder helpers — the public way to construct Program IR.
# ---------------------------------------------------------------------------


def const_int(value: int) -> ProgramNode:
  return ProgramNode(ProgramOp.CONST_INT, (), attrs={"value": int(value)}, dtype=dtypes.int64)


def const_float(value: float, dtype: DType = dtypes.float64) -> ProgramNode:
  return ProgramNode(ProgramOp.CONST_FLOAT, (), attrs={"value": float(value)}, dtype=dtype)


def var(name: str, dtype: DType = dtypes.int64) -> ProgramNode:
  return ProgramNode(ProgramOp.VAR, (), attrs={"name": name}, dtype=dtype)


def buffer(name: str, dtype: DType, shape: tuple[int, ...], *, address_space: str = "global", device: DeviceSpec | str | None = None) -> ProgramNode:
  if address_space not in ADDRESS_SPACES:
    raise ValueError(f"unknown address space {address_space!r}; expected one of {sorted(ADDRESS_SPACES)}")
  dev = DeviceSpec.parse(device)
  return ProgramNode(
    ProgramOp.BUFFER,
    (),
    attrs={"name": name, "shape": tuple(int(d) for d in shape), "address_space": address_space, "device": dev},
    dtype=dtype,
  )


def const_buffer(name: str, dtype: DType, shape: tuple[int, ...], values: Sequence[float]) -> ProgramNode:
  """A read-only ``constant``-address-space BUFFER carrying its initializer ``values``.

  The renderer emits it as a ``static const`` array; loads read it like any buffer.
  Lets constants of any size lower without inline-serializing each element.
  """
  return ProgramNode(
    ProgramOp.BUFFER,
    (),
    attrs={
      "name": name,
      "shape": tuple(int(d) for d in shape),
      "address_space": "constant",
      "device": DeviceSpec.parse(None),
      "values": tuple(values),  # numeric type preserved; the renderer formats by dtype
    },
    dtype=dtype,
  )


def view(buf: ProgramNode, index: Sequence[ProgramNode]) -> ProgramNode:
  if buf.op != ProgramOp.BUFFER:
    raise TypeError(f"view requires a BUFFER, got {buf.op}")
  if len(index) != 1:
    raise ValueError("view requires exactly one flat index")
  if index[0].op not in SCALAR_OPS:
    raise TypeError(f"view index must be a scalar ProgramNode, got {index[0].op}")
  return ProgramNode(ProgramOp.VIEW, tuple(index), attrs={"buffer": buf.attrs["name"]}, dtype=buf.dtype)


def load(view_node: ProgramNode) -> ProgramNode:
  if view_node.op != ProgramOp.VIEW:
    raise TypeError(f"load expects a VIEW, got {view_node.op}")
  return ProgramNode(ProgramOp.LOAD, (view_node,), dtype=view_node.dtype)


def store(view_node: ProgramNode, value: ProgramNode) -> ProgramNode:
  if view_node.op != ProgramOp.VIEW:
    raise TypeError(f"store expects a VIEW target, got {view_node.op}")
  if value.op not in SCALAR_OPS:
    raise TypeError(f"store value must be a scalar ProgramNode, got {value.op}")
  return ProgramNode(ProgramOp.STORE, (view_node, value), dtype=value.dtype)


def store_pair(view_node: ProgramNode, first: ProgramNode, second: ProgramNode) -> ProgramNode:
  """Store two float64 values at a view and its next scalar element."""
  if view_node.op != ProgramOp.VIEW:
    raise TypeError(f"store_pair expects a VIEW target, got {view_node.op}")
  if first.op not in SCALAR_OPS or second.op not in SCALAR_OPS:
    raise TypeError("store_pair values must be scalar ProgramNodes")
  if view_node.dtype != dtypes.float64 or first.dtype != dtypes.float64 or second.dtype != dtypes.float64:
    raise TypeError("store_pair requires float64 target and values")
  return ProgramNode(ProgramOp.STORE_PAIR, (view_node, first, second), dtype=dtypes.float64)


def assign(target: str, value: ProgramNode, dtype: DType | None = None, *, declare: bool = False) -> ProgramNode:
  """Assign a scalar value, optionally declaring a new typed local variable."""
  if value.op not in SCALAR_OPS:
    raise TypeError(f"assign value must be a scalar ProgramNode, got {value.op}")
  return ProgramNode(ProgramOp.ASSIGN, (value,), attrs={"target": target, **({"declare": True} if declare else {})}, dtype=dtype or value.dtype)


def range_(
  name: str, start: ProgramNode | int, stop: ProgramNode | int, *, step: ProgramNode | int = 1, kind: RangeKind = RangeKind.SERIAL
) -> ProgramNode:
  s = start if isinstance(start, ProgramNode) else const_int(start)
  e = stop if isinstance(stop, ProgramNode) else const_int(stop)
  st = step if isinstance(step, ProgramNode) else const_int(step)
  return ProgramNode(ProgramOp.RANGE, (s, e, st), attrs={"name": name, "kind": RangeKind(kind)}, dtype=dtypes.int64)


def for_(rng: ProgramNode, body: Sequence[ProgramNode]) -> ProgramNode:
  if rng.op != ProgramOp.RANGE:
    raise TypeError(f"for_ requires a RANGE, got {rng.op}")
  return ProgramNode(ProgramOp.FOR, (rng, *body), attrs={"body_len": len(body)})


def block(*statements: ProgramNode) -> ProgramNode:
  return ProgramNode(ProgramOp.BLOCK, tuple(statements))


def call(callee: str, args: Sequence[ProgramNode], *, returns: Sequence[str] = ()) -> ProgramNode:
  return ProgramNode(ProgramOp.CALL, tuple(args), attrs={"callee": callee, "returns": tuple(returns)})


def proc(name: str, params: Sequence[ProgramNode], body: Sequence[ProgramNode], *, device: DeviceSpec | str | None = None) -> ProgramNode:
  for p in params:
    if p.op != ProgramOp.BUFFER:
      raise TypeError(f"proc params must be BUFFER nodes, got {p.op}")
  return ProgramNode(ProgramOp.PROC, (*params, *body), attrs={"name": name, "param_count": len(params), "device": DeviceSpec.parse(device)})


def program(procs: Sequence[ProgramNode]) -> ProgramNode:
  for p in procs:
    if p.op != ProgramOp.PROC:
      raise TypeError(f"program procs must be PROC nodes, got {p.op}")
  return ProgramNode(ProgramOp.PROGRAM, tuple(procs), attrs={"proc_count": len(procs)})


# Binary scalar helpers — used by the lowerer (Phase 5+) and by tests.


def _scalar_binop(op: ProgramOp, x: ProgramNode, y: ProgramNode) -> ProgramNode:
  if x.op not in SCALAR_OPS or y.op not in SCALAR_OPS:
    raise TypeError(f"{op.value} requires scalar ProgramNode operands, got {x.op} and {y.op}")
  if x.dtype != y.dtype:
    raise TypeError(f"{op.value} operand dtype mismatch: {x.dtype} vs {y.dtype}")
  return ProgramNode(op, (x, y), dtype=x.dtype)


def add(x: ProgramNode, y: ProgramNode) -> ProgramNode:
  return _scalar_binop(ProgramOp.ADD, x, y)


def sub(x: ProgramNode, y: ProgramNode) -> ProgramNode:
  return _scalar_binop(ProgramOp.SUB, x, y)


def mul(x: ProgramNode, y: ProgramNode) -> ProgramNode:
  return _scalar_binop(ProgramOp.MUL, x, y)


def div(x: ProgramNode, y: ProgramNode) -> ProgramNode:
  return _scalar_binop(ProgramOp.DIV, x, y)


def mod(x: ProgramNode, y: ProgramNode) -> ProgramNode:
  return _scalar_binop(ProgramOp.MOD, x, y)


def neg(x: ProgramNode) -> ProgramNode:
  if x.op not in SCALAR_OPS:
    raise TypeError(f"neg requires a scalar ProgramNode, got {x.op}")
  return ProgramNode(ProgramOp.NEG, (x,), dtype=x.dtype)
