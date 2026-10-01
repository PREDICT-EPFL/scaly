"""Expression dialect vocabulary: the op registry (``OpDef``, ``register_op``), ``Expr``, interning, builders, topo.

The verify rules are ``ir/expr_spec.py`` and the one builder that needs a ``Function`` — ``vmap``
— is ``function/sugar.py``. The printers are ``ir/text.py``, with one exception: ``format_expr``
stays here because ``Expr.debug`` calls it, and moving it would make ``ir/expr.py`` import
``ir/text.py``, which imports ``ir/expr.py``.
"""

from __future__ import annotations

import math
import struct
import weakref
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .program import ProgramOp
from .types import DType, Lowering, TensorType, as_dtype, as_shape, broadcast_shape, dtypes


class ExprOp(StrEnum):
  """The names of the builtin expression ops.

  A ``StrEnum``, so ``ExprOp.ADD`` is the string ``"add"`` and compares, hashes and prints as it.
  The op set itself is the registry (``register_op``): every op, builtin or registered by an
  extension, has an ``OpDef`` there, and an ``Expr`` holds its op's registered name. Extension ops
  have no member here.
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
  TAKE = "take"
  PUT_ADD = "put_add"
  PUT = "put"
  RESHAPE = "reshape"
  TRANSPOSE = "transpose"
  SLICE = "slice"
  GATHER = "gather"
  SEGMENT_REDUCE = "segment_reduce"
  STACK = "stack"
  CONCAT = "concat"
  MATMUL = "matmul"
  CALL = "call"
  VMAP = "vmap"
  SCAN = "scan"
  WHILE = "while"
  EXTERN_CALL = "extern_call"


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
  ExprOp.TAKE,
  ExprOp.PUT_ADD,
  ExprOp.PUT,
  ExprOp.RESHAPE,
  ExprOp.TRANSPOSE,
  ExprOp.SLICE,
  ExprOp.GATHER,
  ExprOp.SEGMENT_REDUCE,
  ExprOp.STACK,
  ExprOp.CONCAT,
  ExprOp.MATMUL,
  ExprOp.CALL,
  ExprOp.VMAP,
  ExprOp.SCAN,
  ExprOp.WHILE,
  ExprOp.EXTERN_CALL,
}

# Ops whose ``callee`` attr names a ``Function`` the graph runs (a ``while`` also runs its ``cond``);
# ``callees_of`` lists them per node.
CALLEE_OPS = {ExprOp.CALL, ExprOp.VMAP, ExprOp.SCAN, ExprOp.WHILE}

# Deliberately not in the MVP set: expm1/log1p (nice but low priority), splines/interpolants
# (important but require carefully specified extrapolation, knots, derivatives, and codegen tables),
# matrix exponentials/decompositions, and solver/control-flow ops.
COMMON_OPS = COMMON_STRUCTURAL | COMMON_ELEMENTWISE_UNARY | COMMON_ELEMENTWISE_BINARY | COMMON_CONTROL


_RULE_KINDS = ("jvp", "jvp_many", "vjp", "sparsity", "fold", "verify", "lower")


class OpDef:
  """One expression op: its name, arity (``None`` for variadic), NumPy evaluation rule for constant
  folding (``None`` when it has none), whether a derivative can flow through it, and its rules.

  The rules are what the compiler asks of an op; each is ``None`` until a module defines it with
  ``define_rules``, and an op without one gets the default named here:

  - ``jvp(expr, d)``: the tangent of ``expr`` from ``d``, one tangent per argument. None: an error.
  - ``jvp_many(expr, tan, nseed)``: tangents with a leading seed axis; ``tan(arg)`` forms an
    argument's, so a rule forms only those it reads. None: ``jvp`` once per seed, stacked.
  - ``vjp(expr, cot)``: one cotangent per argument. None: an error.
  - ``sparsity(expr, mask, ncols)``: the structural Jacobian pattern, a boolean CSR array of
    ``(expr.size, ncols)`` from ``mask(arg)``, each argument's. None: every entry of the output on
    every column any argument reads.
  - ``fold(expr, values)``: the value from constant arguments, or ``None`` to leave the node. None:
    ``numpy`` applied to the values, when there is one.
  - ``verify``: rules (``ir.spec.Rule``) a node with this op must pass, beyond the shared ones.
  - ``lower(ctx, node)``: the lowering to the program dialect (``passes/lowering.py``). None: the
    op cannot be lowered.

  ``traits`` are what passes ask about an op beyond its rules (``define_traits`` sets them):

  - ``elementwise``: entry ``i`` of the result reads entry ``i`` of each (broadcast) argument, as
    the program op this names computes it; such an op needs no lowering rule of its own.
  - ``expensive``: an elementwise op that costs a libm call, which fusion does not duplicate.
  - ``runtime_index``: addresses memory through an index known only at run time.
  - ``exact_reads``: its structural pattern is exactly the entries it reads, so an in-place loop
    body may be proven safe through it.
  - ``update(node, value, length)``: an op that writes some entries of its first argument and keeps
    the rest; the callback gives the positions it writes at each of ``length`` steps.
  - ``reads(node, position, value, length)``: the positions it reads of argument ``position`` at
    each step, when that argument is a loop carry updated in place.

  Positions are a ``(length, lanes)`` integer array, -1 for a dropped lane, or an object with
  ``bounds()`` and ``explicit()`` (``passes/lowering.py``); ``None`` when they cannot be bounded.
  ``runtime_index`` and ``exact_reads`` may be a predicate on the node instead of a flag, when the
  answer depends on the node's arguments: a ``put`` at constant indices has no run-time index.
  ``expr_has_trait`` asks about one node, ``has_trait`` about the op.

  Compared and hashed by identity: the registry holds one per name."""

  __slots__ = ("name", "arity", "numpy", "differentiable", "traits", *_RULE_KINDS)

  def __init__(self, name: str, arity: int | None, numpy: Callable[..., Any] | None = None, differentiable: bool = True) -> None:
    self.name = name
    self.arity = arity
    self.numpy = numpy
    self.differentiable = differentiable
    self.jvp: Callable[..., Any] | None = None
    self.jvp_many: Callable[..., Any] | None = None
    self.vjp: Callable[..., Any] | None = None
    self.sparsity: Callable[..., Any] | None = None
    self.fold: Callable[..., Any] | None = None
    self.verify: tuple[Any, ...] = ()
    self.lower: Callable[..., Any] | None = None
    self.traits: dict[str, Any] = {}

  def __repr__(self) -> str:
    return f"OpDef({str(self.name)!r})"


_OPS: dict[str, OpDef] = {}
_VERSION = [0]  # bumped by every registration, rule and trait: a key for caches derived from the registry


def registry_version() -> int:
  """A number that changes whenever an op, a rule or a trait is added, for caches derived from them."""
  return _VERSION[0]


def register_op(
  name: str,
  *,
  arity: int | None,
  numpy: Callable[..., Any] | None = None,
  differentiable: bool = True,
  traits: dict[str, Any] | None = None,
  **rules: Any,
) -> OpDef:
  """Add the op ``name`` to the expression dialect, with any of its rules (``OpDef``); registering
  a name twice raises.

  Registration has to happen before an ``Expr`` with the op is built, which holds by construction
  when the module that registers the op is the one that provides its builder.

  The JIT's key for a library it built before takes an op's definition as it stands: its arity,
  its traits by value, its rules by module and name, and the files of the package that defines
  them. Keep what shapes an op's C in that package, and its rules plain functions: one that is a
  closure, a bound method or a partial application carries state no file holds, and a Function
  with such an op is rendered at each start."""
  if name in _OPS:
    raise ValueError(f"expression op {name!r} is already registered")
  definition = OpDef(name, arity, numpy, differentiable)
  _OPS[name] = definition
  _VERSION[0] += 1
  if rules:
    define_rules(name, **rules)
  if traits:
    define_traits(name, **traits)
  return definition


def define_rules(op: str, **rules: Any) -> OpDef:
  """Give the registered op ``op`` rules it does not have yet (``OpDef`` names them); defining one
  twice raises, so two modules cannot silently disagree about an op."""
  definition = op_def(op)
  for kind, rule in rules.items():
    if kind not in _RULE_KINDS:
      raise TypeError(f"unknown op rule {kind!r}; the rules are {', '.join(_RULE_KINDS)}")
    if getattr(definition, kind):
      raise ValueError(f"the {kind} rule of expression op {op!r} is already defined")
    setattr(definition, kind, tuple(rule) if kind == "verify" else rule)
  _VERSION[0] += 1
  return definition


_TRAITS = ("elementwise", "expensive", "runtime_index", "exact_reads", "update", "reads")


def define_traits(op: str, **traits: Any) -> OpDef:
  """Give the registered op ``op`` traits (``OpDef``); setting one twice raises."""
  definition = op_def(op)
  for name, value in traits.items():
    if name not in _TRAITS:
      raise TypeError(f"unknown op trait {name!r}; the traits are {', '.join(_TRAITS)}")
    if name in definition.traits:
      raise ValueError(f"the {name} trait of expression op {op!r} is already defined")
    definition.traits[name] = value
  _VERSION[0] += 1
  return definition


def has_trait(op: str, name: str) -> bool:
  """Whether the registered op ``op`` has the trait ``name``; for a flag given as a predicate, whether
  some node of the op may have it (``expr_has_trait`` asks about one)."""
  return bool(op_def(op).traits.get(name))


def expr_has_trait(expr: Expr, name: str) -> bool:
  """Whether the node ``expr`` has the flag trait ``name`` (``runtime_index``, ``exact_reads``): its
  op's flag, or the predicate's answer for this node where the op gives one (``OpDef``)."""
  value = op_def(expr.op).traits.get(name)
  return bool(value(expr)) if callable(value) else bool(value)


def op_def(op: str) -> OpDef:
  """The registered definition of ``op``."""
  definition = _OPS.get(op)
  if definition is None:
    raise ValueError(f"unknown expression op {op!r}; register it with register_op first")
  return definition


def registered_ops() -> tuple[str, ...]:
  """Every registered op name, builtins first, in registration order."""
  return tuple(d.name for d in _OPS.values())


# name, arity, NumPy rule[, differentiable]
_BUILTIN_OPS: tuple[tuple[Any, ...], ...] = (
  (ExprOp.INPUT, 0, None),
  (ExprOp.CONST, 0, None, False),
  (ExprOp.NEG, 1, np.negative),
  (ExprOp.SIN, 1, np.sin),
  (ExprOp.COS, 1, np.cos),
  (ExprOp.TAN, 1, np.tan),
  (ExprOp.ASIN, 1, np.arcsin),
  (ExprOp.ACOS, 1, np.arccos),
  (ExprOp.ATAN, 1, np.arctan),
  (ExprOp.SINH, 1, np.sinh),
  (ExprOp.COSH, 1, np.cosh),
  (ExprOp.TANH, 1, np.tanh),
  # NumPy has no erf; frompyfunc keeps constant folding vectorized without adding SciPy.
  (ExprOp.ERF, 1, lambda x: np.asarray(np.frompyfunc(math.erf, 1, 1)(x), dtype=np.float64)),
  (ExprOp.EXP, 1, np.exp),
  (ExprOp.LOG, 1, np.log),
  (ExprOp.SQRT, 1, np.sqrt),
  (ExprOp.ABS, 1, np.abs),
  (ExprOp.FLOOR, 1, np.floor),
  (ExprOp.CEIL, 1, np.ceil),
  (ExprOp.ADD, 2, np.add),
  (ExprOp.SUB, 2, np.subtract),
  (ExprOp.MUL, 2, np.multiply),
  (ExprOp.DIV, 2, np.divide),
  (ExprOp.POW, 2, np.power),
  (ExprOp.ATAN2, 2, np.arctan2),
  (ExprOp.MINIMUM, 2, np.fmin),  # as C's fmin: a NaN operand gives the other
  (ExprOp.MAXIMUM, 2, np.fmax),
  (ExprOp.COPYSIGN, 2, np.copysign),
  (ExprOp.LT, 2, np.less, False),
  (ExprOp.LE, 2, np.less_equal, False),
  (ExprOp.EQ, 2, np.equal, False),
  (ExprOp.NE, 2, np.not_equal, False),
  (ExprOp.AND, 2, np.logical_and, False),
  (ExprOp.OR, 2, np.logical_or, False),
  (ExprOp.NOT, 1, np.logical_not, False),
  (ExprOp.ISFINITE, 1, np.isfinite, False),
  (ExprOp.SELECT, 3, np.where),
  (ExprOp.CAST, 1, None),
  (ExprOp.SUM, 1, np.sum),
  (ExprOp.MAX, 1, np.max),
  (ExprOp.MIN, 1, np.min),
  (ExprOp.TAKE, 2, None),
  (ExprOp.PUT_ADD, 3, None),
  (ExprOp.PUT, 3, None),
  (ExprOp.RESHAPE, 1, np.reshape),
  (ExprOp.TRANSPOSE, 1, np.transpose),
  (ExprOp.SLICE, 1, None),
  (ExprOp.GATHER, 1, None),
  (ExprOp.SEGMENT_REDUCE, 1, None),
  (ExprOp.STACK, None, np.stack),
  (ExprOp.CONCAT, None, np.concatenate),
  (ExprOp.MATMUL, 2, np.matmul),
  (ExprOp.CALL, None, None),
  (ExprOp.VMAP, None, None),
  (ExprOp.SCAN, None, None),
  (ExprOp.WHILE, None, None),
  (ExprOp.EXTERN_CALL, None, None, False),
)
for _op, _arity, _numpy, *_diff in _BUILTIN_OPS:
  register_op(_op, arity=_arity, numpy=_numpy, differentiable=_diff[0] if _diff else True)
del _op, _arity, _numpy, _diff

# The ops that lower one entry at a time, each to the program op named (the shared elementwise
# lowering in ``passes/lowering.py``).
for _op, _program_op in (
  (ExprOp.NEG, ProgramOp.NEG),
  (ExprOp.SIN, ProgramOp.SIN),
  (ExprOp.COS, ProgramOp.COS),
  (ExprOp.TAN, ProgramOp.TAN),
  (ExprOp.ASIN, ProgramOp.ASIN),
  (ExprOp.ACOS, ProgramOp.ACOS),
  (ExprOp.ATAN, ProgramOp.ATAN),
  (ExprOp.SINH, ProgramOp.SINH),
  (ExprOp.COSH, ProgramOp.COSH),
  (ExprOp.TANH, ProgramOp.TANH),
  (ExprOp.ERF, ProgramOp.ERF),
  (ExprOp.EXP, ProgramOp.EXP),
  (ExprOp.LOG, ProgramOp.LOG),
  (ExprOp.SQRT, ProgramOp.SQRT),
  (ExprOp.ABS, ProgramOp.ABS),
  (ExprOp.FLOOR, ProgramOp.FLOOR),
  (ExprOp.CEIL, ProgramOp.CEIL),
  (ExprOp.NOT, ProgramOp.NOT),
  (ExprOp.ISFINITE, ProgramOp.ISFINITE),
  (ExprOp.CAST, ProgramOp.CAST),
  (ExprOp.ADD, ProgramOp.ADD),
  (ExprOp.SUB, ProgramOp.SUB),
  (ExprOp.MUL, ProgramOp.MUL),
  (ExprOp.DIV, ProgramOp.DIV),
  (ExprOp.POW, ProgramOp.POW),
  (ExprOp.ATAN2, ProgramOp.ATAN2),
  (ExprOp.MINIMUM, ProgramOp.MINIMUM),
  (ExprOp.MAXIMUM, ProgramOp.MAXIMUM),
  (ExprOp.COPYSIGN, ProgramOp.COPYSIGN),
  (ExprOp.LT, ProgramOp.LT),
  (ExprOp.LE, ProgramOp.LE),
  (ExprOp.EQ, ProgramOp.EQ),
  (ExprOp.NE, ProgramOp.NE),
  (ExprOp.AND, ProgramOp.AND),
  (ExprOp.OR, ProgramOp.OR),
):
  define_traits(_op, elementwise=_program_op)
del _program_op
# Elementwise ops that cost a libm call.
for _op in (ExprOp.SIN, ExprOp.COS, ExprOp.TAN, ExprOp.ASIN, ExprOp.ACOS, ExprOp.ATAN, ExprOp.SINH, ExprOp.COSH, ExprOp.TANH):
  define_traits(_op, expensive=True)
for _op in (ExprOp.ERF, ExprOp.EXP, ExprOp.LOG, ExprOp.SQRT, ExprOp.POW, ExprOp.ATAN2):
  define_traits(_op, expensive=True)
define_traits(ExprOp.TAKE, runtime_index=True)
# A put at constant indices addresses fixed entries, which lowering resolves as it does a gather's;
# its structural pattern is then exactly what it reads, as below.
for _op in (ExprOp.PUT_ADD, ExprOp.PUT):
  define_traits(_op, runtime_index=lambda node: node.args[1].op != ExprOp.CONST, exact_reads=lambda node: node.args[1].op == ExprOp.CONST)
# Ops whose structural pattern is exactly the entries they read, so the pattern can stand in for a
# read set. Everything else (predicates, ``select``'s condition, ``copysign``'s sign, ``floor`` and
# ``ceil``, whose derivative is zero, casts, calls, maps and loops) may read entries its pattern
# omits.
for _op in (
  ExprOp.INPUT,
  ExprOp.CONST,
  *((COMMON_ELEMENTWISE_UNARY - {ExprOp.FLOOR, ExprOp.CEIL}) | (COMMON_ELEMENTWISE_BINARY - {ExprOp.COPYSIGN})),
  ExprOp.SUM,
  ExprOp.MAX,
  ExprOp.MIN,
  ExprOp.RESHAPE,
  ExprOp.TRANSPOSE,
  ExprOp.SLICE,
  ExprOp.GATHER,
  ExprOp.SEGMENT_REDUCE,
  ExprOp.STACK,
  ExprOp.CONCAT,
  ExprOp.MATMUL,
):
  define_traits(_op, exact_reads=True)
del _op


def _asarray(value: Any, *, dtype: DType | str | None = None) -> np.ndarray:
  np_dtype = as_dtype(dtype).numpy() if dtype is not None else np.float64
  return np.asarray(value, dtype=np_dtype)


# Construction-time hash-consing cache. Two ``Expr(...)`` constructions with the same
# (op, args-by-identity, type, name, value-bytes, attrs, lowering) collapse to a single
# Python object — so structural equality becomes identity, ``is`` works as a fast equality
# check, and any pass that builds new graphs gets CSE for free.
#
# The key tells apart what the generated C can: an attribute that is a float keys by its bits and
# a number by its kind (``_attrs_key``), a constant by its bytes. What it still merges (two equal
# ``TensorType`` objects, a list with a tuple, ``np.int64(1)`` with ``1``) is equal and not
# identical, so a hit returns the node exactly as it was first built: ``Expr.__new__`` assigns
# the fields of a new node only, and the class has no ``__init__`` to assign them again.
#
# WeakValueDictionary lets nodes be garbage-collected when no live reference remains; the
# cache shrinks automatically. Arg refs in cache keys are weakrefs, not raw ``id(...)``
# integers, so CPython id reuse cannot alias a new subgraph to a still-cached old node.
_NODE_CACHE: weakref.WeakValueDictionary[tuple[Any, ...], "Expr"] = weakref.WeakValueDictionary()


def _intern_key(
  op: str,
  args: tuple["Expr", ...],
  type_: "TensorType",
  name: str | None,
  value: np.ndarray | None,
  attrs: dict[str, Any],
  lowering: Lowering,
) -> tuple[Any, ...]:
  args_key = tuple(weakref.ref(a) for a in args)
  value_key = None if value is None else (value.shape, str(value.dtype), value.tobytes())
  return (str(op), args_key, type_, name, value_key, _attrs_key(attrs), lowering)


# ``init=False``: ``__new__`` is the whole constructor. A generated ``__init__`` would run on the
# node ``__new__`` returns, which on an interning hit is one that Functions already hold.
@dataclass(frozen=True, slots=True, weakref_slot=True, eq=False, init=False)
class Expr:
  op: str  # a registered name: the ``ExprOp`` member for a builtin
  args: tuple[Expr, ...] = ()
  type: TensorType = field(default_factory=TensorType)
  name: str | None = None
  value: np.ndarray | None = None
  attrs: dict[str, Any] = field(default_factory=dict)
  lowering: Lowering = "auto"
  # Frozen → safe to cache. Populated lazily by structural_key on first call.
  _key_cache: tuple[Any, ...] | None = field(default=None, init=False, repr=False, compare=False)

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
    op: str | None = None,
    args: tuple["Expr", ...] = (),
    type: TensorType | None = None,
    name: str | None = None,
    value: np.ndarray | None = None,
    attrs: dict[str, Any] | None = None,
    lowering: Lowering = "auto",
  ) -> "Expr":
    if op is None:  # callers like ``copy.copy`` / pickling instantiate w/o args
      return object.__new__(cls)
    op_name = op_def(op).name
    type_eff = type if type is not None else TensorType()
    attrs_eff = attrs if attrs is not None else {}
    key = _intern_key(op_name, args, type_eff, name, value, attrs_eff, lowering)
    cached = _NODE_CACHE.get(key)
    if cached is not None:
      return cached  # as it was built: the arguments of this construction are dropped
    instance = object.__new__(cls)
    put = object.__setattr__
    put(instance, "op", op_name)
    put(instance, "args", args)
    put(instance, "type", type_eff)
    put(instance, "name", name)
    put(instance, "value", value)
    put(instance, "attrs", attrs_eff)
    put(instance, "lowering", lowering)
    put(instance, "_key_cache", None)
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
      str(self.op),
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
    """``other`` as an operand of ``+``, ``-`` or ``*`` beside this expression. A Python number takes
    its dtype: any number beside a floating expression, and an integer beside an integer one, so
    index arithmetic needs no casts. A float beside an integer expression stays a float64 constant,
    which the mixed-dtype check refuses rather than truncating it."""
    dtype = self.type.dtype
    if _is_number(other) and (dtype.is_floating or dtype.is_integer and not isinstance(other, (float, np.floating))):
      return Expr.const(other, dtype=dtype)
    return as_expr(other)

  def _float_operand(self, other: Any) -> Expr:
    """``other`` as an operand of ``/`` or ``**``: a Python number takes a floating expression's dtype.
    Beside an integer expression it stays float64 and is refused, since integer division is not what
    ``/`` means."""
    return Expr.const(other, dtype=self.type.dtype) if _is_number(other) and self.type.dtype.is_floating else as_expr(other)

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
    return binary(ExprOp.DIV, self, self._float_operand(other))

  def __rtruediv__(self, other: Any) -> Expr:
    return binary(ExprOp.DIV, self._float_operand(other), self)

  def __pow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, self, self._float_operand(other))

  def __rpow__(self, other: Any) -> Expr:
    return binary(ExprOp.POW, self._float_operand(other), self)

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
  """``attrs`` as a hashable key: equal keys mean attributes no generated code can tell apart."""

  def key(v: Any) -> Any:
    kind = type(v)
    if kind is int or kind is str or v is None:
      return v
    # ``==`` merges what C does not: ``-0.0`` with ``0.0``, and ``1`` with ``True`` and ``1.0``. A
    # float keys by its bits, as a Program IR constant does (``ir.program._attr_key``), which also
    # lets a NaN match itself; a bool and a float carry their kind, and an int is the bare value.
    if kind is float or isinstance(v, (float, np.floating)):
      return (float, struct.pack("<d", v))
    if kind is bool or isinstance(v, np.bool_):
      return (bool, bool(v))
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


def _is_number(x: Any) -> bool:
  return isinstance(x, (int, float, np.integer, np.floating)) and not isinstance(x, (bool, np.bool_))


def common_lowering(*exprs: Expr) -> Lowering:
  explicit = {e.lowering for e in exprs if e.lowering != "auto"}
  if len(explicit) == 1:
    return next(iter(explicit))
  if len(explicit) > 1:
    return "auto"
  return "auto"


def diff_any(*exprs: Expr) -> bool:
  return any(e.type.diff for e in exprs)


def op_diff(op: str, *exprs: Expr) -> bool:
  return op_def(op).differentiable and diff_any(*exprs)


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
  return binary(ExprOp.ATAN2, *_operands(y, x))


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
  return binary(ExprOp.COPYSIGN, *_operands(x, y))


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

  A cast between floating types keeps the derivative; any other cast has none. A float to integer
  cast truncates toward zero, as in C, and a NaN, an infinity or a value out of the integer type's
  range is undefined behaviour in the generated C: clamp with ``minimum`` and ``maximum`` first,
  which map NaN to the other operand and so leave a finite value.
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

  Repeated indices accumulate. ``indices`` must have as many entries as ``values``. The ``add``
  reduction of the ``segment_reduce`` op, as ``segment_sum`` is.
  """
  values = as_expr(values)
  shape = as_shape(shape)
  idx = _index_array(indices, int(np.prod(shape, dtype=int)))
  if idx.size != values.size:
    raise ValueError(f"scatter has {idx.size} indices but values shape {values.shape} has {values.size} entries")
  return _segment_reduce("add", values, idx, shape, 0.0)


SEGMENT_REDUCTIONS = ("add", "max", "min")
"""The reductions of the ``segment_reduce`` op: ``scatter`` and ``segment_sum`` add, ``segment_max``
and ``segment_min`` keep the extremum."""


def _segment_reduce(reduce: str, values: Expr, idx: np.ndarray, shape: tuple[int, ...], fill: float) -> Expr:
  """``out = full(shape, fill)``, then ``out.flat[idx.flat[k]] = reduce(out.flat[idx.flat[k]], values.flat[k])`` in order."""
  return Expr(
    ExprOp.SEGMENT_REDUCE,
    (values,),
    TensorType(shape, dtype=values.type.dtype, diff=values.type.diff),
    attrs={"reduce": reduce, "indices": idx, "fill": float(fill)},
    lowering=values.lowering,
  )


def index_add(base: Any, indices: Any, values: Any) -> Expr:
  """``base`` with ``values`` added at flat ``indices`` (repeated indices accumulate).

  ``put_add`` at constant indices, through the flat view of a base with more than one axis: the
  same value as ``base + scatter(values, indices, base.shape)``, kept as one update so that a loop
  whose carry is changed only this way can update the carry in place: see ``sc.scan``.
  """
  return _index_update("index_add", base, indices, values)


def index_set(base: Any, indices: Any, values: Any) -> Expr:
  """``base`` with the entries at flat ``indices`` replaced by ``values``; the indices must be distinct.

  ``put`` at constant indices, through the flat view of a base with more than one axis."""
  return _index_update("index_set", base, indices, values)


def _index_update(name: str, base: Any, indices: Any, values: Any) -> Expr:
  base, values = _operands(base, values)
  idx = _index_array(np.asarray(indices).reshape(-1), base.size)
  if idx.size != values.size:
    raise ValueError(f"{name} has {idx.size} indices for {values.size} values")
  if name == "index_set" and np.unique(idx).size != idx.size:
    raise ValueError("index_set indices must be distinct")
  promote_dtype(base, values)
  flat = base if len(base.shape) == 1 else base.reshape((base.size,))
  lanes = values if len(values.shape) == 1 else values.reshape((values.size,))
  out = (put_add if name == "index_add" else put)(flat, idx, lanes)
  return out if out.shape == base.shape else out.reshape(base.shape)


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


def put_lanes(expr: Expr) -> tuple[np.ndarray, np.ndarray] | None:
  """For a ``put`` or ``put_add`` at constant indices, the lanes whose values land, in lane order,
  and the entries of the last axis they land at; None when the indices are known only at run time.

  A lane whose index is outside ``[0, n)`` drops its value, and of the ``put`` lanes writing one
  entry only the last lands: a ``put``'s entries are distinct, a ``put_add``'s may repeat."""
  idx = expr.args[1]
  if idx.op != ExprOp.CONST or idx.value is None:
    return None
  known = np.asarray(idx.value, dtype=np.int64).reshape(-1)
  lanes = np.flatnonzero((known >= 0) & (known < expr.args[0].shape[-1]))
  if expr.op == ExprOp.PUT and lanes.size:
    _, last = np.unique(known[lanes][::-1], return_index=True)
    lanes = np.sort(lanes[::-1][last])
  return lanes, known[lanes]


def segment_sum(values: Any, segment_ids: Any, num_segments: int) -> Expr:
  """Sum ``values`` into ``num_segments`` bins: entry ``k`` adds into bin ``segment_ids[k]``.

  The ids are fixed when the graph is built, which is what lets code generation pick an
  implementation for this exact pattern. Empty bins are zero. The same as ``scatter`` into a
  vector, spelled the way sparse kernels read.
  """
  values = as_expr(values)
  return scatter(values.reshape((values.size,)), np.asarray(segment_ids).reshape(-1), (int(num_segments),))


def _segment_extremum(reduce: str, values: Any, segment_ids: Any, num_segments: int, fill: float | None) -> Expr:
  values = as_expr(values)
  ids = _index_array(np.asarray(segment_ids).reshape(-1), int(num_segments))
  if ids.size != values.size:
    raise ValueError(f"segment_{reduce} has {ids.size} segment ids for {values.size} values")
  if fill is None:
    fill = -math.inf if reduce == "max" else math.inf
  return _segment_reduce(reduce, values.reshape((values.size,)), ids, (int(num_segments),), fill)


def segment_max(values: Any, segment_ids: Any, num_segments: int, *, fill: float | None = None) -> Expr:
  """Largest value in each of ``num_segments`` bins; ``fill`` (default ``-inf``) where a bin is empty.

  NaN propagates within its bin. Ties follow ``sc.options(nonsmooth=...)``. The ``max`` reduction
  of the ``segment_reduce`` op.
  """
  return _segment_extremum("max", values, segment_ids, num_segments, fill)


def segment_min(values: Any, segment_ids: Any, num_segments: int, *, fill: float | None = None) -> Expr:
  """Smallest value in each of ``num_segments`` bins; ``fill`` (default ``inf``) where a bin is empty.

  The ``min`` reduction of the ``segment_reduce`` op."""
  return _segment_extremum("min", values, segment_ids, num_segments, fill)


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


def tangent_dtype(x: Expr) -> DType:
  """The dtype of ``x``'s tangents and cotangents: its own when floating, float64 otherwise (an
  integer or bool value has no derivative, and its zero tangent is float64 as it always was)."""
  return x.type.dtype if x.type.dtype.is_floating else dtypes.float64


def zeros_like(x: Expr) -> Expr:
  """Zeros shaped like ``x``, of its tangent dtype."""
  return Expr.const(np.zeros(x.shape), dtype=tangent_dtype(x), lowering=x.lowering)


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
      rhs = f"{e.op}({', '.join(f'%{loc[a.id]}' for a in e.args)})"
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
