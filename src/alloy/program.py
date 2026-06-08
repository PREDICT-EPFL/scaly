"""Program IR: explicit loops, buffers, loads/stores, calls, kernel launches.

Phase 4 of the roadmap. Program IR is the lower representation that looks like
code. Where ``Expr`` (semantic IR) preserves mathematical meaning, Program IR
chooses an implementation: which loops run, which buffer holds which value,
which kernel runs on which device.

Design choices:

- A single flat ``PNode`` class (frozen dataclass, hash-consed by op + args + attrs)
  with an ``op`` tag from ``POps``. Statement vs scalar-expression vs declaration
  is encoded by op tag, matching the tinygrad UOp style and keeping the
  PatternMatcher infrastructure from Phase 2 reusable.
- Address spaces and device placement live on ``BUFFER`` nodes via attrs
  (``address_space`` ∈ ``global`` / ``local`` / ``private`` / ``constant``;
  ``device`` is a ``DeviceSpec``).
- Range kinds borrow from tinygrad's ``AxisType``: ``SERIAL``, ``VECTOR``,
  ``GLOBAL``, ``THREAD``, ``LOCAL``, ``WARP``, ``REDUCE``, ``GROUP_REDUCE``,
  ``UNROLL``. The kind is a verifier-checked attribute of ``RANGE`` and ``FOR``,
  not a separate op; backend scheduling uses it to bind loops to launch axes.
- No execution / lowering yet (those are Phase 5+). Phase 4 ships the
  vocabulary, the verifier, and the pretty-printer.

Per the roadmap, Program IR is a real language: every construct must verify,
every malformed node must produce a clear diagnostic, and the pretty-printer
must show enough to debug schedule decisions.
"""

from __future__ import annotations

import weakref
from collections import defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from .types import DType, DeviceSpec, dtypes


class POps(StrEnum):
  """Program IR opcode set. Statement-shaped ops sit alongside scalar/index ops."""

  # Declarations
  PROGRAM = "program"
  PROC = "proc"
  KERNEL = "kernel"
  BUFFER = "buffer"
  VIEW = "view"
  PARAM = "param"

  # Statement-level
  BLOCK = "block"
  FOR = "for"
  RANGE = "range"
  ASSIGN = "assign"
  STORE = "store"
  CALL = "call"
  LAUNCH = "launch"
  BARRIER = "barrier"

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

  Backends use these to bind loops to launch axes (``GLOBAL``/``THREAD``),
  to choose between vectorized or unrolled emission (``VECTOR``/``UNROLL``),
  or to lower reductions (``REDUCE``/``GROUP_REDUCE``).
  """

  SERIAL = "serial"
  VECTOR = "vector"
  GLOBAL = "global"
  THREAD = "thread"
  LOCAL = "local"
  WARP = "warp"
  REDUCE = "reduce"
  GROUP_REDUCE = "group_reduce"
  UNROLL = "unroll"


ADDRESS_SPACES = frozenset({"global", "local", "private", "constant"})


SCALAR_OPS: frozenset[POps] = frozenset(
  {
    POps.CONST_INT,
    POps.CONST_FLOAT,
    POps.VAR,
    POps.LOAD,
    POps.ADD,
    POps.SUB,
    POps.MUL,
    POps.DIV,
    POps.MOD,
    POps.NEG,
    POps.SIN,
    POps.COS,
    POps.TAN,
    POps.ASIN,
    POps.ACOS,
    POps.ATAN,
    POps.SINH,
    POps.COSH,
    POps.TANH,
    POps.EXP,
    POps.LOG,
    POps.SQRT,
    POps.ABS,
    POps.FLOOR,
    POps.CEIL,
    POps.POW,
    POps.ATAN2,
    POps.MINIMUM,
    POps.MAXIMUM,
  }
)


# Scalar unary/binary POps that render as a C function call (libm), keyed for the
# pretty-printer and the C renderer. NEG/ADD/SUB/MUL/DIV/MOD render as operators.
UNARY_FN_OPS: frozenset[POps] = frozenset(
  {
    POps.SIN,
    POps.COS,
    POps.TAN,
    POps.ASIN,
    POps.ACOS,
    POps.ATAN,
    POps.SINH,
    POps.COSH,
    POps.TANH,
    POps.EXP,
    POps.LOG,
    POps.SQRT,
    POps.ABS,
    POps.FLOOR,
    POps.CEIL,
  }
)
BINARY_FN_OPS: frozenset[POps] = frozenset({POps.POW, POps.ATAN2, POps.MINIMUM, POps.MAXIMUM})


HOST_ONLY_OPS: frozenset[POps] = frozenset({POps.LAUNCH})
DEVICE_ONLY_OPS: frozenset[POps] = frozenset({POps.BARRIER})


# Hash-cons Program IR nodes the same way ``Expr`` is hash-consed: structurally-
# equal nodes collapse to the same Python object so identity == equality.
_PNODE_CACHE: weakref.WeakValueDictionary[tuple[Any, ...], "PNode"] = weakref.WeakValueDictionary()


def _attrs_key(attrs: dict[str, Any]) -> tuple[Any, ...]:
  out: list[tuple[str, Any]] = []
  for k, v in sorted(attrs.items()):
    if isinstance(v, dict):
      out.append((k, tuple(sorted(v.items()))))
    elif isinstance(v, list):
      out.append((k, tuple(v)))
    else:
      out.append((k, v))
  return tuple(out)


@dataclass(frozen=True, slots=True, weakref_slot=True)
class PNode:
  """A flat Program IR node.

  ``op`` selects the variant (statement-shaped, declaration-shaped, or scalar-
  shaped). ``args`` holds child nodes; ``attrs`` holds non-node metadata; the
  ``dtype`` field only matters for scalar-typed nodes (``CONST_*``, ``LOAD``,
  ``ADD``, ``MUL``, ...).
  """

  op: POps
  args: tuple["PNode", ...] = ()
  attrs: dict[str, Any] = field(default_factory=dict)
  dtype: DType = dtypes.float64
  _initialized: bool = field(default=False, init=False, repr=False, compare=False)

  def __new__(
    cls,
    op: POps,
    args: tuple["PNode", ...] = (),
    attrs: dict[str, Any] | None = None,
    dtype: DType = dtypes.float64,
  ) -> "PNode":
    attrs = dict(attrs) if attrs else {}
    key = (op.value, tuple(id(a) for a in args), _attrs_key(attrs), dtype.name)
    cached = _PNODE_CACHE.get(key)
    if cached is not None:
      return cached
    instance = super().__new__(cls)
    object.__setattr__(instance, "op", op)
    object.__setattr__(instance, "args", tuple(args))
    object.__setattr__(instance, "attrs", attrs)
    object.__setattr__(instance, "dtype", dtype)
    _PNODE_CACHE[key] = instance
    return instance

  def __post_init__(self) -> None:
    if self._initialized:
      return
    if not isinstance(self.op, POps):
      object.__setattr__(self, "op", POps(self.op))
    object.__setattr__(self, "_initialized", True)

  @property
  def id(self) -> int:
    return id(self)

  def __repr__(self) -> str:  # pragma: no cover - cosmetic
    attr_str = "" if not self.attrs else f" {self.attrs}"
    return f"PNode({self.op.value}, {len(self.args)} args{attr_str})"


# ---------------------------------------------------------------------------
# Builder helpers — the public way to construct Program IR.
# ---------------------------------------------------------------------------


def const_int(value: int) -> PNode:
  return PNode(POps.CONST_INT, (), attrs={"value": int(value)}, dtype=dtypes.int64)


def const_float(value: float, dtype: DType = dtypes.float64) -> PNode:
  return PNode(POps.CONST_FLOAT, (), attrs={"value": float(value)}, dtype=dtype)


def var(name: str, dtype: DType = dtypes.int64) -> PNode:
  return PNode(POps.VAR, (), attrs={"name": name}, dtype=dtype)


def buffer(name: str, dtype: DType, shape: tuple[int, ...], *, address_space: str = "global", device: DeviceSpec | str | None = None) -> PNode:
  if address_space not in ADDRESS_SPACES:
    raise ValueError(f"unknown address space {address_space!r}; expected one of {sorted(ADDRESS_SPACES)}")
  dev = DeviceSpec.parse(device)
  return PNode(
    POps.BUFFER,
    (),
    attrs={"name": name, "shape": tuple(int(d) for d in shape), "address_space": address_space, "device": dev},
    dtype=dtype,
  )


def const_buffer(name: str, dtype: DType, shape: tuple[int, ...], values: Sequence[float]) -> PNode:
  """A read-only ``constant``-address-space BUFFER carrying its initializer ``values``.

  The renderer emits it as a ``static const`` array; loads read it like any buffer.
  Lets constants of any size lower without inline-serializing each element.
  """
  return PNode(
    POps.BUFFER,
    (),
    attrs={
      "name": name,
      "shape": tuple(int(d) for d in shape),
      "address_space": "constant",
      "device": DeviceSpec.parse(None),
      "values": tuple(float(v) for v in values),
    },
    dtype=dtype,
  )


def view(buf: PNode, index: Sequence[PNode]) -> PNode:
  if buf.op != POps.BUFFER:
    raise TypeError(f"view requires a BUFFER, got {buf.op}")
  for n in index:
    if n.op not in SCALAR_OPS:
      raise TypeError(f"view index components must be scalar PNodes, got {n.op}")
  return PNode(POps.VIEW, tuple(index), attrs={"buffer": buf.attrs["name"], "rank": len(index)}, dtype=buf.dtype)


def load(view_node: PNode) -> PNode:
  if view_node.op != POps.VIEW:
    raise TypeError(f"load expects a VIEW, got {view_node.op}")
  return PNode(POps.LOAD, (view_node,), dtype=view_node.dtype)


def store(view_node: PNode, value: PNode) -> PNode:
  if view_node.op != POps.VIEW:
    raise TypeError(f"store expects a VIEW target, got {view_node.op}")
  if value.op not in SCALAR_OPS:
    raise TypeError(f"store value must be a scalar PNode, got {value.op}")
  return PNode(POps.STORE, (view_node, value), dtype=value.dtype)


def assign(target: str, value: PNode, dtype: DType | None = None) -> PNode:
  if value.op not in SCALAR_OPS:
    raise TypeError(f"assign value must be a scalar PNode, got {value.op}")
  return PNode(POps.ASSIGN, (value,), attrs={"target": target}, dtype=dtype or value.dtype)


def range_(name: str, start: PNode | int, stop: PNode | int, *, step: PNode | int = 1, kind: RangeKind = RangeKind.SERIAL) -> PNode:
  s = start if isinstance(start, PNode) else const_int(start)
  e = stop if isinstance(stop, PNode) else const_int(stop)
  st = step if isinstance(step, PNode) else const_int(step)
  return PNode(POps.RANGE, (s, e, st), attrs={"name": name, "kind": RangeKind(kind)}, dtype=dtypes.int64)


def for_(rng: PNode, body: Sequence[PNode]) -> PNode:
  if rng.op != POps.RANGE:
    raise TypeError(f"for_ requires a RANGE, got {rng.op}")
  return PNode(POps.FOR, (rng, *body), attrs={"body_len": len(body)})


def block(*statements: PNode) -> PNode:
  return PNode(POps.BLOCK, tuple(statements))


def call(callee: str, args: Sequence[PNode], *, returns: Sequence[str] = ()) -> PNode:
  return PNode(POps.CALL, tuple(args), attrs={"callee": callee, "returns": tuple(returns)})


def launch(kernel: str, grid: Sequence[PNode | int], block_dims: Sequence[PNode | int], args: Sequence[PNode]) -> PNode:
  gr = tuple(g if isinstance(g, PNode) else const_int(g) for g in grid)
  bl = tuple(b if isinstance(b, PNode) else const_int(b) for b in block_dims)
  return PNode(POps.LAUNCH, (*gr, *bl, *args), attrs={"kernel": kernel, "grid_dims": len(gr), "block_dims": len(bl)})


def barrier(kind: str = "device") -> PNode:
  if kind not in {"device", "group", "warp"}:
    raise ValueError(f"barrier kind {kind!r} not in device/group/warp")
  return PNode(POps.BARRIER, (), attrs={"kind": kind})


def proc(name: str, params: Sequence[PNode], body: Sequence[PNode], *, device: DeviceSpec | str | None = None) -> PNode:
  for p in params:
    if p.op != POps.BUFFER:
      raise TypeError(f"proc params must be BUFFER nodes, got {p.op}")
  return PNode(POps.PROC, (*params, *body), attrs={"name": name, "param_count": len(params), "device": DeviceSpec.parse(device)})


def kernel(name: str, params: Sequence[PNode], body: Sequence[PNode], *, grid_dims: int = 1, device: DeviceSpec | str | None = None) -> PNode:
  for p in params:
    if p.op != POps.BUFFER:
      raise TypeError(f"kernel params must be BUFFER nodes, got {p.op}")
  return PNode(
    POps.KERNEL,
    (*params, *body),
    attrs={"name": name, "param_count": len(params), "grid_dims": int(grid_dims), "device": DeviceSpec.parse(device)},
  )


def program(procs: Sequence[PNode], kernels: Sequence[PNode] = ()) -> PNode:
  for p in procs:
    if p.op != POps.PROC:
      raise TypeError(f"program procs must be PROC nodes, got {p.op}")
  for k in kernels:
    if k.op != POps.KERNEL:
      raise TypeError(f"program kernels must be KERNEL nodes, got {k.op}")
  return PNode(POps.PROGRAM, (*procs, *kernels), attrs={"proc_count": len(procs), "kernel_count": len(kernels)})


# Binary scalar helpers — used by the lowerer (Phase 5+) and by tests.


def _scalar_binop(op: POps, x: PNode, y: PNode) -> PNode:
  if x.op not in SCALAR_OPS or y.op not in SCALAR_OPS:
    raise TypeError(f"{op.value} requires scalar PNode operands, got {x.op} and {y.op}")
  if x.dtype != y.dtype:
    raise TypeError(f"{op.value} operand dtype mismatch: {x.dtype} vs {y.dtype}")
  return PNode(op, (x, y), dtype=x.dtype)


def add(x: PNode, y: PNode) -> PNode:
  return _scalar_binop(POps.ADD, x, y)


def sub(x: PNode, y: PNode) -> PNode:
  return _scalar_binop(POps.SUB, x, y)


def mul(x: PNode, y: PNode) -> PNode:
  return _scalar_binop(POps.MUL, x, y)


def div(x: PNode, y: PNode) -> PNode:
  return _scalar_binop(POps.DIV, x, y)


def mod(x: PNode, y: PNode) -> PNode:
  return _scalar_binop(POps.MOD, x, y)


def neg(x: PNode) -> PNode:
  if x.op not in SCALAR_OPS:
    raise TypeError(f"neg requires a scalar PNode, got {x.op}")
  return PNode(POps.NEG, (x,), dtype=x.dtype)


# ---------------------------------------------------------------------------
# Verifier: spec tables for Program IR.
# ---------------------------------------------------------------------------


class ProgramVerifyError(Exception):
  pass


@dataclass(frozen=True, slots=True)
class PRule:
  op: POps | None
  description: str
  check: Callable[[PNode], str | None]

  def applies(self, node: PNode) -> bool:
    return self.op is None or node.op == self.op


class PSpec:
  def __init__(self, rules: Iterable[PRule]) -> None:
    self.any: list[PRule] = []
    self.by_op: dict[POps, list[PRule]] = defaultdict(list)
    for r in rules:
      if r.op is None:
        self.any.append(r)
      else:
        self.by_op[r.op].append(r)

  def candidates(self, op: POps) -> Iterable[PRule]:
    yield from self.any
    yield from self.by_op.get(op, ())

  def check(self, node: PNode) -> tuple[PRule, str] | None:
    for rule in self.candidates(node.op):
      diag = rule.check(node)
      if diag is not None:
        return rule, diag
    return None


def _walk(root: PNode) -> Iterable[PNode]:
  seen: set[int] = set()
  stack: list[PNode] = [root]
  while stack:
    n = stack.pop()
    if id(n) in seen:
      continue
    seen.add(id(n))
    yield n
    stack.extend(n.args)


def verify_program(root: PNode, spec: "PSpec | None" = None) -> None:
  if spec is None:
    spec = spec_program_full
  for node in _walk(root):
    result = spec.check(node)
    if result is not None:
      rule, diag = result
      raise ProgramVerifyError(f"verify_program: node op={node.op.value} failed rule {rule.description!r}: {diag}")


# ---------- shared rules ----------


def _dtype_is_dtype(n: PNode) -> str | None:
  if not isinstance(n.dtype, DType):
    return f"dtype is {type(n.dtype).__name__}, expected DType"
  return None


def _const_int_has_value(n: PNode) -> str | None:
  if "value" not in n.attrs or not isinstance(n.attrs["value"], int):
    return "CONST_INT missing integer 'value' attr"
  if not n.dtype.is_integer:
    return f"CONST_INT must have integer dtype, got {n.dtype}"
  return None


def _const_float_has_value(n: PNode) -> str | None:
  if "value" not in n.attrs:
    return "CONST_FLOAT missing 'value' attr"
  if not n.dtype.is_floating:
    return f"CONST_FLOAT must have floating dtype, got {n.dtype}"
  return None


def _var_has_name(n: PNode) -> str | None:
  if "name" not in n.attrs:
    return "VAR missing 'name' attr"
  return None


def _buffer_attrs(n: PNode) -> str | None:
  for k in ("name", "shape", "address_space", "device"):
    if k not in n.attrs:
      return f"BUFFER missing {k!r} attr"
  if n.attrs["address_space"] not in ADDRESS_SPACES:
    return f"BUFFER address_space {n.attrs['address_space']!r} not in {sorted(ADDRESS_SPACES)}"
  if not isinstance(n.attrs["device"], DeviceSpec):
    return f"BUFFER 'device' must be DeviceSpec, got {type(n.attrs['device']).__name__}"
  return None


def _view_args_scalar(n: PNode) -> str | None:
  for a in n.args:
    if a.op not in SCALAR_OPS:
      return f"VIEW index component op={a.op} is not scalar"
  if "buffer" not in n.attrs:
    return "VIEW missing 'buffer' name attr"
  return None


def _load_takes_view(n: PNode) -> str | None:
  if len(n.args) != 1 or n.args[0].op != POps.VIEW:
    return "LOAD must wrap exactly one VIEW arg"
  return None


def _store_attrs(n: PNode) -> str | None:
  if len(n.args) != 2:
    return f"STORE expects 2 args (view, value), got {len(n.args)}"
  if n.args[0].op != POps.VIEW:
    return "STORE target must be VIEW"
  if n.args[1].op not in SCALAR_OPS:
    return f"STORE value op {n.args[1].op} not scalar"
  return None


def _range_kind(n: PNode) -> str | None:
  if "kind" not in n.attrs or not isinstance(n.attrs["kind"], RangeKind):
    return "RANGE missing valid 'kind' attr"
  if "name" not in n.attrs:
    return "RANGE missing 'name' attr"
  if len(n.args) != 3:
    return f"RANGE expects 3 args (start, stop, step), got {len(n.args)}"
  for arg, label in zip(n.args, ("start", "stop", "step"), strict=True):
    if arg.op not in SCALAR_OPS:
      return f"RANGE {label} op {arg.op} not scalar"
  return None


def _for_body(n: PNode) -> str | None:
  if not n.args or n.args[0].op != POps.RANGE:
    return "FOR first arg must be RANGE"
  return None


def _call_attrs(n: PNode) -> str | None:
  if "callee" not in n.attrs:
    return "CALL missing 'callee' name attr"
  return None


def _launch_attrs(n: PNode) -> str | None:
  for k in ("kernel", "grid_dims", "block_dims"):
    if k not in n.attrs:
      return f"LAUNCH missing {k!r} attr"
  return None


def _barrier_kind(n: PNode) -> str | None:
  if n.attrs.get("kind") not in {"device", "group", "warp"}:
    return f"BARRIER kind {n.attrs.get('kind')!r} not in device/group/warp"
  return None


def _proc_or_kernel_params(n: PNode) -> str | None:
  pc = n.attrs.get("param_count")
  if pc is None:
    return "PROC/KERNEL missing 'param_count'"
  for i in range(pc):
    if n.args[i].op != POps.BUFFER:
      return f"PROC/KERNEL param {i} op={n.args[i].op} is not BUFFER"
  return None


spec_program_shared = PSpec(
  [
    PRule(None, "dtype-is-DType", _dtype_is_dtype),
    PRule(POps.CONST_INT, "const-int-value", _const_int_has_value),
    PRule(POps.CONST_FLOAT, "const-float-value", _const_float_has_value),
    PRule(POps.VAR, "var-name", _var_has_name),
    PRule(POps.BUFFER, "buffer-attrs", _buffer_attrs),
    PRule(POps.VIEW, "view-scalar-args", _view_args_scalar),
    PRule(POps.LOAD, "load-takes-view", _load_takes_view),
    PRule(POps.STORE, "store-attrs", _store_attrs),
    PRule(POps.RANGE, "range-attrs", _range_kind),
    PRule(POps.FOR, "for-body", _for_body),
    PRule(POps.CALL, "call-attrs", _call_attrs),
    PRule(POps.LAUNCH, "launch-attrs", _launch_attrs),
    PRule(POps.BARRIER, "barrier-kind", _barrier_kind),
    PRule(POps.PROC, "proc-params", _proc_or_kernel_params),
    PRule(POps.KERNEL, "kernel-params", _proc_or_kernel_params),
  ]
)


def _host_proc_no_device_only(n: PNode) -> str | None:
  for sub in _walk(n):
    if sub is n:
      continue
    if sub.op in DEVICE_ONLY_OPS:
      return f"host PROC contains device-only op {sub.op.value}"
  return None


def _kernel_no_host_only(n: PNode) -> str | None:
  for sub in _walk(n):
    if sub is n:
      continue
    if sub.op in HOST_ONLY_OPS:
      return f"KERNEL contains host-only op {sub.op.value}"
  return None


spec_host_program = PSpec(
  [
    *spec_program_shared.any,
    *(r for rs in spec_program_shared.by_op.values() for r in rs),
    PRule(POps.PROC, "host-proc-no-device-only", _host_proc_no_device_only),
  ]
)


spec_kernel_program = PSpec(
  [
    *spec_program_shared.any,
    *(r for rs in spec_program_shared.by_op.values() for r in rs),
    PRule(POps.KERNEL, "kernel-no-host-only", _kernel_no_host_only),
  ]
)


# Full spec: shared + host + kernel checks together. Suitable for whole-program verify.
spec_program_full = PSpec(
  [
    *spec_program_shared.any,
    *(r for rs in spec_program_shared.by_op.values() for r in rs),
    PRule(POps.PROC, "host-proc-no-device-only", _host_proc_no_device_only),
    PRule(POps.KERNEL, "kernel-no-host-only", _kernel_no_host_only),
  ]
)


# ---------------------------------------------------------------------------
# Pretty printer: backend-neutral debug dump, decoupled from C syntax.
# ---------------------------------------------------------------------------


def format_program(root: PNode, indent: int = 0) -> str:
  lines: list[str] = []
  _format_node(root, indent, lines)
  return "\n".join(lines)


def _format_node(n: PNode, indent: int, lines: list[str]) -> None:
  pad = "  " * indent
  if n.op == POps.PROGRAM:
    lines.append(f"{pad}program")
    pc = n.attrs.get("proc_count", 0)
    for sub in n.args[:pc]:
      _format_node(sub, indent + 1, lines)
    for sub in n.args[pc:]:
      _format_node(sub, indent + 1, lines)
  elif n.op in (POps.PROC, POps.KERNEL):
    label = "proc" if n.op == POps.PROC else "kernel"
    params = ", ".join(f"{p.attrs['name']}:{p.dtype.name}{p.attrs['shape']}" for p in n.args[: n.attrs["param_count"]])
    device = n.attrs.get("device")
    suffix = f" device={device}" if device and str(device) != "host" else ""
    lines.append(f"{pad}{label} {n.attrs['name']}({params}){suffix}:")
    for sub in n.args[n.attrs["param_count"] :]:
      _format_node(sub, indent + 1, lines)
  elif n.op == POps.BLOCK:
    for sub in n.args:
      _format_node(sub, indent, lines)
  elif n.op == POps.FOR:
    rng = n.args[0]
    kind = rng.attrs["kind"].value
    start = _format_scalar(rng.args[0])
    stop = _format_scalar(rng.args[1])
    step = _format_scalar(rng.args[2])
    lines.append(f"{pad}for {rng.attrs['name']} in [{start}, {stop}) step {step} kind={kind}:")
    for sub in n.args[1:]:
      _format_node(sub, indent + 1, lines)
  elif n.op == POps.STORE:
    target = _format_view(n.args[0])
    value = _format_scalar(n.args[1])
    lines.append(f"{pad}{target} <- {value}")
  elif n.op == POps.ASSIGN:
    lines.append(f"{pad}{n.attrs['target']} = {_format_scalar(n.args[0])}")
  elif n.op == POps.CALL:
    args = ", ".join(_format_scalar_or_view(a) for a in n.args)
    rets = n.attrs.get("returns", ())
    ret_prefix = f"{', '.join(rets)} = " if rets else ""
    lines.append(f"{pad}{ret_prefix}call {n.attrs['callee']}({args})")
  elif n.op == POps.LAUNCH:
    gd = int(n.attrs["grid_dims"])
    bd = int(n.attrs["block_dims"])
    grid = ", ".join(_format_scalar(a) for a in n.args[:gd])
    block_dims = ", ".join(_format_scalar(a) for a in n.args[gd : gd + bd])
    kargs = ", ".join(_format_scalar_or_view(a) for a in n.args[gd + bd :])
    lines.append(f"{pad}launch {n.attrs['kernel']}<<<({grid}), ({block_dims})>>>({kargs})")
  elif n.op == POps.BARRIER:
    lines.append(f"{pad}barrier {n.attrs['kind']}")
  else:
    # fallback for unknown / scalar at statement scope
    lines.append(f"{pad}{n.op.value}")


def _format_view(v: PNode) -> str:
  if v.op != POps.VIEW:
    return v.op.value
  idx = ", ".join(_format_scalar(a) for a in v.args)
  return f"{v.attrs['buffer']}[{idx}]"


def _format_scalar(n: PNode) -> str:
  if n.op == POps.CONST_INT:
    return str(n.attrs["value"])
  if n.op == POps.CONST_FLOAT:
    return f"{n.attrs['value']:g}"
  if n.op == POps.VAR:
    return str(n.attrs["name"])
  if n.op == POps.LOAD:
    return _format_view(n.args[0])
  if n.op in (POps.ADD, POps.SUB, POps.MUL, POps.DIV, POps.MOD):
    sym = {POps.ADD: "+", POps.SUB: "-", POps.MUL: "*", POps.DIV: "/", POps.MOD: "%"}[n.op]
    return f"({_format_scalar(n.args[0])} {sym} {_format_scalar(n.args[1])})"
  if n.op == POps.NEG:
    return f"(-{_format_scalar(n.args[0])})"
  if n.op in UNARY_FN_OPS:
    return f"{n.op.value}({_format_scalar(n.args[0])})"
  if n.op in BINARY_FN_OPS:
    return f"{n.op.value}({_format_scalar(n.args[0])}, {_format_scalar(n.args[1])})"
  return f"<{n.op.value}>"


def _format_scalar_or_view(n: PNode) -> str:
  if n.op == POps.VIEW:
    return _format_view(n)
  return _format_scalar(n)


__all__ = [
  "ADDRESS_SPACES",
  "BINARY_FN_OPS",
  "DEVICE_ONLY_OPS",
  "HOST_ONLY_OPS",
  "UNARY_FN_OPS",
  "PNode",
  "POps",
  "PRule",
  "PSpec",
  "ProgramVerifyError",
  "RangeKind",
  "SCALAR_OPS",
  "add",
  "assign",
  "barrier",
  "block",
  "buffer",
  "call",
  "const_buffer",
  "const_float",
  "const_int",
  "div",
  "for_",
  "format_program",
  "kernel",
  "launch",
  "load",
  "mod",
  "mul",
  "neg",
  "proc",
  "program",
  "range_",
  "spec_host_program",
  "spec_kernel_program",
  "spec_program_full",
  "spec_program_shared",
  "store",
  "sub",
  "var",
  "verify_program",
  "view",
]
