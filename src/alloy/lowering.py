"""Lower semantic IR (``Expr`` / ``Function``) into Program IR (``PNode``).

This is the migration's heart (see ``docs/program_ir_migration.md``). The goal is
for this module + ``codegen/program_c.py`` to become the *sole* path to C, with
no silent fallback to the legacy tape renderer.

Dispatch is a **registry** keyed by semantic ``Ops``: each op's lowering is a
self-contained rule registered with ``@lowers(...)``. Adding/deepening an op (or,
later, a GPU schedule) is a local change — a new rule, not an edit to a monolith.

Covered so far: elementwise unary/binary (with numpy broadcasting), ``RESHAPE``
(alias), ``CONST`` (any size, via ``const_buffer``), general ``SLICE`` (integer /
multi-dim / strided), ``SUM``, ``MATMUL`` (rank <= 2), ``TRANSPOSE`` (rank <= 4),
``GATHER``/``SCATTER`` (any size, ``static const`` index table), ``STACK``/``CONCAT``
(axis 0), ``CALL`` (multi-PROC, deduped) and ``MAP``. The forward tracking and
unbumpercars workloads render and match the interpreter. Remaining: non-axis-0
STACK/CONCAT, workspace packing, then GPU placement and the deferred new ops — all
tracked in the migration roadmap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from . import program as p
from .expr import Expr, topo
from .function import Function
from .ops import Ops
from .passes import optimize_program
from .program import PNode, POps, RangeKind, verify_program
from .types import DeviceSpec, DType, dtypes


class LoweringError(NotImplementedError):
  """The lowerer (or Program-IR renderer) does not yet cover this op / case."""


# Semantic Ops -> Program IR scalar POps. The op vocabulary grows here as ops migrate.
_UNARY: dict[Ops, POps] = {
  Ops.NEG: POps.NEG,
  Ops.SIN: POps.SIN,
  Ops.COS: POps.COS,
  Ops.TAN: POps.TAN,
  Ops.ASIN: POps.ASIN,
  Ops.ACOS: POps.ACOS,
  Ops.ATAN: POps.ATAN,
  Ops.SINH: POps.SINH,
  Ops.COSH: POps.COSH,
  Ops.TANH: POps.TANH,
  Ops.EXP: POps.EXP,
  Ops.LOG: POps.LOG,
  Ops.SQRT: POps.SQRT,
  Ops.ABS: POps.ABS,
  Ops.FLOOR: POps.FLOOR,
  Ops.CEIL: POps.CEIL,
}

_BINARY: dict[Ops, POps] = {
  Ops.ADD: POps.ADD,
  Ops.SUB: POps.SUB,
  Ops.MUL: POps.MUL,
  Ops.DIV: POps.DIV,
  Ops.POW: POps.POW,
  Ops.ATAN2: POps.ATAN2,
  Ops.MINIMUM: POps.MINIMUM,
  Ops.MAXIMUM: POps.MAXIMUM,
}

LowerRule = Callable[["LowerCtx", Expr], None]
_RULES: dict[Ops, LowerRule] = {}


def lowers(*ops: Ops) -> Callable[[LowerRule], LowerRule]:
  """Register ``fn`` as the lowering rule for each semantic op in ``ops``."""

  def deco(fn: LowerRule) -> LowerRule:
    for op in ops:
      _RULES[op] = fn
    return fn

  return deco


def lower_function(fun: Function) -> PNode:
  """Lower ``fun`` into a Program IR ``PROGRAM`` node (verified before return).

  Host placement only for now: the returned PROGRAM holds every lowered callee
  PROC in topological order followed by ``fun``'s main PROC last. Non-host
  placement raises ``LoweringError`` — GPU backends re-land from the reference
  branch after CPU parity (see ``docs/program_ir_migration.md``).
  """
  if fun.device.kind != "host":
    raise LoweringError(f"non-host placement {fun.device} is not lowered yet (GPU backends are deferred to a later migration step)")
  callees: dict[str, PNode] = {}
  root = _lower_to_proc(fun, callees)
  prog = p.program([*callees.values(), root])
  prog = optimize_program(prog)  # fusion + workspace packing (see passes.py)
  verify_program(prog)
  return prog


def main_proc(program_node: PNode) -> PNode:
  """Return the main (last) PROC inside a lowered PROGRAM."""
  if program_node.op != POps.PROGRAM:
    raise TypeError(f"main_proc expects a PROGRAM, got {program_node.op}")
  pc = int(program_node.attrs.get("proc_count", 0))
  if pc <= 0:
    raise ValueError("lowered program contains no procs")
  return program_node.args[pc - 1]


def _shape_or_scalar(shape: tuple[int, ...]) -> tuple[int, ...]:
  return shape or (1,)


def _size_of(shape: tuple[int, ...]) -> int:
  n = 1
  for d in shape:
    n *= int(d)
  return n or 1


def _lower_to_proc(fun: Function, callees: dict[str, PNode]) -> PNode:
  ctx = LowerCtx(fun, callees)
  ctx.emit_inputs()
  ctx.register_outputs()
  ctx.emit_body()
  ctx.emit_outputs()
  return p.proc(fun.name, ctx.params, ctx.statements)


class LowerCtx:
  """Per-Function lowering state: buffers, statements, and the Expr-id -> buffer map."""

  def __init__(self, fun: Function, callees: dict[str, PNode]) -> None:
    self.fun = fun
    self.callees = callees
    self.params: list[PNode] = []
    self.statements: list[PNode] = []
    self.buffers: dict[str, PNode] = {}
    # Expr.id -> name of the buffer holding that value at runtime.
    self.value_buffers: dict[int, str] = {}
    # Output Expr.id -> output buffer name, so the body writes outputs in place.
    self._output_alias: dict[int, str] = {}
    # (callee_name, arg_buffer_names) -> output buffer names, to dedup repeated CALL invocations.
    self.call_invocations: dict[tuple[str, tuple[str, ...]], tuple[str, ...]] = {}
    self._tmp = 0

  # --- declarations ---------------------------------------------------------

  def emit_inputs(self) -> None:
    for name, expr in zip(self.fun.input_names, self.fun.inputs, strict=True):
      buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      self.value_buffers[expr.id] = name

  def register_outputs(self) -> None:
    """Register output BUFFER params and alias each unique computed output Expr to
    its output buffer, so its rule writes directly into the output (no copy)."""
    seen: set[int] = set()
    for name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      if expr.op in (Ops.INPUT, Ops.CONST) or expr.id in seen:
        continue  # INPUT/CONST or shared output: emit_outputs inserts the copy
      seen.add(expr.id)
      self._output_alias[expr.id] = name

  def emit_outputs(self) -> None:
    for name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      src = self.value_buffers.get(expr.id)
      if src is None:
        raise LoweringError(f"output {name!r} expression was not lowered")
      if src == name:
        continue  # already written in place via the output alias
      self.statements.append(_copy_loop(self.buffers[src], self.buffers[name], expr.shape))

  # --- body -----------------------------------------------------------------

  def emit_body(self) -> None:
    for node in topo(self.fun.outputs):
      if node.id in self.value_buffers:
        continue  # input (or already lowered)
      rule = _RULES.get(Ops(node.op))
      if rule is None:
        raise LoweringError(f"semantic op {node.op!r} is not yet lowered to Program IR")
      rule(self, node)

  # --- helpers --------------------------------------------------------------

  def buf_of(self, expr: Expr) -> PNode:
    return self.buffers[self.value_buffers[expr.id]]

  def new_private(self, dtype: DType, shape: tuple[int, ...]) -> PNode:
    """Allocate a fresh private scratch BUFFER (declared as a local array by the renderer)."""
    name = f"t{self._tmp}"
    self._tmp += 1
    buf = p.buffer(name, dtype, _shape_or_scalar(shape), address_space="private")
    self.buffers[name] = buf
    self.statements.append(buf)  # marks the local-array declaration for the renderer
    return buf

  def alloc_tmp(self, expr: Expr) -> PNode:
    """Buffer to hold ``expr``'s value: the output buffer if aliased, else a fresh private temp."""
    alias = self._output_alias.get(expr.id)
    if alias is not None:
      self.value_buffers[expr.id] = alias
      return self.buffers[alias]
    buf = self.new_private(expr.type.dtype, expr.shape)
    self.value_buffers[expr.id] = buf.attrs["name"]
    return buf

  def new_alias(self, dtype: DType, shape: tuple[int, ...], src_name: str, offset: int) -> PNode:
    """A zero-copy private BUFFER that aliases ``src_name`` at a flat ``offset`` (rendered
    ``const T* tN = <src> + offset;``). Carries ``alias_of`` / ``alias_offset`` so the workspace
    pass leaves it unpacked and keeps its source live. Port of ``codegen/c.py``'s contiguous
    SLICE / RESHAPE pointer aliasing."""
    name = f"t{self._tmp}"
    self._tmp += 1
    buf = PNode(
      POps.BUFFER,
      (),
      attrs={
        "name": name,
        "shape": _shape_or_scalar(shape),
        "address_space": "private",
        "device": DeviceSpec.parse(None),
        "alias_of": src_name,
        "alias_offset": int(offset),
      },
      dtype=dtype,
    )
    self.buffers[name] = buf
    self.statements.append(buf)
    return buf

  def new_const_index(self, idx: Iterable[int]) -> PNode:
    """A read-only int64 index table (for GATHER/SCATTER), declared ``static const``."""
    values = [int(v) for v in idx]
    buf = p.const_buffer(f"k{self._tmp}", dtypes.int64, (len(values),), values)
    self._tmp += 1
    self.buffers[buf.attrs["name"]] = buf
    self.statements.append(buf)
    return buf

  def emit_elementwise(self, node: Expr, pop: POps, *, arity: int) -> None:
    out = self.alloc_tmp(node)
    vname = f"i_{out.attrs['name']}"
    rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
    i = p.var(vname)
    loads = tuple(p.load(p.view(self.buf_of(a), [_broadcast_index_p(i, a.shape, node.shape)])) for a in node.args[:arity])
    computed = PNode(pop, loads, dtype=node.type.dtype)
    self.statements.append(p.for_(rng, [p.store(p.view(out, [i]), computed)]))


def _stride(shape: tuple[int, ...], dim: int) -> int:
  s = 1
  for d in shape[dim + 1 :]:
    s *= int(d)
  return s


def _coord_p(flat: PNode, shape: tuple[int, ...], dim: int) -> PNode:
  """Decompose flat output index ``flat`` into the ``dim``-th coordinate of ``shape``."""
  stride = _stride(shape, dim)
  v = flat if stride == 1 else p.div(flat, p.const_int(stride))
  # The leading dim needs no modulo: flat < size guarantees (flat // stride) < shape[0].
  return v if dim == 0 else p.mod(v, p.const_int(int(shape[dim])))


def _flat_index_p(coords: list[PNode], shape: tuple[int, ...]) -> PNode:
  terms = [c if (st := _stride(shape, i)) == 1 else p.mul(c, p.const_int(st)) for i, c in enumerate(coords)]
  if not terms:
    return p.const_int(0)
  acc = terms[0]
  for t in terms[1:]:
    acc = p.add(acc, t)
  return acc


def _affine_sum(vars_: list[PNode], coeffs: list[int]) -> PNode:
  """Build ``Σ coeffs[i] * vars_[i]`` as a PNode, dropping zero coeffs and unit multiplies."""
  acc: PNode | None = None
  for v, c in zip(vars_, coeffs, strict=True):
    if c == 0:
      continue
    term = v if c == 1 else p.mul(v, p.const_int(c))
    acc = term if acc is None else p.add(acc, term)
  return acc if acc is not None else p.const_int(0)


def _row_major_strides(shape: tuple[int, ...]) -> list[int]:
  strides = [1] * len(shape)
  for i in range(len(shape) - 2, -1, -1):
    strides[i] = strides[i + 1] * int(shape[i + 1])
  return strides


def _broadcast_index_p(flat: PNode, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> PNode:
  """Map an output flat index to the source flat index under numpy broadcasting
  (right-aligned; size-1 dims and missing leading dims read index 0)."""
  if in_shape == out_shape or not out_shape:
    return flat
  if not in_shape:
    return p.const_int(0)
  offset = len(out_shape) - len(in_shape)
  coords = [p.const_int(0) if d == 1 else _coord_p(flat, out_shape, offset + i) for i, d in enumerate(in_shape)]
  return _flat_index_p(coords, in_shape)


def _copy_loop(src: PNode, dst: PNode, shape: tuple[int, ...]) -> PNode:
  vname = f"c_{dst.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(shape), kind=RangeKind.GLOBAL)
  i = p.var(vname)
  return p.for_(rng, [p.store(p.view(dst, [i]), p.load(p.view(src, [i])))])


# ---------------------------------------------------------------------------
# Lowering rules.
# ---------------------------------------------------------------------------


@lowers(*_UNARY)
def _lower_unary(ctx: LowerCtx, node: Expr) -> None:
  ctx.emit_elementwise(node, _UNARY[Ops(node.op)], arity=1)


@lowers(*_BINARY)
def _lower_binary(ctx: LowerCtx, node: Expr) -> None:
  ctx.emit_elementwise(node, _BINARY[Ops(node.op)], arity=2)


@lowers(Ops.RESHAPE)
def _lower_reshape(ctx: LowerCtx, node: Expr) -> None:
  # Metadata-only: the result aliases its source buffer (no copy).
  ctx.value_buffers[node.id] = ctx.value_buffers[node.args[0].id]


@lowers(Ops.CONST)
def _lower_const(ctx: LowerCtx, node: Expr) -> None:
  value = node.value
  assert value is not None
  # A constant of any size materializes as a read-only ``constant``-space buffer
  # (rendered ``static const``). Output-aliasing never applies to CONST, so
  # emit_outputs inserts a copy when a CONST is itself an output.
  name = f"k{ctx._tmp}"
  ctx._tmp += 1
  buf = p.const_buffer(name, node.type.dtype, _shape_or_scalar(node.shape), [float(v) for v in value.reshape(-1)])
  ctx.buffers[name] = buf
  ctx.value_buffers[node.id] = name
  ctx.statements.append(buf)


def _contiguous_slice_offset(index: tuple[object, ...], in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> int | None:
  """If the slice selects a contiguous sub-block of ``src`` at a constant flat offset (no stride,
  at most one partial leading slice with full trailing dims), return that offset; else None. Port
  of ``codegen/c.py::_contiguous_slice_offset`` — the precondition for pointer aliasing."""
  if not out_shape:
    offset = 0
    for dim, item in enumerate(index):
      if not isinstance(item, int):
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * _stride(in_shape, dim)
    return offset
  first_slice = None
  offset = 0
  out_dim = 0
  for dim, item in enumerate(index):
    stride = _stride(in_shape, dim)
    if isinstance(item, int):
      if first_slice is not None and in_shape[dim] != 1:
        return None
      offset += (item if item >= 0 else in_shape[dim] + item) * stride
      continue
    assert isinstance(item, slice)
    start, stop, step = item.indices(int(in_shape[dim]))
    if step != 1:
      return None
    if first_slice is None:
      first_slice = dim
      offset += start * stride
      out_dim += 1
      continue
    if start != 0 or stop != int(in_shape[dim]) or out_shape[out_dim] != int(in_shape[dim]):
      return None
    out_dim += 1
  return offset


@lowers(Ops.SLICE)
def _lower_slice(ctx: LowerCtx, node: Expr) -> None:
  """General SLICE: integer indices drop a dim, slices keep one. A contiguous slice (constant
  flat offset, no stride) becomes a zero-copy pointer alias of its source; otherwise each output
  element reads the source via flat-index arithmetic. Covers rank-1, multi-dim, integer, strided."""
  src = node.args[0]
  src_shape = src.shape
  index = node.attrs["index"]
  # Contiguous + not an output: alias the source pointer instead of copying (legacy parity).
  if node.id not in ctx._output_alias and (offset := _contiguous_slice_offset(index, src_shape, node.shape)) is not None:
    alias = ctx.new_alias(node.type.dtype, node.shape, ctx.value_buffers[src.id], offset)
    ctx.value_buffers[node.id] = alias.attrs["name"]
    return
  out = ctx.alloc_tmp(node)
  vname = f"i_{out.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  k = p.var(vname)
  coords: list[PNode] = []
  out_dim = 0
  for dim, item in enumerate(index):
    if isinstance(item, int):
      coords.append(p.const_int(item if item >= 0 else int(src_shape[dim]) + item))
      continue
    start, _stop, step = item.indices(int(src_shape[dim]))
    c = _coord_p(k, node.shape, out_dim)
    if step != 1:
      c = p.mul(c, p.const_int(step))
    coords.append(c if (start == 0 and step == 1) else p.add(p.const_int(start), c))
    out_dim += 1
  src_idx = _flat_index_p(coords, src_shape)
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


@lowers(Ops.SUM)
def _lower_sum(ctx: LowerCtx, node: Expr) -> None:
  """Full reduction to a scalar: zero the accumulator, then a REDUCE loop adds every element."""
  src = node.args[0]
  acc = ctx.alloc_tmp(node)
  z = p.const_int(0)
  ctx.statements.append(p.store(p.view(acc, [z]), p.const_float(0.0, dtype=node.type.dtype)))
  name = f"i_{acc.attrs['name']}"
  rng = p.range_(name, 0, _size_of(src.shape), kind=RangeKind.REDUCE)
  i = p.var(name)
  ctx.statements.append(p.for_(rng, [p.store(p.view(acc, [z]), p.add(p.load(p.view(acc, [z])), p.load(p.view(ctx.buf_of(src), [i]))))]))


@lowers(Ops.TRANSPOSE)
def _lower_transpose(ctx: LowerCtx, node: Expr) -> None:
  """Permuted copy: ``out[Σ o_i·out_stride_i] = src[Σ o_i·src_stride_{axes[i]}]``, one loop per output axis."""
  src = node.args[0]
  axes = tuple(int(a) for a in node.attrs["axes"])
  src_shape, out_shape = src.shape, node.shape
  if len(src_shape) > 4:
    raise LoweringError(f"TRANSPOSE lowering handles rank <= 4; got {src_shape}")
  out = ctx.alloc_tmp(node)
  src_strides = _row_major_strides(src_shape)
  out_strides = _row_major_strides(out_shape)
  ranges, loop_vars = [], []
  for i, d in enumerate(out_shape):
    name = f"d{i}_{out.attrs['name']}"
    ranges.append(p.range_(name, 0, int(d), kind=RangeKind.GLOBAL))
    loop_vars.append(p.var(name))
  out_idx = _affine_sum(loop_vars, out_strides)
  src_idx = _affine_sum(loop_vars, [src_strides[axes[i]] for i in range(len(out_shape))])
  stmt: PNode = p.store(p.view(out, [out_idx]), p.load(p.view(ctx.buf_of(src), [src_idx])))
  for rng in reversed(ranges):
    stmt = p.for_(rng, [stmt])
  ctx.statements.append(stmt)


def _mm_accumulate(ctx: LowerCtx, out: PNode, out_idx: PNode, a_load: PNode, b_load: PNode, dtype: DType, outer: list[PNode], k_rng: PNode) -> None:
  """Common matmul shape: zero out[out_idx], then a REDUCE-k loop adds a*b, nested in ``outer`` loops."""
  init = p.store(p.view(out, [out_idx]), p.const_float(0.0, dtype=dtype))
  accum = p.store(p.view(out, [out_idx]), p.add(p.load(p.view(out, [out_idx])), p.mul(a_load, b_load)))
  body = [init, p.for_(k_rng, [accum])]
  if not outer:
    ctx.statements.extend(body)
    return
  stmt = p.for_(outer[-1], body)
  for rng in reversed(outer[:-1]):
    stmt = p.for_(rng, [stmt])
  ctx.statements.append(stmt)


@lowers(Ops.MATMUL)
def _lower_matmul(ctx: LowerCtx, node: Expr) -> None:
  a, b = node.args
  a_buf, b_buf, out = ctx.buf_of(a), ctx.buf_of(b), ctx.alloc_tmp(node)
  dt = node.type.dtype
  sa, sb = a.shape, b.shape
  nm = out.attrs["name"]
  if len(sa) == 1 and len(sb) == 1:  # dot
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, sa[0], kind=RangeKind.REDUCE)
    _mm_accumulate(ctx, out, p.const_int(0), p.load(p.view(a_buf, [k])), p.load(p.view(b_buf, [k])), dt, [], krng)
  elif len(sa) == 2 and len(sb) == 1:  # mat @ vec
    m, kk = sa
    i = p.var(f"i_{nm}")
    irng = p.range_(f"i_{nm}", 0, m, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    a_idx = p.add(p.mul(i, p.const_int(kk)), k)
    _mm_accumulate(ctx, out, i, p.load(p.view(a_buf, [a_idx])), p.load(p.view(b_buf, [k])), dt, [irng], krng)
  elif len(sa) == 1 and len(sb) == 2:  # vec @ mat
    kk, n = sb
    j = p.var(f"j_{nm}")
    jrng = p.range_(f"j_{nm}", 0, n, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    b_idx = p.add(p.mul(k, p.const_int(n)), j)
    _mm_accumulate(ctx, out, j, p.load(p.view(a_buf, [k])), p.load(p.view(b_buf, [b_idx])), dt, [jrng], krng)
  elif len(sa) == 2 and len(sb) == 2:  # mat @ mat
    m, kk = sa
    n = sb[1]
    i = p.var(f"i_{nm}")
    irng = p.range_(f"i_{nm}", 0, m, kind=RangeKind.GLOBAL)
    j = p.var(f"j_{nm}")
    jrng = p.range_(f"j_{nm}", 0, n, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    out_idx = p.add(p.mul(i, p.const_int(n)), j)
    a_idx = p.add(p.mul(i, p.const_int(kk)), k)
    b_idx = p.add(p.mul(k, p.const_int(n)), j)
    _mm_accumulate(ctx, out, out_idx, p.load(p.view(a_buf, [a_idx])), p.load(p.view(b_buf, [b_idx])), dt, [irng, jrng], krng)
  else:
    raise LoweringError(f"matmul shapes {sa}@{sb} not lowered (batched / higher-rank deferred)")


def _ensure_callee(ctx: LowerCtx, callee: Function) -> None:
  if callee.device.kind != ctx.fun.device.kind:
    raise LoweringError(f"mixed-device CALL ({ctx.fun.device} -> {callee.device}) is deferred to a later migration step")
  if callee.name not in ctx.callees:
    ctx.callees[callee.name] = _lower_to_proc(callee, ctx.callees)


@lowers(Ops.CALL)
def _lower_call(ctx: LowerCtx, node: Expr) -> None:
  """A semantic CALL output: emit one Program-IR CALL writing all callee outputs into scratch
  buffers (deduped per unique invocation), then map this node to the selected output buffer."""
  callee: Function = node.attrs["callee"]
  out_idx = int(node.attrs["output"])
  arg_names = tuple(ctx.value_buffers[a.id] for a in node.args)
  key = (callee.name, arg_names)
  if key not in ctx.call_invocations:
    _ensure_callee(ctx, callee)
    out_bufs = [ctx.new_private(o.type.dtype, o.shape) for o in callee.outputs]
    in_bufs = [ctx.buffers[n] for n in arg_names]
    ctx.statements.append(
      PNode(POps.CALL, tuple(in_bufs + out_bufs), attrs={"callee": callee.name, "n_in": len(in_bufs), "n_out": len(out_bufs), "returns": ()})
    )
    ctx.call_invocations[key] = tuple(b.attrs["name"] for b in out_bufs)
  ctx.value_buffers[node.id] = ctx.call_invocations[key][out_idx]


@lowers(Ops.MAP)
def _lower_map(ctx: LowerCtx, node: Expr) -> None:
  """A ``length``-iteration loop calling the callee with pointer-offset VIEW args. Iteration ``it``
  reads ``outer_k[start_k + it·stride_k ...]`` and writes the selected output into ``out[it·slice_size ...]``."""
  callee: Function = node.attrs["callee"]
  out_idx = int(node.attrs["output"])
  length = int(node.attrs["length"])
  starts = tuple(int(s) for s in node.attrs["starts"])
  strides = tuple(int(s) for s in node.attrs["strides"])
  slice_size = int(node.attrs["slice_size"])
  _ensure_callee(ctx, callee)
  out = ctx.alloc_tmp(node)
  # Other callee outputs are written every iteration but discarded: one reused scratch each.
  scratch = [out if i == out_idx else ctx.new_private(o.type.dtype, o.shape) for i, o in enumerate(callee.outputs)]
  if length == 0:
    return
  loop = f"it_{out.attrs['name']}"
  rng = p.range_(loop, 0, length, kind=RangeKind.GLOBAL)
  it = p.var(loop)
  in_args = []
  for k, outer in enumerate(node.args):
    off = p.add(p.const_int(starts[k]), p.mul(p.const_int(strides[k]), it)) if strides[k] else p.const_int(starts[k])
    in_args.append(p.view(ctx.buf_of(outer), [off]))
  out_args = []
  for i, sbuf in enumerate(scratch):
    if i == out_idx:
      off = p.mul(it, p.const_int(slice_size)) if slice_size != 1 else it
      out_args.append(p.view(out, [off]))
    else:
      out_args.append(sbuf)
  call = PNode(POps.CALL, tuple(in_args + out_args), attrs={"callee": callee.name, "n_in": len(in_args), "n_out": len(out_args), "returns": ()})
  ctx.statements.append(p.for_(rng, [call]))


@lowers(Ops.GATHER)
def _lower_gather(ctx: LowerCtx, node: Expr) -> None:
  """``out[k] = src[indices[k]]`` via a ``static const`` index table + one GLOBAL loop (any size)."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  idx_buf = ctx.new_const_index(idx)
  vname = f"i_{out.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  k = p.var(vname)
  src_idx = p.load(p.view(idx_buf, [k]))
  ctx.statements.append(p.for_(rng, [p.store(p.view(out, [k]), p.load(p.view(ctx.buf_of(src), [src_idx])))]))


@lowers(Ops.SCATTER)
def _lower_scatter(ctx: LowerCtx, node: Expr) -> None:
  """Zero the output, then ``out[indices[k]] = src[k]`` via a ``static const`` index table."""
  src = node.args[0]
  idx = node.attrs["indices"].reshape(-1)
  out = ctx.alloc_tmp(node)
  zname = f"z_{out.attrs['name']}"
  zrng = p.range_(zname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
  z = p.var(zname)
  ctx.statements.append(p.for_(zrng, [p.store(p.view(out, [z]), p.const_float(0.0, dtype=node.type.dtype))]))
  idx_buf = ctx.new_const_index(idx)
  iname = f"i_{out.attrs['name']}"
  irng = p.range_(iname, 0, len(idx), kind=RangeKind.GLOBAL)
  i = p.var(iname)
  dst = p.load(p.view(idx_buf, [i]))
  ctx.statements.append(p.for_(irng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [i])))]))


@lowers(Ops.STACK)
def _lower_stack(ctx: LowerCtx, node: Expr) -> None:
  """Stack ``n`` rank-r inputs along a new ``axis`` into a rank-(r+1) output: each input
  occupies index ``i`` along the new axis. Per element, decompose the input flat index into
  its coords, insert ``i`` at ``axis``, recombine against the output shape."""
  axis = int(node.attrs.get("axis", 0))
  out_shape = node.shape
  out = ctx.alloc_tmp(node)
  for i, src in enumerate(node.args):
    src_shape = src.shape
    name = f"j_{out.attrs['name']}_{i}"
    rng = p.range_(name, 0, _size_of(src_shape), kind=RangeKind.GLOBAL)
    j = p.var(name)
    src_coords = [_coord_p(j, src_shape, d) for d in range(len(src_shape))]
    out_coords = src_coords[:axis] + [p.const_int(i)] + src_coords[axis:]
    dst = _flat_index_p(out_coords, out_shape)
    ctx.statements.append(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))


@lowers(Ops.CONCAT)
def _lower_concat(ctx: LowerCtx, node: Expr) -> None:
  """Concatenate inputs along ``axis``: each input keeps its shape but its ``axis`` coordinate is
  shifted by the running offset. Per element, decompose / shift / recombine against the output."""
  axis = int(node.attrs.get("axis", 0))
  out_shape = node.shape
  out = ctx.alloc_tmp(node)
  offset = 0
  for i, src in enumerate(node.args):
    src_shape = src.shape
    name = f"j_{out.attrs['name']}_{i}"
    rng = p.range_(name, 0, _size_of(src_shape), kind=RangeKind.GLOBAL)
    j = p.var(name)
    coords = [_coord_p(j, src_shape, d) for d in range(len(src_shape))]
    if offset:
      coords[axis] = p.add(p.const_int(offset), coords[axis])
    dst = _flat_index_p(coords, out_shape)
    ctx.statements.append(p.for_(rng, [p.store(p.view(out, [dst]), p.load(p.view(ctx.buf_of(src), [j])))]))
    offset += int(src_shape[axis])


__all__ = ["LoweringError", "LowerCtx", "lower_function", "lowers", "main_proc"]
