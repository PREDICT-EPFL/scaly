"""Lower expression IR (``Expr`` / ``Function``) into Program IR (``ProgramNode``).

This module + ``codegen/c.py`` are the **sole** CPU path to C (see
``docs/how_it_works/lowering.md``); the legacy tape-based scalar renderer is gone.
The one sanctioned non-Program-IR escape is the ``codegen/solver`` wrapper for a
``solver Function`` — and even there the oracle Functions it drives lower through here.

Dispatch is a **registry** keyed by expression ``ExprOp``: each op's lowering is a
self-contained rule registered with ``@lowers(...)``. Adding/deepening an op (or,
later, a GPU schedule) is a local change — a new rule, not an edit to a monolith.

Covered: elementwise unary/binary (with numpy broadcasting), ``RESHAPE`` (alias),
``CONST`` (any size, via ``const_buffer``), general ``SLICE`` (integer / multi-dim /
strided), ``SUM``, ``MATMUL`` (rank <= 2), ``TRANSPOSE`` (rank <= 4), ``GATHER`` /
``SCATTER`` (any size, ``static const`` index table), ``STACK`` / ``CONCAT`` (any axis),
``CALL`` (multi-PROC, deduped) and ``VMAP``; a ``solver Function`` ``CALL`` is opaque
(see ``lower_function``). The tracking and unbumpercars workloads (forward + ``jac`` +
``spjac``) render and match generated-code / external numeric references. Deferred (re-land from the reference branch):
GPU placement and the new ops tracked in the migration roadmap.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

from ..ir import program as p
from ..ir.expr import Expr, ExprOp, topo
from ..function import Function
from .program import ProgramObserver, optimize_program
from ..ir.program import ProgramNode, ProgramOp, RangeKind
from ..ir.program_spec import verify_program
from ..ir.types import DeviceSpec, DType, dtypes


class LoweringError(NotImplementedError):
  """The lowerer (or Program-IR renderer) does not yet cover this op / case."""


# Expression ExprOp -> Program IR scalar ProgramOp. The op vocabulary grows here as ops migrate.
_UNARY: dict[ExprOp, ProgramOp] = {
  ExprOp.NEG: ProgramOp.NEG,
  ExprOp.SIN: ProgramOp.SIN,
  ExprOp.COS: ProgramOp.COS,
  ExprOp.TAN: ProgramOp.TAN,
  ExprOp.ASIN: ProgramOp.ASIN,
  ExprOp.ACOS: ProgramOp.ACOS,
  ExprOp.ATAN: ProgramOp.ATAN,
  ExprOp.SINH: ProgramOp.SINH,
  ExprOp.COSH: ProgramOp.COSH,
  ExprOp.TANH: ProgramOp.TANH,
  ExprOp.ERF: ProgramOp.ERF,
  ExprOp.EXP: ProgramOp.EXP,
  ExprOp.LOG: ProgramOp.LOG,
  ExprOp.SQRT: ProgramOp.SQRT,
  ExprOp.ABS: ProgramOp.ABS,
  ExprOp.FLOOR: ProgramOp.FLOOR,
  ExprOp.CEIL: ProgramOp.CEIL,
}

_BINARY: dict[ExprOp, ProgramOp] = {
  ExprOp.ADD: ProgramOp.ADD,
  ExprOp.SUB: ProgramOp.SUB,
  ExprOp.MUL: ProgramOp.MUL,
  ExprOp.DIV: ProgramOp.DIV,
  ExprOp.POW: ProgramOp.POW,
  ExprOp.ATAN2: ProgramOp.ATAN2,
  ExprOp.MINIMUM: ProgramOp.MINIMUM,
  ExprOp.MAXIMUM: ProgramOp.MAXIMUM,
}

LowerRule = Callable[["LowerCtx", Expr], None]
_RULES: dict[ExprOp, LowerRule] = {}


def lowers(*ops: ExprOp) -> Callable[[LowerRule], LowerRule]:
  """Register ``fn`` as the lowering rule for each expression op in ``ops``."""

  def deco(fn: LowerRule) -> LowerRule:
    for op in ops:
      _RULES[op] = fn
    return fn

  return deco


def lower_function(fun: Function, observe: ProgramObserver | None = None) -> ProgramNode:
  """Lower ``fun`` into a Program IR ``PROGRAM`` node (verified before return).

  Host placement only for now: the returned PROGRAM holds every lowered callee
  PROC in topological order followed by ``fun``'s main PROC last. Non-host
  placement raises ``LoweringError`` — GPU backends re-land from the reference
  branch after CPU parity (see ``internal/notes/program_ir_migration.md``).

  A ``solver Function`` callee is **opaque**: its ``ExprOp.SOLVER_CALL`` body is not
  lowered — the solver wrapper is rendered by the sanctioned ``codegen/solver``
  path (rule 6) — but its oracle Functions *are* lowered to PROCs (the wrapper
  calls them as ``<oracle>_raw``). The solver→oracle-name map is recorded on the
  PROGRAM (``solver_oracles`` attr) so ``pack_workspace`` can size the caller's
  ``w[]`` to fit the oracle and the CALL-to-solver gets ``callee_needs_w`` right.
  """
  if fun.device.kind != "host":
    raise LoweringError(f"non-host placement {fun.device} is not lowered yet (GPU backends are deferred to a later migration step)")
  callees: dict[str, ProgramNode] = {}
  solver_fns: dict[str, Function] = {}
  from ..solvers.graph import is_solver_function, solver_callees

  if is_solver_function(fun):
    solver_fns[fun.name] = fun
    for oracle in solver_callees(fun):
      if oracle.name not in callees:
        callees[oracle.name] = _lower_to_proc(oracle, callees, solver_fns)
    prog = p.program([*callees.values()])
    prog = ProgramNode(ProgramOp.PROGRAM, prog.args, {**prog.attrs, "solver_root": fun.name}, prog.dtype)
  else:
    root = _lower_to_proc(fun, callees, solver_fns, auto_scalarize=False)
    prog = p.program([*callees.values(), root])
  if solver_fns:
    solver_oracles = {name: tuple(o.name for o in solver_callees(sf)) for name, sf in solver_fns.items()}
    from ..solvers.model import ExternalOracle

    solver_external_workspace = {}
    for name, sf in solver_fns.items():
      desc = getattr(sf, "descriptor")
      solver_external_workspace[name] = max(
        (oracle.workspace_size for oracle in (desc.base, desc.grad, desc.jac, desc.hess, desc.bounds) if isinstance(oracle, ExternalOracle)),
        default=0,
      )
    prog = ProgramNode(
      ProgramOp.PROGRAM,
      prog.args,
      {**prog.attrs, "solver_oracles": solver_oracles, "solver_external_workspace": solver_external_workspace},
      prog.dtype,
    )
  if observe is not None:
    observe("lowered", prog)
  prog = optimize_program(prog, observe=observe)
  verify_program(prog)
  return prog


def main_proc(program_node: ProgramNode) -> ProgramNode:
  """Return the main (last) PROC inside a lowered PROGRAM."""
  if program_node.op != ProgramOp.PROGRAM:
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
  return n


def _lower_to_proc(fun: Function, callees: dict[str, ProgramNode], solver_fns: dict[str, Function], *, auto_scalarize: bool = True) -> ProgramNode:
  ctx = LowerCtx(fun, callees, solver_fns)
  ctx.emit_inputs()
  ctx.register_outputs()
  ctx.emit_body()
  ctx.emit_outputs()
  proc = p.proc(fun.name, ctx.params, ctx.statements)
  # ``input_count`` lets the renderer ``const``-qualify the first N (input) params of a ``_raw``
  # callee; emit_inputs runs before register_outputs, so inputs are the leading params.
  nodes = topo(fun.outputs)
  lowering = fun._effective_lowering()
  # Narrower stores round or truncate; scalar substitution must not erase those conversions.
  return ProgramNode(
    ProgramOp.PROC,
    proc.args,
    {
      **proc.attrs,
      "input_count": len(fun.inputs),
      "lowering": lowering,
      "scalarize": all(n.type.dtype == dtypes.float64 for n in (*fun.inputs, *nodes))
      and (lowering == "scalar" or (lowering == "auto" and auto_scalarize)),
    },
    proc.dtype,
  )


class LowerCtx:
  """Per-Function lowering state: buffers, statements, and the Expr-id -> buffer map."""

  def __init__(self, fun: Function, callees: dict[str, ProgramNode], solver_fns: dict[str, Function]) -> None:
    self.fun = fun
    self.callees = callees
    self.solver_fns = solver_fns  # name -> solver Function (opaque callees; rendered by codegen/solver)
    self.params: list[ProgramNode] = []
    self.statements: list[ProgramNode] = []
    self.buffers: dict[str, ProgramNode] = {}
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
      if expr.op in (ExprOp.INPUT, ExprOp.CONST) or expr.id in seen:
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
      rule = _RULES.get(ExprOp(node.op))
      if rule is None:
        raise LoweringError(f"Expression op {node.op!r} is not yet lowered to Program IR")
      rule(self, node)

  # --- helpers --------------------------------------------------------------

  def buf_of(self, expr: Expr) -> ProgramNode:
    return self.buffers[self.value_buffers[expr.id]]

  def new_private(self, dtype: DType, shape: tuple[int, ...]) -> ProgramNode:
    """Allocate a fresh private scratch BUFFER (declared as a local array by the renderer)."""
    name = f"t{self._tmp}"
    self._tmp += 1
    buf = p.buffer(name, dtype, _shape_or_scalar(shape), address_space="private")
    self.buffers[name] = buf
    self.statements.append(buf)  # marks the local-array declaration for the renderer
    return buf

  def alloc_tmp(self, expr: Expr) -> ProgramNode:
    """Buffer to hold ``expr``'s value: the output buffer if aliased, else a fresh private temp."""
    alias = self._output_alias.get(expr.id)
    if alias is not None:
      self.value_buffers[expr.id] = alias
      return self.buffers[alias]
    buf = self.new_private(expr.type.dtype, expr.shape)
    self.value_buffers[expr.id] = buf.attrs["name"]
    return buf

  def new_alias(self, dtype: DType, shape: tuple[int, ...], src_name: str, offset: int) -> ProgramNode:
    """A zero-copy private BUFFER that aliases ``src_name`` at a flat ``offset`` (rendered
    ``const T* tN = <src> + offset;``). Carries ``alias_of`` / ``alias_offset`` so the workspace
    pass leaves it unpacked and keeps its source live. Port of ``codegen/c.py``'s contiguous
    SLICE / RESHAPE pointer aliasing."""
    name = f"t{self._tmp}"
    self._tmp += 1
    buf = ProgramNode(
      ProgramOp.BUFFER,
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

  def new_const_index(self, idx: Iterable[int]) -> ProgramNode:
    """A read-only int64 index table (for GATHER/SCATTER), declared ``static const``."""
    values = [int(v) for v in idx]
    buf = p.const_buffer(f"k{self._tmp}", dtypes.int64, (len(values),), values)
    self._tmp += 1
    self.buffers[buf.attrs["name"]] = buf
    self.statements.append(buf)
    return buf

  def emit_elementwise(self, node: Expr, pop: ProgramOp, *, arity: int) -> None:
    out = self.alloc_tmp(node)
    vname = f"i_{out.attrs['name']}"
    rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
    i = p.var(vname)
    loads = tuple(p.load(p.view(self.buf_of(a), [_broadcast_index_p(i, a.shape, node.shape)])) for a in node.args[:arity])
    computed = ProgramNode(pop, loads, dtype=node.type.dtype)
    self.statements.append(p.for_(rng, [p.store(p.view(out, [i]), computed)]))


def _stride(shape: tuple[int, ...], dim: int) -> int:
  s = 1
  for d in shape[dim + 1 :]:
    s *= int(d)
  return s


def _coord_p(flat: ProgramNode, shape: tuple[int, ...], dim: int) -> ProgramNode:
  """Decompose flat output index ``flat`` into the ``dim``-th coordinate of ``shape``."""
  stride = _stride(shape, dim)
  v = flat if stride == 1 else p.div(flat, p.const_int(stride))
  # The leading dim needs no modulo: flat < size guarantees (flat // stride) < shape[0].
  return v if dim == 0 else p.mod(v, p.const_int(int(shape[dim])))


def _flat_index_p(coords: list[ProgramNode], shape: tuple[int, ...]) -> ProgramNode:
  terms = [c if (st := _stride(shape, i)) == 1 else p.mul(c, p.const_int(st)) for i, c in enumerate(coords)]
  if not terms:
    return p.const_int(0)
  acc = terms[0]
  for t in terms[1:]:
    acc = p.add(acc, t)
  return acc


def _affine_sum(vars_: list[ProgramNode], coeffs: list[int]) -> ProgramNode:
  """Build ``Σ coeffs[i] * vars_[i]`` as a ProgramNode, dropping zero coeffs and unit multiplies."""
  acc: ProgramNode | None = None
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


def _broadcast_index_p(flat: ProgramNode, in_shape: tuple[int, ...], out_shape: tuple[int, ...]) -> ProgramNode:
  """Map an output flat index to the source flat index under numpy broadcasting
  (right-aligned; size-1 dims and missing leading dims read index 0)."""
  if in_shape == out_shape or not out_shape:
    return flat
  if not in_shape:
    return p.const_int(0)
  offset = len(out_shape) - len(in_shape)
  coords = [p.const_int(0) if d == 1 else _coord_p(flat, out_shape, offset + i) for i, d in enumerate(in_shape)]
  return _flat_index_p(coords, in_shape)


def _copy_loop(src: ProgramNode, dst: ProgramNode, shape: tuple[int, ...]) -> ProgramNode:
  vname = f"c_{dst.attrs['name']}"
  rng = p.range_(vname, 0, _size_of(shape), kind=RangeKind.GLOBAL)
  i = p.var(vname)
  return p.for_(rng, [p.store(p.view(dst, [i]), p.load(p.view(src, [i])))])


# ---------------------------------------------------------------------------
# Lowering rules.
# ---------------------------------------------------------------------------


@lowers(*_UNARY)
def _lower_unary(ctx: LowerCtx, node: Expr) -> None:
  ctx.emit_elementwise(node, _UNARY[ExprOp(node.op)], arity=1)


@lowers(*_BINARY)
def _lower_binary(ctx: LowerCtx, node: Expr) -> None:
  ctx.emit_elementwise(node, _BINARY[ExprOp(node.op)], arity=2)


@lowers(ExprOp.RESHAPE)
def _lower_reshape(ctx: LowerCtx, node: Expr) -> None:
  # Metadata-only: the result aliases its source buffer (no copy).
  ctx.value_buffers[node.id] = ctx.value_buffers[node.args[0].id]


@lowers(ExprOp.CONST)
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


@lowers(ExprOp.SLICE)
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
  coords: list[ProgramNode] = []
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


@lowers(ExprOp.SUM)
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


@lowers(ExprOp.TRANSPOSE)
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
  stmt: ProgramNode = p.store(p.view(out, [out_idx]), p.load(p.view(ctx.buf_of(src), [src_idx])))
  for rng in reversed(ranges):
    stmt = p.for_(rng, [stmt])
  ctx.statements.append(stmt)


def _nest(ranges: list[ProgramNode], body: list[ProgramNode]) -> list[ProgramNode]:
  for rng in reversed(ranges):
    body = [p.for_(rng, body)]
  return body


def _mm_init(out: ProgramNode, idx: ProgramNode, dtype: DType) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.const_float(0.0, dtype=dtype))


def _mm_accum(out: ProgramNode, idx: ProgramNode, a_load: ProgramNode, b_load: ProgramNode) -> ProgramNode:
  return p.store(p.view(out, [idx]), p.add(p.load(p.view(out, [idx])), p.mul(a_load, b_load)))


def _mm_accumulate(
  ctx: LowerCtx,
  out: ProgramNode,
  out_idx: ProgramNode,
  a_load: ProgramNode,
  b_load: ProgramNode,
  dtype: DType,
  outer: list[ProgramNode],
  k_rng: ProgramNode,
) -> None:
  """Zero ``out[out_idx]`` over the ``outer`` loops, then accumulate ``a*b`` with the REDUCE-k loop outermost.

  A dot product per output is a serial add chain the C compiler cannot break without reassociation. With
  k outermost the inner loop runs over independent outputs and vectorizes, and each output still sums its
  terms in the same order, so the result is bit-identical to the dot form. Use it when the reduction axis
  is the matrix's slow axis; a contiguous reduction axis would make the compiler gather under -march=native.
  """
  ctx.statements.extend(_nest(outer, [_mm_init(out, out_idx, dtype)]))
  ctx.statements.extend(_nest([k_rng, *outer], [_mm_accum(out, out_idx, a_load, b_load)]))


@lowers(ExprOp.MATMUL)
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
  elif len(sa) == 2 and len(sb) == 1:  # mat @ vec: the reduction axis is contiguous in ``a``
    m, kk = sa
    i = p.var(f"i_{nm}")
    irng = p.range_(f"i_{nm}", 0, m, kind=RangeKind.GLOBAL)
    k = p.var(f"k_{nm}")
    krng = p.range_(f"k_{nm}", 0, kk, kind=RangeKind.REDUCE)
    row_dot = lambda row: _mm_accum(out, row, p.load(p.view(a_buf, [p.add(p.mul(row, p.const_int(kk)), k)])), p.load(p.view(b_buf, [k])))
    ctx.statements.extend(_nest([irng], [_mm_init(out, i, dt)]))
    # Four output rows per pass as four unrolled statements, so the compiler keeps four independent
    # accumulators; a four-trip inner loop over the rows becomes gathers under -march=native.
    blocks, tail = divmod(m, 4)
    if blocks:
      ib = p.var(f"ib_{nm}")
      ibrng = p.range_(f"ib_{nm}", 0, blocks, kind=RangeKind.GLOBAL)
      rows = [p.add(p.mul(ib, p.const_int(4)), p.const_int(r)) for r in range(4)]
      ctx.statements.extend(_nest([ibrng, krng], [row_dot(row) for row in rows]))
    if tail:
      it = p.var(f"it_{nm}")
      itrng = p.range_(f"it_{nm}", 4 * blocks, m, kind=RangeKind.GLOBAL)
      ctx.statements.extend(_nest([itrng, krng], [row_dot(it)]))
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
  from ..solvers.graph import is_solver_function, solver_callees

  if is_solver_function(callee):
    # Opaque: the solver wrapper is rendered by codegen/solver (rule 6), not lowered. Its body is
    # SOLVER_CALL (no lowering rule). We still lower the oracle Functions the wrapper drives.
    ctx.solver_fns[callee.name] = callee
    for oracle in solver_callees(callee):
      _ensure_callee(ctx, oracle)
    return
  if callee.name not in ctx.callees:
    ctx.callees[callee.name] = _lower_to_proc(callee, ctx.callees, ctx.solver_fns)


@lowers(ExprOp.CALL)
def _lower_call(ctx: LowerCtx, node: Expr) -> None:
  """An expression CALL output: emit one Program-IR CALL writing all callee outputs into scratch
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
      ProgramNode(
        ProgramOp.CALL, tuple(in_bufs + out_bufs), attrs={"callee": callee.name, "n_in": len(in_bufs), "n_out": len(out_bufs), "returns": ()}
      )
    )
    ctx.call_invocations[key] = tuple(b.attrs["name"] for b in out_bufs)
  ctx.value_buffers[node.id] = ctx.call_invocations[key][out_idx]


@lowers(ExprOp.VMAP)
def _lower_vmap(ctx: LowerCtx, node: Expr) -> None:
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
  call = ProgramNode(
    ProgramOp.CALL, tuple(in_args + out_args), attrs={"callee": callee.name, "n_in": len(in_args), "n_out": len(out_args), "returns": ()}
  )
  ctx.statements.append(p.for_(rng, [call]))


@lowers(ExprOp.GATHER)
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


@lowers(ExprOp.SCATTER)
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


@lowers(ExprOp.STACK)
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


@lowers(ExprOp.CONCAT)
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
