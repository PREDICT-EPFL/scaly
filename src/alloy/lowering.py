"""Lower semantic IR (``Expr`` / ``Function``) into Program IR (``PNode``).

This is the migration's heart (see ``docs/program_ir_migration.md``). The goal is
for this module + ``codegen/program_c.py`` to become the *sole* path to C, with
no silent fallback to the legacy tape renderer.

Dispatch is a **registry** keyed by semantic ``Ops``: each op's lowering is a
self-contained rule registered with ``@lowers(...)``. Adding/deepening an op (or,
later, a GPU schedule) is a local change — a new rule, not an edit to a monolith.

Step 1 slice (this commit): elementwise unary/binary with identical operand
shapes, ``RESHAPE`` (alias), and small ``CONST``. Subsequent steps add integer
``SLICE``, ``CONST_BUFFER``, ``MATMUL``/``SUM``/``TRANSPOSE``, ``CALL``/``MAP``,
``GATHER``/``SCATTER``, ``STACK``/``CONCAT``, and elementwise broadcasting. GPU
placement and the deferred ops are tracked in the migration roadmap.
"""

from __future__ import annotations

from collections.abc import Callable

from . import program as p
from .expr import Expr, topo
from .function import Function
from .ops import Ops
from .program import PNode, POps, RangeKind, verify_program


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

  def alloc_tmp(self, expr: Expr) -> PNode:
    """Buffer to hold ``expr``'s value: the output buffer if aliased, else a fresh private temp."""
    alias = self._output_alias.get(expr.id)
    if alias is not None:
      self.value_buffers[expr.id] = alias
      return self.buffers[alias]
    name = f"t{self._tmp}"
    self._tmp += 1
    buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.shape), address_space="private")
    self.buffers[name] = buf
    self.value_buffers[expr.id] = name
    self.statements.append(buf)  # marks the local-array declaration for the renderer
    return buf

  def emit_elementwise(self, node: Expr, pop: POps, *, arity: int) -> None:
    if any(a.shape != node.shape for a in node.args):
      raise LoweringError(f"broadcasting is not yet lowered for {node.op!r}; operand shapes {[a.shape for a in node.args]}")
    out = self.alloc_tmp(node)
    vname = f"i_{out.attrs['name']}"
    rng = p.range_(vname, 0, _size_of(node.shape), kind=RangeKind.GLOBAL)
    i = p.var(vname)
    loads = tuple(p.load(p.view(self.buf_of(a), [i])) for a in node.args[:arity])
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


@lowers(Ops.SLICE)
def _lower_slice(ctx: LowerCtx, node: Expr) -> None:
  """General SLICE: integer indices drop a dim, slices keep one. Each output element
  reads its source via flat-index arithmetic built from the (full-rank, normalized)
  index spec. Covers rank-1, multi-dim, integer, and strided slices."""
  src = node.args[0]
  src_shape = src.shape
  index = node.attrs["index"]
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


__all__ = ["LoweringError", "LowerCtx", "lower_function", "lowers", "main_proc"]
