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

# Above this many elements an inline-serialized CONST is rejected; CONST_BUFFER
# (a dedicated op for large constant tables) lands in Step 2.
_CONST_INLINE_MAX = 16


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
  if int(value.size) > _CONST_INLINE_MAX:
    raise LoweringError(f"CONST of size {value.size} exceeds the inline-serialization cap; CONST_BUFFER lands in a later step")
  buf = ctx.alloc_tmp(node)
  for i, item in enumerate(value.reshape(-1)):
    ctx.statements.append(p.store(p.view(buf, [p.const_int(i)]), p.const_float(float(item), dtype=node.type.dtype)))


__all__ = ["LoweringError", "LowerCtx", "lower_function", "lowers", "main_proc"]
