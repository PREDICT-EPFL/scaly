"""Lower semantic IR (``Expr`` / ``Function``) into Program IR (``PNode``).

Phase 5 of the roadmap. This module owns the explicit choice of how each
semantic op maps to a Program IR construct: buffers, loops, loads/stores,
calls. Today the lowerer only covers a small slice of the op set:

- ``INPUT`` / ``CONST`` → ``BUFFER`` params / constant buffers
- ``NEG`` / ``SIN`` / ``COS`` / ``EXP`` / ``LOG`` / ``SQRT`` → per-element ``FOR`` over a ``GLOBAL`` range that ``STORE``s ``unary(LOAD)`` into a workspace buffer
- ``ADD`` / ``SUB`` / ``MUL`` / ``DIV`` → per-element binary ``FOR`` (broadcasting not yet supported here — semantic IR resolves shape before lowering)
- ``RESHAPE`` → a buffer aliasing the source view (no copy)

This is intentionally a tiny slice; expanding to ``SUM``, ``MATMUL``, ``GATHER``,
``CALL``, ``MAP``, and ``SOLVER_CALL`` comes in subsequent phases. The lowerer
verifies its output via ``verify_program`` so future extensions cannot regress
silently.
"""

from __future__ import annotations

from . import program as p
from .expr import Expr, topo
from .function import Function
from .ops import Ops
from .program import PNode, POps, RangeKind, verify_program
from .types import DType, dtypes


SUPPORTED_UNARY = {
  Ops.NEG: POps.NEG,
  Ops.SIN: POps.SIN,
  Ops.COS: POps.COS,
  Ops.EXP: POps.EXP,
  Ops.LOG: POps.LOG,
  Ops.SQRT: POps.SQRT,
}

SUPPORTED_BINARY = {
  Ops.ADD: POps.ADD,
  Ops.SUB: POps.SUB,
  Ops.MUL: POps.MUL,
  Ops.DIV: POps.DIV,
}


class LoweringError(NotImplementedError):
  """Lowering cannot handle the requested semantic op yet."""


def lower_function(fun: Function) -> PNode:
  """Lower ``fun`` into a host ``PROC`` Program IR node.

  Each semantic input becomes an input ``BUFFER`` param. Each semantic output
  becomes an output ``BUFFER`` param. Internal values get workspace buffers
  named ``t{i}`` and are written by per-element loops. The final body either
  reuses the workspace buffer that holds an output value (via a `view` alias),
  or copies it into the output param's buffer.
  """

  if fun.device.kind != "host":
    raise LoweringError(f"Phase 5 lowerer only emits host PROC; got device={fun.device}")

  builder = _Builder(fun)
  builder.emit_inputs()
  builder.emit_body()
  builder.emit_outputs()
  proc = p.proc(fun.name, builder.params, builder.statements)
  verify_program(proc)
  return proc


class _Builder:
  def __init__(self, fun: Function) -> None:
    self.fun = fun
    self.params: list[PNode] = []
    self.statements: list[PNode] = []
    # Map each Expr id to the *buffer name* that holds its value at runtime.
    self.value_buffers: dict[int, str] = {}
    self.buffers: dict[str, PNode] = {}
    self._tmp_counter = 0

  # --- inputs ---------------------------------------------------------------

  def emit_inputs(self) -> None:
    for name, expr in zip(self.fun.input_names, self.fun.inputs, strict=True):
      buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.shape), address_space="global")
      self.params.append(buf)
      self.buffers[name] = buf
      self.value_buffers[expr.id] = name

  # --- outputs --------------------------------------------------------------

  def emit_outputs(self) -> None:
    for name, expr in zip(self.fun.output_names, self.fun.outputs, strict=True):
      out_buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.shape), address_space="global")
      self.params.append(out_buf)
      self.buffers[name] = out_buf
      src_buf_name = self.value_buffers.get(expr.id)
      if src_buf_name is None:
        raise LoweringError(f"output {name!r} expression was not lowered")
      if src_buf_name == name:
        continue  # already written into the output param
      self.statements.append(_copy_loop(self.buffers[src_buf_name], out_buf, expr.type.shape, expr.type.dtype))

  # --- body -----------------------------------------------------------------

  def emit_body(self) -> None:
    for node in topo(self.fun.outputs):
      if node.id in self.value_buffers:
        continue  # already an input
      if node.op == Ops.CONST:
        self._emit_const(node)
      elif node.op in SUPPORTED_UNARY:
        self._emit_elementwise(node, SUPPORTED_UNARY[Ops(node.op)], arity=1)
      elif node.op in SUPPORTED_BINARY:
        self._emit_elementwise(node, SUPPORTED_BINARY[Ops(node.op)], arity=2)
      elif node.op == Ops.RESHAPE:
        self._emit_reshape(node)
      else:
        raise LoweringError(f"semantic op {node.op!r} is not yet lowered to Program IR (Phase 5 slice)")

  def _alloc_tmp(self, expr: Expr) -> PNode:
    name = f"t{self._tmp_counter}"
    self._tmp_counter += 1
    buf = p.buffer(name, expr.type.dtype, _shape_or_scalar(expr.type.shape), address_space="private")
    self.buffers[name] = buf
    self.value_buffers[expr.id] = name
    return buf

  def _emit_const(self, node: Expr) -> None:
    # Constants are emitted as private-address-space "constant" buffers; lowering
    # doesn't yet have a CONST_ARRAY op so we store each scalar element with a
    # CONST_FLOAT initializer loop. Practical phase 5+ work will introduce a
    # dedicated CONST_BUFFER op; for now this keeps the structure simple.
    buf = self._alloc_tmp(node)
    value = node.value
    assert value is not None
    flat = value.reshape(-1)
    shape = node.type.shape or (1,)
    rng = p.range_(f"k_{buf.attrs['name']}", 0, int(value.size), kind=RangeKind.SERIAL)
    k_var = p.var(f"k_{buf.attrs['name']}", dtype=dtypes.int64)
    # Without a dedicated CONST_ARRAY op we serialize the values into a sequence
    # of guarded stores. That's only viable for tiny constants; the real solution
    # comes when CONST_BUFFER lands.
    if int(value.size) > 16:
      raise LoweringError(f"CONST of size {value.size} too large for the Phase 5 lowerer's inline serialization")
    body: list[PNode] = []
    for i, item in enumerate(flat):
      cond_var = p.var(f"k_{buf.attrs['name']}", dtype=dtypes.int64)
      _ = cond_var  # placeholder until IF lands
      body.append(p.store(p.view(buf, [p.const_int(i)]), p.const_float(float(item), dtype=node.type.dtype)))
    _ = shape, rng, k_var  # reserved for the future CONST_BUFFER lowering path
    self.statements.extend(body)

  def _emit_elementwise(self, node: Expr, pop: POps, *, arity: int) -> None:
    if any(a.shape != node.shape for a in node.args):
      raise LoweringError(f"Phase 5 lowerer requires identical shapes for elementwise op {node.op!r}; got {[a.shape for a in node.args]}")
    out_buf = self._alloc_tmp(node)
    size = node.size or 1
    rng = p.range_(f"i_{out_buf.attrs['name']}", 0, size, kind=RangeKind.GLOBAL)
    i = p.var(f"i_{out_buf.attrs['name']}", dtype=dtypes.int64)
    operand_loads = [p.load(p.view(self.buffers[self.value_buffers[a.id]], [i])) for a in node.args]
    if arity == 1:
      computed = PNode(pop, (operand_loads[0],), dtype=node.type.dtype)
    else:
      computed = PNode(pop, (operand_loads[0], operand_loads[1]), dtype=node.type.dtype)
    body = [p.store(p.view(out_buf, [i]), computed)]
    self.statements.append(p.for_(rng, body))

  def _emit_reshape(self, node: Expr) -> None:
    # RESHAPE is metadata-only at this layer: the result aliases the source buffer.
    src_buf_name = self.value_buffers[node.args[0].id]
    self.value_buffers[node.id] = src_buf_name


def _shape_or_scalar(shape: tuple[int, ...]) -> tuple[int, ...]:
  return shape if shape else (1,)


def _copy_loop(src_buf: PNode, dst_buf: PNode, shape: tuple[int, ...], dtype: DType) -> PNode:
  size = 1
  for d in shape:
    size *= d
  size = size or 1
  rng = p.range_(f"i_{dst_buf.attrs['name']}", 0, size, kind=RangeKind.GLOBAL)
  i = p.var(f"i_{dst_buf.attrs['name']}", dtype=dtypes.int64)
  return p.for_(rng, [p.store(p.view(dst_buf, [i]), p.load(p.view(src_buf, [i])))])


__all__ = ["LoweringError", "lower_function"]
