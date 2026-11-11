"""Schedule ordered reductions from their scalar accesses and emit their accumulators."""

from __future__ import annotations

from collections.abc import Sequence

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp, walk_program
from ...ir.types import dtypes
from ..arith import constant
from ..program._common import substitute_var, trip_count
from .ctx import LowerCtx, _coefficients


def emit_reduction(
  ctx: LowerCtx,
  outputs: Sequence[ProgramNode],
  reductions: Sequence[ProgramNode],
  target: ProgramNode,
  value: ProgramNode,
  *,
  kind: str = "add",
  batch_axes: int = 0,
  identity: ProgramNode | None = None,
) -> None:
  """Use four scalar output lanes for unit-stride reductions, otherwise put reduction axes first.

  Four output lanes expand during emission, each with its own named accumulator. A strided
  reduction carries its running sums in the destination between iterations, with batch axes outside. Loaded bounds keep
  their output coordinates outside the reduction and disable lane expansion.
  """
  if identity is None:
    identity = constant(9223372036854775807 if value.dtype == dtypes.int64 and kind == "min" else {"add": 0, "max": -float("inf"), "min": float("inf")}[kind], value.dtype)

  def combine(a: ProgramNode, b: ProgramNode) -> ProgramNode:
    return combine_reduction(a, b, kind)

  names = frozenset(r.attrs["name"] for r in reductions)
  accesses = [_coefficients(n.args[0].args[0], names) for n in walk_program(value) if n.op == ProgramOp.LOAD]
  strided = any(c is None or any(c.get(name, 0) not in (0, 1) for name in names) for c in accesses)

  def accumulator() -> tuple[str, ProgramNode, ProgramNode]:
    name = ctx.names.allocate(f"acc_{target.attrs['buffer']}")
    return name, p.var(name, value.dtype), p.assign(name, identity, declare=True)

  loaded = any(n.op == ProgramOp.LOAD for r in reductions for bound in r.args[:2] for n in walk_program(bound))
  if strided and not loaded:
    ctx.emit(outputs, [p.store(target, identity)])
    ctx.emit([*outputs[:batch_axes], *reductions, *outputs[batch_axes:]], [p.store(target, combine(p.load(target), value))])
    return

  def emit(outer: Sequence[ProgramNode], coordinates: Sequence[ProgramNode] = ()) -> None:
    values = [substitute_var(value, outputs[-1].attrs["name"], coord) for coord in coordinates] if coordinates else [value]
    targets = [substitute_var(target, outputs[-1].attrs["name"], coord) for coord in coordinates] if coordinates else [target]
    accumulators = [accumulator() for _ in values]
    updates = [p.assign(name, combine(acc, term)) for (name, acc, _), term in zip(accumulators, values, strict=True)]
    # An empty reduction must not remove its identity or final store.
    loops = ctx.nest_body(reductions, updates)
    stores = [p.store(dst, acc) for dst, (_, acc, _) in zip(targets, accumulators, strict=True)]
    ctx.emit(outer, [*(initial for _, _, initial in accumulators), *loops, *stores])

  count = trip_count(outputs[-1]) if outputs else 1
  if count is not None and count >= 4 and not loaded:
    axis = outputs[-1]
    blocks, tail = divmod(count, 4)
    block_name = ctx.names.allocate(f"{axis.attrs['name']}_block")
    block = p.range_(block_name, 0, blocks, kind=axis.attrs["kind"])
    coordinates = [p.add(axis.args[0], p.add(p.mul(p.var(block_name), p.const_int(4)), p.const_int(lane))) for lane in range(4)]
    emit([*outputs[:-1], block], coordinates)
    if tail:
      emit([*outputs[:-1], p.range_(axis.attrs["name"], axis.args[0].attrs["value"] + 4 * blocks, axis.args[1], kind=axis.attrs["kind"])])
  else:
    emit(outputs)


def combine_reduction(a: ProgramNode, b: ProgramNode, kind: str) -> ProgramNode:
  """Combine extrema with SELECT so either operand's NaN propagates."""
  if kind == "add":
    return p.add(a, b)
  chosen = ProgramNode(ProgramOp.LT, (b, a) if kind == "max" else (a, b), dtype=dtypes.bool_)
  nan = ProgramNode(ProgramOp.NE, (a, a), dtype=dtypes.bool_)
  condition = ProgramNode(ProgramOp.OR, (nan, chosen), dtype=dtypes.bool_)
  return ProgramNode(ProgramOp.SELECT, (condition, a, b), dtype=a.dtype)
