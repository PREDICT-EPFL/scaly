"""Schedule ordered reductions from their scalar accesses and emit their accumulators."""

from __future__ import annotations

from collections.abc import Sequence

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp, walk_program
from ..program._common import substitute_var, trip_count
from .ctx import LowerCtx, _coefficients


def emit_reduction(ctx: LowerCtx, outputs: Sequence[ProgramNode], reductions: Sequence[ProgramNode], target: ProgramNode, value: ProgramNode) -> None:
  """Use four scalar output lanes for unit-stride reductions, otherwise put reduction axes first.

  Four output lanes expand during emission, each with its own named accumulator. A strided
  reduction carries its running sums in the destination between iterations. Loaded bounds keep
  their output coordinates outside the reduction and disable lane expansion.
  """
  identity = p.const_float(0.0, dtype=value.dtype)
  names = frozenset(r.attrs["name"] for r in reductions)
  accesses = [_coefficients(n.args[0].args[0], names) for n in walk_program(value) if n.op == ProgramOp.LOAD]
  strided = any(c is None or any(c.get(name, 0) not in (0, 1) for name in names) for c in accesses)

  def accumulator() -> tuple[str, ProgramNode, ProgramNode]:
    name = ctx.names.allocate(f"acc_{target.attrs['buffer']}")
    return name, p.var(name, value.dtype), p.assign(name, identity, declare=True)

  loaded = any(n.op == ProgramOp.LOAD for r in reductions for bound in r.args[:2] for n in walk_program(bound))
  if strided and not loaded:
    ctx.emit(outputs, [p.store(target, identity)])
    ctx.emit([*reductions, *outputs], [p.store(target, p.add(p.load(target), value))])
    return

  def emit(outer: Sequence[ProgramNode], coordinates: Sequence[ProgramNode] = ()) -> None:
    values = [substitute_var(value, outputs[-1].attrs["name"], coord) for coord in coordinates] if coordinates else [value]
    targets = [substitute_var(target, outputs[-1].attrs["name"], coord) for coord in coordinates] if coordinates else [target]
    accumulators = [accumulator() for _ in values]
    updates = [p.assign(name, p.add(acc, term)) for (name, acc, _), term in zip(accumulators, values, strict=True)]
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
