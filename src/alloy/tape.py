from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from operator import mul
from typing import Any, Iterable, Mapping

import numpy as np

from .expr import Expr, topo
from .ops import OP_INFO, Ops
from .types import Lowering


@dataclass(frozen=True, slots=True)
class Instruction:
  index: int
  expr_id: int
  op: Ops
  inputs: tuple[int, ...]
  shape: tuple[int, ...]
  lowering: Lowering
  attrs: dict[str, Any]
  value: np.ndarray | None = None
  name: str | None = None

  @property
  def size(self) -> int:
    return reduce(mul, self.shape, 1)


@dataclass(frozen=True, slots=True)
class WorkspaceSlot:
  instruction: int
  offset: int
  size: int


@dataclass(frozen=True, slots=True)
class WorkspacePlan:
  slots: tuple[WorkspaceSlot, ...]
  size: int


@dataclass(frozen=True, slots=True)
class TapeRegion:
  lowering: Lowering
  instructions: tuple[int, ...]

  @property
  def start(self) -> int:
    return self.instructions[0]

  @property
  def end(self) -> int:
    return self.instructions[-1] + 1


@dataclass(frozen=True, slots=True)
class Tape:
  instructions: tuple[Instruction, ...]
  outputs: tuple[int, ...]

  def __iter__(self):
    return iter(self.instructions)

  def __len__(self) -> int:
    return len(self.instructions)

  def debug(self) -> str:
    return format_tape(self)

  def lowering_regions(self) -> tuple[TapeRegion, ...]:
    if not self.instructions:
      return ()
    regions: list[TapeRegion] = []
    lowering = self.instructions[0].lowering
    current: list[int] = []
    for inst in self.instructions:
      if inst.lowering != lowering:
        regions.append(TapeRegion(lowering, tuple(current)))
        lowering = inst.lowering
        current = []
      current.append(inst.index)
    regions.append(TapeRegion(lowering, tuple(current)))
    return tuple(regions)

  def evaluate(self, env: Mapping[str, Any]) -> list[np.ndarray]:
    values: list[np.ndarray] = []
    for inst in self.instructions:
      out = _eval_instruction(inst, values, env)
      if out.shape != inst.shape:
        raise ValueError(f"instruction {inst.index} {inst.op.value!r} produced shape {out.shape}, expected {inst.shape}")
      values.append(out)
    return [values[i] for i in self.outputs]

  def plan_workspace(self) -> WorkspacePlan:
    last_use = {inst.index: inst.index for inst in self.instructions}
    end = len(self.instructions)
    for inst in self.instructions:
      for src in inst.inputs:
        last_use[src] = max(last_use[src], inst.index)
    for out in self.outputs:
      last_use[out] = end

    free: list[tuple[int, int]] = []
    active: list[tuple[int, int, int]] = []
    high_water = 0
    slots: list[WorkspaceSlot] = []
    for inst in self.instructions:
      still_active: list[tuple[int, int, int]] = []
      for death, offset, size in active:
        if death < inst.index:
          free.append((offset, size))
        else:
          still_active.append((death, offset, size))
      active = still_active
      if inst.op in {Ops.INPUT, Ops.CONST}:
        continue
      offset, free = _alloc_slot(inst.size, free)
      if offset < 0:
        offset = high_water
        high_water += inst.size
      slots.append(WorkspaceSlot(inst.index, offset, inst.size))
      active.append((last_use[inst.index], offset, inst.size))
    return WorkspacePlan(tuple(slots), high_water)


def _alloc_slot(size: int, free: list[tuple[int, int]]) -> tuple[int, list[tuple[int, int]]]:
  for i, (offset, free_size) in enumerate(free):
    if free_size < size:
      continue
    rest = free[:i] + free[i + 1 :]
    if free_size > size:
      rest.append((offset + size, free_size - size))
    return offset, rest
  return -1, free


def _instruction_attrs(e: Expr) -> dict[str, Any]:
  attrs = dict(e.attrs)
  if e.op == Ops.SLICE:
    attrs["source_shape"] = e.args[0].shape
  return attrs


def linearize(outputs: Iterable[Expr]) -> Tape:
  nodes = topo(outputs)
  loc = {e.id: i for i, e in enumerate(nodes)}
  instructions = tuple(
    Instruction(
      index=i,
      expr_id=e.id,
      op=Ops(e.op),
      inputs=tuple(loc[a.id] for a in e.args),
      shape=e.shape,
      lowering=e.lowering,
      attrs=_instruction_attrs(e),
      value=e.value.copy() if e.value is not None else None,
      name=e.name,
    )
    for i, e in enumerate(nodes)
  )
  return Tape(instructions, tuple(loc[e.id] for e in outputs))


def format_tape(tape: Tape) -> str:
  lines: list[str] = []
  for inst in tape.instructions:
    lhs = f"%{inst.index}"
    args = ", ".join(f"%{i}" for i in inst.inputs)
    if inst.op == Ops.INPUT:
      rhs = f"input {inst.name}"
    elif inst.op == Ops.CONST:
      assert inst.value is not None
      rhs = f"const {np.array2string(inst.value, threshold=6)}"
    elif inst.op == Ops.CALL:
      callee = inst.attrs["callee"]
      rhs = f"call {callee.name}[{inst.attrs['output']}]({args})"
    elif inst.op == Ops.MAP:
      callee = inst.attrs["callee"]
      length = inst.attrs["length"]
      slice_size = inst.attrs["slice_size"]
      bindings = ", ".join(f"%{src}[{s}::{st}]" for src, s, st in zip(inst.inputs, inst.attrs["starts"], inst.attrs["strides"], strict=True))
      rhs = f"map[{length}x{slice_size}] {callee.name}[{inst.attrs['output']}]({bindings})"
    else:
      rhs = f"{inst.op.value}({args})"
    lowering = "" if inst.lowering == "auto" else f" [{inst.lowering}]"
    lines.append(f"{lhs} = {rhs} : float64{inst.shape}{lowering}")
  lines.append("outputs " + ", ".join(f"%{i}" for i in tape.outputs))
  return "\n".join(lines)


def _asarray(value: Any) -> np.ndarray:
  return np.asarray(value, dtype=np.float64)


def _eval_map(inst: Instruction, args: list[np.ndarray]) -> np.ndarray:
  callee = inst.attrs["callee"]
  output_idx = inst.attrs["output"]
  length = inst.attrs["length"]
  starts = inst.attrs["starts"]
  strides = inst.attrs["strides"]
  slice_size = inst.attrs["slice_size"]
  out = np.empty((length * slice_size,), dtype=np.float64)
  for it in range(length):
    callee_args = [
      args[i][starts[i] + it * strides[i] : starts[i] + it * strides[i] + callee.inputs[i].size].reshape(callee.inputs[i].shape)
      for i in range(len(callee.inputs))
    ]
    res = _asarray(callee.eval_interpreter(*callee_args)[output_idx])
    out[it * slice_size : (it + 1) * slice_size] = res.reshape(-1)
  return _asarray(out)


def _check_input(inst: Instruction, env: Mapping[str, Any]) -> np.ndarray:
  if inst.name is None:
    raise ValueError(f"input instruction {inst.index} has no name")
  if inst.name not in env:
    raise KeyError(f"missing input {inst.name!r}")
  value = _asarray(env[inst.name])
  if value.shape != inst.shape:
    raise ValueError(f"input {inst.name!r} has shape {value.shape}, expected {inst.shape}")
  return value


def _eval_instruction(inst: Instruction, values: list[np.ndarray], env: Mapping[str, Any]) -> np.ndarray:
  if inst.op == Ops.INPUT:
    return _check_input(inst, env)
  if inst.op == Ops.CONST:
    if inst.value is None:
      raise ValueError(f"const instruction {inst.index} has no value")
    return inst.value

  args = [values[i] for i in inst.inputs]
  if inst.op == Ops.RESHAPE:
    return args[0].reshape(inst.attrs["shape"])
  if inst.op == Ops.TRANSPOSE:
    return _asarray(np.transpose(args[0], axes=inst.attrs["axes"]))
  if inst.op == Ops.SLICE:
    return _asarray(args[0][inst.attrs["index"]])
  if inst.op == Ops.GATHER:
    indices = inst.attrs["indices"]
    return _asarray(np.take(args[0].reshape(-1), indices).reshape(indices.shape))
  if inst.op == Ops.SCATTER:
    out = np.zeros(inst.shape, dtype=np.float64).reshape(-1)
    out[inst.attrs["indices"].reshape(-1)] = args[0].reshape(-1)
    return out.reshape(inst.shape)
  if inst.op == Ops.STACK:
    return _asarray(np.stack(args, axis=inst.attrs.get("axis", 0)))
  if inst.op == Ops.CONCAT:
    return _asarray(np.concatenate(args, axis=inst.attrs.get("axis", 0)))
  if inst.op == Ops.SUM:
    return _asarray(np.sum(args[0]))
  if inst.op == Ops.MATMUL:
    return _asarray(args[0] @ args[1])
  if inst.op == Ops.CALL:
    callee = inst.attrs["callee"]
    return _asarray(callee.eval_interpreter(*args)[inst.attrs["output"]])
  if inst.op == Ops.MAP:
    return _eval_map(inst, args)

  info = OP_INFO[inst.op]
  if info.numpy is None:
    raise NotImplementedError(f"no tape evaluator for op {inst.op.value!r}")
  return _asarray(info.numpy(*args))
