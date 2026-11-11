"""Prove buffer reuse from the expression schedule and the entries each operation reads."""

from __future__ import annotations

import numpy as np

from ...ad.rules import ELEMENTWISE
from ...ir.expr import Expr, ExprOp, _index_values
from ...function.concrete import ConcreteFunction
from ...ir.types import dtypes


def index_batches(nodes: tuple[Expr, ...], length: int, step: Expr | None) -> dict[int, np.ndarray]:
  """Evaluate all known indices for all steps together, without evaluating carry data."""
  values: dict[int, np.ndarray] = {}
  for node in nodes:
    if node is step:
      values[node.id] = np.arange(length, dtype=np.int64).reshape((length,) + (1,) * len(node.shape))
    elif node.value is not None:
      values[node.id] = np.broadcast_to(np.asarray(node.value), (length, *node.shape))
    elif node.op in ELEMENTWISE and all(arg.id in values for arg in node.args):
      rank = len(node.shape)
      args = [values[arg.id].reshape((length, *(1,) * (rank - len(arg.shape)), *arg.shape)) for arg in node.args]
      with np.errstate(all="ignore"):
        values[node.id] = np.asarray(ELEMENTWISE[node.op].numpy(*args)).astype(node.type.dtype.numpy())
    elif node.op == ExprOp.RESHAPE and node.args[0].id in values:
      values[node.id] = values[node.args[0].id].reshape((length, *node.shape))
  return values


def _indices(node: Expr, batches: dict[int, np.ndarray], length: int) -> np.ndarray | None:
  base, index = node.args[:2]
  if index.value is not None:
    return np.broadcast_to(_index_values(node).reshape(1, -1), (length, node.args[2].size if node.op == ExprOp.SCATTER else node.size))
  if index.id not in batches:
    return None
  entries = batches[index.id].reshape(length, index.size)
  axis = node.attrs.get("axis")
  if axis is None:
    return entries
  before = int(np.prod(base.shape[:axis]))
  after = int(np.prod(base.shape[axis + 1 :]))
  addresses = (np.arange(before)[None, :, None, None] * base.shape[axis] + entries[:, None, :, None]) * after + np.arange(after)[None, None, None, :]
  return addresses.reshape(length, before * index.size * after)


def _reads(node: Expr, operand: Expr, batches: dict[int, np.ndarray], length: int) -> np.ndarray | None:
  # These rules describe value reads. In particular FLOOR and CEIL read every operand entry.
  if node.op == ExprOp.GATHER and node.args[0] is operand and operand.op in (ExprOp.INPUT, ExprOp.SCATTER):
    return _indices(node, batches, length)
  return None


_ALIASES = {ExprOp.RESHAPE, ExprOp.STOP_GRADIENT, ExprOp.SLICE, ExprOp.PACK, ExprOp.VALUES}


def _aliases(nodes: tuple[Expr, ...], versions: set[Expr]) -> set[Expr]:
  aliases = set(versions)
  for node in nodes:
    if node.op in _ALIASES and node.args[0] in aliases:
      aliases.add(node)
  return aliases


def _safe_link(
  nodes: tuple[Expr, ...], link: Expr, versions: set[Expr], outputs: tuple[Expr, ...], batches: dict[int, np.ndarray], length: int
) -> bool:
  versions = _aliases(nodes, versions)
  if any(arg in versions for arg in link.args[1:]):
    return False
  position = nodes.index(link)
  later = [node for node in nodes[position + 1 :] if any(arg in versions for arg in node.args)]
  if any(out in versions for out in outputs):
    return False
  if not later:
    return True
  writes = _indices(link, batches, length)
  if writes is None:
    return False
  for node in later:
    for operand in node.args:
      if operand not in versions:
        continue
      reads = _reads(node, operand, batches, length)
      if reads is None or np.any(np.any(reads[:, :, None] == writes[:, None, :], axis=(1, 2))):
        return False
  return True


def carry_reuse(fun: ConcreteFunction, n_carry: int, length: int, index: bool, excluded: set[int]) -> dict[int, tuple[Expr, ...]]:
  """Return proved scatter chains, keyed by the carry input position."""
  nodes = tuple(fun.nodes)
  batches: dict[int, np.ndarray] | None = None
  proved: dict[int, tuple[Expr, ...]] = {}
  for k in range(n_carry):
    if k in excluded or fun.inputs[k].type.pattern is not None:
      continue
    chain: list[Expr] = []
    current = fun.outputs[k]
    while current.op == ExprOp.SCATTER and current.attrs["mode"] != "fill" and current.type.pattern is None:
      chain.append(current)
      current = current.args[0]
    if current is not fun.inputs[k]:
      continue
    chain.reverse()
    versions = {current}
    safe = True
    for link in chain:
      later = nodes[nodes.index(link) + 1 :]
      if any(any(arg in versions for arg in node.args) for node in later):
        if batches is None:
          batches = index_batches(nodes, length, fun.inputs[n_carry] if index else None)
      if not _safe_link(nodes, link, versions, tuple(fun.outputs), batches or {}, length):
        safe = False
        break
      versions.add(link)
    if safe:
      proved[k] = tuple(chain)
  return proved


def scatter_reuse(ctx, node: Expr) -> bool:
  """A dead private base can be overwritten after its update values have been computed."""
  base = node.args[0]
  if node.attrs["mode"] == "fill" or base.type.pattern is not None or base.id not in ctx.value_buffers:
    return False
  buf = ctx.raw_buf_of(base)
  if buf.attrs.get("address_space") != "private" or "values" in buf.attrs or "alias_of" in buf.attrs:
    return False
  nodes = tuple(ctx.fun.nodes)

  def owner(name: str) -> str:
    while "alias_of" in ctx.buffers[name].attrs:
      name = ctx.buffers[name].attrs["alias_of"]
    return name

  aliases = _aliases(nodes, {expr for expr in nodes if expr.id in ctx.value_buffers and owner(ctx.value_buffers[expr.id]) == buf.attrs["name"]})
  if any(arg in aliases for arg in node.args[1:]):
    return False
  if any(expr in aliases for expr in ctx.fun.outputs) or any(any(arg in aliases for arg in later.args) for later in nodes[nodes.index(node) + 1 :]):
    return False
  ctx.value_buffers[node.id] = ctx.value_buffers[base.id]
  return True


def same_index_sources(fun: ConcreteFunction, scatter: Expr) -> set[Expr]:
  """Find a single-use elementwise tree that can read and write one index in one loop."""
  base, index, source = scatter.args
  if scatter.attrs["mode"] == "fill" or (index.size != 1 and (index.value is None or len(np.unique(index.value)) != index.size)):
    return set()
  nodes = tuple(fun.nodes)
  selected: set[Expr] = set()
  found = False

  def visit(node: Expr) -> bool:
    nonlocal found
    if node.type.dtype not in (dtypes.float64, dtypes.int64, dtypes.bool_):
      return False
    if node.op == ExprOp.CONST:
      return True
    if node in fun.outputs or any(node in other.args for other in nodes if other is not scatter and other not in selected):
      return False
    if node.op == ExprOp.GATHER:
      if node.args != (base, index) or node.attrs.get("axis") != scatter.attrs.get("axis") or node.attrs["mode"] != scatter.attrs["mode"]:
        return False
      found = True
      selected.add(node)
      return True
    if node.op not in ELEMENTWISE:
      return False
    selected.add(node)
    return all(visit(arg) for arg in node.args)

  return selected if visit(source) and found else set()


def fused_value(ctx, node: Expr, scatter: Expr, address, coords):
  """Render the proved elementwise update, loading its gather at the destination address."""
  from ...ir import program as p
  from ...ir.program import ProgramNode

  if node.op == ExprOp.GATHER:
    return p.load(p.view(ctx.buf_of(scatter.args[0]), [address]))
  if node.op == ExprOp.CONST:
    return ctx.read(
      node, [coord if dim != 1 else p.const_int(0) for coord, dim in zip(coords[len(coords) - len(node.shape) :], node.shape, strict=True)]
    )
  op = ELEMENTWISE[ExprOp(node.op)].program
  return ProgramNode(op, tuple(fused_value(ctx, arg, scatter, address, coords) for arg in node.args), dtype=node.type.dtype)
