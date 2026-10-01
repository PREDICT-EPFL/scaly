"""Split a flat loop whose indices divide the loop variable into nested loops that do not.

Lowering reads a gather's source and writes a scatter's destination at arithmetic on one flat loop
variable ``k``, and each range of the index map costs a division (``k / n``) and often a remainder;
fusing two such loops composes their maps, so one load can carry four divisions. When every index
in the loop is affine in the coordinates of one factorization of ``k``'s range, the loop becomes
one loop per coordinate and each index a sum of coordinates times constants: no divisions, and the
innermost loop reads and writes at constant strides the C compiler can vectorize.

An index that reads a table (a gather whose map repeats with a period: ``t[k % 13] + (k / 13) * 12``)
is taken apart at the table: the arithmetic around the read and the index of the read itself are
each made affine, and the read stays, at its coordinate.
"""

from __future__ import annotations

import numpy as np

from ...ir import program as p
from ...ir.program import ProgramNode, ProgramOp, RangeKind
from ._common import _map_procs, _proc_parts, _rebuild_proc, trip_count

_KINDS = frozenset({RangeKind.GLOBAL, RangeKind.REDUCE})
_INDEX_OPS = frozenset({ProgramOp.CONST_INT, ProgramOp.VAR, ProgramOp.ADD, ProgramOp.SUB, ProgramOp.MUL, ProgramOp.DIV, ProgramOp.MOD, ProgramOp.NEG})


def delinearize_loops(prog: ProgramNode) -> ProgramNode:
  """Rewrite every eligible loop in every procedure (see the module docstring)."""
  return _map_procs(prog, _delinearize_proc)


def _delinearize_proc(proc: ProgramNode) -> ProgramNode:
  params, body = _proc_parts(proc)
  new_body = [_delinearize_stmt(stmt) for stmt in body]
  return proc if all(a is b for a, b in zip(new_body, body, strict=True)) else _rebuild_proc(proc, params, new_body)


def _delinearize_stmt(stmt: ProgramNode) -> ProgramNode:
  if stmt.op != ProgramOp.FOR:
    return stmt
  rng, *body = stmt.args
  new_body = [_delinearize_stmt(sub) for sub in body]
  if any(a is not b for a, b in zip(new_body, body, strict=True)):
    stmt = ProgramNode(ProgramOp.FOR, (rng, *new_body), stmt.attrs, stmt.dtype)
  return _split(stmt) or stmt


def _evaluate(node: ProgramNode, name: str, k: np.ndarray) -> np.ndarray | None:
  """The integer value of an index expression at every ``k``, with C's truncating division, or None
  when it reads anything but ``k`` and constants."""
  if node.op not in _INDEX_OPS:
    return None
  if node.op == ProgramOp.CONST_INT:
    return np.full(k.shape, int(node.attrs["value"]), dtype=np.int64)
  if node.op == ProgramOp.VAR:
    return k if node.attrs["name"] == name else None
  values: list[np.ndarray] = []
  for arg in node.args:
    value = _evaluate(arg, name, k)
    if value is None:
      return None
    values.append(value)
  if node.op == ProgramOp.NEG:
    return -values[0]
  a, b = values
  if node.op == ProgramOp.ADD:
    return a + b
  if node.op == ProgramOp.SUB:
    return a - b
  if node.op == ProgramOp.MUL:
    return a * b
  if np.any(b == 0):
    return None
  quotient = np.sign(a) * np.sign(b) * (np.abs(a) // np.abs(b))
  return quotient if node.op == ProgramOp.DIV else a - b * quotient


def _divides(node: ProgramNode) -> bool:
  stack = [node]
  while stack:
    n = stack.pop()
    if n.op in (ProgramOp.DIV, ProgramOp.MOD):
      return True
    stack.extend(n.args)
  return False


def _views(node: ProgramNode, out: list[ProgramNode]) -> bool:
  """Collect the VIEW nodes under ``node``; False if the loop's body holds anything this pass keeps
  away from (calls, nested loops, early exits)."""
  if node.op in (ProgramOp.CALL, ProgramOp.FOR, ProgramOp.BREAK_IF):
    return False
  if node.op == ProgramOp.VIEW:
    out.append(node)
  return all(_views(a, out) for a in node.args)


def _uses_var(node: ProgramNode, name: str, skip: set[int]) -> bool:
  if id(node) in skip:
    return False
  if node.op == ProgramOp.VAR and node.attrs["name"] == name:
    return True
  return any(_uses_var(a, name, skip) for a in node.args)


def _factor(arrays: list[np.ndarray]) -> tuple[list[int], list[list[int]], list[int]] | None:
  """The finest factorization of the range into nested coordinates in which every array is affine:
  ``(dims, coefficients per array, base per array)``, outermost first; None if there is none."""
  rest = arrays
  dims: list[int] = []
  coeffs: list[list[int]] = [[] for _ in arrays]
  while len(rest[0]) > 1:
    n = len(rest[0])
    for m in [d for d in range(1, n) if n % d == 0]:
      rows = [a.reshape(-1, m) for a in rest]
      deltas = [int(r[1, 0] - r[0, 0]) for r in rows]
      steps = np.arange(n // m, dtype=np.int64)[:, None]
      if all(np.array_equal(r, r[0] + d * steps) for r, d in zip(rows, deltas, strict=True)):
        dims.append(n // m)
        for c, d in zip(coeffs, deltas, strict=True):
          c.append(d)
        rest = [r[0] for r in rows]
        break
    else:
      return None
  return dims, coeffs, [int(a[0]) for a in rest]


def _loads(node: ProgramNode) -> bool:
  return node.op == ProgramOp.LOAD or any(_loads(a) for a in node.args)


def _pieces(component: ProgramNode, out: list[ProgramNode]) -> None:
  """The parts of an index that must be affine in the new coordinates: the index itself, or, when
  it reads a table, the largest subexpressions around the reads that do not (a read's own index is
  the component of its view)."""
  if not _loads(component):
    out.append(component)
  elif component.op != ProgramOp.LOAD:
    for arg in component.args:
      _pieces(arg, out)


def _split(loop: ProgramNode) -> ProgramNode | None:
  rng, *body = loop.args
  if loop.attrs.get("exit_var") or rng.attrs["kind"] not in _KINDS:
    return None
  n = trip_count(rng)
  if n is None or n < 2 or rng.args[0].op != ProgramOp.CONST_INT or int(rng.args[0].attrs["value"]) != 0:
    return None
  views: list[ProgramNode] = []
  if not all(_views(stmt, views) for stmt in body):
    return None
  components: list[ProgramNode] = []
  for v in views:
    for c in v.args:
      _pieces(c, components)
  if not any(_divides(c) for c in components):
    return None
  name = rng.attrs["name"]
  if any(_uses_var(stmt, name, {id(v) for v in views}) for stmt in body):
    return None  # the loop variable is used as a value, not only as an index
  # The variable's values, not its trip numbers: a strided loop (a reduction's partial sums) runs
  # k = 0, s, 2s, ...; the new coordinates count trips, so each index is affine in them either way.
  k = np.arange(n, dtype=np.int64) * int(rng.args[2].attrs["value"])
  arrays = [_evaluate(c, name, k) for c in components]
  if any(a is None for a in arrays):
    return None
  factored = _factor([a for a in arrays if a is not None])
  if factored is None:
    return None
  dims, coeffs, bases = factored
  coords = [p.var(f"{name}_{i}") for i in range(len(dims))]
  replaced: dict[int, ProgramNode] = {}
  for component, cs, base in zip(components, coeffs, bases, strict=True):
    expr = p.const_int(base)
    for coord, c in zip(coords, cs, strict=True):
      if not c:
        continue
      term = coord if abs(c) == 1 else p.mul(p.const_int(abs(c)), coord)
      if expr.op == ProgramOp.CONST_INT and expr.attrs["value"] == 0:
        expr = term if c > 0 else p.neg(term)
      else:
        expr = p.add(expr, term) if c > 0 else p.sub(expr, term)
    replaced[id(component)] = expr

  def rebuild(node: ProgramNode) -> ProgramNode:
    if id(node) in replaced:
      return replaced[id(node)]
    args = tuple(rebuild(a) for a in node.args)
    return node if all(a is b for a, b in zip(args, node.args, strict=True)) else ProgramNode(node.op, args, node.attrs, node.dtype)

  inner: list[ProgramNode] = [rebuild(stmt) for stmt in body]
  for i in reversed(range(len(dims))):
    inner = [p.for_(p.range_(f"{name}_{i}", 0, dims[i], kind=rng.attrs["kind"]), inner)]
  return inner[0]
