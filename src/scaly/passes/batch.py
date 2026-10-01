"""Maps as batched expressions: a ``VMAP`` whose body has a batched form for every op, and gains by it, becomes one expression over all its trips."""

from __future__ import annotations

from ..function import ConcreteFunction
from ..ir.expr import Expr, ExprOp, concat, has_trait, matmul, stack, topo
from ..ir.target import Target
from ..ir.types import TensorType

# A value of the body, rewritten: the expression, and whether it changes with the trip. One that
# does holds the trips as its leading axis.
type _Value = tuple[Expr, bool]


class _NoForm(Exception):
  """The map keeps its loop: an op of its body has no batched form, or the form would not be faster."""


def batch_maps(fun: ConcreteFunction, target: Target) -> ConcreteFunction:
  """``fun`` with each map that gains by it written as one expression over all its trips.

  A mapped body multiplies by its weights once a trip, so a matrix every trip shares is read as
  many times as there are trips, and each product is as narrow as one trip's value: a vector, or a
  matrix of a few tangent columns. Batched, every value that changes with the trip takes the
  trips as a leading axis: an elementwise op runs over the whole axis, a move of data (``reshape``,
  ``transpose``, a slice, ``stack``, ``concat``) carries the axis along, and a product with a
  shared matrix becomes one matrix product over all the trips' rows, which reads the matrix once
  and fills the register tiles a single trip's product could not.

  The loop is kept where the body holds an op with no such form (a product of two values that
  both change with the trip, a reduction, a gather, a call, a nested loop), where a mapped input's
  windows overlap, where an output does not change with the trip, and where the batched form is
  not expected faster for ``target`` (``_gains``). Sibling outputs of one map share one batched
  body.
  """
  rebuilt: dict[int, Expr] = {}
  bodies: dict[tuple[object, ...], dict[int, _Value] | None] = {}
  changed = False
  for node in topo(fun.outputs):
    args = tuple(rebuilt[a.id] for a in node.args)
    out = _batched_output(node, args, target, bodies) if node.op == ExprOp.VMAP else None
    if out is not None:
      changed = True
    elif all(before is after for before, after in zip(node.args, args, strict=True)):
      out = node
    else:
      out = Expr(node.op, args, node.type, node.name, node.value, dict(node.attrs), node.lowering)
    rebuilt[node.id] = out
  return fun._with_outputs(tuple(rebuilt[o.id] for o in fun.outputs)) if changed else fun


def _batched_output(node: Expr, args: tuple[Expr, ...], target: Target, bodies: dict[tuple[object, ...], dict[int, _Value] | None]) -> Expr | None:
  callee: ConcreteFunction = node.attrs["callee"]
  trips = int(node.attrs["length"])
  starts, strides = tuple(int(s) for s in node.attrs["starts"]), tuple(int(s) for s in node.attrs["strides"])
  key = (id(callee), tuple(a.id for a in args), trips, starts, strides)
  if key not in bodies:
    try:
      bodies[key] = _batched_body(callee, args, trips, starts, strides, target)
    except _NoForm:
      bodies[key] = None
  body = bodies[key]
  if body is None:
    return None
  out, batched = body[callee.outputs[int(node.attrs["output"])].id]
  # An output no trip changes would have to be repeated: its map keeps the loop, which does that.
  return out.reshape(node.shape) if batched else None


def _batched_body(
  callee: ConcreteFunction, args: tuple[Expr, ...], trips: int, starts: tuple[int, ...], strides: tuple[int, ...], target: Target
) -> dict[int, _Value]:
  """Every value of ``callee``'s body over all ``trips``, by the id of the node it stands for."""
  if trips < 2 or callee._effective_lowering() != "auto":
    raise _NoForm  # one trip is the body itself; a body asked to be scalar or opaque stays a procedure
  values: dict[int, _Value] = {}
  for formal, outer, start, stride in zip(callee.inputs, args, starts, strides, strict=True):
    flat = outer if len(outer.shape) == 1 else outer.reshape((outer.size,))
    if stride == 0:
      window = flat if (start, formal.size) == (0, flat.size) else flat[start : start + formal.size]
      values[formal.id] = (window.reshape(formal.shape), False)
    elif stride == formal.size:
      window = flat if (start, trips * formal.size) == (0, flat.size) else flat[start : start + trips * formal.size]
      values[formal.id] = (window.reshape((trips, *formal.shape)), True)
    else:
      raise _NoForm  # windows that overlap or leave gaps are not the rows of one array
  nodes = topo(callee.outputs)
  if not _gains(nodes, {formal.id for formal, stride in zip(callee.inputs, strides, strict=True) if stride}, trips, target):
    raise _NoForm
  for n in nodes:
    if n.id in values:
      continue
    found = [values[a.id] for a in n.args]
    if not any(batched for _, batched in found):
      same = all(x is a for (x, _), a in zip(found, n.args, strict=True))
      values[n.id] = (n if same else Expr(n.op, tuple(x for x, _ in found), n.type, n.name, n.value, dict(n.attrs), n.lowering), False)
      continue
    out = _form(n, found, trips)
    assert out.shape == (trips, *n.shape) and out.type.dtype == n.type.dtype, (n.op, out.shape, n.shape)
    values[n.id] = (out, True)
  return values


def _tiled(rows: int, k: int, columns: int, target: Target) -> bool:
  """Whether a product of ``rows`` by ``k`` with ``k`` by ``columns`` fills register tiles: a tile's
  rows, a reduction of four terms or more, and more columns than half a tile's (and than four), as
  lowering judges a product worth its loops (``lowering._product_in_loops``)."""
  tile_rows, tile_columns = target.product_tile
  return 1 < tile_rows <= rows and k >= 4 and columns > max(tile_columns // 2, 4)


def _gains(nodes: list[Expr], mapped: set[int], trips: int, target: Target) -> bool:
  """Whether the batched form of a body is expected faster than its loop for ``target``.

  What batching changes is the products of a value that changes with the trip by a matrix that
  does not. Such a product gains in two cases. The trips' rows together fill the register tiles
  (``_tiled``) where one trip's product, a vector's or a narrow matrix's, did not: measured at
  0.6 to 0.9 of the loop's time from a 16 by 16 matrix on. Or the shared matrix is larger than
  ``Target.panel_bytes``, half the level-1 data cache, so the loop read it from the level-2 cache
  again at every trip: 0.4 to 0.9. A product that already fills the tiles one trip at a time, by
  a matrix that stays in the cache, runs the same multiply-adds either way and pays for the
  copies that bring the trips' columns together: 1.0 to 1.07. The body is batched when the
  products that gain are at least half of the multiply-adds of all its products with a mapped
  operand."""
  batched = set(mapped)
  gain = total = 0
  for n in nodes:
    if not any(a.id in batched for a in n.args):
      continue
    batched.add(n.id)
    if n.op != ExprOp.MATMUL:
      continue
    a, b = n.args
    if a.id in batched and b.id in batched:
      return False  # a product of two mapped values has no batched form (``_product``)
    work = a.size * b.size // int(a.shape[-1]) if a.size and b.size else 0  # the multiply-adds of one trip's product
    total += work
    shared, value, left = (b, a, False) if a.id in batched else (a, b, True)
    if len(shared.shape) != 2:
      continue
    m, k = shared.shape if left else reversed(shared.shape)  # the reduction runs over ``k``
    if left:  # shared (m, k) @ value (k,) or (k, columns)
      columns = 1 if len(value.shape) == 1 else int(value.shape[1])
      once = len(value.shape) == 2 and _tiled(int(m), int(k), columns, target)
      # Batched (``_product``): the trips' columns as rows against a constant transposed, or side by side.
      together = _tiled(trips * columns, int(k), int(m), target) if shared.op == ExprOp.CONST else _tiled(int(m), int(k), trips * columns, target)
    else:  # value (k,) or (rows, k) @ shared (k, m): batched, the trips' rows together
      rows = 1 if len(value.shape) == 1 else int(value.shape[0])
      once, together = _tiled(rows, int(k), int(m), target), _tiled(trips * rows, int(k), int(m), target)
    streamed = shared.size * shared.type.dtype.itemsize > target.panel_bytes
    if together and (streamed or not once):
      gain += work
  return gain > 0 and 2 * gain >= total


def _form(n: Expr, found: list[_Value], trips: int) -> Expr:
  """The batched form of ``n``, some argument of which changes with the trip."""
  if has_trait(n.op, "elementwise") or n.op == ExprOp.SELECT:
    # Operands align at their last axes, as they did in the body: a batched one of lower rank
    # takes axes of one between the trips and its own, and a shared one broadcasts as it is.
    rank = len(n.shape)
    operands = tuple(_lead(x, rank, trips) if batched else x for x, batched in found)
    return Expr(n.op, operands, TensorType((trips, *n.shape), dtype=n.type.dtype, diff=n.type.diff), attrs=dict(n.attrs), lowering=n.lowering)
  if n.op == ExprOp.MATMUL:
    return _product(found[0], found[1], trips)
  x = found[0][0]
  if n.op == ExprOp.RESHAPE:
    return x.reshape((trips, *n.shape))
  if n.op == ExprOp.TRANSPOSE:
    axes = n.attrs.get("axes") or tuple(reversed(range(len(n.args[0].shape))))
    return x.transpose((0, *(int(axis) + 1 for axis in axes)))
  if n.op == ExprOp.SLICE:
    return x[(slice(None), *n.attrs["index"])]  # an index holds integers and slices only, one for each axis
  if n.op in (ExprOp.STACK, ExprOp.CONCAT) and all(batched for _, batched in found):
    join = stack if n.op == ExprOp.STACK else concat
    return join([value for value, _ in found], axis=int(n.attrs["axis"]) + 1)
  raise _NoForm


def _lead(x: Expr, rank: int, trips: int) -> Expr:
  """A batched operand with its own axes as the last ones of a result of ``rank`` axes."""
  inner = x.shape[1:]
  return x if len(inner) == rank else x.reshape((trips, *([1] * (rank - len(inner))), *inner))


def _product(left: _Value, right: _Value, trips: int) -> Expr:
  """A product of a batched value and a shared one, as one product over the trips. Each output is
  the sum it was, in the order of the reduction, so the batched product computes what the loop's
  did."""
  (a, a_batched), (b, b_batched) = left, right
  if a_batched and b_batched:
    raise _NoForm  # a product of two values that both change with the trip is a loop of products
  if a_batched:
    inner = a.shape[1:]
    if len(inner) == 1:
      return matmul(a, b)  # (trips, k) @ (k, n), or @ (k,)
    rows, k = inner
    return matmul(a.reshape((trips * rows, k)), b).reshape((trips, rows, *b.shape[1:]))
  inner = b.shape[1:]
  if len(a.shape) == 1:
    if len(inner) == 1:
      return matmul(b, a)  # a dot product a trip: (trips, k) @ (k,)
    k, columns = inner
    return matmul(b.transpose((0, 2, 1)).reshape((trips * columns, k)), a).reshape((trips, columns))
  m, k = a.shape
  if a.op == ExprOp.CONST:
    # The trips' columns as rows against the constant transposed, which is another constant: no
    # copy of it at run time, and the product reads it as a product reads its right operand.
    assert a.value is not None
    transposed = Expr.const(a.value.T.copy(), dtype=a.type.dtype, lowering=a.lowering)
    if len(inner) == 1:
      return matmul(b, transposed)
    columns = inner[1]
    out = matmul(b.transpose((0, 2, 1)).reshape((trips * columns, k)), transposed)
    return out.reshape((trips, columns, m)).transpose((0, 2, 1))
  # A matrix known only at run time is not copied: it multiplies the trips' columns side by side.
  if len(inner) == 1:
    return matmul(a, b.transpose()).transpose()
  columns = inner[1]
  out = matmul(a, b.transpose((1, 0, 2)).reshape((k, trips * columns)))
  return out.reshape((m, trips, columns)).transpose((1, 0, 2))


__all__ = ["batch_maps"]
