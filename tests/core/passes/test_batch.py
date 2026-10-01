"""Maps as batched expressions (``passes/batch.py``): each form against the body called once a trip, and the maps that keep their loop."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly.ir.program import ProgramOp
from scaly.passes.lowering import lower_function

TRIPS = 7
RNG = np.random.default_rng(11)


def _calls(fn: sc.Function, target: str = "apple-m3") -> int:
  """The calls left in ``fn``'s lowered program: a map that keeps its loop calls its body in it."""
  stack, found = [lower_function(fn, target=target)], 0
  while stack:
    node = stack.pop()
    found += node.op == ProgramOp.CALL
    stack.extend(node.args)
  return found


def _mapped(
  name: str, body: sc.Function, shared: tuple[int, ...] = (), trips: int = TRIPS
) -> tuple[sc.Function, list[np.ndarray], list[list[np.ndarray]]]:
  """``body`` mapped over ``trips`` windows of each input but those at ``shared``, which every trip
  reads whole; with random values for the mapped Function's inputs and, trip by trip, for the body's."""
  outer, specs, values = [], [], []
  for k, formal in enumerate(body.concrete.inputs):
    size = formal.size
    if k in shared:
      sym = sc.sym(f"s{k}", formal.shape)
      specs.append((sym if len(formal.shape) == 1 else sym.reshape((size,)), 0, 0))
      values.append(RNG.standard_normal(formal.shape))
    else:
      sym = sc.sym(f"m{k}", trips * size)
      specs.append((sym, 0, size))
      values.append(RNG.standard_normal(trips * size))
    outer.append(sym)
  fn = sc.Function.from_exprs(name, outer, [sc.vmap(body, trips, specs)], [f"a{k}" for k in range(len(outer))], ["y"])
  per_trip = [
    [v if k in shared else v.reshape(trips, *formal.shape)[i] for k, (v, formal) in enumerate(zip(values, body.concrete.inputs, strict=True))]
    for i in range(trips)
  ]
  return fn, values, per_trip


def _check(fn: sc.Function, body: sc.Function, values, per_trip, *, batched: bool, targets=("apple-m3", "x86-64-v3", "generic")) -> None:
  """``fn`` against ``body`` called once a trip, on each target; and whether it kept its loop.
  ``generic`` has one lane and no register tiles for the trips' rows to fill: it keeps every loop."""
  want = np.concatenate([np.asarray(_call(body, trip)).reshape(-1) for trip in per_trip])
  for target in targets:
    assert (_calls(fn, target) == 0) == (batched and target != "generic"), target
    with sc.target(target):
      np.testing.assert_allclose(np.asarray(_call(fn, values)).reshape(-1), want, rtol=1e-13, atol=1e-13)


def _call(fn: sc.Function, values):
  """``fn`` at ``values``: a Function of one input takes it bare."""
  return fn(values[0]) if len(values) == 1 else fn(tuple(values))


K, N, R = 5, 9, 3

# Each of the shapes a product of a mapped value and a shared one takes: (the body's inputs, which
# of them every trip shares, the product, whether it gains alone on a target with register tiles).
PRODUCTS: dict[str, tuple[list[tuple[str, tuple[int, ...]]], tuple[int, ...], Callable[..., sc.Expr], bool]] = {
  "vectors @ matrix": ([("x", (K,)), ("w", (K, N))], (1,), lambda x, w: x @ w, True),
  "matrix @ vectors": ([("w", (N, K)), ("x", (K,))], (0,), lambda w, x: w @ x, True),
  "matrices @ matrix": ([("a", (R, K)), ("w", (K, N))], (1,), lambda a, w: a @ w, True),
  "matrix @ matrices": ([("w", (N, K)), ("a", (K, R))], (0,), lambda w, a: w @ a, True),
  "matrices @ vector": ([("a", (R, K)), ("p", (K,))], (1,), lambda a, p: a @ p, False),
  "vector @ matrices": ([("p", (K,)), ("a", (K, R))], (0,), lambda p, a: p @ a, False),
  "vectors @ vector": ([("x", (K,)), ("p", (K,))], (1,), lambda x, p: (x @ p).reshape((1,)), False),
  "vector @ vectors": ([("p", (K,)), ("x", (K,))], (0,), lambda p, x: (p @ x).reshape((1,)), False),
}

# A product that gains whatever stands beside it: a mapped vector by a shared matrix of more
# multiply-adds than the bodies below hold otherwise, so that they are batched and every form is run.
CARRIER = [("cx", (6,)), ("cw", (60, 6))]


def _body(name: str, inputs, build, *, carried: bool = False) -> sc.Function:
  """A body of ``build`` over ``inputs``, flattened; ``carried`` appends the carrier product."""
  syms = [sc.sym(n, shape) for n, shape in inputs]
  out = build(*syms)
  if carried:
    cx, cw = (sc.sym(n, shape) for n, shape in CARRIER)
    out = sc.concat([out.reshape((out.size,)), cw @ cx])
    syms, inputs = [*syms, cx, cw], [*inputs, *CARRIER]
  return sc.Function.from_exprs(name, syms, [out], [n for n, _ in inputs], ["y"])


@pytest.mark.parametrize("form", sorted(PRODUCTS))
@pytest.mark.parametrize("constant", [False, True], ids=["input", "constant"])
def test_a_product_with_an_operand_every_trip_shares_is_one_product_over_the_trips(form: str, constant: bool) -> None:
  """Each of the eight shapes a product of a mapped value and a shared one takes, as one product
  over all the trips' rows or columns, with the shared operand an input and a constant (a constant
  matrix on the left is transposed when the code is generated): the body called once a trip gives
  the same numbers. A product by a shared matrix is batched for its own sake; one by a shared
  vector gains nothing alone, and is batched beside a product that does."""
  inputs, shared, build, gains = PRODUCTS[form]
  tag = form.replace(" @ ", "_").replace(" ", "_") + ("_const" if constant else "")
  if constant:
    (at,) = shared
    value = np.random.default_rng(len(form)).standard_normal(inputs[at][1])
    inner = build
    build = lambda *syms: inner(*syms[:at], sc.const(value), *syms[at:])  # noqa: E731
    inputs, shared = [spec for k, spec in enumerate(inputs) if k != at], ()
  alone = _body(f"alone_{tag}", inputs, build)
  fn, values, per_trip = _mapped(f"map_alone_{tag}", alone, shared)
  _check(fn, alone, values, per_trip, batched=gains)
  carried = _body(f"carried_{tag}", inputs, build, carried=True)
  fn, values, per_trip = _mapped(f"map_carried_{tag}", carried, (*shared, len(inputs) + 1))
  _check(fn, carried, values, per_trip, batched=True)


def test_a_network_and_its_jacobian_mapped_over_nodes_is_batched() -> None:
  """The shape the pass was written for: an MLP with constant weights and its forward-mode
  Jacobian at each node. Every layer is one product over the nodes' values and one over their
  tangents, and the result is the node function's, node by node."""
  from scaly import nn

  rng = np.random.default_rng(3)
  widths = [3, 12, 12, 3]
  net = [(rng.standard_normal((o, i)) / np.sqrt(i), rng.standard_normal(o) * 0.1) for i, o in zip(widths[:-1], widths[1:], strict=True)]
  a = sc.sym("a", 3)
  y = nn.mlp(a, net)
  body = sc.Function.from_exprs("net_node", [a], [sc.concat([a, y, sc.jacobian(y, a).T.reshape((9,))])], ["a"], ["y"])
  fn, values, per_trip = _mapped("net_nodes", body)
  _check(fn, body, values, per_trip, batched=True)


def test_elementwise_ops_run_over_the_trips_and_broadcast_against_shared_operands() -> None:
  """A unary op, binary ops between two mapped values of different rank, between a mapped value and
  a shared one of higher rank, a comparison and the select on it, and a cast, around a product."""
  x, a, p, w, g = sc.sym("x", K), sc.sym("a", (R, K)), sc.sym("p", K), sc.sym("w", (N, K)), sc.sym("g", (R, K))
  h = (w @ x).tanh()  # (N,)
  mixed = a * x + g  # a mapped matrix, a mapped vector across its columns, a shared matrix
  chosen = sc.where(x < p, x * p, -x)  # mapped against shared
  flags = sc.cast(sc.logical_and(x < 0.0, p < 0.5), "float64")
  out = sc.concat([h, mixed.reshape((R * K,)), chosen, flags, (x.sum() * 0.0 + x)[1:3]])
  body = sc.Function.from_exprs("elementwise_body", [x, a, p, w, g], [out], ["x", "a", "p", "w", "g"], ["y"])
  fn, values, per_trip = _mapped("elementwise_map", body, shared=(2, 3, 4))
  # x.sum() has no batched form: the whole map keeps its loop, and still computes the same.
  _check(fn, body, values, per_trip, batched=False)
  out = sc.concat([h, mixed.reshape((R * K,)), chosen, flags, x[1:3]])
  body = sc.Function.from_exprs("elementwise_body_2", [x, a, p, w, g], [out], ["x", "a", "p", "w", "g"], ["y"])
  fn, values, per_trip = _mapped("elementwise_map_2", body, shared=(2, 3, 4))
  _check(fn, body, values, per_trip, batched=True)


def test_reshape_transpose_slice_stack_and_concat_keep_the_trips_in_front() -> None:
  a, w = sc.sym("a", (R, K)), sc.sym("w", (K, N))
  prod = a @ w  # (R, N)
  parts = [
    prod.T.reshape((N * R,)),
    prod[1],
    prod[:, 2],
    prod[0:2, 1:8:3].reshape((6,)),
    sc.stack([prod[0], prod[2]], axis=1).reshape((2 * N,)),
    sc.stack([prod[0], prod[2]], axis=0).reshape((2 * N,)),
    sc.concat([prod, prod * 2.0], axis=1).reshape((2 * R * N,)),
    sc.concat([prod, prod * 2.0], axis=0).reshape((2 * R * N,)),
  ]
  body = sc.Function.from_exprs("moves_body", [a, w], [sc.concat(parts)], ["a", "w"], ["y"])
  fn, values, per_trip = _mapped("moves_map", body, shared=(1,))
  _check(fn, body, values, per_trip, batched=True)


def _kept(name: str, inputs, shared, build, trips: int = TRIPS) -> None:
  body = _body(f"{name}_body", inputs, build)
  fn, values, per_trip = _mapped(f"{name}_map", body, shared, trips)
  _check(fn, body, values, per_trip, batched=False)


def _taken(name: str, inputs, shared, build, trips: int = TRIPS) -> None:
  body = _body(f"{name}_body", inputs, build)
  fn, values, per_trip = _mapped(f"{name}_map", body, shared, trips)
  _check(fn, body, values, per_trip, batched=True)


def test_a_body_with_an_op_that_has_no_batched_form_keeps_its_loop() -> None:
  """A product of two mapped values, a reduction, a gather at run-time indices and a call each have
  no form over the trips: the map lowers as the loop it was, and computes the same."""
  wx = [("w", (N, K)), ("x", (K,))]
  _kept("both_mapped", [("a", (R, K)), ("x", (K,)), ("w", (N, K))], (2,), lambda a, x, w: sc.concat([a @ x, w @ x]))
  _kept("reduction", wx, (0,), lambda w, x: (w @ x).sum().reshape((1,)))
  inner = _body("inner_fn", [("v", (N,))], lambda v: v.tanh())
  _kept("call", wx, (0,), lambda w, x: inner(w @ x))
  _kept("stack_with_shared", [("w", (N, K)), ("x", (K,)), ("p", (N,))], (0, 2), lambda w, x, p: sc.concat([w @ x, p]))


def test_a_map_without_a_shared_matrix_to_multiply_keeps_its_loop() -> None:
  """Batching pays by reading a shared matrix once. A body of elementwise work alone, and one whose
  only products are of mapped values with a shared vector, keep their loops."""
  _kept("no_product", [("x", (K,)), ("p", (K,))], (1,), lambda x, p: (x * p).tanh())
  _kept("vector_only", [("a", (R, K)), ("p", (K,))], (1,), lambda a, p: a @ p)


def test_a_product_that_fills_the_tiles_one_trip_at_a_time_is_batched_only_past_the_cache() -> None:
  """A shared matrix times a mapped matrix of six columns runs in register tiles at every trip
  already, and batched it would run the same multiply-adds after copying the trips' columns
  together: the loop is kept. Unless the shared matrix is larger than half the level-1 data cache
  (64 KiB on the M3, 16 KiB on ``x86-64-v3``), which the loop reads from the level-2 cache at every
  trip: 16 by 16 fits both, 48 by 48 (18 KiB) the M3's alone, 96 by 96 (72 KiB) neither."""
  for width, batched_on in ((16, ()), (48, ("x86-64-v3",)), (96, ("apple-m3", "x86-64-v3"))):
    body = _body(f"tiled_{width}", [("w", (width, width)), ("t", (width, 6))], lambda w, t: (w @ t).reshape((width * 6,)))
    fn, values, per_trip = _mapped(f"map_tiled_{width}", body, (0,))
    for target in ("apple-m3", "x86-64-v3"):
      _check(fn, body, values, per_trip, batched=target in batched_on, targets=(target,))
    # The same with the mapped matrix on the left: six rows of it fill the tiles at every trip.
    body = _body(f"tiled_left_{width}", [("t", (6, width)), ("w", (width, width))], lambda t, w: (t @ w).reshape((6 * width,)))
    fn, values, per_trip = _mapped(f"map_tiled_left_{width}", body, (1,))
    for target in ("apple-m3", "x86-64-v3"):
      _check(fn, body, values, per_trip, batched=target in batched_on, targets=(target,))


def test_a_body_is_batched_when_the_products_that_gain_are_half_of_its_products() -> None:
  """A network's value is a vector a trip, and its product gains; its six tangent columns fill the
  tiles at every trip, and theirs does not. With the tangents six times the value's multiply-adds
  the loop is kept; with the value's product the larger part, the body is batched."""
  w, x, t = [("w", (16, 16)), ("x", (16,)), ("t", (16, 6))], None, None
  del x, t
  _kept("mostly_tiled", w, (0,), lambda w, x, t: sc.concat([w @ x, (w @ t).reshape((96,))]))
  wide = [("w", (16, 16)), ("x", (16,)), ("v", (96, 16)), ("y", (16,))]
  _taken("mostly_gaining", [*wide, ("t", (16, 6))], (0, 2), lambda w, x, v, y, t: sc.concat([w @ x, v @ y, (w @ t).reshape((96,))]))


def test_a_product_too_small_for_the_tiles_keeps_the_loop() -> None:
  """The batched product has to fill a tile: four rows, a reduction of four terms, and more than
  four columns (half a tile's on the M3). With a constant matrix the trips are its rows and the
  matrix's rows its columns; with a matrix that is an input, the other way around. Below any of
  the three the batched product is no better than the loop's."""
  x4, x3 = [("x", (4,))], [("x", (3,))]
  wx = lambda x, w: w @ x  # noqa: E731
  for trips, rows, k, taken in ((5, 4, 4, True), (4, 4, 4, False), (5, 3, 4, False), (5, 4, 3, False), (1, 9, 4, False)):
    case = _taken if taken else _kept
    case(f"input_{trips}_{rows}_{k}", [*(x4 if k == 4 else x3), ("w", (rows, k))], (1,), wx, trips=trips)
  rng = np.random.default_rng(8)
  for trips, rows, k, taken in ((4, 5, 4, True), (3, 5, 4, False), (4, 4, 4, False), (4, 5, 3, False)):
    case = _taken if taken else _kept
    value = rng.standard_normal((rows, k))
    case(f"constant_{trips}_{rows}_{k}", x4 if k == 4 else x3, (), lambda x, value=value: sc.const(value) @ x, trips=trips)


def test_gaining_products_of_exactly_half_the_multiply_adds_are_enough_and_one_trip_is_not_a_map() -> None:
  half = [("v", (80, 16)), ("y", (16,)), ("w", (16, 16)), ("t", (16, 5))]  # 1 280 that gain, 1 280 that do not
  _taken("exactly_half", half, (0, 2), lambda v, y, w, t: sc.concat([v @ y, (w @ t).reshape((80,))]))
  less = [("v", (79, 16)), ("y", (16,)), ("w", (16, 16)), ("t", (16, 5))]
  _kept("under_half", less, (0, 2), lambda v, y, w, t: sc.concat([v @ y, (w @ t).reshape((80,))]))
  # One trip of a product past the cache, wide enough to fill the tiles: there is nothing to share.
  value = np.random.default_rng(4).standard_normal((96, 96))
  _kept("one_trip_past_the_cache", [("t", (96, 6))], (), lambda t: (sc.const(value) @ t).reshape((576,)), trips=1)
  _taken("two_trips_past_the_cache", [("t", (96, 6))], (), lambda t: (sc.const(value) @ t).reshape((576,)), trips=2)


def test_windows_start_where_the_map_says_and_a_batched_map_feeds_what_follows() -> None:
  """The mapped and the shared inputs are windows of larger arrays, at offsets; and the batched
  result is an operand like any other, of an elementwise op and of a second map, which is batched
  in its turn."""
  x, w = sc.sym("x", K), sc.sym("w", (N, K))
  layer = sc.Function.from_exprs("offset_layer", [x, w], [(w @ x).tanh()], ["x", "w"], ["y"])
  y, v = sc.sym("y", N), sc.sym("v", (K, N))
  back = sc.Function.from_exprs("offset_back", [y, v], [v @ y], ["y", "v"], ["x"])
  xs, ws, vs = sc.sym("xs", 3 + TRIPS * K + 2), sc.sym("ws", 4 + N * K), sc.sym("vs", K * N + 1)
  first = sc.vmap(layer, TRIPS, [(xs, 3, K), (ws, 4, 0)])
  second = sc.vmap(back, TRIPS, [(first * 2.0 + 1.0, 0, N), (vs, 1, 0)])
  fn = sc.Function.from_exprs("offset_maps", [xs, ws, vs], [second, first.sum()], ["xs", "ws", "vs"], ["z", "s"])
  xv, wv, vv = RNG.standard_normal(xs.size), RNG.standard_normal(ws.size), RNG.standard_normal(vs.size)
  h = np.tanh(xv[3 : 3 + TRIPS * K].reshape(TRIPS, K) @ wv[4:].reshape(N, K).T)
  for target in ("apple-m3", "generic"):
    with sc.target(target):
      z, total = fn((xv, wv, vv))
    np.testing.assert_allclose(np.asarray(z).reshape(TRIPS, K), (h * 2.0 + 1.0) @ vv[1:].reshape(K, N).T, rtol=1e-13, atol=1e-13)
    np.testing.assert_allclose(total, h.sum(), rtol=1e-13)
  assert _calls(fn) == 0 and _calls(fn, "generic") == 2


def test_a_body_asked_to_be_scalar_or_in_blocks_keeps_its_procedure() -> None:
  """A lowering hint on the body is a request about its own procedure, and is kept."""
  _taken("no_hint", [("w", (9, 5)), ("x", (5,))], (0,), lambda w, x: w @ x)
  _kept("block_hint", [("w", (9, 5)), ("x", (5,))], (0,), lambda w, x: (w @ x).block())
  _kept("scalar_hint", [("w", (9, 5)), ("x", (5,))], (0,), lambda w, x: (w @ x).scalar())


def test_portable_rounding_batches_as_the_reference_machine_does() -> None:
  """Whether a map is batched reads the target's tile and its cache, which are choices like any
  other: under ``rounding="portable"`` they are the reference machine's, and the C is the same for
  every target."""
  from scaly.codegen import render_c_source

  body = _body("portable_body", [("w", (9, 5)), ("x", (5,))], lambda w, x: (w @ x).tanh())
  fn, _, _ = _mapped("portable_map", body, (0,))
  reference = render_c_source(fn, target=sc.Target.preset("apple-m3", rounding="portable"))
  assert "portable_body" not in reference.split("portable_map", 1)[1]
  for name in ("generic", "x86-64-v3", "cortex-a53"):
    assert render_c_source(fn, target=sc.Target.preset(name, rounding="portable")) == reference, name
  assert "portable_body_raw(" in render_c_source(fn, target="generic")


@pytest.mark.parametrize("seed", range(40))
def test_random_bodies_batched_compute_what_the_body_does_trip_by_trip(seed: int) -> None:
  """A differential fuzz: a random body over a mapped vector, a mapped matrix and shared operands,
  from the ops that have a batched form, beside a product large enough for the map to be batched.
  Mapped, it gives what the body gives called once a trip."""
  rng = np.random.default_rng(1000 + seed)
  nx, r, c, k = (int(rng.integers(lo, hi)) for lo, hi in ((2, 7), (1, 4), (2, 6), (2, 7)))
  x, m, p, w, w2 = sc.sym("x", nx), sc.sym("m", (r, c)), sc.sym("p", nx), sc.sym("w", (k, nx)), sc.sym("w2", (c, k))
  unary = [lambda v: v.sin(), lambda v: v.tanh(), lambda v: -v, lambda v: v.abs(), lambda v: (v.abs() + 1.0).sqrt(), lambda v: (v * 0.1).exp()]
  pick = lambda pool: pool[int(rng.integers(len(pool)))]  # noqa: E731
  vecs, mats, scalars = [x, x * p, pick(unary)(x)], [m, pick(unary)(m)], []
  for _ in range(int(rng.integers(4, 14))):
    kind, v, a = int(rng.integers(13)), pick(vecs), pick(mats)
    if kind == 0:
      vecs.append(pick(unary)(v))
    elif kind == 1 and v.shape == (other := pick(vecs)).shape:
      vecs.append(v * other + 0.5 * v)
    elif kind == 2 and v.shape == (nx,):
      vecs.append(w @ v)
    elif kind == 3 and v.shape == (k,):
      vecs.append(v @ w)
    elif kind == 4 and a.shape[1] == c:
      mats.append(a @ w2)
    elif kind == 5 and a.shape[0] in (nx, k):
      mats.append(w @ a if a.shape[0] == nx else w2 @ a)
    elif kind == 6 and nx in a.shape:
      vecs.append(a @ p if a.shape[1] == nx else p @ a)
    elif kind == 7 and v.shape == (nx,):
      scalars += [v @ p, p @ v]
    elif kind == 8:
      mats.append(a.T)
    elif kind == 9:
      vecs += [a.reshape((a.size,)), a[int(rng.integers(a.shape[0]))], a[:, int(rng.integers(a.shape[1]))]]
    elif kind == 10 and v.size > 1:
      lo = int(rng.integers(0, v.size - 1))
      vecs.append(v[lo:])
      scalars.append(v[lo])
    elif kind == 11:
      mats.append(sc.stack([v, pick(unary)(v)], axis=int(rng.integers(2))))
      vecs.append(sc.concat([v, v * 2.0]))
    elif kind == 12:
      vecs.append(sc.where(v < p, v, p * 2.0) if v.shape == (nx,) else sc.where(v < 0.0, -v, v * v))
      if a.shape[1] == v.shape[0]:
        mats.append(a * v + (scalars[-1] if scalars else 1.0))
  parts = [v.reshape((v.size,)) for v in (*vecs[-3:], *mats[-3:])] + [s.reshape((1,)) for s in scalars[-2:]]
  # The carrier: more multiply-adds than every other product of the body together.
  others = sum(
    n.args[0].size * n.args[1].size // n.args[0].shape[-1]
    for n in sc.Function.from_exprs("probe", [x, m, p, w, w2], [sc.concat(parts)]).concrete.outputs[0:1]
    for n in _exprs(n)
  )
  cx, cw = sc.sym("cx", 6), sc.sym("cw", (others // 6 + 8, 6))
  body = sc.Function.from_exprs(
    f"fuzz_body_{seed}", [x, m, p, w, w2, cx, cw], [sc.concat([*parts, cw @ cx])], ["x", "m", "p", "w", "w2", "cx", "cw"], ["y"]
  )
  fn, values, per_trip = _mapped(f"fuzz_map_{seed}", body, (2, 3, 4, 6), trips=int(rng.integers(5, 12)))
  _check(fn, body, values, per_trip, batched=True, targets=("apple-m3", "generic"))


def _exprs(root):
  """The products under ``root``."""
  from scaly.ir.expr import ExprOp, topo

  return [n for n in topo((root,)) if n.op == ExprOp.MATMUL]


def test_windows_that_overlap_and_an_output_no_trip_changes_keep_the_loop() -> None:
  x, w = sc.sym("x", K), sc.sym("w", (N, K))
  body = sc.Function.from_exprs("window_body", [x, w], [w @ x], ["x", "w"], ["y"])
  xs = sc.sym("xs", TRIPS + K - 1)
  ws = sc.sym("ws", (N, K))
  fn = sc.Function.from_exprs("window_map", [xs, ws], [sc.vmap(body, TRIPS, [(xs, 0, 1), (ws.reshape((N * K,)), 0, 0)])], ["xs", "ws"], ["y"])
  xv, wv = RNG.standard_normal(TRIPS + K - 1), RNG.standard_normal((N, K))
  assert _calls(fn) > 0
  np.testing.assert_allclose(np.asarray(fn((xv, wv))).reshape(TRIPS, N), np.stack([wv @ xv[i : i + K] for i in range(TRIPS)]), rtol=1e-13, atol=1e-13)
  # Two outputs, one of which reads shared inputs only: every trip writes the same, and the map of that one keeps its loop.
  p = sc.sym("p", K)
  two = sc.Function.from_exprs("two_body", [x, w, p], [w @ x, w @ p], ["x", "w", "p"], ["y", "z"])
  xs2, ps = sc.sym("xs", TRIPS * K), sc.sym("ps", K)
  specs = [(xs2, 0, K), (ws.reshape((N * K,)), 0, 0), (ps, 0, 0)]
  both = sc.Function.from_exprs(
    "two_map", [xs2, ws, ps], [sc.vmap(two, TRIPS, specs, output=0), sc.vmap(two, TRIPS, specs, output=1)], ["xs", "ws", "ps"], ["y", "z"]
  )
  xv2, pv = RNG.standard_normal(TRIPS * K), RNG.standard_normal(K)
  y, z = both((xv2, wv, pv))
  np.testing.assert_allclose(np.asarray(y).reshape(TRIPS, N), xv2.reshape(TRIPS, K) @ wv.T, rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(np.asarray(z).reshape(TRIPS, N), np.tile(wv @ pv, (TRIPS, 1)), rtol=1e-13, atol=1e-13)
  assert _calls(both) > 0


def test_the_outputs_of_one_map_share_one_batched_body() -> None:
  """Two outputs of one body, each a ``VMAP`` node of its own: batched once, the product they
  share is computed once, where the two loops called the body twice a trip."""
  x, w = sc.sym("x", K), sc.sym("w", (N, K))
  h = w @ x
  two = sc.Function.from_exprs("shared_body", [x, w], [h.tanh(), h * h], ["x", "w"], ["y", "z"])
  xs, ws = sc.sym("xs", TRIPS * K), sc.sym("ws", (N, K))
  specs = [(xs, 0, K), (ws.reshape((N * K,)), 0, 0)]
  fn = sc.Function.from_exprs(
    "shared_map", [xs, ws], [sc.vmap(two, TRIPS, specs, output=0), sc.vmap(two, TRIPS, specs, output=1)], ["xs", "ws"], ["y", "z"]
  )
  xv, wv = RNG.standard_normal(TRIPS * K), RNG.standard_normal((N, K))
  y, z = fn((xv, wv))
  hv = xv.reshape(TRIPS, K) @ wv.T
  np.testing.assert_allclose(np.asarray(y).reshape(TRIPS, N), np.tanh(hv), rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(np.asarray(z).reshape(TRIPS, N), hv * hv, rtol=1e-13, atol=1e-13)
  assert _calls(fn) == 0
  one = sc.Function.from_exprs("shared_one", [xs, ws], [sc.vmap(two, TRIPS, specs, output=0)], ["xs", "ws"], ["y"])
  assert _reductions(fn) == _reductions(one) > 0


def test_a_map_inside_a_loop_body_is_batched_where_the_body_is_lowered() -> None:
  """The pass runs on every Function as it is lowered, a scan's body included."""
  x, w, c = sc.sym("x", K), sc.sym("w", (K, K)), sc.sym("c", TRIPS * K)
  layer = sc.Function.from_exprs("scan_layer", [x, w], [(w @ x).tanh()], ["x", "w"], ["y"])
  wq = sc.sym("wq", (K, K))
  step = sc.Function.from_exprs("scan_step", [c, wq], [sc.vmap(layer, TRIPS, [(c, 0, K), (wq.reshape((K * K,)), 0, 0)])], ["c", "wq"], ["c"])
  c0, ws = sc.sym("c0", TRIPS * K), sc.sym("ws", (K, K))
  fn = sc.Function.from_exprs("scan_of_maps", [c0, ws], [sc.scan(step, c0, [(ws.reshape((K * K,)), 0, 0)], length=3)[0]], ["c0", "ws"], ["c"])
  cv, wv = RNG.standard_normal(TRIPS * K), RNG.standard_normal((K, K)) * 0.5
  want = cv.reshape(TRIPS, K)
  for _ in range(3):
    want = np.tanh(want @ wv.T)
  np.testing.assert_allclose(np.asarray(fn((cv, wv))).reshape(TRIPS, K), want, rtol=1e-13, atol=1e-13)
  names = {n.attrs["callee"] for n in _walk(lower_function(fn, target="apple-m3")) if n.op == ProgramOp.CALL}
  assert names == {"scan_step"} or names == {"scan_step_inplace"}, names


def _reductions(fn: sc.Function) -> int:
  """The reductions over ``k`` in ``fn``'s lowered program: one for each tile of a product."""
  return len({id(n) for n in _walk(lower_function(fn, target="apple-m3")) if n.op == ProgramOp.RANGE and n.attrs["kind"] == "reduce"})


def _walk(node):
  stack = [node]
  out = []
  while stack:
    n = stack.pop()
    out.append(n)
    stack.extend(n.args)
  return out
