"""Sparse Jacobians and Hessians of models written over the edges of a graph (``ad/sparse.py``):
derivatives with respect to the gathered operands, and the constant part of a Jacobian computed
when the graph is built, each against the dense derivative."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

import scaly as sc
from scaly.ad import sparse as sparse_module
from scaly.ad.sparse import SparseJacobian, sparse_hessian, sparse_jacobian_colored
from scaly.ad.sparsity import column_coloring, star_coloring


def _network(nodes: int, hub: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
  """The two ends of each edge of a connected graph: a random tree, some chords, and node 0 joined
  to ``hub`` others, so that its degree is what a coloring of the columns has to pay for."""
  rng = np.random.default_rng(seed)
  edges = {(int(rng.integers(0, k)), k) for k in range(1, nodes)}
  edges |= {(0, int(k)) for k in rng.choice(np.arange(1, nodes), hub, replace=False)}
  while len(edges) < nodes + hub + nodes // 3:
    a, b = sorted(int(v) for v in rng.integers(0, nodes, 2))
    if a != b:
      edges.add((a, b))
  f, t = (np.array(v) for v in zip(*sorted(edges), strict=True))
  return f, t


def _dense(sj: SparseJacobian, inputs: list[sc.Expr], values: list[np.ndarray], name: str) -> np.ndarray:
  """The matrix a compact derivative stands for, at ``values``."""
  fn = sc.Function.from_exprs(name, inputs, [sj.values], [str(e.name) for e in inputs], ["v"])
  got = np.asarray(fn(values[0]) if len(values) == 1 else fn(tuple(values))).reshape(-1)
  out = np.zeros(sj.sparsity.shape)
  np.add.at(out, (np.asarray(sj.sparsity.rows), np.asarray(sj.sparsity.cols)), got)
  return out


def _reference(expr: sc.Expr, wrt: sc.Expr, inputs: list[sc.Expr], values: list[np.ndarray], name: str, *, second: bool) -> np.ndarray:
  dense = sc.hessian(expr, wrt) if second else sc.jacobian(expr, wrt)
  fn = sc.Function.from_exprs(name, inputs, [dense], [str(e.name) for e in inputs], ["d"])
  return np.asarray(fn(values[0]) if len(values) == 1 else fn(tuple(values))).reshape(dense.shape)


def _plain(build, *args, **kwargs) -> SparseJacobian:
  """``build`` with neither the cut at the selections nor the linear part: one coloring of ``wrt``."""
  with (
    mock.patch.object(sparse_module, "_at_selections", lambda expr, wrt: None),
    mock.patch.object(sparse_module, "_linear_part", lambda expr, wrt: (None, expr)),
  ):
    return build(*args, **kwargs)


@pytest.mark.parametrize(("nodes", "hub"), [(12, 6), (40, 25), (90, 60)])
def test_a_hessian_over_edges_takes_the_colors_of_an_edge_not_of_the_largest_degree(nodes: int, hub: int) -> None:
  """An energy summed over the edges of a graph, each term a function of the values at the edge's
  two ends, read by ``gather``. The Hessian's column for a node couples with every neighbour's, so
  star-coloring the nodes takes more colors the larger the hub. With respect to the gathered
  operands each term is one small block, and four colors do whatever the degree: the Hessian is
  that one with its entries added up where they belong, the same matrix as the dense one."""
  f, t = _network(nodes, hub, nodes)
  rng = np.random.default_rng(nodes)
  x, w = sc.sym("x", nodes), sc.sym("w", len(f))
  xf, xt = sc.gather(x, f), sc.gather(x, t)
  energy = (w * (xf - xt).cos() * xf * xt).sum() + (sc.const(rng.uniform(0.5, 1.5, nodes)) * x * x * x).sum()
  cut = sparse_hessian(energy, x)
  plain = _plain(sparse_hessian, energy, x)
  assert cut.coloring_width is not None and plain.coloring_width is not None
  assert cut.coloring_width <= 4 < plain.coloring_width
  assert sorted(zip(cut.sparsity.rows, cut.sparsity.cols, strict=True)) == sorted(zip(plain.sparsity.rows, plain.sparsity.cols, strict=True))
  values = [rng.uniform(0.5, 1.5, nodes), rng.uniform(0.5, 1.5, len(f))]
  want = _reference(energy, x, [x, w], values, f"edge_hess_dense_{nodes}", second=True)
  assert np.abs(want).max() > 0.5
  for label, sj in (
    ("cut", cut),
    ("plain", plain),
    ("lower", sparse_hessian(energy, x, triangle="lower")),
    ("upper", sparse_hessian(energy, x, triangle="upper")),
  ):
    got = _dense(sj, [x, w], values, f"edge_hess_{label}_{nodes}")
    expected = want if label in ("cut", "plain") else np.tril(want) if label == "lower" else np.triu(want)
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-12, err_msg=label)


def test_selections_of_every_kind_and_a_direct_read_are_cut_together() -> None:
  """A matrix variable read through a slice, a transpose, a reshape, a gather of a slice and
  directly: each is a selection of the same entries, and the Hessian with respect to all of them
  at once, added up, is the Hessian."""
  rng = np.random.default_rng(5)
  x = sc.sym("x", (4, 3))
  rows, columns = x[1:3], x.T[2]  # two rows; the last column
  flat = x.reshape((12,))
  picked = sc.gather(flat[2:], np.array([0, 0, 3, 7, 9, 3]))
  energy = (
    (rows * rows.sin()).sum()
    + (columns * columns * picked[:4]).sum()
    + (picked.exp() * sc.gather(x, np.array([11, 0, 5, 5, 2, 1]))).sum()
    + (x * x * x).sum()
  )
  cut = sparse_hessian(energy, x)
  values = [rng.uniform(0.5, 1.5, (4, 3))]
  want = _reference(energy, x, [x], values, "kinds_dense", second=True)
  np.testing.assert_allclose(_dense(cut, [x], values, "kinds_cut"), want.reshape(12, 12), rtol=1e-12, atol=1e-12)
  found = sparse_module._at_selections(energy, x)
  assert found is not None
  _, at, index = found
  assert at.size == index.size > x.size and set(index.tolist()) == set(range(12))  # every entry, some several times


def test_selections_that_read_each_entry_once_are_not_cut() -> None:
  """Disjoint slices are the variable in another order: the same pattern, the same colors, and no
  cut. The stage variables of a transcription are read this way."""
  x = sc.sym("x", 12)
  a, b, c = x[:4], x[4:8], sc.gather(x, np.array([11, 10, 9, 8]))
  energy = (a * b * c).sum()
  assert sparse_module._at_selections(energy, x) is None
  sj = sparse_hessian(energy, x)
  assert sj.coloring_width == _plain(sparse_hessian, energy, x).coloring_width


def test_a_cut_that_saves_no_color_is_not_taken() -> None:
  """One edge whose two ends are each read twice: the operands take two colors, and so do the two
  nodes, so the Hessian is colored as before. The cut exists (an entry is read twice); it is the
  comparison of colors that turns it down."""
  x = sc.sym("x", 2)
  a, b = sc.gather(x, np.array([0, 0])), sc.gather(x, np.array([1, 1]))
  energy = (a * b).sum()
  assert sparse_module._at_selections(energy, x) is not None
  assert sparse_module._sparse_hessian_at_selections(energy, x) is None
  np.testing.assert_allclose(_dense(sparse_hessian(energy, x), [x], [np.array([2.0, 3.0])], "no_gain"), [[0.0, 2.0], [2.0, 0.0]])
  # The same for a Jacobian: each row reads both entries, two colors over the operands and over x.
  rows = a * b.sin()
  assert sparse_module._sparse_jacobian_at_selections(rows, x) is None
  v = np.array([2.0, 3.0])
  np.testing.assert_allclose(_dense(sparse_jacobian_colored(rows, x), [x], [v], "no_gain_jac"), [[np.sin(3.0), 2.0 * np.cos(3.0)]] * 2)


def test_a_linear_part_that_saves_no_color_is_left_in() -> None:
  """The linear rows couple the same pairs of columns as the others: two colors with them or
  without, so the Jacobian is colored whole, as before."""
  x = sc.sym("x", 4)
  g = sc.concat([x[:2] * x[2:], x[:2] + 3.0 * x[2:]])
  assert sparse_module._linear_part(g, x)[0] is not None
  assert sparse_module._sparse_jacobian_linear_part(g, x) is None
  sj = sparse_jacobian_colored(g, x)
  assert sj.coloring_width == 2
  v = np.array([1.0, 2.0, 3.0, 4.0])
  want = np.array([[3.0, 0, 1.0, 0], [0, 4.0, 0, 2.0], [1.0, 0, 3.0, 0], [0, 1.0, 0, 3.0]])
  np.testing.assert_allclose(_dense(sj, [x], [v], "linear_left_in"), want)


@pytest.mark.parametrize(("nodes", "hub"), [(12, 6), (40, 25), (90, 60)])
def test_a_jacobian_takes_its_constant_part_at_build_time_and_colors_the_rest_over_edges(nodes: int, hub: int) -> None:
  """Constraints of a network: a balance at each node that sums the flows of its edges (linear in
  the flows, with as many entries a row as the node's degree), plus a term in the node's own
  value; and for each edge an equation between its flow and the values at its ends. Coloring the
  columns pays the hub's degree for the balance rows, all of whose flow entries are constants.
  They are computed here; the rest is colored over the edges' operands, four colors."""
  f, t = _network(nodes, hub, 100 + nodes)
  edges = len(f)
  rng = np.random.default_rng(nodes)
  z, p = sc.sym("z", nodes + 2 * edges), sc.sym("p", edges)  # the nodes' values, then two flows an edge
  zx, zflow = z[:nodes], z[nodes:]
  xf, xt = sc.gather(zx, f), sc.gather(zx, t)
  arc_node = np.concatenate([f, t])
  balance = sc.segment_sum(zflow, arc_node, nodes) * 2.0 - sc.const(rng.uniform(0.5, 1.5, nodes)) * zx * zx + sc.const(np.ones(nodes))
  forward = zflow[:edges] - p * xf * xt * (xf - xt).cos()
  backward = zflow[edges:] / sc.const(rng.uniform(1.0, 2.0, edges)) + xf * (xt - xf).sin()
  window = xf - xt
  g = sc.concat([balance, forward, backward, window])
  taken = sparse_jacobian_colored(g, z)
  plain = _plain(sparse_jacobian_colored, g, z)
  assert taken.coloring_width is not None and plain.coloring_width is not None
  assert taken.coloring_width <= 4 and plain.coloring_width > hub
  values = [rng.uniform(0.5, 1.5, z.size), rng.uniform(0.5, 1.5, edges)]
  want = _reference(g, z, [z, p], values, f"net_jac_dense_{nodes}", second=False)
  np.testing.assert_allclose(_dense(taken, [z, p], values, f"net_jac_taken_{nodes}"), want, rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(_dense(plain, [z, p], values, f"net_jac_plain_{nodes}"), want, rtol=1e-12, atol=1e-12)
  # The pattern is the plain one, less nothing: every constant entry is there.
  assert sorted(zip(taken.sparsity.rows, taken.sparsity.cols, strict=True)) == sorted(zip(plain.sparsity.rows, plain.sparsity.cols, strict=True))


def test_the_linear_part_goes_through_every_linear_op_and_stops_at_a_coefficient_that_is_not_a_constant() -> None:
  """The linear part of an expression, as a matrix: selections, sums and differences with
  broadcasting, negation, products with and division by constants on either side, ``segment_sum``,
  ``sum``, ``concat`` and ``stack`` on either axis. A product with a parameter is linear, but its
  coefficient is not known here, so it stays in the rest, as a product of two variables does."""
  rng = np.random.default_rng(9)
  x, q = sc.sym("x", (3, 4)), sc.sym("q", 4)
  c = sc.const(rng.uniform(1.0, 2.0, 4))
  rows = x[0] * c - (c * x[2]) / sc.const(np.array([2.0, 4.0, 5.0, 8.0])) + 1.5  # (4,)
  spread = x - x[1]  # (3, 4) minus a row, broadcast
  parts = [
    rows,
    -sc.segment_sum(x.reshape((12,)), np.arange(12) % 4, 4),
    sc.stack([x[:, 0], x[:, 3]], axis=1).reshape((6,)),
    sc.concat([x.T, x.T * 3.0], axis=0).reshape((24,)),
    spread.reshape((12,)),
    (x[2] * 0.5 + x).reshape((12,)),  # the smaller operand first
    x.sum().reshape((1,)),
    q * x[1],  # a parameter's coefficient: not a constant
    x[0] * x[1],  # a product of two variables
    sc.gather(x, np.array([5, 5, 0, 11])).sin(),
  ]
  g = sc.concat(parts)
  matrix, rest = sparse_module._linear_part(g, x)
  assert matrix is not None and rest is not None
  values = [rng.uniform(0.5, 1.5, (3, 4)), rng.uniform(0.5, 1.5, 4)]
  want = _reference(g, x, [x, q], values, "linear_dense", second=False).reshape(g.size, 12)
  rest_jac = _reference(rest, x, [x, q], values, "linear_rest_dense", second=False).reshape(g.size, 12)
  np.testing.assert_allclose(matrix.toarray() + rest_jac, want, rtol=1e-13, atol=1e-13)
  linear_rows = sum(part.size for part in parts[:7])
  np.testing.assert_array_equal(rest_jac[:linear_rows], 0.0)  # nothing linear is left in the rest
  np.testing.assert_array_equal(matrix.toarray()[linear_rows:], 0.0)  # and nothing else is in the matrix
  np.testing.assert_allclose(_dense(sparse_jacobian_colored(g, x), [x, q], values, "linear_taken"), want, rtol=1e-13, atol=1e-13)


def test_an_expression_that_is_all_linear_has_a_jacobian_of_constants() -> None:
  x = sc.sym("x", 6)
  g = sc.concat([x[:3] - x[3:], sc.segment_sum(x, np.array([0, 0, 1, 1, 1, 0]), 2) * 0.5])
  sj = sparse_jacobian_colored(g, x)
  assert sj.coloring_width == 0 and sj.values.op == "const"
  want = np.zeros((5, 6))
  want[[0, 1, 2], [0, 1, 2]], want[[0, 1, 2], [3, 4, 5]] = 1.0, -1.0
  want[3, [0, 1, 5]], want[4, [2, 3, 4]] = 0.5, 0.5
  np.testing.assert_array_equal(_dense(sj, [x], [np.arange(6.0)], "all_linear"), want)


def test_terms_that_cancel_leave_the_pattern() -> None:
  """``x - x`` has no Jacobian entry: the constant part is summed before the pattern is read. (The
  last row, a sum of every entry, is what makes the constant part worth taking out here.)"""
  x = sc.sym("x", 3)
  g = sc.concat([x - x + x[::-1] * x[::-1], x.sum().reshape((1,))])
  sj = sparse_jacobian_colored(g, x)
  assert sj.coloring_width == 1
  assert sorted(zip(sj.sparsity.rows, sj.sparsity.cols, strict=True)) == [(0, 2), (1, 1), (2, 0), (3, 0), (3, 1), (3, 2)]
  v = np.array([1.0, 2.0, 3.0])
  np.testing.assert_allclose(_dense(sj, [x], [v], "cancel"), np.vstack([np.diag(2.0 * v[::-1])[:, ::-1], np.ones((1, 3))]))


@pytest.mark.parametrize("seed", range(12))
def test_random_edge_models_match_the_dense_derivatives(seed: int) -> None:
  """A differential fuzz: random graphs, random terms over their edges and nodes, linear and not,
  the sparse Jacobian and the sparse Hessian of a weighted sum against the dense ones."""
  rng = np.random.default_rng(300 + seed)
  nodes = int(rng.integers(5, 14))
  f, t = _network(nodes, int(rng.integers(2, nodes - 1)), seed)
  edges = len(f)
  z = sc.sym("z", nodes + edges)  # a value at each node, then one at each edge
  zx, zy = z[:nodes], z[nodes:]
  ends = [sc.gather(zx, f), sc.gather(zx, t), sc.gather(zx, (f + t) % nodes)]
  unary = [lambda v: v.sin(), lambda v: v.tanh(), lambda v: v * v, lambda v: (v * 0.3).exp(), lambda v: -v]
  pick = lambda pool: pool[int(rng.integers(len(pool)))]  # noqa: E731
  terms = [zy, zy * 2.0 - pick(ends), sc.segment_sum(zy, f, nodes) - zx, sc.segment_sum(pick(unary)(pick(ends)), t, nodes)]
  for _ in range(int(rng.integers(2, 6))):
    a, b = pick(ends), pick(ends)
    terms.append(pick(unary)(a) * b + sc.const(rng.uniform(0.5, 1.5, edges)) * zy * pick([a, b, zy]))
    terms.append(pick(unary)(zx) - sc.segment_sum(a * b, pick([f, t]), nodes))
  g = sc.concat(terms)
  values = [rng.uniform(0.5, 1.5, z.size)]
  np.testing.assert_allclose(
    _dense(sparse_jacobian_colored(g, z), [z], values, f"fuzz_jac_{seed}"),
    _reference(g, z, [z], values, f"fuzz_jac_dense_{seed}", second=False),
    rtol=1e-11,
    atol=1e-11,
  )
  weights = sc.const(rng.uniform(-1.0, 1.0, g.size))
  scalar = (weights * g).sum()
  np.testing.assert_allclose(
    _dense(sparse_hessian(scalar, z), [z], values, f"fuzz_hess_{seed}"),
    _reference(scalar, z, [z], values, f"fuzz_hess_dense_{seed}", second=True),
    rtol=1e-10,
    atol=1e-10,
  )


def test_a_graph_deeper_than_the_interpreters_stack_is_split_without_recursion() -> None:
  """Four thousand terms in a chain, all linear: the linear part is found walking the graph in
  order, as every pass over a graph does, and the Jacobian is a constant (a recursive walk
  overflowed on a benchmark problem)."""
  x = sc.sym("x", 3)
  y = x * 2.0
  for k in range(4000):
    y = y + (x[::-1] if k % 2 else x) * 0.25
  sj = sparse_jacobian_colored(sc.concat([y, x.sum().reshape((1,))]), x)
  assert sj.coloring_width == 0
  want = np.vstack([502.0 * np.eye(3) + 500.0 * np.eye(3)[::-1], np.ones((1, 3))])
  np.testing.assert_allclose(_dense(sj, [x], [np.array([1.0, 2.0, 3.0])], "deep_linear"), want, rtol=1e-12)


def test_a_pattern_of_a_graph_deeper_than_the_interpreters_stack_is_found() -> None:
  """``jacobian_sparsity``'s rules ask for their arguments' patterns by calling back, which
  overflows on a chain of a few hundred ops (a multistage QP's Lagrangian, with a few nodes more
  under each of its selections, was one). The patterns are then filled in arguments first."""
  from scaly.ad.sparsity import _depends_on, jacobian_sparsity

  x = sc.sym("x", 3)
  y = x.sin()
  for k in range(1500):
    y = (y * x[::-1]) if k % 2 else y + x[k % 3]
  assert _depends_on(y, x, {})
  pattern = jacobian_sparsity(y, x)
  assert sorted(zip(pattern.rows, pattern.cols, strict=True)) == [(i, j) for i in range(3) for j in range(3)]


def test_the_colorings_compared_are_the_ones_named() -> None:
  """The cut is taken on fewer colors than a coloring of ``wrt`` itself would need: of its columns
  for a Jacobian, a star coloring for a Hessian."""
  f, t = _network(20, 12, 3)
  x = sc.sym("x", 20)
  xf, xt = sc.gather(x, f), sc.gather(x, t)
  hess = sparse_hessian((xf * xt * xf).sum(), x)
  jac = sparse_jacobian_colored(xf * xt.sin(), x)
  assert hess.coloring_width is not None and jac.coloring_width is not None
  assert hess.coloring_width < max(star_coloring(hess.sparsity)) + 1
  assert jac.coloring_width < max(column_coloring(jac.sparsity)) + 1
