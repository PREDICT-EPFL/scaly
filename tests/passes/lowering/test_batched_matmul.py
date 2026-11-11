"""NumPy matrix products across vector, broadcast and empty batch shapes."""

import numpy as np
import pytest

import scaly as sc


SHAPES = [
  ((3,), (3,)),
  ((3,), (2, 3, 4)),
  ((2, 4, 3), (3,)),
  ((2, 4, 3), (2, 3, 5)),
  ((1, 4, 3), (2, 3, 5)),
  ((2, 1, 4, 3), (3, 3, 5)),
  ((2, 1, 1, 4, 3), (3, 1, 3, 5)),
  ((0, 4, 3), (1, 3, 5)),
  ((2, 4, 0), (1, 0, 5)),
  ((2, 0, 3), (3, 5)),
  ((2, 4, 3), (3, 0)),
  ((1, 1, 1), (2, 1, 1)),
  ((0,), (2, 0, 4)),
]


@pytest.mark.parametrize("shapes", SHAPES)
@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_numpy(shapes, lowering):
  a, b = shapes

  @sc.function(sc.group(sc.arg("a", a), sc.arg("b", b)), outputs=sc.arg("out"))
  def f(inputs):
    x, y = inputs
    return sc.matmul(x, y).with_lowering(lowering)

  rng = np.random.default_rng(67)
  x, y = rng.normal(size=a), rng.normal(size=b)
  np.testing.assert_allclose(f((x, y)), np.matmul(x, y), atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("shapes", SHAPES)
@pytest.mark.parametrize("nseed", [1, 3])
def test_derivatives(shapes, nseed):
  from scaly.ad.reverse import vjp

  a, b = shapes
  rng = np.random.default_rng(68)
  x, y = rng.normal(size=a), rng.normal(size=b)
  dx, dy = rng.normal(size=(nseed, *a)), rng.normal(size=(nseed, *b))
  cot = rng.normal(size=np.matmul(x, y).shape)

  @sc.function(sc.group(sc.arg("a", a), sc.arg("b", b)), outputs=sc.group(sc.arg("dleft"), sc.arg("dright"), sc.arg("ra"), sc.arg("rb")))
  def f(inputs):
    a, b = inputs
    out = a @ b
    ra, rb = vjp([out], [a, b], [sc.const(cot)])
    return sc.jvp_many(out, a, sc.const(dx)), sc.jvp_many(out, b, sc.const(dy)), ra, rb

  da, db, ra, rb = f((x, y))
  expected_a = np.stack([np.matmul(seed, y) for seed in dx])
  expected_b = np.stack([np.matmul(x, seed) for seed in dy])
  np.testing.assert_allclose(da, expected_a, atol=1e-12)
  np.testing.assert_allclose(db, expected_b, atol=1e-12)
  for i in range(nseed):
    np.testing.assert_allclose(np.sum(ra * dx[i]), np.sum(cot * expected_a[i]), atol=1e-10)
    np.testing.assert_allclose(np.sum(rb * dy[i]), np.sum(cot * expected_b[i]), atol=1e-10)
  eps = 1e-5
  finite = (np.matmul(x + eps * dx[0], y + eps * dy[0]) - np.matmul(x - eps * dx[0], y - eps * dy[0])) / (2 * eps)
  np.testing.assert_allclose(da[0] + db[0], finite, atol=1e-9)


def test_seed_axis_is_one_matmul():
  from scaly.ir.expr import ExprOp, topo

  a, b = sc.sym("a", (2, 3)), sc.sym("b", (3, 4))
  d = sc.jvp_many(a @ b, a, sc.sym("seed", (5, 2, 3)))
  products = [e for e in topo([d]) if e.op == ExprOp.MATMUL]
  assert len(products) == 1
  assert products[0].shape == (5, 2, 4)


@pytest.mark.parametrize("shapes", SHAPES)
@pytest.mark.parametrize("left", [False, True])
def test_sparsity_finite_difference(shapes, left):
  from scaly.ad.sparsity import jacobian_sparsity

  a, b = shapes
  rng = np.random.default_rng(69)
  x, y = rng.normal(size=a), rng.normal(size=b)
  x[x < 0] = 0
  y[y < 0] = 0
  variable = sc.sym("variable", a if left else b)
  expression = variable @ sc.const(y) if left else sc.const(x) @ variable
  pattern = jacobian_sparsity(expression, variable)
  actual = np.zeros(pattern.shape, dtype=bool)
  actual[pattern.rows, pattern.cols] = True
  expected = np.zeros(pattern.shape, dtype=bool)
  for column in range(variable.size):
    delta = np.zeros(variable.shape)
    delta.reshape(-1)[column] = 1e-5
    high = np.matmul(x + delta, y) if left else np.matmul(x, y + delta)
    low = np.matmul(x - delta, y) if left else np.matmul(x, y - delta)
    expected[:, column] = np.abs((high - low).reshape(-1)) > 1e-10
  np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(2, 3, 4), (2, 1, 3, 4), (0, 3, 4)])
@pytest.mark.parametrize("left", [False, True])
def test_vector_folds(shape, left):
  from scaly.ir.expr import ExprOp
  from scaly.passes.expr import simplify

  x = sc.sym("x", shape)
  axes = (*range(len(shape) - 2), len(shape) - 1, len(shape) - 2)
  vector = sc.sym("v", (shape[-1] if left else shape[-2],))
  folded = simplify(vector @ x.transpose(axes) if left else x.transpose(axes) @ vector)
  if folded.size:
    assert folded.op == ExprOp.MATMUL
    assert all(arg.op != ExprOp.TRANSPOSE for arg in folded.args)

  @sc.function(sc.arg("x", shape), outputs=sc.arg("out"))
  def f(x):
    return sc.const(np.ones(shape[-2])) @ x if left else x @ sc.const(np.ones(shape[-1]))

  data = np.arange(np.prod(shape), dtype=float).reshape(shape)
  np.testing.assert_array_equal(f(data), data.sum(-2 if left else -1))


def test_sparse_batch_error():
  from scaly.ir.types import CanonicalPattern, TensorType

  pattern = CanonicalPattern((2, 3), (0, 1), (1, 2))
  x = sc.Expr(sc.ExprOp.INPUT, type=TensorType((4, 2, 3), pattern=pattern), name="x")
  with pytest.raises(NotImplementedError, match="#403"):
    x @ sc.sym("y", (3, 2))


def test_reverse_broadcast_reduces_batch_axes():
  from scaly.ad.reverse import vjp
  from scaly.ir.expr import ExprOp, topo

  x, y = sc.sym("x", (1, 2, 3)), sc.sym("y", (4, 3, 2))
  out = x @ y
  (dx,) = vjp([out], [x], [sc.sym("cot", out.shape)])
  nodes = topo([dx])
  assert any(e.op == ExprOp.REDUCE and e.attrs["axes"] == (0,) for e in nodes)
  assert all(e.op != ExprOp.SCATTER for e in nodes)


@pytest.mark.parametrize("shapes", [((), (2, 3)), ((2, 3), ()), ((2, 3), (4, 2)), ((2, 2, 3), (3, 3, 2))])
def test_invalid_shapes(shapes):
  with pytest.raises(ValueError):
    sc.matmul(sc.sym("a", shapes[0]), sc.sym("b", shapes[1]))


@pytest.mark.parametrize("lowering", ["scalar", "block"])
def test_reference_summation_order(lowering):
  @sc.function(sc.arg("a", (2, 2, 4)), sc.arg("b", (1, 4, 5)), outputs=sc.arg("out"))
  def f(a, b):
    return (a @ b).with_lowering(lowering)

  a = np.tile(np.array([1e16, 1, -1e16, 2.0]), (2, 2, 1))
  b = np.ones((1, 4, 5))
  b[..., 1, :] = 2
  expected = np.zeros((2, 2, 5))
  for k in range(4):
    expected += a[..., k, None] * b[..., k, :]
  np.testing.assert_array_equal(f(a, b), expected)


def test_verifier_checks_batched_result_shape():
  a, b = sc.sym("a", (2, 2, 3)), sc.sym("b", (1, 3, 4))
  sc.verify_expr(sc.matmul(a, b))
  malformed = sc.Expr(sc.ExprOp.MATMUL, (a, b), sc.TensorType((2, 2, 5)))
  with pytest.raises(sc.VerifyError, match="MATMUL shape mismatch"):
    sc.verify_expr(malformed)
