"""Sparse-sparse tangents and sampled adjoints agree with SciPy and finite differences."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.ad.forward import jvp_many
from scaly.ad.derivatives import finite_difference
from scaly.ad.reverse import vjp
from scaly.function.model import as_concrete
from scaly.ir.expr import ExprOp, matmul, substitute
from scaly.ir.program import ProgramOp, walk_program
from scaly.ir.types import CanonicalPattern
from scaly.passes.lowering import lower_function
from scaly.passes.lowering.plans import SampledProductPlan


@pytest.mark.parametrize(("left", "coordinate_mask"), [(True, False), (False, False), (False, True)])
def test_masked_sparse_product_singular_partials(left, coordinate_mask):
  pattern = CanonicalPattern((2, 2), (0, 0, 1, 1), (0, 1, 0, 1))
  coefficients = np.array([[-2.0, 2.0], [3.0, -3.0]])
  weights = np.array([[1.0, 2.0], [3.0, 5.0]])

  @sc.function(sc.arg("v", 4), sc.arg("c", 4 if coordinate_mask else ()), name=f"masked_product_{left}_{coordinate_mask}")
  def loss(v, c):
    x = sc.pack(v, pattern).sqrt()
    b = sc.const(sparse.coo_array(coefficients))
    product = x @ b if left else b @ x
    condition = sc.pack(c > 0, pattern) if coordinate_mask else c > 0
    selected = sc.where(condition, product, sc.pack(sc.const(np.zeros(4)), pattern))
    return (sc.values(selected) * sc.const(weights.ravel())).sum().block()

  grad = sc.gradient(loss, wrt="v")
  inactive = np.full(4, -1.0) if coordinate_mask else np.asarray(-1.0)
  np.testing.assert_array_equal(grad(np.zeros(4), inactive), np.zeros(4))
  point = np.array([0.4, 0.7, 1.1, 1.6])
  conditions = [inactive, np.ones(4) if coordinate_mask else np.asarray(1.0)]
  if coordinate_mask:
    conditions.append(np.array([1.0, -1.0, -1.0, 1.0]))

  def reference(v, c):
    x = np.sqrt(v.reshape(2, 2))
    product = x @ coefficients if left else coefficients @ x
    condition = np.asarray(c).reshape(2, 2) if coordinate_mask else c
    return (np.where(condition > 0, product, 0) * weights).sum()

  hess = sc.hessian(loss, wrt="v")

  @sc.function(sc.arg("v", 4), sc.arg("c", 4 if coordinate_mask else ()), name=f"masked_product_reverse_twice_{left}_{coordinate_mask}")
  def twice(v, c):
    output = as_concrete(loss).outputs[0]
    inputs = as_concrete(loss).inputs
    inner = vjp((output,), (inputs[0],), (sc.const(1.0),))[0]
    result = vjp((inner.sum(),), (inputs[0],), (sc.const(1.0),))[0]
    return substitute(result, dict(zip(inputs, (v, c), strict=True)))

  for c in conditions:
    np.testing.assert_allclose(grad(point, c), finite_difference(lambda v: reference(v, c), point).ravel(), atol=1e-8)
    selected_weights = np.where((c.reshape(2, 2) if coordinate_mask else c) > 0, weights, 0)
    factors = selected_weights @ coefficients.T if left else coefficients.T @ selected_weights
    expected_hess = finite_difference(lambda v: factors.ravel() * 0.5 / np.sqrt(v), point)
    np.testing.assert_allclose(hess(point, c), expected_hess, atol=1e-8)
    np.testing.assert_allclose(twice(point, c), expected_hess.sum(axis=0), atol=1e-8)


@pytest.mark.parametrize(("left", "lowering"), [(True, "scalar"), (False, "block")])
def test_masked_sparse_product_guards_infinite_coefficients_per_term(left, lowering):
  pattern = CanonicalPattern((2, 2), (0, 0, 1, 1), (0, 1, 0, 1))
  coefficients = np.array([[-2.0, 2.0, np.inf], [3.0, -3.0, -np.inf]])
  if not left:
    coefficients = coefficients.T
  coefficient_pattern = CanonicalPattern(coefficients.shape, *map(tuple, np.nonzero(coefficients)))
  output = pattern.matmul(coefficient_pattern) if left else coefficient_pattern.matmul(pattern)

  @sc.function(sc.arg("v", 4), sc.arg("c", output.nnz), name=f"masked_product_inf_{left}_{lowering}")
  def loss(v, c):
    x = sc.pack(v, pattern).sqrt()
    b = sc.const(sparse.coo_array(coefficients))
    product = x @ b if left else b @ x
    selected = sc.where(sc.pack(c > 0, output), product.sin(), sc.pack(sc.const(np.zeros(output.nnz)), output))
    weights = np.tile([1.0, 2.0, 1.0], (2, 1))
    return (sc.values(selected) * sc.const((weights if left else weights.T).ravel())).sum().with_lowering(lowering)

  mask = np.tile([1.0, 1.0, -1.0], (2, 1))
  if not left:
    mask = mask.T
  point = np.array([0.4, 0.7, 1.1, 1.6])

  def reference_grad(v):
    x = np.sqrt(v.reshape(2, 2))
    active_coefficients = coefficients[:, :2] if left else coefficients[:2, :]
    product = x @ active_coefficients if left else active_coefficients @ x
    weights = np.tile([1.0, 2.0], (2, 1))
    weights = weights if left else weights.T
    partial = weights * np.cos(product)
    factors = partial @ active_coefficients.T if left else active_coefficients.T @ partial
    return factors.ravel() * 0.5 / np.sqrt(v)

  grad = sc.gradient(loss, wrt="v")
  np.testing.assert_allclose(grad(point, mask.ravel()), reference_grad(point), atol=1e-12)
  np.testing.assert_array_equal(grad(np.zeros(4), -np.ones(output.nnz)), np.zeros(4))
  hess = sc.hessian(loss, wrt="v")
  expected_hess = finite_difference(reference_grad, point)
  np.testing.assert_allclose(hess(point, mask.ravel()), expected_hess, atol=1e-8)
  np.testing.assert_array_equal(hess(np.zeros(4), -np.ones(output.nnz)), np.zeros((4, 4)))

  concrete = as_concrete(loss)
  first = vjp(concrete.outputs, (concrete.inputs[0],), (sc.const(1.0),))[0]
  second = vjp((first.sum(),), (concrete.inputs[0],), (sc.const(1.0),))[0]

  @sc.function(sc.arg("v", 4), sc.arg("c", output.nnz), name=f"masked_product_inf_twice_{left}_{lowering}")
  def twice(v, c):
    return substitute(second, dict(zip(concrete.inputs, (v, c), strict=True)))

  np.testing.assert_allclose(twice(point, mask.ravel()), expected_hess.sum(axis=0), atol=1e-8)


@pytest.mark.parametrize("left", [False, True])
def test_sparse_product_coordinate_mask_isolates_singular_inputs(left):
  pattern = CanonicalPattern((2, 2), (0, 1), (0, 1))

  @sc.function(sc.arg("v", 2), sc.arg("c", 2), name=f"masked_product_diagonal_{left}")
  def loss(v, c):
    x = sc.pack(v, pattern).sqrt()
    identity = sc.const(sparse.eye_array(2))
    product = x @ identity if left else identity @ x
    return sc.where(sc.pack(c > 0, pattern), product, sc.pack(sc.const(np.zeros(2)), pattern)).sum().block()

  np.testing.assert_array_equal(sc.gradient(loss, wrt="v")(np.array([1.0, 0.0]), np.array([1.0, -1.0])), [0.5, 0.0])


@pytest.mark.parametrize("case", ["rectangular", "banded", "empty_product", "empty_operand", "empty_dimension"])
@pytest.mark.parametrize("layout", ["csc", "csr", "coordinates"])
def test_sparse_product_forward_and_reverse(case, layout):
  rng = np.random.default_rng(47)
  a = sparse.coo_array(rng.random((5, 7)) < 0.3)
  b = sparse.coo_array(rng.random((7, 4)) < 0.4)
  if case == "banded":
    a = sparse.diags_array([np.ones(6), np.ones(6)], offsets=[0, 1], shape=(6, 7), format="coo")
    b = sparse.diags_array([np.ones(6), np.ones(6)], offsets=[0, -1], shape=(7, 6), format="coo")
  elif case == "empty_product":
    a = sparse.coo_array(([1.0], ([0], [0])), shape=(5, 7))
    b = sparse.coo_array(([1.0], ([6], [3])), shape=(7, 4))
  elif case == "empty_operand":
    a = sparse.coo_array((5, 7))
  elif case == "empty_dimension":
    a, b = sparse.coo_array((5, 0)), sparse.coo_array((0, 4))
  pa = CanonicalPattern(a.shape, tuple(map(int, a.row)), tuple(map(int, a.col)))
  pb = CanonicalPattern(b.shape, tuple(map(int, b.row)), tuple(map(int, b.col)))
  pc = pa.matmul(pb)

  def leaf(name, pattern, batch=()):
    order = sc.SparsityPattern(pattern.shape, pattern.rows[::-1], pattern.cols[::-1]) if layout == "coordinates" else layout
    return sc.arg(name, sc.TensorType((*batch, *pattern.shape), pattern=pattern), layout=order)

  @sc.function(
    leaf("a", pa),
    leaf("b", pb),
    leaf("da", pa, (2,)),
    leaf("db", pb, (2,)),
    leaf("cot", pc),
    outputs=sc.group(leaf("dc", pc, (2,)), leaf("abar", pa), leaf("bbar", pb)),
    name=f"sparse_product_ad_{case}_{layout}",
  )
  def derivatives(a, b, da, db, cot):
    c = a @ b
    tangent = jvp_many(c, a, da) + jvp_many(c, b, db)
    abar, bbar = vjp((c,), (a, b), (cot,))
    assert tangent.type.pattern is c.type.pattern
    assert abar.type.pattern is a.type.pattern and bbar.type.pattern is b.type.pattern
    return tangent, abar, bbar

  def matrix(pattern, data):
    return sparse.coo_array((data, (pattern.rows, pattern.cols)), shape=pattern.shape)

  av, bv, cv = (rng.normal(size=p.nnz) for p in (pa, pb, pc))
  a, b, cot = matrix(pa, av), matrix(pb, bv), matrix(pc, cv)
  dav, dbv = rng.normal(size=(2, pa.nnz)), rng.normal(size=(2, pb.nnz))

  def batch(pattern, data):
    return sparse.coo_array(
      (data.ravel(), (np.repeat(np.arange(2), pattern.nnz), np.tile(pattern.rows, 2), np.tile(pattern.cols, 2))), shape=(2, *pattern.shape)
    )

  dc, abar, bbar = derivatives(a, b, batch(pa, dav), batch(pb, dbv), cot)
  eps = 1e-5
  for i in range(2):
    da, db = matrix(pa, dav[i]), matrix(pb, dbv[i])
    reference = da @ b + a @ db
    difference = ((a + eps * da) @ (b + eps * db) - (a - eps * da) @ (b - eps * db)) / (2 * eps)
    np.testing.assert_allclose(dc.toarray()[i], reference.toarray(), atol=1e-12)
    np.testing.assert_allclose(dc.toarray()[i], difference.toarray(), atol=1e-8)
    dual = (abar.toarray() * da.toarray()).sum() + (bbar.toarray() * db.toarray()).sum()
    np.testing.assert_allclose(dual, (cot.toarray() * difference.toarray()).sum(), atol=1e-8)
  expected_a, expected_b = (cot @ b.T).toarray(), (a.T @ cot).toarray()
  np.testing.assert_allclose(abar.toarray()[pa.rows, pa.cols], expected_a[pa.rows, pa.cols], atol=1e-12)
  np.testing.assert_allclose(bbar.toarray()[pb.rows, pb.cols], expected_b[pb.rows, pb.cols], atol=1e-12)
  assert abar.nnz == pa.nnz and bbar.nnz == pb.nnz


@pytest.mark.parametrize("masked", [False, True])
def test_sampled_sparse_adjoint_does_not_materialize_the_unsampled_product(monkeypatch, masked):
  n = 97
  pa = CanonicalPattern((n, n), tuple(range(n)), tuple(range(n)))
  pb = CanonicalPattern((n, 2), tuple(np.repeat(np.arange(n), 2)), (0, 1) * n)
  pc = pa.matmul(pb)
  matmul = CanonicalPattern.matmul

  def guarded(a, b):
    assert (a.shape[0], b.shape[1]) != (n, n), "the sampled adjoint must not build its full Boolean pattern"
    return matmul(a, b)

  monkeypatch.setattr(CanonicalPattern, "matmul", guarded)

  @sc.function(
    sc.arg("a", sc.TensorType(pa.shape, pattern=pa)),
    sc.arg("b", sc.TensorType(pb.shape, pattern=pb)),
    sc.arg("cot", sc.TensorType(pc.shape, pattern=pc)),
    sc.arg("condition", pc.nnz),
    name="sampled_sparse_adjoint",
  )
  def fn(a, b, cot, condition):
    if masked:
      cot = sc.where(sc.pack(condition > 0, pc), cot, sc.pack(sc.const(np.zeros(pc.nnz)), pc))
    adjoint = vjp((a @ b,), (a,), (cot,))[0]
    assert adjoint.type.pattern is pa
    return sc.values(adjoint).block()

  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    buffers = [b for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
    assert max((np.prod(b.attrs["shape"]) for b in buffers), default=0) <= 2 * n
  a = sparse.coo_array((np.ones(n), (pa.rows, pa.cols)), shape=pa.shape)
  b = sparse.coo_array((np.arange(1.0, 2 * n + 1), (pb.rows, pb.cols)), shape=pb.shape)
  cot = sparse.coo_array((np.arange(pc.nnz, dtype=float), (pc.rows, pc.cols)), shape=pc.shape)
  condition = np.where(np.arange(pc.nnz) % 3 == 0, -1.0, 1.0)
  expected_cot = cot.copy()
  if masked:
    expected_cot.data *= condition > 0
  np.testing.assert_allclose(fn(a, b, cot, condition), (expected_cot @ b.T).diagonal())


@pytest.mark.parametrize(("n", "masked"), [(32, False), (64, False), (64, True)])
def test_weighted_gram_with_a_dense_column_has_stored_size_reverse_and_second_order_work(n, masked, monkeypatch):
  rows = np.tile(np.arange(2 * n), 3)
  cols = np.concatenate((np.arange(2 * n) // 2, np.minimum(np.arange(2 * n) // 2 + 1, n - 1), np.full(2 * n, n - 1)))
  g = sparse.coo_array((np.ones(len(rows)), (rows, cols)), shape=(2 * n, n))
  g.sum_duplicates()
  pg = CanonicalPattern(g.shape, tuple(map(int, g.row)), tuple(map(int, g.col)))
  pw = CanonicalPattern((2 * n, 2 * n), tuple(range(2 * n)), tuple(range(2 * n)))
  pc = CanonicalPattern((n, 2 * n), pg.cols, pg.rows).matmul(pw).matmul(pg)
  matmul = CanonicalPattern.matmul

  def guarded(a, b):
    result = matmul(a, b)
    assert result.nnz <= 12 * n, "an adjoint built the unsampled Boolean product"
    return result

  monkeypatch.setattr(CanonicalPattern, "matmul", guarded)

  @sc.function(sc.arg("v", pg.nnz), sc.arg("w", 2 * n), sc.arg("seeds", (3, 2 * n)), sc.arg("condition", pc.nnz), name=f"sparse_dense_column_ad_{n}")
  def fn(v, w, seeds, condition):
    matrix = sc.pack(v, pg)
    c = matrix.T @ sc.pack(w, pw) @ matrix
    selected = sc.where(sc.pack(condition > 0, pc), c.sin(), sc.pack(sc.const(np.zeros(pc.nnz)), pc)) if masked else c.sin()
    loss = sc.values(selected).sum()
    grad = vjp((loss,), (w,), (sc.const(1.0),))[0]
    return grad, jvp_many(grad, w, seeds)

  concrete = as_concrete(fn)
  sampled_work = 0
  for node in concrete.nodes:
    if node.op == ExprOp.MATMUL:
      assert node.type.storage_size <= 12 * n
      if node.attrs.get("sampled") is not None:
        a, b = node.args[0].type.pattern, node.args[1].type.pattern
        assert a is not None and b is not None and node.type.pattern is not None
        plan = SampledProductPlan.build(a, b, node.type.pattern)
        sampled_work += (plan.right_stops - plan.right_starts).sum()
        assert sum(v.size for v in vars(plan).values()) <= 6 * (a.nnz + b.nnz + node.type.pattern.nnz)
  assert sampled_work <= (320 if masked else 160) * n
  observed = {}
  lower_function(fn, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    sizes = [np.prod(b.attrs["shape"]) for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
    assert max(sizes, default=0) <= 36 * n
  v, w = np.linspace(0.2, 0.7, pg.nnz), np.linspace(0.4, 1.2, 2 * n)
  seeds = np.random.default_rng(47).normal(size=(3, 2 * n))
  condition = np.where(np.arange(pc.nnz) % 3 == 0, -1.0, 1.0)
  selected = np.ones(pc.shape)
  if masked:
    selected[pc.rows, pc.cols] = condition > 0
  matrix = sparse.coo_array((v, (pg.rows, pg.cols)), shape=pg.shape).toarray()

  def reference(w):
    c = matrix.T @ (w[:, None] * matrix)
    return np.einsum("ij,jk,ik->i", matrix, np.cos(c) * selected, matrix)

  grad, tangent = fn(v, w, seeds, condition)
  np.testing.assert_allclose(grad, reference(w), atol=1e-11)
  eps = 1e-5
  for direction, actual in zip(seeds, tangent, strict=True):
    expected = (reference(w + eps * direction) - reference(w - eps * direction)) / (2 * eps)
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_sparse_product_hessian_keeps_sampled_patterns():
  n = 24
  pattern = CanonicalPattern((n, n), tuple(range(n)), tuple(range(n)))
  b = sparse.coo_array(np.linspace(0.1, 1.0, 2 * n).reshape(n, 2))

  @sc.function(sc.arg("w", n), name="sampled_sparse_hessian")
  def loss(w):
    return sc.values(sc.pack(w, pattern) @ sc.const(b)).sin().sum()

  hess = sc.hessian(loss)
  observed = {}
  lower_function(hess, observe=observed.__setitem__)
  for program in (observed["lowered"], observed["pass:pack_workspace"]):
    sizes = [np.prod(b.attrs["shape"]) for b in walk_program(program) if b.op == ProgramOp.BUFFER and b.attrs["address_space"] == "private"]
    assert max(sizes, default=0) <= 2 * n * n
  w = np.linspace(0.2, 0.8, n)
  expected = np.diag(-(np.sin(w[:, None] * b.toarray()) * b.toarray() ** 2).sum(axis=1))
  np.testing.assert_allclose(hess(w), expected, atol=1e-12)


def test_sampled_product_both_operand_derivatives_and_hessian_match_differences():
  from scaly.ad.derivatives import finite_difference

  a = CanonicalPattern((3, 5), (0, 0, 1, 2, 2), (0, 2, 1, 0, 4))
  b = CanonicalPattern((5, 4), (0, 1, 2, 3, 4), (0, 2, 3, 1, 1))
  requested = CanonicalPattern((3, 4), (0, 1, 1, 2, 2), (3, 0, 2, 0, 1))

  @sc.function(sc.arg("v", a.nnz + b.nnz), name="sampled_sparse_derivatives")
  def loss(v):
    product = matmul(sc.pack(v[: a.nnz], a), sc.pack(v[a.nnz :], b), sampled=requested)
    return sc.values(product).sin().sum()

  def reference(v):
    x = sparse.coo_array((v[: a.nnz], (a.rows, a.cols)), shape=a.shape)
    y = sparse.coo_array((v[a.nnz :], (b.rows, b.cols)), shape=b.shape)
    return np.sin((x @ y).toarray()[requested.rows, requested.cols]).sum()

  v = np.linspace(0.2, 0.9, a.nnz + b.nnz)
  grad = sc.gradient(loss)
  np.testing.assert_allclose(grad(v), finite_difference(reference, v).ravel(), atol=1e-8)
  expected = finite_difference(grad, v)
  np.testing.assert_allclose(sc.hessian(loss)(v), expected, atol=1e-8)
  reverse_grad = sc.function(sc.arg("v", len(v)), name="sampled_sparse_reverse_twice")(lambda v: grad(v).sum())
  np.testing.assert_allclose(sc.gradient(reverse_grad)(v), expected.sum(axis=0), atol=1e-8)
  hess = sc.sparse_hessian(loss)
  pattern = hess.sparsity()
  assert pattern is not None
  actual = sparse.coo_array(hess(v)).toarray()
  np.testing.assert_allclose(actual, expected, atol=1e-8)


def test_sampled_product_derivative_sparsity_does_not_join_unsampled_rows(monkeypatch):
  from scaly.ad import sparsity

  n = 128
  pattern = CanonicalPattern((n, n), tuple(range(n)), tuple(range(n)))
  b = sparse.coo_array(np.linspace(0.1, 1.0, 2 * n).reshape(n, 2))

  @sc.function(sc.arg("w", n), name="sampled_sparse_analysis")
  def loss(w):
    return sc.values(sc.pack(w, pattern) @ sc.const(b)).sin().sum()

  incidence = sparsity._incidence

  def guarded(shape, rows, cols):
    assert rows.size <= 16 * n, "sampled derivative analysis joined unsampled output rows"
    return incidence(shape, rows, cols)

  monkeypatch.setattr(sparsity, "_incidence", guarded)
  hess = sc.sparse_hessian(loss)
  hp = hess.sparsity()
  assert hp is not None and hp.nnz == n
  w = np.linspace(0.2, 0.8, n)
  expected = -(np.sin(w[:, None] * b.toarray()) * b.toarray() ** 2).sum(axis=1)
  np.testing.assert_allclose(sparse.coo_array(hess(w)).toarray(), np.diag(expected), atol=1e-12)
