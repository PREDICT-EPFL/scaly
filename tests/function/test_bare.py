"""Bare templates: ``@sc.function`` with nothing declared binds structure, shapes and dtypes at each call."""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import pytest
from scipy import sparse

import scaly as sc


def test_one_body_many_shapes() -> None:
  traced = []

  @sc.function
  def product(A, B):
    traced.append((A.shape, B.shape))
    return A @ B

  a, b = np.arange(16.0).reshape(4, 4), np.ones((4, 2))
  np.testing.assert_allclose(product(a, b), a @ b)
  c, d = np.eye(3), np.arange(3.0).reshape(3, 1)
  np.testing.assert_allclose(product(c, d), d)
  np.testing.assert_allclose(product(2 * a, b), 2 * a @ b)
  assert list(product.instances) == ["product__4x4_4x2", "product__3x3_3x1"]
  assert traced == [((4, 4), (4, 2)), ((3, 3), (3, 1))]
  instance = product.instances["product__4x4_4x2"]
  assert instance.input_names == ("A", "B") and instance.output_names == ("product",)
  # A symbolic call at a bound shape reuses the instance.
  assert product(sc.sym("a", (4, 4)), sc.sym("b", (4, 2))).attrs["callee"] is instance


def test_numbers_and_lists_are_leaves_of_their_shape() -> None:
  @sc.function()
  def twice(x):
    return 2.0 * x

  for value in (1.5, 2, np.float64(2.0), np.array(2.0)):
    assert twice(value) == 2.0 * float(value)
  np.testing.assert_allclose(twice([1.0, 2.0]), [2.0, 4.0])
  assert list(twice.instances) == ["twice__s", "twice__2"]
  with pytest.raises(TypeError, match=r"'x' is a tuple of numbers, which is structure: 2 scalars; pass a list or an array"):
    twice((1.0, 2.0))
  with pytest.raises(TypeError, match="'x' is a list holding expressions; build one with sc.stack"):
    twice([sc.sym("a"), 1.0])
  with pytest.raises(TypeError, match=r"'x' is a SciPy sparse matrix; declare its pattern"):
    twice(sparse.eye_array(2, format="csc"))


def test_tuples_are_structure_and_the_nesting_is_part_of_the_instance() -> None:
  @sc.function(name="step")
  def body(state, dt):
    x, v = state
    return (x + dt * v, v), x

  (x1, v1), x0 = body((np.zeros(2), np.ones(2)), 0.5)
  np.testing.assert_allclose(x1, [0.5, 0.5])
  (instance,) = body.instances.values()
  assert instance.input_names == ("state_0", "state_1", "dt")
  assert instance.output_names == ("step_0_0", "step_0_1", "step_1")
  assert instance.name.startswith("step__2_2_s_t") and len(instance.name) == len("step__2_2_s_t") + 6

  @sc.function
  def total(p):
    return p[0].sum() + p[1][0].sum()

  total((np.ones(2), (np.ones(1), np.ones(3))))
  total((np.ones(2), (np.ones(1), (np.ones(3),))))  # the same flat shapes, nested differently: another instance
  assert len(total.instances) == 2


def test_an_expr_argument_brings_its_dtype() -> None:
  @sc.function
  def pick(values, k):
    return sc.take(values, sc.stack([k]))

  out = pick(sc.sym("v", 5), sc.sym("k", (), dtype="int64"))
  assert out.shape == (1,) and list(pick.instances) == ["pick__5_sint64"]


def test_a_sparse_matrix_argument_declares_its_pattern() -> None:
  @sc.function
  def apply(A, x):
    return A @ x

  eye, tri = np.eye(3, dtype=bool), np.tril(np.ones((3, 3), dtype=bool))
  y1 = apply(sc.SparseMatrix.symbol("A", eye), sc.sym("x", 3))
  y2 = apply(sc.SparseMatrix.symbol("B", tri), sc.sym("x", 3))
  names = list(apply.instances)
  assert len(names) == 2 and all(name.startswith("apply__p") and name.endswith("_3") for name in names)
  assert y1.attrs["callee"] is not y2.attrs["callee"]
  rows, cols = np.nonzero(tri)
  csc = sparse.csc_array((np.arange(1.0, 7.0), (rows, cols)), shape=(3, 3))
  np.testing.assert_allclose(apply.instances[names[1]](csc, np.ones(3)), csc @ np.ones(3))


def test_outputs_are_read_off_the_trace_and_named_after_the_function() -> None:
  @sc.function(3)
  def pair(x):
    return x.sum(), (x, 2.0 * x)

  assert type(pair) is sc.ConcreteFunction and pair.output_names == ("pair_0", "pair_1_0", "pair_1_1")
  total, (same, double) = pair(np.ones(3))
  assert total == 3.0 and double.tolist() == [2.0, 2.0, 2.0]

  @sc.function
  def constant():
    return sc.const([1.0, 2.0])

  assert type(constant) is sc.ConcreteFunction and constant.output_names == ("constant",)

  @sc.function(output=sc.G("lo", "hi"))
  def bounds(x):
    return x.min(), x.max()

  lo, hi = bounds(np.array([3.0, -1.0, 2.0]))
  assert (lo, hi) == (-1.0, 3.0) and bounds.instances["bounds__3"].output_names == ("lo", "hi")

  with pytest.raises(TypeError, match="the body returned a list; return an Expr, or a tuple of them for several outputs"):
    sc.function(2)(lambda x: [x, x])


def test_output_default_names_follow_the_template_not_the_instance() -> None:
  @sc.function
  def norm2(x):
    return (x * x).sum()

  norm2(np.ones(2))
  norm2(np.ones(3))
  assert {instance.output_names for instance in norm2.instances.values()} == {("norm2",)}


def test_arity_errors_name_the_body_parameters() -> None:
  @sc.function
  def f(a, b):
    return a + b

  with pytest.raises(TypeError, match=r"f\(\) takes 2 arguments \(a, b\), got 1"):
    cast(Any, f)(np.ones(2))
  np.testing.assert_allclose(cast(Any, f)(b=np.ones(2), a=np.ones(2)), [2.0, 2.0])
  with pytest.raises(sc.NotConcrete, match="f leaves every parameter's structure and shape to its calls"):
    f.concrete
  assert repr(f) == "Function('f', (a, b) -> inferred, instances=['f__2_2'])"


def test_a_named_output_takes_the_kind_of_leaf_the_body_returns() -> None:
  @sc.function(3, output=sc.G("K", "y"))
  def stiffness(x):
    return sc.SparseMatrix.diag(x), 2.0 * x

  K, y = cast(Any, stiffness(np.arange(1.0, 4.0)))  # the sparse leaf comes back as a scipy.sparse.csc_array
  assert stiffness.output_names == ("K", "y") and stiffness.output_sparsities[0] is not None
  np.testing.assert_allclose(K.toarray(), np.diag([1.0, 2.0, 3.0]))
  np.testing.assert_allclose(y, [2.0, 4.0, 6.0])
