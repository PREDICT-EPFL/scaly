"""NumPy matrix products across vector, broadcast and empty batch shapes."""

import numpy as np
import pytest

import scaly as sc
from tests.matmul_cases import SHAPES


@pytest.mark.parametrize("shapes", SHAPES)
def test_numpy(shapes):
  a, b = shapes

  @sc.function(sc.group(sc.arg("a", a), sc.arg("b", b)), outputs=sc.arg("out"))
  def f(inputs):
    x, y = inputs
    return sc.matmul(x, y).with_lowering("scalar")

  rng = np.random.default_rng(67)
  x, y = rng.normal(size=a), rng.normal(size=b)
  np.testing.assert_allclose(f((x, y)), np.matmul(x, y), atol=1e-12, rtol=1e-12)


def test_reference_summation_order():
  @sc.function(sc.arg("a", (2, 2, 4)), sc.arg("b", (1, 4, 5)), outputs=sc.arg("out"))
  def f(a, b):
    return (a @ b).with_lowering("scalar")

  a = np.tile(np.array([1e16, 1, -1e16, 2.0]), (2, 2, 1))
  b = np.ones((1, 4, 5))
  b[..., 1, :] = 2
  expected = np.zeros((2, 2, 5))
  for k in range(4):
    expected += a[..., k, None] * b[..., k, :]
  np.testing.assert_array_equal(f(a, b), expected)
