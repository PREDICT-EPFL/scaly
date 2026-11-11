"""NaN-propagating extrema keep generated source and operand evaluation bounded."""

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_nested_extrema_source_grows_linearly(dtype):
  sizes = []
  for depth in (4, 12):

    @sc.function(sc.arg("x", depth), outputs=sc.arg("y"))
    def f(x):
      x = sc.cast(x, dtype)
      y = x[0]
      for i in range(1, depth):
        y = (sc.maximum if i % 2 else sc.minimum)(y, x[i])
      return sc.cast(y, "float64").scalar()

    source = render_c_source(f)
    sizes.append(len(source))
    point = np.linspace(-2.0, 3.0, depth).astype(dtype)
    expected = point[0]
    for i in range(1, depth):
      expected = (np.maximum if i % 2 else np.minimum)(expected, point[i])
    np.testing.assert_equal(f(point.astype(float)), expected)
  assert sizes[1] < 4 * sizes[0]
  assert sizes[1] < 10000


@pytest.mark.parametrize("dtype", ["float64", "float32"])
def test_extremum_operands_appear_once_in_scalar_source(dtype):
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def f(x):
    x = sc.cast(x, dtype)
    return sc.cast(sc.maximum(x[0].sin(), x[1].cos()), "float64").scalar()

  source = render_c_source(f)
  suffix = "f" if dtype == "float32" else ""
  assert source.count(f"sin{suffix}(") == 1
  assert source.count(f"cos{suffix}(") == 1
