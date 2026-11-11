"""NaN-propagating extrema keep generated source and operand evaluation bounded."""

import numpy as np

import scaly as sc
from scaly.codegen import render_c_source


def test_nested_extrema_source_grows_linearly():
  sizes = []
  for depth in (4, 12):

    @sc.function(sc.arg("x", depth), outputs=sc.arg("y"))
    def f(x):
      y = x[0]
      for i in range(1, depth):
        y = (sc.maximum if i % 2 else sc.minimum)(y, x[i])
      return y.scalar()

    source = render_c_source(f)
    sizes.append(len(source))
    point = np.linspace(-2.0, 3.0, depth)
    expected = point[0]
    for i in range(1, depth):
      expected = (np.maximum if i % 2 else np.minimum)(expected, point[i])
    np.testing.assert_equal(f(point), expected)
  assert sizes[1] < 4 * sizes[0]
  assert sizes[1] < 10000


def test_extremum_operands_appear_once_in_scalar_source():
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"))
  def f(x):
    return sc.maximum(x[0].sin(), x[1].cos()).scalar()

  source = render_c_source(f)
  assert source.count("sin(") == 1
  assert source.count("cos(") == 1
