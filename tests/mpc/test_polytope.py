"""Polytopes in halfspace form: membership, support, redundancy, centers, vertices."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from scaly import mpc


def test_a_box_and_its_linear_programs() -> None:
  square = mpc.Polytope.box([-1, -2], [1, 2])
  assert square.H.shape == (4, 2) and square.dim == 2 and repr(square) == "Polytope(4 rows in 2 dimensions)"
  assert square.contains([0.5, -1.9]) and not square.contains([1.1, 0.0])
  np.testing.assert_array_equal(square.contains(np.array([[0, 0], [2, 0], [0, -2]])), [True, False, True])
  assert square.support([1, 1]) == pytest.approx(3.0) and square.support([0, -1]) == pytest.approx(2.0)
  center, radius = square.chebyshev_center()
  assert radius == pytest.approx(1.0) and center[0] == pytest.approx(0.0, abs=1e-9) and abs(center[1]) <= 1 + 1e-9  # a 2x4 box: a segment of centers
  vertices = square.vertices()
  assert vertices.shape == (4, 2) and {tuple(np.round(v, 9)) for v in vertices} == {(-1, -2), (1, -2), (1, 2), (-1, 2)}
  half = mpc.Polytope.box([-np.inf, 0], [np.inf, 1])
  assert half.H.shape == (2, 2) and half.support([1, 0]) == np.inf


def test_redundant_rows_go_and_empty_sets_are_found() -> None:
  square = mpc.Polytope.box([-1, -1], [1, 1])
  padded = square.intersect(mpc.Polytope([[1, 1], [2, 0], [0, 0]], [5, 2, 1]))  # a loose row, a duplicate, a zero row
  trimmed = padded.remove_redundancy()
  assert trimmed.H.shape[0] == 4
  assert not square.is_empty() and mpc.Polytope([[1.0], [-1.0]], [0.0, -1.0]).is_empty()  # x <= 0 and x >= 1
  assert mpc.Polytope([[0.0, 0.0]], [-1.0]).remove_redundancy().is_empty()  # 0 <= -1
  cut = square.intersect(mpc.Polytope([[1, 1]], [1.0])).remove_redundancy()
  assert cut.H.shape[0] == 5  # a corner cut off stays


def test_preimages_describe_where_a_map_lands_inside() -> None:
  square = mpc.Polytope.box([-1, -1], [1, 1])
  stretched = square.preimage(np.diag([2.0, 0.5]))
  assert stretched.contains([0.49, 1.9]) and not stretched.contains([0.51, 0.0])
  sheared = square.preimage(np.array([[1.0, 2.0], [0.0, 1.0]]))  # not symmetric: M, not its transpose
  assert sheared.contains([-1.0, 0.9]) and not sheared.contains([0.9, 0.9])
  control = mpc.Polytope.box([-1], [1]).preimage(np.array([[0.5, -2.0]]))  # |K x| <= 1
  assert control.contains([1.0, 0.2]) and not control.contains([0.0, 0.6])
  with pytest.raises(ValueError, match="takes a matrix with 2 rows"):
    square.preimage(np.eye(3))
  with pytest.raises(ValueError, match="one bound per row"):
    mpc.Polytope(np.eye(2), [1.0])
  with pytest.raises(ValueError, match="no Chebyshev center"):
    mpc.Polytope([[1.0], [-1.0]], [0.0, -1.0]).chebyshev_center()


def test_as_a_terminal_set_it_is_one_group_of_rows() -> None:
  square = mpc.Polytope.box([-1, -1], [1, 1])
  x = sc.sym("x", 2)
  eq, ineq = square.constraints(x)
  assert eq == [] and len(ineq) == 1 and ineq[0].expr.shape == (4,) and ineq[0].lo is None
