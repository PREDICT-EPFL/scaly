"""``QPStructure`` and ``QPValues``: which bounds exist, and PIQP's preprocessing of the values."""

from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.solvers.ipm import INF, QPStructure, QPValues


def _structure() -> QPStructure:
  P = sparse.csc_array(np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]]))
  G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
  return QPStructure.from_patterns(
    P, np.zeros((0, 3)), G, h_l=[0.0, -np.inf, -INF], h_u=[np.inf, 2.0, INF], x_l=[-1.0, -np.inf, 0.0], x_u=[1.0, np.inf, np.inf]
  )


def test_the_patterns_and_which_bounds_exist() -> None:
  s = _structure()
  assert (s.n, s.p, s.m) == (3, 0, 3)
  np.testing.assert_array_equal(np.stack([s.P_rows, s.P_cols]), [[0, 0, 1, 2], [0, 1, 1, 2]])  # the upper triangle, CSC order
  np.testing.assert_array_equal(s.free_rows, [False, False, True])  # values at INF count as absent
  np.testing.assert_array_equal(s.h_l_idx, [0, 2])  # free rows keep both bounds, at -1 and 1
  np.testing.assert_array_equal(s.h_u_idx, [1, 2])
  np.testing.assert_array_equal(s.x_l_idx, [0, 2])
  np.testing.assert_array_equal(s.x_u_idx, [0])
  assert s.n_bounds == 7


def test_boolean_masks_declare_the_same_structure() -> None:
  s = QPStructure.from_patterns(
    np.eye(2),
    np.zeros((0, 2)),
    np.ones((1, 2)),
    h_l=np.array([True]),
    h_u=np.array([False]),
    x_l=np.array([True, False]),
    x_u=np.array([False, True]),
  )
  np.testing.assert_array_equal(s.h_l_idx, [0])
  assert s.h_u_idx.size == 0 and s.free_rows.sum() == 0
  np.testing.assert_array_equal(s.x_l_idx, [0])
  np.testing.assert_array_equal(s.x_u_idx, [1])


def test_preprocessing_zeroes_free_rows_and_packs_the_box_bounds() -> None:
  s = _structure()
  syms = {
    k: sc.sym(k, shape) for k, shape in (("P", 4), ("c", 3), ("A", 0), ("b", 0), ("G", s.G_rows.size), ("h_l", 3), ("h_u", 3), ("x_l", 3), ("x_u", 3))
  }
  v = QPValues.preprocess(s, **syms)
  names = ["G", "h_l", "h_u", "x_l", "x_u"]
  fn = sc.Function.from_exprs("qp_pre", list(syms.values()), [v.G, v.h_l, v.h_u, v.x_l, v.x_u], list(syms), names)
  G = np.arange(1.0, s.G_rows.size + 1.0)  # G's entries in the structure's order
  out = dict(
    zip(
      names,
      fn((np.ones(4), np.zeros(3), np.zeros(0), np.zeros(0), G, [0.0, -np.inf, 5.0], [np.inf, 2.0, 5.0], [-1.0, 9.0, 0.0], [1.0, 9.0, 9.0])),
      strict=True,
    )
  )
  np.testing.assert_array_equal(out["G"], np.where(s.G_rows == 2, 0.0, G))  # the free row is zeroed
  np.testing.assert_array_equal(out["h_l"], [0.0, -INF, -1.0])  # an absent bound at -INF, whatever arrived
  np.testing.assert_array_equal(out["h_u"], [INF, 2.0, 1.0])
  np.testing.assert_array_equal(out["x_l"], [-1.0, 0.0])  # the finite ones, in index order
  np.testing.assert_array_equal(out["x_u"], [1.0])


@pytest.mark.parametrize("bad", [np.inf, 1e30])
def test_a_value_at_inf_counts_as_absent(bad: float) -> None:
  s = QPStructure.from_patterns(np.eye(1), np.zeros((0, 1)), np.zeros((0, 1)), h_l=np.zeros(0), h_u=np.zeros(0), x_l=[-bad], x_u=[bad])
  assert s.x_l_idx.size == 0 and s.x_u_idx.size == 0


def test_every_pattern_form_declares_the_same_structure() -> None:
  """``from_patterns`` reads patterns as ``SparseMatrix`` does: SciPy, masks, ``SparsityType`` and
  ``SparseMatrix`` give one structure, and ``P``'s lower triangle is ignored."""
  from scaly.linalg import SparseMatrix

  P = np.array([[2.0, 1.0, 0.0], [1.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
  G = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
  bounds = dict(h_l=np.zeros(2), h_u=np.ones(2), x_l=np.zeros(3), x_u=np.ones(3))
  want = QPStructure.from_patterns(sparse.csc_array(P), np.zeros((0, 3)), sparse.csc_array(G), **bounds)
  for form in (lambda a: a != 0, lambda a: SparseMatrix.symbol("M", a != 0), lambda a: SparseMatrix.symbol("M", a != 0).sparsity):
    got = QPStructure.from_patterns(form(P), np.zeros((0, 3), dtype=bool), form(G), **bounds)
    for field in ("P_rows", "P_cols", "G_rows", "G_cols"):
      np.testing.assert_array_equal(getattr(got, field), getattr(want, field))
  np.testing.assert_array_equal(np.stack([want.P_rows, want.P_cols]), [[0, 0, 1, 2], [0, 1, 1, 2]])
