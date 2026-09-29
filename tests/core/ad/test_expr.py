from __future__ import annotations

import numpy as np

import scaly as sc


def test_structural_transpose_concat_vec_eval_and_ad() -> None:
  x = sc.sym("x", (2, 2))
  y = sc.concat([x.T, x + 1.0], axis=1).vec()
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])
  jf = sc.jacobian(f, "y", "x")
  xv = np.array([[1.0, 2.0], [3.0, 4.0]])

  np.testing.assert_allclose(f(xv), np.concatenate([xv.T, xv + 1.0], axis=1).reshape(8))
  np.testing.assert_allclose(
    jf(xv),
    np.array(
      [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ]
    ),
  )


def test_slice_split_eval_and_ad() -> None:
  x = sc.sym("x", 4)
  left, right = sc.split(x, [2, 2])
  y = sc.stack([x[0], x[2:4].sum(), sc.concat([left, right])[3]])
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])
  jf = sc.jacobian(f, "y", "x")
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  np.testing.assert_allclose(f(xv), np.array([1.0, 7.0, 4.0]))
  np.testing.assert_allclose(jf(xv), np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))

  try:
    _ = sc.split(x, [1, 2])
  except ValueError as e:
    assert "do not sum to axis length 4" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid split should fail")

  try:
    _ = sc.split(x, [-1, 5])
  except ValueError as e:
    assert "cannot contain negative entries" in str(e)
  else:  # pragma: no cover
    raise AssertionError("negative split size should fail")


def test_gather_scatter_eval_and_ad() -> None:
  x = sc.sym("x", 5)
  y = sc.scatter(x.gather([3, 1, 4]), [0, 2, 3], 5)
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])
  jf = sc.jacobian(f, "y", "x")
  xv = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

  np.testing.assert_allclose(f(xv), np.array([13.0, 0.0, 11.0, 14.0, 0.0]))
  np.testing.assert_allclose(
    jf(xv),
    np.array(
      [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
      ]
    ),
  )

  try:
    _ = sc.gather(x, [5])
  except IndexError as e:
    assert "indices must be in [0, 5)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("out-of-range gather should fail")

  v = sc.sym("v", 2)
  repeated = sc.Function.from_exprs("scatter_repeat", [v], [sc.scatter(v, [1, 1], 3)], ["v"], ["y"])
  np.testing.assert_allclose(repeated(np.array([2.0, 5.0])), [0.0, 7.0, 0.0])  # repeated indices accumulate
