"""``scaly.linalg`` in float32: values against the same routine in float64, the pivot test in the
matrix's own precision, derivatives against finite differences taken in float64, int32 statuses,
and a lowered program with no float64 value in it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import scipy.linalg

import scaly as sc
from scaly import linalg
from scaly.codegen import write_module
from scaly.ir.program import ProgramOp, walk_program

from .test_dense import N, _spd, _triangular
from .test_ldl_lu import _needs_swaps, _quasi_definite, _symmetric

EPS32 = float(np.finfo(np.float32).eps)
MATRICES = {
  "triangular": _triangular,
  "cholesky": _spd,
  "pos": _spd,
  "ldl": _quasi_definite,
  "sym": _quasi_definite,
  "lu": _needs_swaps,
  "gen": _needs_swaps,
}


def _routine(name: str, A: sc.Expr, b: sc.Expr) -> tuple[sc.Expr, ...]:
  if name == "triangular":
    return (linalg.solve_triangular(A, b, lower=True),)
  if name == "cholesky":
    return linalg.cholesky(A)
  if name == "ldl":
    return linalg.ldl(A)
  if name == "lu":
    return linalg.lu_factor(A)
  return linalg.solve(A, b, assume=name)


def _typed(name: str, batch: tuple[int, ...], dtype: sc.DType) -> sc.Function:
  @sc.function(sc.arg("A", sc.TensorType((*batch, N, N), dtype)), sc.arg("b", sc.TensorType((*batch, N, 2), dtype)), name=f"{name}_{dtype}")
  def fn(A, b):
    return _routine(name, A, b)

  return fn


@pytest.mark.parametrize("batch", [(), (2, 3)])
@pytest.mark.parametrize("name", list(MATRICES))
def test_float32_results_match_float64_on_the_same_data(name, batch) -> None:
  rng = np.random.default_rng(20)
  count = int(np.prod(batch))
  a = np.stack([MATRICES[name](rng) for _ in range(count)]).reshape((*batch, N, N)).astype(np.float32)
  b = rng.normal(size=(*batch, N, 2)).astype(np.float32)
  single = _typed(name, batch, sc.dtypes.float32)
  wide = _typed(name, batch, sc.dtypes.float64)
  for got, expected in zip(single(a, b), wide(a.astype(np.float64), b.astype(np.float64)), strict=True):
    got, expected = np.asarray(got), np.asarray(expected)
    if expected.dtype.kind == "f":
      assert got.dtype == np.float32
      np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)
    else:
      assert got.dtype == expected.dtype
      np.testing.assert_array_equal(got, expected)
  statuses = [s for s in single(a, b) if np.asarray(s).shape == batch and np.asarray(s).dtype.kind == "i"]
  assert name == "triangular" or [np.asarray(s).dtype for s in statuses] == [np.int32]


# Statements carry a placeholder dtype; only declarations of buffers and scalar values have one that matters.
STATEMENTS = {
  ProgramOp.PROGRAM,
  ProgramOp.PROC,
  ProgramOp.BLOCK,
  ProgramOp.FOR,
  ProgramOp.RANGE,
  ProgramOp.CALL,
  ProgramOp.PRINT,
  ProgramOp.BREAK_IF,
  ProgramOp.IF,
}


@pytest.mark.parametrize("name", list(MATRICES))
def test_an_all_float32_call_lowers_without_float64(tmp_path: Path, name) -> None:
  module = write_module(_typed(name, (2,), sc.dtypes.float32), tmp_path, dialect="c")
  wide = {str(n.op) for n in walk_program(module.program) if n.op not in STATEMENTS and n.dtype == sc.dtypes.float64}
  assert not wide, f"float64 values in the lowered program: {sorted(wide)}"


@pytest.mark.parametrize(
  ("name", "a"),
  [
    ("cholesky", [[1.0, 1.0 - EPS32 / 2], [1.0 - EPS32 / 2, 1.0]]),  # in float32 the pivot 1 - fl((1 - eps/2)^2) is exactly eps
    ("ldl", [[-1.0, 0.0], [1.0 - EPS32 / 2, -1.0]]),  # the same pivot, negative, against a negative diagonal
    ("lu", [[1.0, 1.0], [1.0, 1.0 + EPS32]]),  # in float32 the pivot fl(1 + eps) - 1 is eps times its entry
    ("pos", [[1.0, 1.0 - EPS32 / 2], [1.0 - EPS32 / 2, 1.0]]),
  ],
)
def test_the_pivot_test_uses_the_epsilon_of_the_matrix_dtype(name, a) -> None:
  """Each pivot is exactly float32's epsilon times its entry: unusable in float32, usable in float64."""
  a = np.asarray(a)
  n = a.shape[0]

  def status(dtype: sc.DType) -> int:
    @sc.function(
      sc.arg("A", sc.TensorType((n, n), dtype)), sc.arg("b", sc.TensorType((n,), dtype)), outputs=sc.arg("status", sc.TensorType((), sc.dtypes.int32))
    )
    def fn(A, b):
      return _routine(name, A, b)[-1]

    return int(np.asarray(fn(a.astype(dtype.numpy()), np.ones(n, dtype=dtype.numpy()))))

  assert status(sc.dtypes.float32) == 2
  assert status(sc.dtypes.float64) == 0


# --- derivatives against finite differences in float64 --------------------------------------


def _reference(name: str, p: np.ndarray) -> np.ndarray:
  k, b = p[: N * N].reshape((N, N)), p[N * N :]
  if name == "triangular":
    return scipy.linalg.solve_triangular(k, b, lower=True)
  return np.linalg.solve(_symmetric(k) if name in ("pos", "sym") else k, b)


def _cost(x):
  return (x * x * x).sum() / 3.0 + x[0] * x[-1]


def _finite_differences(name: str, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
  """The Jacobian of ``x`` and the Hessian of the cost, by central differences in float64."""
  size, h, eye = p.size, 1e-6, np.eye(p.size)
  jac = np.stack([(_reference(name, p + h * e) - _reference(name, p - h * e)) / (2 * h) for e in eye], axis=1)

  def cost(q: np.ndarray) -> float:
    return float(_cost(_reference(name, q)))

  h = 1e-4
  hess = np.empty((size, size))
  for i in range(size):
    for j in range(size):
      di, dj = h * eye[i], h * eye[j]
      hess[i, j] = (cost(p + di + dj) - cost(p + di - dj) - cost(p - di + dj) + cost(p - di - dj)) / (4 * h * h)
  return jac, hess


@pytest.mark.parametrize("name", ["triangular", "pos", "sym", "gen"])
def test_float32_derivatives_match_float64_finite_differences(name) -> None:
  size = N * N + N
  f32 = sc.TensorType((size,), sc.dtypes.float32)

  @sc.function(sc.arg("p", f32), outputs=sc.group(sc.arg("x"), sc.arg("c")))
  def solve(p):
    x = _routine(name, p[: N * N].reshape((N, N)), p[N * N :])[0]
    return x, _cost(x)

  rng = np.random.default_rng(21)
  p = np.concatenate([MATRICES[name](rng).reshape(-1), rng.normal(size=N)]).astype(np.float32)
  jac, hess = _finite_differences(name, p.astype(np.float64))
  forward, gradient, second = sc.jacobian(solve, "x", "p")(p), sc.gradient(solve, "c", "p")(p), sc.hessian(solve, "c", "p")(p)
  assert np.asarray(forward).dtype == np.asarray(gradient).dtype == np.asarray(second).dtype == np.float32
  np.testing.assert_allclose(forward, jac, rtol=1e-3, atol=1e-4 * np.abs(jac).max())
  x = _reference(name, p.astype(np.float64))
  cost_x = x * x + x[-1] * np.eye(N)[0] + x[0] * np.eye(N)[-1]
  np.testing.assert_allclose(np.asarray(gradient), jac.T @ cost_x, rtol=1e-3, atol=1e-4 * np.abs(jac.T @ cost_x).max())
  np.testing.assert_allclose(second, hess, rtol=2e-3, atol=1e-3 * np.abs(hess).max())


# --- arguments ----------------------------------------------------------------------------------


def test_a_matrix_and_a_right_hand_side_of_different_float_dtypes_are_refused() -> None:
  k32, b64 = sc.sym("K", (N, N), dtype=sc.dtypes.float32), sc.sym("b", N)
  with pytest.raises(TypeError, match="mixed dtypes float32 and float64"):
    linalg.solve(k32, b64, assume="pos")
  with pytest.raises(TypeError, match="mixed dtypes float32 and float64"):
    linalg.solve_triangular(k32, b64, lower=True)


def test_a_python_right_hand_side_takes_the_matrix_dtype() -> None:
  x, status = linalg.solve(sc.sym("K", (N, N), dtype=sc.dtypes.float32), [1.0] * N, assume="gen")
  assert x.type.dtype == sc.dtypes.float32
  assert status.type.dtype == sc.dtypes.int32


def test_an_integer_matrix_is_refused() -> None:
  with pytest.raises(TypeError, match="float32 or float64 matrix"):
    linalg.cholesky(sc.sym("A", (N, N), dtype=sc.dtypes.int64))


@pytest.mark.parametrize("route", ["solve", "triangular"])
def test_a_python_matrix_takes_the_dtype_of_a_numpy_right_hand_side(route) -> None:
  b = np.ones(2, dtype=np.float32)
  k = [[2.0, 0.0], [0.0, 2.0]]
  x = linalg.solve(k, b, assume="pos")[0] if route == "solve" else linalg.solve_triangular(k, b, lower=True)
  assert x.type.dtype == sc.dtypes.float32


def test_a_complex_python_right_hand_side_is_refused() -> None:
  with pytest.raises(TypeError, match="cannot combine"):
    linalg.solve(np.eye(2, dtype=np.float32), [1 + 2j, 2 + 3j], assume="pos")


@pytest.mark.parametrize("lower", [True, False])
@pytest.mark.parametrize("trans", [True, False])
@pytest.mark.parametrize("unit", [True, False])
def test_every_float32_triangular_variant_matches_float64(lower, trans, unit) -> None:
  rng = np.random.default_rng(22)
  t, b = _triangular(rng).astype(np.float32), rng.normal(size=(N, 3)).astype(np.float32)

  def solve(dtype: sc.DType) -> np.ndarray:
    @sc.function(sc.arg("T", (N, N), dtype=...), sc.arg("b", (N, 3), dtype=...), name="variant")
    def fn(T, b):
      return linalg.solve_triangular(T, b, lower=lower, trans=trans, unit_diagonal=unit)

    return np.asarray(fn(t.astype(dtype.numpy()), b.astype(dtype.numpy())))

  single = solve(sc.dtypes.float32)
  assert single.dtype == np.float32
  np.testing.assert_allclose(single, solve(sc.dtypes.float64), rtol=1e-5, atol=1e-5)
