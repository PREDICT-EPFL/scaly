"""Phase 10: batched matmul ``(B,M,K) @ (B,K,N) -> (B,M,N)``.

Built up in slices: forward semantics here, AD and Program IR lowering follow.
"""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest

import alloy as al
from alloy.ops import Ops
from alloy.spec import verify_expr


def _has_compiler() -> bool:
  cc = os.environ.get("ALLOY_CC", "cc")
  try:
    subprocess.run([cc, "--version"], check=True, capture_output=True)
    return True
  except (FileNotFoundError, subprocess.CalledProcessError):
    return False


# --------------------------------------------------------------------------- forward


def test_bmm_construct_and_shape() -> None:
  a = al.sym("a", (4, 2, 3))
  b = al.sym("b", (4, 3, 5))
  c = a @ b
  assert c.op == Ops.MATMUL
  assert c.shape == (4, 2, 5)


def test_bmm_verifies() -> None:
  a = al.sym("a", (2, 3, 4))
  b = al.sym("b", (2, 4, 6))
  verify_expr(a @ b)


def test_bmm_interpreter_matches_numpy() -> None:
  a = al.sym("a", (4, 2, 3))
  b = al.sym("b", (4, 3, 5))
  fn = al.Function("f_bmm", [a, b], [a @ b], ["a", "b"], ["c"])
  a_val = np.arange(4 * 2 * 3, dtype=np.float64).reshape(4, 2, 3)
  b_val = np.linspace(-1.0, 1.0, 4 * 3 * 5).reshape(4, 3, 5)
  out = fn.eval_interpreter(a_val, b_val)[0]
  np.testing.assert_allclose(out, a_val @ b_val)


def test_bmm_rejects_batch_mismatch() -> None:
  a = al.sym("a", (4, 2, 3))
  b = al.sym("b", (5, 3, 6))
  with pytest.raises(ValueError, match="batch-matmul"):
    _ = a @ b


def test_bmm_rejects_contraction_mismatch() -> None:
  a = al.sym("a", (4, 2, 3))
  b = al.sym("b", (4, 7, 6))
  with pytest.raises(ValueError, match="batch-matmul"):
    _ = a @ b


# ------------------------------------------------------------------------------- AD


def _bmm_jac_fd(a_val: np.ndarray, b_val: np.ndarray, wrt: str, eps: float = 1e-6) -> np.ndarray:
  """Finite-difference Jacobian of vec(a@b) w.r.t. vec(a) or vec(b)."""
  base = (a_val @ b_val).reshape(-1)
  src = a_val if wrt == "a" else b_val
  cols = []
  flat = src.reshape(-1).copy()
  for i in range(flat.size):
    pert = flat.copy()
    pert[i] += eps
    p = pert.reshape(src.shape)
    out = (p @ b_val) if wrt == "a" else (a_val @ p)
    cols.append((out.reshape(-1) - base) / eps)
  return np.stack(cols, axis=1)


def _bmm_jac_fn(name: str, wrt: str):
  # c = a @ b depends on both inputs, so the Jacobian function must declare both
  # (al.jacobian only keeps the single wrt input — fine when the rest are constants).
  a = al.sym("a", (2, 2, 3))
  b = al.sym("b", (2, 3, 2))
  fn = al.Function(name, [a, b], [a @ b], ["a", "b"], ["c"])
  return fn.factory(f"{name}_jac", ["a", "b"], [f"jac:c:{wrt}"])


def test_bmm_jacobian_wrt_a_matches_fd() -> None:
  jf = _bmm_jac_fn("f_bmm_ja", "a")
  a_val = np.linspace(-1.0, 1.0, 12).reshape(2, 2, 3)
  b_val = np.linspace(0.5, 2.0, 12).reshape(2, 3, 2)
  np.testing.assert_allclose(jf(a_val, b_val), _bmm_jac_fd(a_val, b_val, "a"), atol=1e-5)


def test_bmm_jacobian_wrt_b_matches_fd() -> None:
  jf = _bmm_jac_fn("f_bmm_jb", "b")
  a_val = np.linspace(-1.0, 1.0, 12).reshape(2, 2, 3)
  b_val = np.linspace(0.5, 2.0, 12).reshape(2, 3, 2)
  np.testing.assert_allclose(jf(a_val, b_val), _bmm_jac_fd(a_val, b_val, "b"), atol=1e-5)


def test_bmm_jacobian_is_batch_block_diagonal() -> None:
  # Cross-batch entries must be zero: output batch b only sees inputs in batch b.
  a = al.sym("a", (2, 2, 2))
  b = al.sym("b", (2, 2, 2))
  fn = al.Function("f_bmm_bd", [a, b], [a @ b], ["a", "b"], ["c"])
  jf = fn.factory("f_bmm_bd_jac", ["a", "b"], ["jac:c:a"])
  a_val = np.arange(8.0).reshape(2, 2, 2) + 1.0
  b_val = np.arange(8.0).reshape(2, 2, 2) + 1.0
  J = jf(a_val, b_val)
  # rows 0..3 are batch 0 outputs, cols 0..3 are batch 0 inputs; off-diagonal blocks zero.
  assert np.all(J[:4, 4:] == 0.0)
  assert np.all(J[4:, :4] == 0.0)
