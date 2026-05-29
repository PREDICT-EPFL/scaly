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
