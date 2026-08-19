from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import alloy as al
import alloy.codegen.jit as jit


def _cc_available() -> bool:
  return shutil.which(os.environ.get("ALLOY_CC", "cc")) is not None


pytestmark = pytest.mark.skipif(not _cc_available(), reason="cc is required for JIT smoke tests")


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
  monkeypatch.setenv("ALLOY_CACHE_DIR", str(tmp_path))
  # Drop any process-local artifacts so the cache key derivation runs fresh.
  jit._artifact_cache.clear()
  yield tmp_path


def _simple_fn() -> al.Function:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  return al.Function("smoke_jit", [x], [y], ["x"], ["y"])


def test_call_uses_jit_and_matches_numpy(isolated_cache) -> None:
  fn = _simple_fn()
  xv = np.array([0.1, -0.7, 2.5])

  jit_out = fn(xv)
  np.testing.assert_allclose(jit_out, (np.sin(xv) + xv * xv).sum())
  assert fn._compiled is not None
  assert Path(fn._compiled.lib_path).exists()


def test_jit_cache_key_stable_across_function_instances(isolated_cache) -> None:
  a = _simple_fn()
  b = _simple_fn()
  a(np.zeros(3))
  b(np.zeros(3))
  assert a._compiled.cache_key == b._compiled.cache_key


def test_recompile_clears_cache_and_recompiles(isolated_cache) -> None:
  fn = _simple_fn()
  fn(np.zeros(3))
  assert fn._compiled is not None
  lib_path = Path(fn._compiled.lib_path)
  assert lib_path.exists()
  cache_dir = lib_path.parent
  fn.recompile()
  assert fn._compiled is None
  assert not cache_dir.exists()
  xv = np.array([0.5, 0.0, -1.0])
  out = fn(xv)
  np.testing.assert_allclose(out, (np.sin(xv) + xv * xv).sum())


def test_jit_handles_multi_output_and_kwargs(isolated_cache) -> None:
  x = al.sym("x", 2)
  y = al.sym("y", 2)
  fn = al.Function("kw_jit", [x, y], [x + y, x * y, (x - y).sum()], ["x", "y"], ["sum", "prod", "diff"])
  xv = np.array([1.0, 2.0])
  yv = np.array([3.0, -1.0])
  s, p, d = fn(x=xv, y=yv)
  np.testing.assert_allclose(s, xv + yv)
  np.testing.assert_allclose(p, xv * yv)
  np.testing.assert_allclose(d, (xv - yv).sum())


def test_jit_handles_sparse_jacobian_factory_output(isolated_cache) -> None:
  x = al.sym("x", 4)
  y = al.stack([x[0], x[2:4].sum(), x[1] * x[3]])
  f = al.Function("f_sj", [x], [y], ["x"], ["y"])
  spjf = al.spjacobian(f, "x", "y", name="f_sj_jac")
  xv = np.array([2.0, 3.0, 5.0, 7.0])
  values = spjf(xv)
  np.testing.assert_allclose(values, np.array([1.0, 1.0, 1.0, xv[3], xv[1]]))


def test_jit_handles_nested_call_nodes(isolated_cache) -> None:
  x = al.sym("x", 3)
  inner = al.Function("inner_jit", [x], [x * x], ["x"], ["sq"])
  z = al.sym("z", 3)
  (inner_sq,) = inner.call([z + 1.0])
  outer = al.Function("outer_jit", [z], [inner_sq.sum()], ["z"], ["s"])
  zv = np.array([0.25, -0.75, 2.0])
  np.testing.assert_allclose(outer(zv), ((zv + 1.0) ** 2).sum())


def test_invalidate_cache_handles_missing_directory(isolated_cache) -> None:
  fn = _simple_fn()
  fn.recompile()  # nothing to remove yet
  assert fn._compiled is None


def test_jit_input_shape_mismatch_raises(isolated_cache) -> None:
  fn = _simple_fn()
  with pytest.raises(ValueError, match="shape"):
    fn(np.zeros(5))
