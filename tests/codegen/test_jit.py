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
  return al.Function._from_exprs("smoke_jit", [x], [y], ["x"], ["y"])


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


def test_jit_handles_multi_output(isolated_cache) -> None:
  x = al.sym("x", 2)
  y = al.sym("y", 2)
  fn = al.Function._from_exprs("kw_jit", [x, y], [x + y, x * y, (x - y).sum()], ["x", "y"], ["sum", "prod", "diff"])
  xv = np.array([1.0, 2.0])
  yv = np.array([3.0, -1.0])
  s, p, d = fn((xv, yv))
  np.testing.assert_allclose(s, xv + yv)
  np.testing.assert_allclose(p, xv * yv)
  np.testing.assert_allclose(d, (xv - yv).sum())


def test_jit_handles_sparse_jacobian_factory_output(isolated_cache) -> None:
  x = al.sym("x", 4)
  y = al.stack([x[0], x[2:4].sum(), x[1] * x[3]])
  f = al.Function._from_exprs("f_sj", [x], [y], ["x"], ["y"])
  spjf = al.sparse_jacobian(f, "y", "x", name="f_sj_jac")
  xv = np.array([2.0, 3.0, 5.0, 7.0])
  values = spjf(xv)
  np.testing.assert_allclose(values, np.array([1.0, 1.0, 1.0, xv[3], xv[1]]))


def test_jit_handles_nested_call_nodes(isolated_cache) -> None:
  x = al.sym("x", 3)
  inner = al.Function._from_exprs("inner_jit", [x], [x * x], ["x"], ["sq"])
  z = al.sym("z", 3)
  inner_sq = inner(z + 1.0)
  outer = al.Function._from_exprs("outer_jit", [z], [inner_sq.sum()], ["z"], ["s"])
  zv = np.array([0.25, -0.75, 2.0])
  np.testing.assert_allclose(outer(zv), ((zv + 1.0) ** 2).sum())


def test_invalidate_cache_handles_missing_directory(isolated_cache) -> None:
  fn = _simple_fn()
  fn.recompile()  # nothing to remove yet
  assert fn._compiled is None


def test_jit_compile_command_targets_host(isolated_cache, monkeypatch) -> None:
  commands: list[list[str]] = []
  real_run = jit.subprocess.run
  monkeypatch.setattr(jit.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd) or real_run(cmd, **kwargs))
  _simple_fn()(np.zeros(3))
  (cmd,) = commands
  flags = list(jit.compile_flags())
  assert cmd[1 : 1 + len(flags)] == flags
  assert "-fno-math-errno" in flags and any(flag.endswith("=native") for flag in flags)


def test_jit_input_shape_mismatch_raises(isolated_cache) -> None:
  fn = _simple_fn()
  with pytest.raises(ValueError, match="shape"):
    fn(np.zeros(5))


@pytest.mark.solver("ipopt")
@pytest.mark.parametrize("nested", [False, True])
def test_hoisted_solver_oracles_compile_and_run(isolated_cache, nested: bool) -> None:
  from alloy.codegen import render_c_module
  from tests.solvers.problem_helpers import build_nlp

  value, target = al.sym("value", 1), al.sym("target", 1)
  stage = al.Function._from_exprs("oracle_stage", [value, target], [((value - target.exp()) ** 2).sum().block()], ["value", "target"], ["cost"])
  x, param = al.sym("x", 3), al.sym("param", 1)
  cost = al.vmap(stage, 3, [(x, 0, 1), (param, 0, 0)]).sum()
  solver = build_nlp(x=x, f=cost, p=param, name="hoisted_oracle_solver")
  if nested:
    inputs = (al.const(np.zeros(3)), al.const(np.zeros(3)), al.const(np.zeros(0)), al.const(np.zeros(0)), param)
    fun = al.Function._from_exprs("hoisted_oracle_host", [param], [solver.symbolic_call(inputs)[0]], ["param"], ["solution"])
  else:
    fun = solver
  module = render_c_module(fun)
  procs = module.program.args[: module.program.attrs["proc_count"]]
  assert any(proc.attrs.get("hoisted_from") == stage.name for proc in procs)
  names = {proc.attrs["name"] for proc in procs}
  assert all(name in names for oracles in module.program.attrs["solver_oracles"].values() for name in oracles)
  pv = np.array([0.2])
  result = fun(pv) if nested else fun((np.zeros(3), np.zeros(3), np.zeros(0), np.zeros(0), pv))[0]
  np.testing.assert_allclose(result, np.full(3, np.exp(pv[0])), atol=1e-7)


@pytest.mark.parametrize("input_name", ["_h0", "v0", "v:0"])
def test_deep_block_callee_temporaries_do_not_shadow_inputs(isolated_cache, input_name: str) -> None:
  x = al.sym(input_name, 1)
  value = x
  for _ in range(40):
    value = value.sin() + 0.1
  stage = al.Function._from_exprs("named_deep_stage", [x], [value.block()], [input_name], ["y"])
  z = al.sym("z", 1)
  root = al.Function._from_exprs("named_deep_root", [z], [stage(z)], ["z"], ["y"])
  expected = np.array([0.3])
  for _ in range(40):
    expected = np.sin(expected) + 0.1
  np.testing.assert_allclose(root(np.array([0.3])), expected)
