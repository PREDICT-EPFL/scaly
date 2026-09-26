from __future__ import annotations

import os
import shutil
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
import scaly.codegen.jit as jit


def _cc_available() -> bool:
  return shutil.which(os.environ.get("SCALY_CC", "cc")) is not None


pytestmark = pytest.mark.skipif(not _cc_available(), reason="cc is required for JIT smoke tests")


@pytest.fixture
def isolated_cache(tmp_path, monkeypatch):
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))
  # Drop any process-local artifacts so the cache key derivation runs fresh.
  jit._artifact_cache.clear()
  yield tmp_path


def _simple_fn() -> sc.Function:
  x = sc.sym("x", 3)
  y = (x.sin() + x * x).sum()
  return sc.Function._from_exprs("smoke_jit", [x], [y], ["x"], ["y"])


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
  x = sc.sym("x", 2)
  y = sc.sym("y", 2)
  fn = sc.Function._from_exprs("kw_jit", [x, y], [x + y, x * y, (x - y).sum()], ["x", "y"], ["sum", "prod", "diff"])
  xv = np.array([1.0, 2.0])
  yv = np.array([3.0, -1.0])
  s, p, d = fn((xv, yv))
  np.testing.assert_allclose(s, xv + yv)
  np.testing.assert_allclose(p, xv * yv)
  np.testing.assert_allclose(d, (xv - yv).sum())


def test_jit_handles_sparse_jacobian_factory_output(isolated_cache) -> None:
  x = sc.sym("x", 4)
  y = sc.stack([x[0], x[2:4].sum(), x[1] * x[3]])
  f = sc.Function._from_exprs("f_sj", [x], [y], ["x"], ["y"])
  spjf = sc.sparse_jacobian(f, "y", "x", name="f_sj_jac")
  xv = np.array([2.0, 3.0, 5.0, 7.0])
  values = spjf(xv)
  np.testing.assert_allclose(values, np.array([1.0, 1.0, 1.0, xv[3], xv[1]]))


def test_jit_handles_nested_call_nodes(isolated_cache) -> None:
  x = sc.sym("x", 3)
  inner = sc.Function._from_exprs("inner_jit", [x], [x * x], ["x"], ["sq"])
  z = sc.sym("z", 3)
  inner_sq = inner(z + 1.0)
  outer = sc.Function._from_exprs("outer_jit", [z], [inner_sq.sum()], ["z"], ["s"])
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
  from scaly.codegen import render_c_module
  from tests.solvers.problem_helpers import build_nlp

  value, target = sc.sym("value", 1), sc.sym("target", 1)
  stage = sc.Function._from_exprs("oracle_stage", [value, target], [((value - target.exp()) ** 2).sum().block()], ["value", "target"], ["cost"])
  x, param = sc.sym("x", 3), sc.sym("param", 1)
  cost = sc.vmap(stage, 3, [(x, 0, 1), (param, 0, 0)]).sum()
  solver = build_nlp(x=x, f=cost, p=param, name="hoisted_oracle_solver")
  if nested:
    inputs = (sc.const(np.zeros(3)), sc.const(np.zeros(3)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), param)
    fun = sc.Function._from_exprs("hoisted_oracle_host", [param], [solver.symbolic_call(inputs)[0]], ["param"], ["solution"])
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
  x = sc.sym(input_name, 1)
  value = x
  for _ in range(40):
    value = value.sin() + 0.1
  stage = sc.Function._from_exprs("named_deep_stage", [x], [value.block()], [input_name], ["y"])
  z = sc.sym("z", 1)
  root = sc.Function._from_exprs("named_deep_root", [z], [stage(z)], ["z"], ["y"])
  expected = np.array([0.3])
  for _ in range(40):
    expected = np.sin(expected) + 0.1
  np.testing.assert_allclose(root(np.array([0.3])), expected)


def test_each_calls_workspace_is_released_without_a_garbage_collection(isolated_cache) -> None:
  """A call's workspace must be freed when the call returns. ``ctypes.cast`` of a ctypes array puts
  the array in a reference cycle, which kept every workspace alive until the next collection."""
  import gc
  import tracemalloc

  z0, us = sc.sym("z0", 4), sc.sym("us", 3000)
  c, u = sc.sym("c", 4), sc.sym("u", 1)
  step = sc.Function._from_exprs("ws_step", [c, u], [c * 0.99 + u[0]], ["c", "u"], ["cn"])
  (final,) = sc.scan(step, z0, [(us, 0, 1)], length=3000)
  (grad,) = sc.vjp((final,), (us,), (sc.const(np.ones(4)),))  # stores 3000 carries: a large workspace
  fn = sc.Function._from_exprs("ws_grad", [z0, us], [grad], ["z0", "us"], ["g"])
  point = (np.ones(4), np.zeros(3000))
  fn(point)
  workspace_bytes = 8 * fn._compiled._sz_w
  assert workspace_bytes >= 8 * 4 * 3000
  gc.disable()
  tracemalloc.start()
  try:
    before = tracemalloc.get_traced_memory()[0]
    for _ in range(20):
      fn(point)
    grown = tracemalloc.get_traced_memory()[0] - before
  finally:
    tracemalloc.stop()
    gc.enable()
  assert grown < 2 * workspace_bytes


def test_integer_inputs_reach_callees_as_integers(isolated_cache) -> None:
  """The entry point takes ``double`` arrays for every input. A callee, a scan or a map reads an
  ``int64`` input through an ``int64_t`` pointer, so the entry converts it once; passing the
  ``double`` buffer through made the callee read raw bits (4.6e18 instead of 2)."""
  c, k = sc.sym("c", 1), sc.sym("k", 1, dtype="int64")
  body = sc.Function._from_exprs("int_body", [c, k], [c + k.cast("float64")], ["c", "k"], ["cn"])
  c0, one, three = sc.sym("c0", 1), sc.sym("K1", 1, dtype="int64"), sc.sym("K3", 3, dtype="int64")
  (called,) = body._flat_symbolic_call([c0, one])
  (scanned,) = sc.scan(body, c0, [(three, 0, 1)], length=3)
  mapped = sc.vmap(body, 3, [(sc.const(np.zeros(3)), 0, 1), (three, 0, 1)])
  fn = sc.Function._from_exprs("int_host", [c0, one, three], [called, scanned, mapped], ["c0", "K1", "K3"], ["called", "scanned", "mapped"])
  got = fn((np.array([0.5]), np.array([2], dtype=np.int64), np.array([1, 2, 3], dtype=np.int64)))
  np.testing.assert_array_equal(got[0], [2.5])
  np.testing.assert_array_equal(got[1], [6.5])
  np.testing.assert_array_equal(got[2], [1.0, 2.0, 3.0])


def test_calls_leave_no_reference_cycles(isolated_cache) -> None:
  """A call passes raw addresses: typed pointers from ``data_as`` left a reference cycle per input,
  output and workspace buffer, which only the garbage collector could free."""
  import gc

  xs = [sc.sym(f"rc{i}", 3) for i in range(4)]
  fn = sc.Function._from_exprs("rc_fn", xs, [x * 2.0 for x in xs], [f"rc{i}" for i in range(4)], [f"y{i}" for i in range(4)])
  point = tuple(np.ones(3) for _ in range(4))
  fn(point)
  gc.collect()
  gc.disable()
  try:
    for _ in range(50):
      fn(point)
    assert gc.collect() == 0
  finally:
    gc.enable()


def test_inputs_of_every_memory_kind() -> None:
  """Addresses come from the buffer protocol where it applies; read-only, empty and strided inputs
  take the fallback (a copy for the strided one) and give the same result."""
  x, e = sc.sym("mk_x", 3), sc.sym("mk_e", 0)
  fn = sc.Function._from_exprs("mem_kinds", [x, e], [x * 2.0, e + 1.0], ["x", "e"], ["y", "z"])
  frozen = np.arange(3.0)
  frozen.setflags(write=False)
  for value in (np.arange(3.0), frozen, np.arange(6.0)[::2] / 2.0, [0.0, 1.0, 2.0]):
    y, z = fn._flat_numerical_call(value, np.zeros(0))
    np.testing.assert_array_equal(y, [0.0, 2.0, 4.0])
    assert z.shape == (0,)


def test_build_survives_its_cache_directory_vanishing(isolated_cache, monkeypatch) -> None:
  """Another process's ``recompile()`` can remove the directory mid-build; the build starts over once."""
  real_replace = Path.replace
  removed: list[Path] = []

  def replace_after_removal(self: Path, target):
    if not removed and self.suffix == ".tmp":
      removed.append(self.parent)
      shutil.rmtree(self.parent)
    return real_replace(self, target)

  monkeypatch.setattr(Path, "replace", replace_after_removal)
  fn = _simple_fn()
  np.testing.assert_allclose(fn(np.array([0.1, -0.7, 2.5])), (np.sin([0.1, -0.7, 2.5]) + np.square([0.1, -0.7, 2.5])).sum())
  assert removed and (removed[0] / "libsmoke_jit").with_suffix(jit.shared_lib_ext()).exists()


def test_a_failing_compile_is_not_retried(isolated_cache, monkeypatch) -> None:
  """Only a vanished cache directory earns a second build; a compiler error is reported at once."""
  calls: list[list[str]] = []

  def failing_run(cmd, **kwargs):
    calls.append(cmd)
    raise jit.subprocess.CalledProcessError(1, cmd, stderr="boom")

  monkeypatch.setattr(jit.subprocess, "run", failing_run)
  with pytest.raises(jit.JitError, match="boom"):
    _simple_fn()(np.zeros(3))
  assert len(calls) == 1


@pytest.mark.parametrize("gcc, opt, vectorize", [(True, None, True), (False, None, False), (True, "-O3", False), (True, "-O2", True)])
def test_gcc_gets_tree_vectorize_at_o2(monkeypatch, gcc: bool, opt: str | None, vectorize: bool) -> None:
  """GCC vectorizes at ``-O2`` by itself only from version 12; clang always does."""
  monkeypatch.setattr(jit, "is_gcc", lambda cc: gcc)
  if opt is None:
    monkeypatch.delenv("SCALY_CC_OPT", raising=False)
  else:
    monkeypatch.setenv("SCALY_CC_OPT", opt)
  flags = jit.compile_flags()
  assert flags[0] == (opt or "-O2")
  assert ("-ftree-vectorize" in flags) == vectorize
  assert flags[-len(jit.HOST_CFLAGS) :] == jit.HOST_CFLAGS


@pytest.mark.parametrize(
  "banner, gcc",
  [
    ("gcc (Ubuntu 11.4.0-1ubuntu1~22.04) 11.4.0\nCopyright (C) 2021 Free Software Foundation, Inc.\n", True),
    ("cc (GCC) 13.2.1 20231011 (Red Hat 13.2.1-4)\nCopyright (C) 2023 Free Software Foundation, Inc.\n", True),
    ("Apple clang version 21.0.0 (clang-2100.1.1.101)\nTarget: arm64-apple-darwin25.6.0\n", False),
    ("Ubuntu clang version 18.1.3 (1ubuntu1)\nTarget: x86_64-pc-linux-gnu\n", False),
    ("", False),
  ],
)
def test_gcc_is_told_from_its_banner(banner: str, gcc: bool) -> None:
  from scaly.codegen.toolchain import _is_gcc_banner

  assert _is_gcc_banner(banner) is gcc


def test_is_gcc_reads_the_compilers_banner(monkeypatch) -> None:
  from scaly.codegen import toolchain

  assert toolchain._version_banner("/definitely/not/a/compiler") == ""
  gcc_banner = "gcc (GCC) 11.4.0\nCopyright (C) 2021 Free Software Foundation, Inc.\n"
  monkeypatch.setattr(toolchain, "_version_banner", lambda cc: gcc_banner if cc == "gcc" else "clang version 18\n")
  assert toolchain.is_gcc("gcc") and not toolchain.is_gcc("clang")
