from __future__ import annotations

import os
import platform
import shlex
import shutil
import sys
from typing import cast
from pathlib import Path

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
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
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="smoke_jit")
  def fn(x):
    return (x.sin() + x * x).sum()

  return fn


def test_call_uses_jit_and_matches_numpy(isolated_cache) -> None:
  fn = _simple_fn()
  xv = np.array([0.1, -0.7, 2.5])

  jit_out = fn(xv)
  np.testing.assert_allclose(jit_out, (np.sin(xv) + xv * xv).sum())
  assert as_concrete(fn)._compiled is not None
  assert Path(as_concrete(fn)._compiled.lib_path).exists()


def test_jit_cache_key_stable_across_function_instances(isolated_cache) -> None:
  a = _simple_fn()
  b = _simple_fn()
  a(np.zeros(3))
  b(np.zeros(3))
  assert as_concrete(a)._compiled.cache_key == as_concrete(b)._compiled.cache_key


@pytest.mark.skipif(sys.platform == "win32", reason="the wrapper is a shell script")
def test_compiler_wrapper_change_misses_the_cache(isolated_cache, monkeypatch) -> None:
  real = shutil.which(os.environ.get("SCALY_CC", "cc"))
  assert real is not None
  wrapper = isolated_cache / "bin" / "cc"
  wrapper.parent.mkdir()
  monkeypatch.setenv("SCALY_CC", str(wrapper))

  def build(script: str | None = None) -> str:
    if script is not None:
      wrapper.write_text(f'#!/bin/sh\n{script}exec {shlex.quote(real)} "$@"\n')
      wrapper.chmod(0o755)
    return jit._build_artifact(_simple_fn()).key

  first = build("")
  assert build() == first
  assert build(": patched\n") != first


def test_recompile_clears_cache_and_recompiles(isolated_cache) -> None:
  fn = _simple_fn()
  fn(np.zeros(3))
  assert as_concrete(fn)._compiled is not None
  lib_path = Path(as_concrete(fn)._compiled.lib_path)
  assert lib_path.exists()
  cache_dir = lib_path.parent
  fn.recompile()
  assert as_concrete(fn)._compiled is None
  assert not cache_dir.exists()
  xv = np.array([0.5, 0.0, -1.0])
  out = fn(xv)
  np.testing.assert_allclose(out, (np.sin(xv) + xv * xv).sum())


def test_jit_handles_multi_output(isolated_cache) -> None:
  @sc.function(sc.group(sc.arg("x", 2), sc.arg("y", 2)), outputs=sc.group(sc.arg("sum"), sc.arg("prod"), sc.arg("diff")), name="kw_jit")
  def fn(inputs):
    x, y = inputs
    return (x + y, x * y, (x - y).sum())

  xv = np.array([1.0, 2.0])
  yv = np.array([3.0, -1.0])
  s, p, d = fn((xv, yv))
  np.testing.assert_allclose(s, xv + yv)
  np.testing.assert_allclose(p, xv * yv)
  np.testing.assert_allclose(d, (xv - yv).sum())


def test_jit_handles_sparse_jacobian_factory_output(isolated_cache) -> None:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("y"), name="f_sj")
  def f(x):
    return sc.stack([x[0], x[2:4].sum(), x[1] * x[3]])

  spjf = sc.sparse_jacobian(f, "y", "x", name="f_sj_jac")
  xv = np.array([2.0, 3.0, 5.0, 7.0])
  values = spjf(xv)
  np.testing.assert_allclose(values, np.array([1.0, 1.0, 1.0, xv[3], xv[1]]))


def test_jit_handles_nested_call_nodes(isolated_cache) -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("sq"), name="inner_jit")
  def inner(x):
    return x * x

  @sc.function(sc.arg("z", 3), outputs=sc.arg("s"), name="outer_jit")
  def outer(z):
    return inner(z + 1.0).sum()

  zv = np.array([0.25, -0.75, 2.0])
  np.testing.assert_allclose(outer(zv), ((zv + 1.0) ** 2).sum())


def test_invalidate_cache_handles_missing_directory(isolated_cache) -> None:
  fn = _simple_fn()
  fn.recompile()  # nothing to remove yet
  assert as_concrete(fn)._compiled is None


def test_jit_compile_command_targets_host(isolated_cache, monkeypatch) -> None:
  commands: list[list[str]] = []
  real_run = jit.subprocess.run
  monkeypatch.setattr(jit.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd) or real_run(cmd, **kwargs))
  _simple_fn()(np.zeros(3))
  (cmd,) = [command for command in commands if "-dM" not in command]
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

  @sc.function(sc.group(sc.arg("value", 1), sc.arg("target", 1)), outputs=sc.arg("cost"), name="oracle_stage")
  def stage(inputs):
    value, target = inputs
    return ((value - target.exp()) ** 2).sum().block()

  x, param = sc.sym("x", 3), sc.sym("param", 1)
  cost = _mapped_call(stage, 3, [x, param]).sum()
  solver = build_nlp(x=x, f=cost, p=param, name="hoisted_oracle_solver")
  if nested:

    @sc.function(sc.arg("param", 1), outputs=sc.arg("solution"), name="hoisted_oracle_host")
    def fun(param):
      return solver(param)[0]

  else:
    fun = cast(sc.Function, solver.function)
  module = render_c_module(fun)
  procs = module.program.args[: module.program.attrs["proc_count"]]
  assert any(proc.attrs.get("hoisted_from") == stage.name for proc in procs)
  names = {proc.attrs["name"] for proc in procs}
  assert all(name in names for oracles in module.program.attrs["solver_oracles"].values() for name in oracles)
  pv = np.array([0.2])
  result = fun(pv) if nested else solver(pv)[0]
  np.testing.assert_allclose(result, np.full(3, np.exp(pv[0])), atol=1e-7)


@pytest.mark.parametrize("input_name", ["_h0", "v0", "v:0"])
def test_deep_block_callee_temporaries_do_not_shadow_inputs(isolated_cache, input_name: str) -> None:
  @sc.function(sc.arg(input_name, 1), outputs=sc.arg("y"), name="named_deep_stage")
  def stage(value):
    for _ in range(40):
      value = value.sin() + 0.1
    return value.block()

  @sc.function(sc.arg("z", 1), outputs=sc.arg("y"), name="named_deep_root")
  def root(z):
    return stage(z)

  expected = np.array([0.3])
  for _ in range(40):
    expected = np.sin(expected) + 0.1
  np.testing.assert_allclose(root(np.array([0.3])), expected)


@pytest.mark.parametrize("override,expected", [(None, "glibc"), ("none", "none"), ("glibc", "glibc")])
def test_native_render_recipe_and_math_override(monkeypatch, override, expected):
  from scaly.codegen.toolchain import BuildRecipe

  if override is None:
    monkeypatch.delenv("SCALY_VECTOR_LIBM", raising=False)
  else:
    monkeypatch.setenv("SCALY_VECTOR_LIBM", override)
  compilers = []
  options = {}
  recipe = BuildRecipe(cpu="native", lanes=4, vector_libm="glibc")
  monkeypatch.setattr(jit, "native_recipe", lambda compiler: compilers.append(compiler) or recipe)
  sentinel = object()

  def render(_fun, **kwargs):
    options.update(kwargs)
    return sentinel

  monkeypatch.setattr(jit, "render_c_module", render)
  assert jit._render_native(_simple_fn(), ("chosen-compiler",)) is sentinel
  assert compilers == [("chosen-compiler",)]
  assert options == {"cpu": "native", "lanes": 4, "dialect": "gnu", "vector_libm": expected, "reciprocal": False}


@pytest.mark.parametrize("value", ["", "auto", "sleef", "GLIBC"])
def test_vector_math_override_rejects_invalid_values(monkeypatch, value):
  from scaly.codegen.toolchain import BuildRecipe
  from scaly.utils.env import ToolchainError

  monkeypatch.setenv("SCALY_VECTOR_LIBM", value)
  monkeypatch.setattr(jit, "native_recipe", lambda _compiler: BuildRecipe(cpu="native", lanes=1))
  with pytest.raises(ToolchainError, match="SCALY_VECTOR_LIBM"):
    jit._render_native(_simple_fn(), ("unused",))


def test_native_recipe_render_policies_separate_cache_entries(monkeypatch):
  from scaly.codegen.toolchain import BuildRecipe

  monkeypatch.delenv("SCALY_VECTOR_LIBM", raising=False)
  fun = _simple_fn()
  keys = []
  for lanes, vector_libm in ((1, "none"), (4, "none"), (4, "glibc")):
    recipe = BuildRecipe(cpu="native", lanes=lanes, vector_libm=vector_libm)
    monkeypatch.setattr(jit, "native_recipe", lambda _compiler: recipe)
    module = jit._render_native(fun, ("unused",))
    assert module.recipe == recipe
    assert "-lmvec" in module.link_flags if vector_libm == "glibc" else "-lmvec" not in module.link_flags
    keys.append(jit._compute_cache_key(module.body, fun_name=fun.name, command=jit._compile_command(("cc",), module, "m.c", "m.so"), toolchain=""))
  assert len(set(keys)) == 3


def test_scalar_math_override_compile_and_invalidation_use_same_recipe(isolated_cache, monkeypatch):
  from scaly.codegen.toolchain import BuildRecipe

  monkeypatch.setenv("SCALY_VECTOR_LIBM", "none")
  monkeypatch.setattr(jit, "native_recipe", lambda _compiler: BuildRecipe(cpu="native", lanes=4, vector_libm="glibc"))
  fun = _simple_fn()
  artifact = jit._build_artifact(fun)
  source = next(artifact.lib_path.parent.glob("*.c")).read_text()
  assert "lanes=4, dialect=gnu, vector_libm=none" in source
  assert "-lmvec" not in artifact.flags
  assert artifact.key in jit._artifact_cache
  jit.invalidate_cache(fun)
  assert artifact.key not in jit._artifact_cache
  assert not artifact.lib_path.parent.exists()


def test_vector_math_environment_variable_is_registered():
  from scaly.utils.env import scaly_env_vars

  setting = next(var for var in scaly_env_vars() if var.name == "SCALY_VECTOR_LIBM")
  assert setting.default is None


@pytest.mark.skipif(platform.libc_ver()[0] != "glibc" or platform.machine().lower() not in ("x86_64", "amd64"), reason="links libmvec")
def test_math_library_does_not_use_solver_namespace(isolated_cache, monkeypatch):
  from scaly.codegen.toolchain import BuildRecipe

  monkeypatch.setattr(jit, "native_recipe", lambda _compiler: BuildRecipe(cpu="native", lanes=1, vector_libm="glibc"))
  monkeypatch.delenv("SCALY_VECTOR_LIBM", raising=False)
  loaded = []
  original = jit.load_library

  def load(path, *, isolated):
    loaded.append(isolated)
    return original(path, isolated=isolated)

  monkeypatch.setattr(jit, "load_library", load)
  compiled = jit.CompiledFunction(_simple_fn())
  assert "-lmvec" in compiled._artifact.flags
  assert loaded == [False]


def test_native_compiler_probe_has_jit_diagnostic(monkeypatch):
  import subprocess

  def fail(_compiler):
    raise subprocess.CalledProcessError(1, ["cc", "-dM"])

  monkeypatch.setattr(jit, "native_recipe", fail)
  with pytest.raises(jit.JitError, match="failed to probe native C compiler"):
    jit._render_native(_simple_fn(), ("cc",))


@pytest.mark.parametrize("dtype", ["bool", "int32", "int64", "float32"])
@pytest.mark.parametrize("side", ["input", "output"])
def test_jit_refuses_non_float64_leaves_before_cache(dtype, side, monkeypatch) -> None:
  from scaly.function.concrete import ConcreteFunction

  x = sc.Expr.sym("x", (), dtype=dtype if side == "input" else "float64")
  y = sc.Expr.const(1, dtype=dtype) if side == "output" else sc.Expr.const(1.0)
  fn = ConcreteFunction._from_exprs("jit_typed_boundary", [x], [y], ["x"], ["y"])
  monkeypatch.setattr(jit, "_compute_cache_key", lambda *args, **kwargs: pytest.fail("invalid leaves reached the artifact cache"))
  with pytest.raises(jit.JitUnavailable, match=rf"{side}.*{dtype}.*float64"):
    jit.CompiledFunction(fn)


def test_numerical_call_refuses_integer_leaf_before_cast(monkeypatch) -> None:
  from scaly.function.concrete import ConcreteFunction

  x = sc.Expr.sym("x", (), dtype="int64")
  fn = ConcreteFunction._from_exprs("integer_cast_boundary", [x], [sc.Expr.const(1.0)], ["x"], ["y"])
  monkeypatch.setattr(np, "asarray", lambda *args, **kwargs: pytest.fail("integer input reached numerical coercion"))
  with pytest.raises(NotImplementedError, match="int64.*float64"):
    fn(1.75)


def test_folded_extrema_nan_match_runtime_and_numpy(isolated_cache) -> None:
  a = np.array([np.nan, 1.0, np.nan, -np.inf, np.inf, -2.0])
  b = np.array([1.0, np.nan, np.nan, np.inf, -np.inf, 3.0])

  @sc.function(sc.group(sc.arg("a", 6), sc.arg("b", 6)), outputs=sc.group(sc.arg("lo"), sc.arg("hi")), name="runtime_extrema")
  def runtime(inputs):
    x, y = inputs
    return sc.minimum(x, y), sc.maximum(x, y)

  @sc.function(sc.arg("unused", 6), outputs=sc.group(sc.arg("lo"), sc.arg("hi")), name="folded_extrema")
  def folded(unused):
    return sc.minimum(sc.const(a), sc.const(b)), sc.maximum(sc.const(a), sc.const(b))

  for got in (runtime((a, b)), folded(np.zeros(6))):
    np.testing.assert_array_equal(got[0], np.fmin(a, b))
    np.testing.assert_array_equal(got[1], np.fmax(a, b))
