from __future__ import annotations

import os
import json
import platform
import shlex
import shutil
import subprocess
import tempfile
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import cast
from pathlib import Path

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
import scaly.codegen.jit as jit
from scaly.codegen.toolchain import find_c_compiler, no_gather_flags
from scaly.utils.env import shared_lib_ext, shared_lib_flag


pytestmark = pytest.mark.skipif(find_c_compiler() is None, reason="cc is required for JIT smoke tests")


def _simple_fn() -> sc.Function[tuple[sc.Expr], tuple[np.ndarray], sc.Expr, np.ndarray]:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", ()), name="smoke_jit")
  def fn(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x * x).sum()

  return fn


def test_jit_cache_in_deep_directory_matches_numpy(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  root = isolated_cache / ("benchmark cache " * 5).rstrip() / ("repeated episode " * 5).rstrip()
  monkeypatch.setenv("SCALY_CACHE_DIR", str(root))
  values = np.array([0.1, -0.7, 2.5])
  fn = _simple_fn()
  np.testing.assert_allclose(fn(values), (np.sin(values) + values * values).sum())
  compiled = jit.get_compiled(fn)
  assert compiled.lib_path.exists()
  assert len(str(compiled.lib_path)) > 260


def test_compiler_wrapper_change_misses_the_cache(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  compiler = find_c_compiler()
  assert compiler is not None
  wrapper = isolated_cache / "bin" / ("cc.exe" if sys.platform == "win32" else "cc")
  wrapper.parent.mkdir()
  monkeypatch.setenv("SCALY_CC", str(wrapper))

  def build(script: str | None = None) -> str:
    if script is not None:
      if sys.platform == "win32":
        source = wrapper.with_suffix(".c")
        command = ", ".join(json.dumps(part) for part in compiler.command)
        source.write_text(
          "#include <process.h>\n#include <stdlib.h>\n"
          f"/* {script} */\nint main(int argc, char** argv) {{\n"
          f"  const char* prefix[] = {{{command}}};\n"
          f"  const int n = {len(compiler.command)};\n"
          "  const char** args = calloc(n + argc, sizeof(char*));\n"
          "  for (int i = 0; i < n; ++i) args[i] = prefix[i];\n"
          "  for (int i = 1; i < argc; ++i) args[n + i - 1] = argv[i];\n"
          "  return (int)_spawnv(_P_WAIT, prefix[0], args);\n}\n"
        )
        subprocess.run([*compiler.command, str(source), "-o", str(wrapper)], check=True)
      else:
        wrapper.write_text(f'#!/bin/sh\n{script}exec {shlex.join(compiler.command)} "$@"\n')
        wrapper.chmod(0o755)
    return jit._build_artifact(as_concrete(_simple_fn())).key

  first = build("")
  assert build() == first
  assert build(": patched\n") != first


_SOLVER_INPUTS_CHILD = """
import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import scaly as sc
import scaly.codegen.jit as jit
import scaly.solvers.paths as paths
import scaly.solvers.registry as registry

backend = SimpleNamespace(name="ipopt", lib_stem="cache_solver", header="solver/api.h", link_flags=("-lcache_solver",))
paths._backends = lambda: {backend.name: backend}
registry.get_backend = lambda _name: backend
paths._plugin_solver_paths = lambda: []
installed = paths.solver_paths()
library = sc.CLibrary(
  "cache_solver", include_dirs=tuple(map(str, installed.include_dirs)), library_dirs=tuple(map(str, installed.lib_dirs)), link=("cache_solver",),
  sources={"bridge.c": '''#include <solver/api.h>
void cache_evaluate(const double* x, double* y, double* w) {
  Settings settings = {0};
  settings.value = x[0] + OFFSET;
  y[0] = evaluate(&settings);
  (void)w;
}
'''},
)
evaluate = sc.extern(library, "cache_evaluate", inputs=(sc.arg("x", 1),), outputs=sc.arg("y", 1))
@sc.function(sc.arg("x", 1), outputs=sc.arg("y", 1), name="solver_cache_inputs")
def fun(x):
  return evaluate(x)
compiles = []
run = jit.subprocess.run
jit.subprocess.run = lambda command, **kwargs: (compiles.append(any(str(arg).endswith("/solver_cache_inputs.c") for arg in command)), run(command, **kwargs))[1]
compiled = jit.get_compiled(fun)
result = compiled.run([np.array([3.0])])[0].item()
print(json.dumps({"key": compiled.cache_key, "compiles": sum(compiles), "result": result}))
"""


@pytest.mark.parametrize("changed", ["library", "symlink", "dangling", "cycle", "unreadable"])
def test_solver_build_change_misses_cache(tmp_path: Path, changed: str) -> None:
  if changed in ("symlink", "dangling", "cycle", "unreadable") and sys.platform == "win32":
    pytest.skip("needs POSIX file permissions or directory symlinks")
  compiler = find_c_compiler()
  assert compiler is not None
  include, lib = tmp_path / "include", tmp_path / "lib"
  (include / "solver").mkdir(parents=True)
  if changed == "symlink":
    target = tmp_path / "native-headers"
    (include / "solver").rename(target)
    (include / "solver").symlink_to(target, target_is_directory=True)
  if changed in ("dangling", "cycle"):
    (include / "unused-link").symlink_to("missing-header" if changed == "dangling" else ".", target_is_directory=changed == "cycle")
  lib.mkdir()
  if changed == "unreadable":
    (lib / "unused-file").write_bytes(b"unrelated to the solver")
    (lib / "unused-file").chmod(0)
  header = include / "solver" / "layout.h"
  api = include / "solver" / "api.h"
  api.write_text('#include "layout.h"\nextern double evaluate(const Settings* settings);\n')
  source = tmp_path / "solver.c"
  library = lib / ("cache_solver.dll" if sys.platform == "win32" else f"libcache_solver{shared_lib_ext()}")

  def install(*, offset: int, revision: int, build_library: bool = True) -> None:
    header.write_text(f"typedef struct {{ double value; double padding; }} Settings;\n#define OFFSET {offset}\n")
    if build_library:
      source.write_text(f"#include <solver/api.h>\ndouble evaluate(const Settings* settings) {{ return settings->value + {revision}; }}\n")
      command = [*compiler.command, "-fPIC", shared_lib_flag(), f"-I{include}", str(source), "-o", str(library)]
      if sys.platform == "win32":
        command += [f"-Wl,--out-implib,{lib / 'libcache_solver.dll.a'}"]
      subprocess.run(command, check=True, capture_output=True, text=True)

  child_env = {**os.environ, "SCALY_CACHE_DIR": str(tmp_path / "cache")}
  for variable in ("SCALY_IPOPT_LIB", "SCALY_SOLVER_INCLUDE_DIR", "SCALY_SOLVER_LIB_DIR", "SCALY_SOLVER_SYSTEM_FALLBACK"):
    child_env.pop(variable, None)
  child_env.update(SCALY_SOLVER_INCLUDE_DIR=str(include), SCALY_SOLVER_LIB_DIR=str(lib))

  def build() -> dict:
    result = subprocess.run([sys.executable, "-c", _SOLVER_INPUTS_CHILD], env=child_env, check=False, capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)

  install(offset=0, revision=0)
  first = build()
  assert first["compiles"] == 1 and first["result"] == 3.0
  assert build() == {**first, "compiles": 0}
  previous = header.stat()
  install(offset=int(changed != "library"), revision=int(changed == "library"), build_library=changed == "library")
  os.utime(header, ns=(previous.st_atime_ns, previous.st_mtime_ns))
  second = build()
  assert second["key"] != first["key"]
  assert second["compiles"] == 1
  assert second["result"] == 4.0
  assert build() == {**second, "compiles": 0}


def test_recompile_uses_another_path_and_preserves_files(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  commands: list[list[str]] = []
  run = jit.subprocess.run
  monkeypatch.setattr(jit.subprocess, "run", lambda command, **kwargs: commands.append(command) or run(command, **kwargs))
  fn = _simple_fn()
  fn(np.zeros(3))
  compiled = as_concrete(fn)._memo.compiled
  assert compiled is not None
  lib_path = Path(compiled.lib_path)
  assert lib_path.exists()
  xv = np.array([0.5, 0.0, -1.0])
  paths = {lib_path}
  for _ in range(2):
    files = {path: path.read_bytes() for path in lib_path.parent.iterdir()}
    fn.recompile()
    assert as_concrete(fn)._memo.compiled is None
    assert all(path.read_bytes() == content for path, content in files.items())
    np.testing.assert_allclose(fn(xv), (np.sin(xv) + xv * xv).sum())
    recompiled = as_concrete(fn)._memo.compiled
    assert recompiled is not None
    assert recompiled.lib_path not in paths
    paths.add(recompiled.lib_path)
    lib_path = recompiled.lib_path
    assert jit.get_compiled(fn).lib_path == recompiled.lib_path
  assert len([command for command in commands if "-fPIC" in command]) == 3


def test_old_handle_stays_callable_after_recompilation(isolated_cache: Path) -> None:
  fn = _simple_fn()
  xv = np.array([0.5, 0.0, -1.0])
  old = jit.get_compiled(fn)
  fn.recompile()
  new = jit.get_compiled(fn)
  for handle in (old, new):
    np.testing.assert_allclose(handle.run([xv])[0], (np.sin(xv) + xv * xv).sum())
  assert new.lib_path != old.lib_path


def test_publication_leaves_a_loaded_library_callable(isolated_cache: Path) -> None:
  compiled = jit.get_compiled(_simple_fn())
  published = compiled.lib_path.read_bytes()
  temporary = compiled.lib_path.with_suffix(".tmp")
  temporary.write_bytes(b"another build")
  jit._publish(temporary, compiled.lib_path)
  assert not temporary.exists()
  # Windows refuses to replace the loaded DLL and keeps the published one, which has the same content.
  assert (compiled.lib_path.read_bytes() == published) == (sys.platform == "win32")
  xv = np.array([0.5, 0.0, -1.0])
  np.testing.assert_allclose(compiled.run([xv])[0], (np.sin(xv) + xv * xv).sum())


_CACHE_CHILD = """
import sys
import time
from pathlib import Path

import numpy as np
import scaly as sc
import scaly.codegen.jit as jit

compiles = []
run = jit.subprocess.run
jit.subprocess.run = lambda command, **kwargs: (compiles.append("-fPIC" in command), run(command, **kwargs))[1]

@sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="shared_cache")
def fn(x):
  return (x.sin() + x * x).sum()

Path(sys.argv[1]).touch()
while not Path(sys.argv[2]).exists():
  time.sleep(0.01)
xv = np.array([0.5, 0.0, -1.0])
np.testing.assert_allclose(fn(xv), (np.sin(xv) + xv * xv).sum())
print(sum(compiles), jit.get_compiled(fn).lib_path)
"""


def test_processes_share_a_cold_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path / "cache"))
  env = {**os.environ, "SCALY_CACHE_DIR": str(tmp_path / "cache")}
  go = tmp_path / "go"
  ready = [tmp_path / f"ready{i}" for i in range(3)]

  def child(i: int) -> subprocess.Popen[str]:
    return subprocess.Popen(
      [sys.executable, "-c", _CACHE_CHILD, str(ready[i]), str(go)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )

  # Both start compiling together, so one may publish while the other still compiles or loads.
  first = [child(0), child(1)]
  while not all(path.exists() or process.poll() is not None for path, process in zip(ready, first)):
    pass
  go.touch()
  outputs = [process.communicate(timeout=300) for process in first]
  assert all(process.returncode == 0 for process in first), outputs

  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="shared_cache")
  def fn(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x * x).sum()

  jit.invalidate_cache(as_concrete(fn))
  (third,) = [child(2)]
  output, error = third.communicate(timeout=300)
  assert third.returncode == 0, error
  reports = [out.split(maxsplit=1) for out, _ in outputs] + [output.split(maxsplit=1)]
  assert len({path for _, path in reports}) == 1
  assert all(int(count) <= 1 for count, _ in reports[:2])
  assert reports[2][0] == "0"


_INVALIDATION_CHILD = """
import sys
from pathlib import Path

import numpy as np
import scaly as sc
import scaly.codegen.jit as jit

ready, phase = Path(sys.argv[1]), sys.argv[2]

def pause(path):
  temporary = ready.with_suffix(".tmp")
  temporary.write_text(str(path))
  temporary.replace(ready)
  assert sys.stdin.readline().strip() == "continue"

if phase == "link":
  run = jit.subprocess.run
  def compile(command, **kwargs):
    if "-fPIC" in command:
      source = next(arg for arg in command if arg.endswith(".c"))
      obj = ready.with_suffix(".o")
      run([*command[:command.index(jit.shared_lib_flag())], "-c", source, "-o", str(obj)], **kwargs)
      pause(Path(source).with_name(f"lib{Path(source).stem}{jit.shared_lib_ext()}"))
      command = [str(obj) if arg == source else arg for arg in command]
    return run(command, **kwargs)
  jit.subprocess.run = compile
elif phase == "publish":
  publish = jit._publish
  def publish_library(temporary, path):
    if path.name.startswith("lib"):
      pause(path)
    publish(temporary, path)
  jit._publish = publish_library
else:
  load = jit.load_library
  def load_library(path, **kwargs):
    pause(path)
    return load(path, **kwargs)
  jit.load_library = load_library

@sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name=f"shared_invalidation_{phase}")
def fn(x):
  return (x.sin() + x * x).sum()

xv = np.array([0.5, 0.0, -1.0])
np.testing.assert_allclose(fn(xv), (np.sin(xv) + xv * xv).sum())
"""


@pytest.mark.parametrize("phase", ["link", "publish", "load"])
def test_invalidation_preserves_another_process_artifact(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch, phase: str) -> None:
  monkeypatch.setenv("SCALY_CACHE_DIR", str(isolated_cache / "shared"))
  ready = isolated_cache / "ready"
  child = subprocess.Popen(
    [sys.executable, "-c", _INVALIDATION_CHILD, str(ready), phase],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
  )
  try:
    deadline = time.monotonic() + 120
    while not ready.exists():
      assert child.poll() is None, child.communicate()
      assert time.monotonic() < deadline, f"child did not reach {phase}"
      time.sleep(0.01)

    @sc.function(sc.arg("x", 3), outputs=sc.arg("y", ()), name=f"shared_invalidation_{phase}")
    def fn(x: sc.Expr) -> sc.Expr:
      return (x.sin() + x * x).sum()

    xv = np.array([0.5, 0.0, -1.0])
    expected = (np.sin(xv) + xv * xv).sum()
    np.testing.assert_allclose(fn(xv), expected)
    old = as_concrete(fn)._memo.compiled
    assert old is not None
    assert old.lib_path == Path(ready.read_text())
    fn.recompile()
    np.testing.assert_allclose(old.run([xv])[0], expected)
    output, error = child.communicate("continue\n", timeout=120)
    assert child.returncode == 0, output + error
    np.testing.assert_allclose(fn(xv), expected)
  finally:
    if child.poll() is None:
      child.kill()
      child.communicate()


def test_threads_share_a_cold_cache(isolated_cache: Path) -> None:
  fun = as_concrete(_simple_fn())
  start = threading.Barrier(2)

  def build(_: int) -> Path:
    start.wait()
    return jit._build_artifact(fun).lib_path

  with ThreadPoolExecutor(2) as pool:
    assert len(set(pool.map(build, range(2)))) == 1
  xv = np.array([0.5, 0.0, -1.0])
  np.testing.assert_allclose(jit.get_compiled(fun).run([xv])[0], (np.sin(xv) + xv * xv).sum())


def test_recompile_before_first_call(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  monkeypatch.setattr(jit, "find_c_compiler", lambda: pytest.fail("invalidation must not probe the compiler"))
  fn = _simple_fn()
  fn.recompile()
  assert as_concrete(fn)._memo.compiled is None


def test_jit_compile_command_targets_host(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  commands: list[list[str]] = []
  real_run = jit.subprocess.run
  monkeypatch.setattr(jit.subprocess, "run", lambda cmd, **kwargs: commands.append(cmd) or real_run(cmd, **kwargs))
  _simple_fn()(np.zeros(3))
  # the compiler fingerprint probes `-dM` and `--version` once per process, so skip those
  (cmd,) = [command for command in commands if "-fPIC" in command]
  flags = list(jit.compile_flags())
  compiler = find_c_compiler()
  assert compiler is not None
  tuning = list(no_gather_flags(compiler.command))
  assert cmd[: len(compiler.command) + len(flags) + len(tuning)] == [*compiler.command, *flags, *tuning]
  assert "-fno-math-errno" in flags and any(flag.endswith("=native") for flag in flags)


def test_jit_compiles_from_the_temporary_directory(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  # zig keeps one runtime build per spelling of its lib directory, so the caller's directory must not matter
  calls: list[tuple[list[str], dict]] = []
  real_run = jit.subprocess.run
  monkeypatch.setattr(jit.subprocess, "run", lambda cmd, **kwargs: calls.append((cmd, kwargs)) or real_run(cmd, **kwargs))
  monkeypatch.chdir(isolated_cache.parent)
  _simple_fn()(np.zeros(3))
  ((cmd, kwargs),) = [(cmd, kwargs) for cmd, kwargs in calls if "-fPIC" in cmd]
  assert kwargs["cwd"] == tempfile.gettempdir()


@pytest.mark.solver("ipopt")
@pytest.mark.parametrize("nested", [False, True])
def test_hoisted_solver_oracles_compile_and_run(isolated_cache: Path, nested: bool) -> None:
  from scaly.codegen import render_c_module
  from tests.solvers.problem_helpers import build_nlp

  @sc.function(sc.group(sc.arg("value", 1), sc.arg("target", 1)), outputs=sc.arg("cost"), name="oracle_stage")
  def stage(inputs):
    value, target = inputs
    return ((value - target.exp()) ** 2).sum().block()

  x, param = sc.sym("x", 3), sc.sym("param", 1)
  cost = _mapped_call(stage, 3, [(x, 0, 1), (param, 0, 0)]).sum()
  solver = build_nlp(x=x, f=cost, p=param, name="hoisted_oracle_solver")
  if nested:

    @sc.function(sc.arg("param", 1), outputs=sc.arg("solution", 3), name="hoisted_oracle_host")
    def fun(param):
      return solver(param)[0]

  else:
    fun = cast(sc.Function, solver.function)
  module = render_c_module(fun, lanes=1)
  source = "\n".join(text for library in module._libraries for _, text in library.sources)
  assert "_hoisted" in source
  assert "oracle_stage" in source
  pv = np.array([0.2])
  result = fun(pv) if nested else solver(pv)[0]
  np.testing.assert_allclose(result, np.full(3, np.exp(pv[0])), atol=1e-7)


@pytest.mark.parametrize("input_name", ["_h0", "v0", "v:0"])
def test_deep_block_callee_temporaries_do_not_shadow_inputs(isolated_cache: Path, input_name: str) -> None:
  @sc.function(sc.arg(input_name, 1), outputs=sc.arg("y"), name="named_deep_stage")
  def stage(value):
    for _ in range(40):
      value = value.sin() + 0.1
    return value.block()

  @sc.function(sc.arg("z", 1), outputs=sc.arg("y", 1), name="named_deep_root")
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
  assert jit._render_native(as_concrete(_simple_fn()), ("chosen-compiler",)) is sentinel
  assert compilers == [("chosen-compiler",)]
  assert options == {"cpu": "native", "lanes": 4, "dialect": "gnu", "vector_libm": expected, "reciprocal": False, "frame_budget": 65536}


@pytest.mark.parametrize("value", ["sleef"])
def test_vector_math_override_rejects_invalid_values(monkeypatch, value):
  from scaly.codegen.toolchain import BuildRecipe
  from scaly.utils.env import ToolchainError

  monkeypatch.setenv("SCALY_VECTOR_LIBM", value)
  monkeypatch.setattr(jit, "native_recipe", lambda _compiler: BuildRecipe(cpu="native", lanes=1))
  with pytest.raises(ToolchainError, match="SCALY_VECTOR_LIBM"):
    jit._render_native(as_concrete(_simple_fn()), ("unused",))


def test_native_recipe_render_policies_separate_cache_entries(monkeypatch):
  from scaly.codegen.toolchain import BuildRecipe

  monkeypatch.delenv("SCALY_VECTOR_LIBM", raising=False)
  monkeypatch.setattr(jit, "no_gather_flags", lambda _compiler: ())
  fun = as_concrete(_simple_fn())
  keys = []
  for lanes, vector_libm in ((1, "none"), (4, "none"), (4, "glibc")):
    recipe = BuildRecipe(cpu="native", lanes=lanes, vector_libm=vector_libm)
    monkeypatch.setattr(jit, "native_recipe", lambda _compiler: recipe)
    module = jit._render_native(fun, ("unused",))
    assert module.recipe == recipe
    assert "-lmvec" in module.link_flags if vector_libm == "glibc" else "-lmvec" not in module.link_flags
    keys.append(
      jit._compute_cache_key(
        module.body, fun_name=fun.name, command=jit._compile_command(("cc",), module.recipe, module.link_flags, "m.c", "m.so"), toolchain=""
      )
    )
  assert len(set(keys)) == 3


def test_scalar_math_override_reaches_compilation(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch):
  from scaly.codegen.toolchain import BuildRecipe

  monkeypatch.setenv("SCALY_VECTOR_LIBM", "none")
  monkeypatch.setattr(jit, "native_recipe", lambda _compiler: BuildRecipe(cpu="native", lanes=4, vector_libm="glibc"))
  fun = as_concrete(_simple_fn())
  artifact = jit._build_artifact(fun)
  source = next(artifact.lib_path.parent.glob("*.c")).read_text()
  assert "lanes=4, dialect=gnu, vector_libm=none" in source
  assert "-lmvec" not in artifact.flags


@pytest.mark.skipif(platform.libc_ver()[0] != "glibc" or platform.machine().lower() not in ("x86_64", "amd64"), reason="links libmvec")
def test_math_library_does_not_use_solver_namespace(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch):
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


@pytest.mark.skipif(platform.libc_ver()[0] != "glibc" or platform.machine().lower() not in ("x86_64", "amd64"), reason="links libmvec")
@pytest.mark.parametrize("compiler", ["clang", "zig cc"])
def test_clang_vectorizes_scalar_math_loops_with_libmvec(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch, compiler: str) -> None:
  from scaly.codegen.toolchain import BuildRecipe

  if shutil.which(compiler.split()[0]) is None:
    pytest.skip(f"{compiler} is not installed")
  monkeypatch.setenv("SCALY_CC", compiler)
  monkeypatch.delenv("SCALY_VECTOR_LIBM", raising=False)
  monkeypatch.setenv("SCALY_CC_OPT", "-O2")
  monkeypatch.setattr(jit, "native_recipe", lambda _compiler: BuildRecipe(cpu="native", lanes=1, vector_libm="glibc"))

  @sc.function(sc.arg("x", 64), outputs=sc.arg("y", 64), name="scalar_sin_loop")
  def fn(x: sc.Expr) -> sc.Expr:
    return x.sin()

  xv = np.linspace(-3.0, 3.0, 64)
  np.testing.assert_allclose(fn(xv), np.sin(xv), rtol=0, atol=1e-15)
  compiled = as_concrete(fn)._memo.compiled
  assert compiled is not None
  undefined = subprocess.run(["nm", "-u", str(compiled.lib_path)], check=True, text=True, capture_output=True).stdout
  assert "_ZGV" in undefined


_ENVIRONMENT_CHANGE_SCRIPT = """
import os
import numpy as np
import scaly as sc

def build(k, backend):
  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", ()), name=f"q{k}")
  def q(x, p):
    return sc.ProblemSpec(minimize=((x - p) * (x - p)).sum() + k, lb=sc.const(-5.0), ub=sc.const(5.0))
  return sc.solver(q, backend)

build(0, "piqp")(np.array(1.0))
for i in range(200):
  os.environ[f"SCALY_REPRO_{i}"] = "x" * 50
build(1, "ipopt")(np.array(1.0))
"""


@pytest.mark.solver("piqp")
@pytest.mark.solver("ipopt")
def test_environment_change_before_next_isolated_load(tmp_path):
  env = {**os.environ, "SCALY_CACHE_DIR": str(tmp_path)}
  result = subprocess.run([sys.executable, "-c", _ENVIRONMENT_CHANGE_SCRIPT], env=env, capture_output=True, text=True, timeout=300)
  assert result.returncode == 0, result.stderr


def test_native_compiler_probe_has_jit_diagnostic(monkeypatch):

  def fail(_compiler):
    raise subprocess.CalledProcessError(1, ["cc", "-dM"])

  monkeypatch.setattr(jit, "native_recipe", fail)
  with pytest.raises(jit.JitError, match="failed to probe native C compiler"):
    jit._render_native(as_concrete(_simple_fn()), ("cc",))


def test_folded_extrema_nan_match_runtime_and_numpy(isolated_cache: Path) -> None:
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
    np.testing.assert_array_equal(got[0], np.minimum(a, b))
    np.testing.assert_array_equal(got[1], np.maximum(a, b))


@pytest.mark.parametrize("name", ["log", "scaly_minimum"])
def test_exported_symbol_clash_is_a_plain_user_error(isolated_cache: Path, name: str) -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name=name)
  def fun(x):
    return x + 1

  with pytest.raises(ValueError) as error:
    fun(np.ones(2))
  assert str(error.value) == f"exported C identifier {name!r} clashes with another declaration; give the function a distinct name"


def test_unsupported_lowering_still_raises_jit_unavailable(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.passes.lowering import LoweringError

  def unsupported(*args, **kwargs):
    raise LoweringError("unsupported construct")

  monkeypatch.setattr(jit, "_render_native", unsupported)
  with pytest.raises(jit.JitUnavailable, match="codegen does not support function 'smoke_jit': unsupported construct"):
    _simple_fn()(np.ones(3))


def test_distinct_solver_functions_with_same_exported_name_are_user_errors(isolated_cache: Path) -> None:
  from tests.solvers.problem_helpers import build_qp
  from scaly.solvers.model import descriptor_function

  left = build_qp(P=np.eye(2), c=np.ones(2), name="same_solver")
  right = build_qp(P=np.eye(2), c=np.ones(2) * 2, name="same_solver")

  left = as_concrete(descriptor_function(as_concrete(left.function).descriptor))
  right = as_concrete(descriptor_function(as_concrete(right.function).descriptor))

  @sc.function(outputs=sc.arg("y", 2), name="solver_host")
  def host():
    zero, empty = sc.const(np.zeros(2)), sc.const(np.zeros(0))
    return left(zero, zero, empty, empty)[0] + right(zero, zero, empty, empty)[0]

  with pytest.raises(ValueError, match="duplicate solver symbol 'same_solver'") as error:
    host()
  assert "codegen does not support" not in str(error.value)
