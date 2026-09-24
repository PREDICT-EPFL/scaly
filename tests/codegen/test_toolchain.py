from __future__ import annotations

import os
import subprocess
import sys

import pytest

from scaly.codegen import aot
from scaly.solvers import paths as solver_paths_module
from scaly.utils import env


def test_solver_dir_override_discovery_and_flags(tmp_path, monkeypatch) -> None:
  include = tmp_path / "include"
  lib = tmp_path / "lib"
  (include / "piqp").mkdir(parents=True)
  (include / "coin-or").mkdir(parents=True)
  lib.mkdir(parents=True)
  (include / "piqp" / "piqp.h").write_text("// piqp")
  (include / "coin-or" / "IpStdCInterface.h").write_text("// ipopt")
  (lib / f"libpiqpc{env.shared_lib_ext()}").write_text("")
  (lib / f"libipopt{env.shared_lib_ext()}").write_text("")

  monkeypatch.setenv("SCALY_SOLVER_INCLUDE_DIR", str(include))
  monkeypatch.setenv("SCALY_SOLVER_LIB_DIR", str(lib))
  paths = solver_paths_module.solver_paths()
  assert paths.include_dirs[0] == include
  assert paths.lib_dirs[0] == lib
  assert solver_paths_module.solver_discoverable("piqp")
  assert solver_paths_module.solver_discoverable("ipopt")
  assert not solver_paths_module.solver_loadable("piqp")
  assert not solver_paths_module.solver_loadable("ipopt")
  assert solver_paths_module.solver_header_include("piqp") == "piqp/piqp.h"
  assert solver_paths_module.solver_header_include("ipopt") == "coin-or/IpStdCInterface.h"

  flags = solver_paths_module.backend_compile_flags(("piqp", "ipopt"))
  assert f"-I{include}" in flags
  assert f"-L{lib}" in flags
  assert f"-Wl,-rpath,{lib}" in flags
  assert "-lpiqpc" in flags
  assert "-lipopt" in flags


def test_solver_paths_required_needs_both_libraries(tmp_path, monkeypatch) -> None:
  include = tmp_path / "include"
  lib = tmp_path / "lib"
  empty_pkg = tmp_path / "empty_pkg"
  (include / "piqp").mkdir(parents=True)
  lib.mkdir(parents=True)
  empty_pkg.mkdir(parents=True)
  (include / "piqp" / "piqp.h").write_text("// piqp")
  (lib / f"libpiqpc{env.shared_lib_ext()}").write_text("")

  monkeypatch.setattr(solver_paths_module, "_package_root", lambda: empty_pkg)
  monkeypatch.setattr(solver_paths_module, "_plugin_solver_paths", lambda: [])
  monkeypatch.setenv("SCALY_SOLVER_SYSTEM_FALLBACK", "0")
  monkeypatch.setenv("SCALY_SOLVER_INCLUDE_DIR", str(include))
  monkeypatch.setenv("SCALY_SOLVER_LIB_DIR", str(lib))
  with pytest.raises(solver_paths_module.SolverLibraryError):
    solver_paths_module.solver_paths(required=True)


def test_aot_cli_writes_the_module_pair(tmp_path, monkeypatch, capsys) -> None:
  (tmp_path / "scaly_aot_cli_target.py").write_text(
    "import scaly as sc\n\n\ndef build():\n  @sc.function(sc.L('x', 2), sc.L('y', ...), name='aot_cli')\n  def f(x):\n    return x * x\n\n  return f\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  out = tmp_path / "generated"
  aot.main(["scaly_aot_cli_target:build", "-o", str(out)])
  assert "#define aot_cli_SZ_W 0" in (out / "aot_cli.h").read_text()
  source = (out / "aot_cli.c").read_text()
  assert source.startswith("/* Scaly build recipe")
  assert '#include "aot_cli.h"' in source
  assert "sz_w: 0" in capsys.readouterr().out

  # The documented invocation goes through codegen/__main__.py. Running aot itself would re-execute
  # an already-imported module (RuntimeWarning, two copies of its render-observer list), so -W error
  # is what pins the shim in place.
  env_vars = {**os.environ, "PYTHONPATH": str(tmp_path)}
  argv = [sys.executable, "-W", "error::RuntimeWarning", "-m", "scaly.codegen", "scaly_aot_cli_target:build", "-o", str(tmp_path / "cli")]
  proc = subprocess.run(argv, check=False, capture_output=True, text=True, env=env_vars)
  assert proc.returncode == 0, proc.stderr
  assert (tmp_path / "cli" / "aot_cli.h").read_text() == (out / "aot_cli.h").read_text()


def test_env_registry_mentions_native_build_controls() -> None:
  names = {v.name for v in env.scaly_env_vars()}
  assert "SCALY_CACHE_DIR" in names
  assert "SCALY_CC" in names
  assert "SCALY_BUILD_SOLVERS" in names


def test_build_recipe_validates_render_controls() -> None:
  from scaly.codegen.toolchain import BuildRecipe

  for build in (
    lambda: BuildRecipe(cpu="avx"),
    lambda: BuildRecipe(lanes=3),
    lambda: BuildRecipe(lanes=True),
    lambda: BuildRecipe(dialect="cpp"),
    lambda: BuildRecipe(vector_libm="sleef"),
    lambda: BuildRecipe(dialect="c", vector_libm="glibc"),
  ):
    with pytest.raises(ValueError):
      build()
  recipe = BuildRecipe(cpu="x86-64-v3", lanes=4, vector_libm="glibc")
  assert recipe.cpu_flags == ("-march=x86-64-v3",)
  assert recipe.link_flags == ("-lmvec",)
  assert "gcc -O3 -march=x86-64-v3 -fno-math-errno -c kernel.c" in recipe.comment("kernel.c")
  assert "glibc >= 2.35" in recipe.comment("kernel.c")


@pytest.mark.parametrize("macros,expected", [("__AVX512F__", 8), ("__AVX__", 4), ("__SSE2__", 2), ("__aarch64__", 2), ("", 1)])
def test_native_recipe_uses_compiler_target_macros(monkeypatch, macros: str, expected: int) -> None:
  from scaly.codegen import toolchain

  toolchain.native_recipe.cache_clear()
  commands = []

  def preprocess(command, **kwargs):
    commands.append(command)
    return subprocess.CompletedProcess(command, 0, f"#define {macros} 1\n" if macros else "", "")

  monkeypatch.setattr(toolchain.subprocess, "run", preprocess)
  monkeypatch.setattr(toolchain.platform, "libc_ver", lambda: ("", ""))
  recipe = toolchain.native_recipe("cc")
  assert recipe.lanes == expected
  assert recipe.vector_libm == "none"
  assert commands[0][-5:] == ["-dM", "-E", "-x", "c", "-"]
  toolchain.native_recipe.cache_clear()


@pytest.mark.parametrize("libc,version,expected", [("glibc", "2.35", "glibc"), ("glibc", "2.34", "none"), ("musl", "1.2", "none")])
def test_native_recipe_checks_vector_libm_host(monkeypatch, libc: str, version: str, expected: str) -> None:
  from scaly.codegen import toolchain

  toolchain.native_recipe.cache_clear()
  monkeypatch.setattr(
    toolchain.subprocess, "run", lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "#define __AVX__ 1\n#define __x86_64__ 1\n", "")
  )
  monkeypatch.setattr(toolchain.platform, "libc_ver", lambda: (libc, version))
  assert toolchain.native_recipe("cc").vector_libm == expected
  toolchain.native_recipe.cache_clear()


def test_native_recipe_uses_fixed_sve_width(monkeypatch):
  from scaly.codegen import toolchain

  toolchain.native_recipe.cache_clear()
  monkeypatch.setattr(
    toolchain.subprocess,
    "run",
    lambda command, **kwargs: subprocess.CompletedProcess(command, 0, "#define __ARM_FEATURE_SVE_BITS 256\n#define __aarch64__ 1\n", ""),
  )
  monkeypatch.setattr(toolchain.platform, "libc_ver", lambda: ("", ""))
  assert toolchain.native_recipe("cc").lanes == 4
  toolchain.native_recipe.cache_clear()


def test_plain_c_module_header_compiles_as_c99(tmp_path):
  import shutil
  import scaly as sc

  compiler = shutil.which("cc")
  if compiler is None:
    pytest.skip("C compiler required")
  x = sc.sym("x", 5)
  fun = sc.Function._from_exprs("plain_c99", [x], [x.sin() * x], ["x"], ["y"])
  module = aot.write_module(fun, tmp_path, dialect="c")
  result = subprocess.run(
    [compiler, "-std=c99", "-pedantic-errors", "-c", str(tmp_path / module.source_name), "-o", str(tmp_path / "kernel.o")],
    capture_output=True,
    text=True,
  )
  assert result.returncode == 0, result.stderr
