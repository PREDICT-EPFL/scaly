from __future__ import annotations

import os
import subprocess
import sys

import pytest

from alloy.codegen import aot
from alloy.solvers import paths as solver_paths_module
from alloy.utils import env


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

  monkeypatch.setenv("ALLOY_SOLVER_INCLUDE_DIR", str(include))
  monkeypatch.setenv("ALLOY_SOLVER_LIB_DIR", str(lib))
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
  monkeypatch.setenv("ALLOY_SOLVER_SYSTEM_FALLBACK", "0")
  monkeypatch.setenv("ALLOY_SOLVER_INCLUDE_DIR", str(include))
  monkeypatch.setenv("ALLOY_SOLVER_LIB_DIR", str(lib))
  with pytest.raises(solver_paths_module.SolverLibraryError):
    solver_paths_module.solver_paths(required=True)


def test_aot_cli_writes_the_module_pair(tmp_path, monkeypatch, capsys) -> None:
  (tmp_path / "alloy_aot_cli_target.py").write_text(
    "import alloy as al\n\n\ndef build():\n  x = al.sym('x', 2)\n  return al.Function('aot_cli', [x], [x * x], ['x'], ['y'])\n"
  )
  monkeypatch.syspath_prepend(str(tmp_path))
  out = tmp_path / "generated"
  aot.main(["alloy_aot_cli_target:build", "-o", str(out)])
  assert "#define aot_cli_SZ_W 0" in (out / "aot_cli.h").read_text()
  assert (out / "aot_cli.c").read_text().startswith('#include "aot_cli.h"')
  assert "sz_w: 0" in capsys.readouterr().out

  # The documented invocation goes through codegen/__main__.py. Running aot itself would re-execute
  # an already-imported module (RuntimeWarning, two copies of its render-observer list), so -W error
  # is what pins the shim in place.
  env_vars = {**os.environ, "PYTHONPATH": str(tmp_path)}
  argv = [sys.executable, "-W", "error::RuntimeWarning", "-m", "alloy.codegen", "alloy_aot_cli_target:build", "-o", str(tmp_path / "cli")]
  proc = subprocess.run(argv, check=False, capture_output=True, text=True, env=env_vars)
  assert proc.returncode == 0, proc.stderr
  assert (tmp_path / "cli" / "aot_cli.h").read_text() == (out / "aot_cli.h").read_text()


def test_env_registry_mentions_native_build_controls() -> None:
  names = {v.name for v in env.alloy_env_vars()}
  assert "ALLOY_CACHE_DIR" in names
  assert "ALLOY_CC" in names
  assert "ALLOY_BUILD_SOLVERS" in names
