"""Where the external solvers' headers and libraries are found: the directory overrides, discovery
without loading, the compile and link flags, the refusal when a required library is missing, and
the build variables in the environment registry."""

from __future__ import annotations

import pytest

from scaly.opt.external import paths as solver_paths_module
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


def test_env_registry_mentions_the_solver_build_mode() -> None:
  assert "SCALY_BUILD_SOLVERS" in {v.name for v in env.scaly_env_vars()}
