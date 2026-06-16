from __future__ import annotations

import pytest

from alloy import toolchain


def test_solver_dir_override_discovery_and_flags(tmp_path, monkeypatch) -> None:
  include = tmp_path / "include"
  lib = tmp_path / "lib"
  (include / "piqp").mkdir(parents=True)
  (include / "coin-or").mkdir(parents=True)
  lib.mkdir(parents=True)
  (include / "piqp" / "piqp.h").write_text("// piqp")
  (include / "coin-or" / "IpStdCInterface.h").write_text("// ipopt")
  (lib / f"libpiqpc{toolchain.shared_lib_ext()}").write_text("")
  (lib / f"libipopt{toolchain.shared_lib_ext()}").write_text("")

  monkeypatch.setenv("ALLOY_SOLVER_INCLUDE_DIR", str(include))
  monkeypatch.setenv("ALLOY_SOLVER_LIB_DIR", str(lib))
  paths = toolchain.solver_paths()
  assert paths.include_dirs[0] == include
  assert paths.lib_dirs[0] == lib
  assert toolchain.solver_discoverable("piqp")
  assert toolchain.solver_discoverable("ipopt")
  assert not toolchain.solver_loadable("piqp")
  assert not toolchain.solver_loadable("ipopt")
  assert toolchain.solver_header_include("piqp") == "piqp/piqp.h"
  assert toolchain.solver_header_include("ipopt") == "coin-or/IpStdCInterface.h"

  flags = toolchain.solver_compile_flags(True, True)
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
  (lib / f"libpiqpc{toolchain.shared_lib_ext()}").write_text("")

  monkeypatch.setattr(toolchain, "_package_root", lambda: empty_pkg)
  monkeypatch.setenv("ALLOY_SOLVER_SYSTEM_FALLBACK", "0")
  monkeypatch.setenv("ALLOY_SOLVER_INCLUDE_DIR", str(include))
  monkeypatch.setenv("ALLOY_SOLVER_LIB_DIR", str(lib))
  with pytest.raises(toolchain.SolverLibraryError):
    toolchain.solver_paths(required=True)


def test_env_registry_mentions_native_build_controls() -> None:
  names = {v.name for v in toolchain.alloy_env_vars()}
  assert "ALLOY_CACHE_DIR" in names
  assert "ALLOY_CC" in names
  assert "ALLOY_BUILD_SOLVERS" in names
