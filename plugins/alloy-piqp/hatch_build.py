"""Custom build hook that vendors PIQP as a shared library inside the alloy-piqp wheel."""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _blasfeo_target(system: str, machine: str) -> tuple[str, str]:
  if machine in {"x86_64", "amd64"}:
    return "X64_INTEL_HASWELL", "x64_avx2"
  if system == "Darwin" and machine in {"arm64", "aarch64"}:
    return "ARMV8A_APPLE_M1", "arm64"
  if system == "Linux" and machine in {"arm64", "aarch64"}:
    return "ARMV8A_ARM_CORTEX_A76", "arm64"
  return "GENERIC", "generic"


def _shared_lib_name(system: str, base: str) -> str:
  if system == "Darwin":
    return f"lib{base}.dylib"
  if system == "Linux":
    return f"lib{base}.so"
  raise RuntimeError(f"Unsupported platform: {system}")


_BUILD_SOLVER_SKIP = {"0", "false", "no", "off", "skip"}
_BUILD_SOLVER_REQUIRE = {"1", "true", "yes", "on", "required", "require", "force"}


def _solver_build_mode() -> str:
  raw = os.environ.get("ALLOY_BUILD_SOLVERS", "auto").strip().lower()
  if raw in _BUILD_SOLVER_SKIP:
    return "skip"
  if raw in _BUILD_SOLVER_REQUIRE:
    return "require"
  if raw in {"", "auto"}:
    return "auto"
  raise RuntimeError("ALLOY_BUILD_SOLVERS must be one of auto, required/1/true, or skip/0/false")


def _has_cxx_compiler() -> bool:
  return any(shutil.which(cmd) for cmd in ("c++", "g++", "clang++"))


def _missing_piqp_tools() -> list[str]:
  missing = [cmd for cmd in ("git", "cmake") if shutil.which(cmd) is None]
  if shutil.which("cc") is None:
    missing.append("cc")
  if not _has_cxx_compiler():
    missing.append("c++")
  return missing


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
  subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _piqp_built(system: str, lib_dir: Path, include_dir: Path) -> bool:
  lib_path = lib_dir / _shared_lib_name(system, "piqpc")
  return lib_path.exists() and (include_dir / "piqp.h").exists() and (include_dir / "piqp_typedef.h").exists()


def _build_piqp(hook: "BuildHook", third_party_dir: Path, lib_dir: Path, include_dir: Path) -> None:
  system = platform.system()
  machine = platform.machine().lower()
  lib_name = _shared_lib_name(system, "piqpc")

  if _piqp_built(system, lib_dir, include_dir):
    hook.app.display_info(f"PIQP C interface already built at {lib_dir}")
    return

  hook.app.display_info("Building PIQP C interface...")

  eigen_dir = third_party_dir / "eigen"
  eigen_install_dir = third_party_dir / "eigen_install"
  eigen_cmake_dir = eigen_install_dir / "share" / "eigen3" / "cmake"
  if not eigen_cmake_dir.exists():
    third_party_dir.mkdir(parents=True, exist_ok=True)
    if not eigen_dir.exists():
      hook.app.display_info(f"Cloning Eigen 3.4.1 to {eigen_dir}")
      subprocess.run(
        ["git", "clone", "--depth=1", "--branch", "3.4.1", "https://gitlab.com/libeigen/eigen.git", str(eigen_dir)],
        check=True,
      )
    eigen_build_dir = eigen_dir / "build"
    eigen_build_dir.mkdir(exist_ok=True)
    eigen_install_dir.mkdir(exist_ok=True)
    eigen_install_path = str(eigen_install_dir.resolve())
    hook.app.display_info(f"Configuring Eigen (install to {eigen_install_path})...")
    subprocess.run(
      [
        "cmake",
        "..",
        f"-DCMAKE_INSTALL_PREFIX={eigen_install_path}",
        "-DBUILD_TESTING=OFF",
        "-DEIGEN_BUILD_DOC=OFF",
        "-DEIGEN_BUILD_BLAS=OFF",
        "-DEIGEN_BUILD_LAPACK=OFF",
      ],
      cwd=eigen_build_dir,
      check=True,
    )
    hook.app.display_info("Installing Eigen...")
    subprocess.run(["cmake", "--install", "."], cwd=eigen_build_dir, check=True)
  else:
    hook.app.display_info(f"Using existing Eigen install at {eigen_install_dir}")

  blasfeo_dir = third_party_dir / "blasfeo"
  blasfeo_install_root = third_party_dir / "blasfeo_install"
  if not blasfeo_dir.exists():
    third_party_dir.mkdir(parents=True, exist_ok=True)
    hook.app.display_info(f"Cloning Blasfeo to {blasfeo_dir}")
    subprocess.run(["git", "clone", "--depth=1", "https://github.com/giaf/blasfeo.git", str(blasfeo_dir)], check=True)

  blasfeo_target, blasfeo_suffix = _blasfeo_target(system, machine)
  blasfeo_install_dir = blasfeo_install_root / blasfeo_suffix
  blasfeo_include = blasfeo_install_dir / "include" / "blasfeo_target.h"
  blasfeo_lib_marker = blasfeo_install_dir / "lib"
  if not blasfeo_include.exists() or not blasfeo_lib_marker.exists():
    blasfeo_build_dir = blasfeo_dir / f"build_{blasfeo_suffix}"
    blasfeo_build_dir.mkdir(exist_ok=True)
    blasfeo_install_dir.mkdir(parents=True, exist_ok=True)
    hook.app.display_info(f"Configuring Blasfeo ({blasfeo_target})...")
    subprocess.run(
      [
        "cmake",
        "..",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
        f"-DTARGET={blasfeo_target}",
        "-DBLAS_API=OFF",
        "-DBLASFEO_EXAMPLES=OFF",
        f"-DCMAKE_INSTALL_PREFIX={blasfeo_install_dir}",
      ],
      cwd=blasfeo_build_dir,
      check=True,
    )
    hook.app.display_info("Building Blasfeo...")
    subprocess.run(["cmake", "--build", "."], cwd=blasfeo_build_dir, check=True)
    hook.app.display_info("Installing Blasfeo...")
    subprocess.run(["cmake", "--install", "."], cwd=blasfeo_build_dir, check=True)
  else:
    hook.app.display_info(f"Using existing Blasfeo install at {blasfeo_install_dir}")

  piqp_dir = third_party_dir / "piqp"
  if not piqp_dir.exists():
    third_party_dir.mkdir(parents=True, exist_ok=True)
    hook.app.display_info(f"Cloning PIQP v0.6.2 to {piqp_dir}")
    subprocess.run(
      ["git", "clone", "--depth=1", "--branch", "v0.6.2", "https://github.com/PREDICT-EPFL/piqp.git", str(piqp_dir)],
      check=True,
    )
  else:
    hook.app.display_info(f"Using existing PIQP source at {piqp_dir}")

  build_dir = piqp_dir / "build"
  build_dir.mkdir(exist_ok=True)
  hook.app.display_info("Configuring PIQP with CMake...")
  cmake_args = [
    "cmake",
    "..",
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_C_INTERFACE=ON",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_PYTHON_INTERFACE=OFF",
    "-DBUILD_SHARED_LIBS=ON",
    f"-DEigen3_DIR={eigen_cmake_dir}",
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    "-DBUILD_WITH_BLASFEO=ON",
    f"-Dblasfeo_DIR={blasfeo_install_dir}",
  ]
  subprocess.run(cmake_args, cwd=build_dir, check=True)
  hook.app.display_info("Building PIQP...")
  subprocess.run(["cmake", "--build", ".", "--config", "Release", "--target", "piqp_c", "-j"], cwd=build_dir, check=True)

  lib_dir.mkdir(parents=True, exist_ok=True)
  include_dir.mkdir(parents=True, exist_ok=True)
  built_lib = None
  for loc in (build_dir / "interfaces" / "c" / lib_name, build_dir / "lib" / lib_name, build_dir / lib_name):
    if loc.exists():
      built_lib = loc
      break
  if built_lib is None:
    for f in build_dir.rglob(lib_name):
      built_lib = f
      break
  if built_lib is None:
    raise RuntimeError(f"Could not find built library {lib_name} in {build_dir}")

  dst_path = lib_dir / lib_name
  hook.app.display_info(f"Copying {built_lib} to {dst_path}")
  shutil.copy2(built_lib, dst_path)
  if system == "Darwin":
    _run(["install_name_tool", "-id", "@rpath/libpiqpc.dylib", str(dst_path)], cwd=lib_dir)

  c_include_dir = piqp_dir / "interfaces" / "c" / "include"
  for header in ("piqp.h", "piqp_typedef.h"):
    src = c_include_dir / header
    dst = include_dir / header
    hook.app.display_info(f"Copying {src} to {dst}")
    shutil.copy2(src, dst)
  hook.app.display_info("PIQP C interface build complete.")


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "alloy"

  def initialize(self, version: str, build_data: dict) -> None:
    if self.target_name == "sdist":
      self.app.display_info("Skipping vendored solver build for sdist target")
      return

    root = Path(self.root)
    third_party_dir = root / "third_party"
    lib_dir = root / "src" / "alloy_piqp" / "lib"
    include_dir = root / "src" / "alloy_piqp" / "include" / "piqp"
    mode = _solver_build_mode()
    strict = mode == "require" or (mode == "auto" and version != "editable")
    system = platform.system()

    if mode != "skip" or any(lib_dir.glob("lib*")):
      build_data["pure_python"] = False
      build_data["infer_tag"] = True
    if mode == "skip":
      self.app.display_info("Skipping vendored solver build because ALLOY_BUILD_SOLVERS=skip/0")
      return
    if system == "Windows":
      msg = "Windows PIQP build not yet supported — track in https://github.com/PREDICT-EPFL/alloy/issues/1"
      if strict:
        raise RuntimeError(msg)
      self.app.display_info(f"Skipping vendored solver build for editable install: {msg}")
      return

    missing = [] if _piqp_built(system, lib_dir, include_dir) else _missing_piqp_tools()
    if missing:
      msg = f"missing native toolchain for PIQP build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(f"{msg}. Install CMake, git, and a C/C++ compiler.")
      self.app.display_info(f"Skipping PIQP build for editable install ({msg}); set ALLOY_BUILD_SOLVERS=required to make this fatal.")
    else:
      _build_piqp(self, third_party_dir, lib_dir, include_dir)

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    for path in (root / "src" / "alloy_piqp" / "lib", root / "src" / "alloy_piqp" / "include", root / "third_party"):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
