"""Custom build hook that vendors PIQP as a shared library inside the scaly-piqp wheel."""

from __future__ import annotations

import json
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
  raw = os.environ.get("SCALY_BUILD_SOLVERS", "auto").strip().lower()
  if raw in _BUILD_SOLVER_SKIP:
    return "skip"
  if raw in _BUILD_SOLVER_REQUIRE:
    return "require"
  if raw in {"", "auto"}:
    return "auto"
  raise RuntimeError("SCALY_BUILD_SOLVERS must be one of auto, required/1/true, or skip/0/false")


def _has_cxx_compiler() -> bool:
  return any(shutil.which(cmd) for cmd in ("c++", "g++", "clang++"))


def _missing_piqp_tools(*, build_piqp: bool) -> list[str]:
  missing = [cmd for cmd in ("git", "cmake") if build_piqp and shutil.which(cmd) is None]
  if build_piqp and shutil.which("cc") is None:
    missing.append("cc")
  if build_piqp and not _has_cxx_compiler():
    missing.append("c++")
  return missing


_BUILD_CONFIG = json.loads((Path(__file__).parent / "src" / "scaly_piqp" / "build_config.json").read_text())
PIQP_TAG = _BUILD_CONFIG["piqp"]["tag"]
EIGEN_TAG = _BUILD_CONFIG["eigen"]["tag"]
BLASFEO_TAG = _BUILD_CONFIG["blasfeo"]["tag"]


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
  subprocess.run(cmd, cwd=cwd, env=env, check=True)


NOTICES = "THIRD_PARTY_NOTICES.md"


def _write_third_party_notices(
  hook: "BuildHook", licenses_dir: Path, distribution: str, entries: list[tuple[str, str, str, str, list[Path]]]
) -> None:
  """Copy each bundled dependency's license texts into `licenses_dir/<name>/` and index them in THIRD_PARTY_NOTICES.md.

  Entries are (name, version, license, upstream URL, license files in the cloned source). Copying
  from the pinned sources at build time keeps the notices from drifting from `build_config.json`."""
  if licenses_dir.exists():
    shutil.rmtree(licenses_dir)
  lines = [
    f"# Third-party notices for {distribution}",
    "",
    "The native libraries under `lib/` in this distribution bundle the software below. The license",
    "texts of each row, copied from the pinned upstream sources when the wheel was built, are in the",
    "named subdirectory.",
    "",
    "| Component | Version | License | Upstream | Texts |",
    "|---|---|---|---|---|",
  ]
  for name, version, license_id, url, files in entries:
    dst = licenses_dir / name
    dst.mkdir(parents=True)
    for src in files:
      if not src.exists():
        raise RuntimeError(f"license text {src} for {name} is missing; upstream moved it, so update the notices entry")
      shutil.copy2(src, dst / src.name)
    lines.append(f"| {name} | {version} | {license_id} | {url} | `{name}/` |")
  (licenses_dir / NOTICES).write_text("\n".join(lines) + "\n")
  hook.app.display_info(f"Wrote {licenses_dir / NOTICES}")


def _piqp_built(system: str, lib_dir: Path, include_dir: Path, licenses_dir: Path) -> bool:
  """Whether the library, its headers and its notices are there, built from the pins in
  ``build_config.json``. The notices record the version of every component they were built from,
  so bumping a pin rebuilds instead of keeping the old library."""
  lib_path = lib_dir / _shared_lib_name(system, "piqpc")
  notices = licenses_dir / NOTICES
  if not (lib_path.exists() and notices.exists() and (include_dir / "piqp.h").exists() and (include_dir / "piqp_typedef.h").exists()):
    return False
  text = notices.read_text()
  return all(f"| {name} | {pin['version']} |" in text for name, pin in _BUILD_CONFIG.items())


def _build_piqp(hook: "BuildHook", third_party_dir: Path, lib_dir: Path, include_dir: Path, licenses_dir: Path) -> None:
  system = platform.system()
  machine = platform.machine().lower()
  lib_name = _shared_lib_name(system, "piqpc")

  if _piqp_built(system, lib_dir, include_dir, licenses_dir):
    hook.app.display_info(f"PIQP C interface already built at {lib_dir}")
    return

  hook.app.display_info("Building PIQP C interface...")

  # Every source and install directory is named by its pinned tag, so a bumped pin clones and
  # builds afresh instead of reusing the previous version's tree.
  eigen_dir = third_party_dir / f"eigen-{EIGEN_TAG}"
  eigen_install_dir = third_party_dir / f"eigen-{EIGEN_TAG}-install"
  eigen_cmake_dir = eigen_install_dir / "share" / "eigen3" / "cmake"
  third_party_dir.mkdir(parents=True, exist_ok=True)
  if not eigen_dir.exists():
    hook.app.display_info(f"Cloning Eigen {EIGEN_TAG} to {eigen_dir}")
    subprocess.run(
      ["git", "clone", "--depth=1", "--branch", EIGEN_TAG, "https://gitlab.com/libeigen/eigen.git", str(eigen_dir)],
      check=True,
    )
  if not eigen_cmake_dir.exists():
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

  blasfeo_dir = third_party_dir / f"blasfeo-{BLASFEO_TAG}"
  blasfeo_install_root = third_party_dir / f"blasfeo-{BLASFEO_TAG}-install"
  if not blasfeo_dir.exists():
    hook.app.display_info(f"Cloning Blasfeo {BLASFEO_TAG} to {blasfeo_dir}")
    subprocess.run(["git", "clone", "--depth=1", "--branch", BLASFEO_TAG, "https://github.com/giaf/blasfeo.git", str(blasfeo_dir)], check=True)

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

  piqp_dir = third_party_dir / f"piqp-{PIQP_TAG}"
  if not piqp_dir.exists():
    hook.app.display_info(f"Cloning PIQP {PIQP_TAG} to {piqp_dir}")
    subprocess.run(
      ["git", "clone", "--depth=1", "--branch", PIQP_TAG, "https://github.com/PREDICT-EPFL/piqp.git", str(piqp_dir)],
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
    # Eigen refuses to compile any file that is not MPL-2.0 under this define, so the build itself
    # proves the shipped library carries no LGPL Eigen code; THIRD_PARTY_NOTICES.md relies on that.
    "-DCMAKE_CXX_FLAGS=-DEIGEN_MPL2_ONLY",
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

  vendored = third_party_dir.parent / "licenses"
  piqp_version = _BUILD_CONFIG["piqp"]["version"]
  _write_third_party_notices(
    hook,
    licenses_dir,
    "scaly-piqp",
    [
      ("piqp", piqp_version, "BSD-2-Clause", "https://github.com/PREDICT-EPFL/piqp", [piqp_dir / "LICENSE"]),
      (
        "ldl",
        f"modified copy inside PIQP {piqp_version}",
        "LGPL-2.1-or-later",
        "https://github.com/DrTimothyAldenDavis/SuiteSparse",
        [piqp_dir / "include" / "piqp" / "sparse" / "LDL_License.txt", vendored / "LGPL-2.1.txt"],
      ),
      (
        "eigen",
        _BUILD_CONFIG["eigen"]["version"],
        "MPL-2.0 (compiled with EIGEN_MPL2_ONLY)",
        "https://gitlab.com/libeigen/eigen",
        [eigen_dir / name for name in ("COPYING.MPL2", "COPYING.BSD", "COPYING.README")],
      ),
      ("blasfeo", _BUILD_CONFIG["blasfeo"]["version"], "BSD-2-Clause", "https://github.com/giaf/blasfeo", [blasfeo_dir / "LICENSE.txt"]),
    ],
  )
  hook.app.display_info("PIQP C interface build complete.")


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "scaly"

  def initialize(self, version: str, build_data: dict) -> None:
    if self.target_name == "sdist":
      self.app.display_info("Skipping vendored solver build for sdist target")
      return

    root = Path(self.root)
    third_party_dir = root / "third_party"
    lib_dir = root / "src" / "scaly_piqp" / "lib"
    include_dir = root / "src" / "scaly_piqp" / "include" / "piqp"
    licenses_dir = root / "src" / "scaly_piqp" / "licenses"
    mode = _solver_build_mode()
    strict = mode == "require" or (mode == "auto" and version != "editable")
    system = platform.system()

    if mode != "skip" or any(lib_dir.glob("lib*")):
      build_data["pure_python"] = False
      build_data["infer_tag"] = True
    if mode == "skip":
      self.app.display_info("Skipping vendored solver build because SCALY_BUILD_SOLVERS=skip/0")
      return
    if system == "Windows":
      msg = "Windows PIQP build not yet supported — track in https://github.com/PREDICT-EPFL/scaly/issues/1"
      if strict:
        raise RuntimeError(msg)
      self.app.display_info(f"Skipping vendored solver build for editable install: {msg}")
      return

    missing = _missing_piqp_tools(build_piqp=not _piqp_built(system, lib_dir, include_dir, licenses_dir))
    if missing:
      msg = f"missing native toolchain for PIQP build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(f"{msg}. Install CMake, git, and a C/C++ compiler.")
      self.app.display_info(f"Skipping PIQP build for editable install ({msg}); set SCALY_BUILD_SOLVERS=required to make this fatal.")
    else:
      _build_piqp(self, third_party_dir, lib_dir, include_dir, licenses_dir)

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    package_dir = root / "src" / "scaly_piqp"
    for path in (package_dir / "lib", package_dir / "include", package_dir / "licenses", root / "third_party"):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
