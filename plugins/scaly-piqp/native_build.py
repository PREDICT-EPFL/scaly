"""Build and bundle the native PIQP libraries, headers and notices."""

from __future__ import annotations

import json
import os
import platform
import shlex
import shutil
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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
  if system == "Windows":
    return f"{base}.dll"
  raise RuntimeError(f"Unsupported platform: {system}")


# Windows has no system C compiler, so the build requirements bring the zig that the JIT also uses.
_DEFAULT_CC, _DEFAULT_CXX = ("zig cc", ("zig c++",)) if platform.system() == "Windows" else ("cc", ("c++", "g++", "clang++"))
if platform.system() == "Windows" and platform.machine().lower() in {"x86_64", "amd64"}:
  # Zig defaults to the host CPU. Match BLASFEO's Haswell baseline so a wheel built on an
  # AVX-512 runner neither requires that CPU nor enters Eigen 3.4.1's broken MinGW AVX-512 path.
  _DEFAULT_CC, _DEFAULT_CXX = "zig cc -mcpu=haswell", ("zig c++ -mcpu=haswell",)


def _compiler_command(value: str) -> list[str]:
  if path := shutil.which(value):
    return [path]
  command = [part.strip('"') for part in shlex.split(value, posix=os.name != "nt")]
  if command and (path := shutil.which(command[0])):
    return [path, *command[1:]]
  return []


_BUILD_CONFIG = json.loads((Path(__file__).parent / "src" / "scaly_piqp" / "build_config.json").read_text())
PIQP_TAG = _BUILD_CONFIG["piqp"]["tag"]
EIGEN_TAG = _BUILD_CONFIG["eigen"]["tag"]
BLASFEO_TAG = _BUILD_CONFIG["blasfeo"]["tag"]


def _run(cmd: list[str], cwd: Path, env: dict | None = None, attempts: int = 1) -> None:
  """Run `cmd`, retrying a failure `attempts - 1` times; downloads use 3 to ride out a busy host."""
  for attempt in range(1, attempts + 1):
    try:
      subprocess.run(cmd, cwd=cwd, env=env, check=True)
      return
    except subprocess.CalledProcessError:
      if attempt == attempts:
        raise
      time.sleep(10 * attempt)


def _clone(hook: "BuildHookInterface", url: str, tag: str, dest: Path) -> None:
  hook.app.display_info(f"Cloning {url}@{tag} to {dest}")
  _run(["git", "clone", "--depth=1", "--branch", tag, url, str(dest)], cwd=dest.parent, attempts=3)


NOTICES = "THIRD_PARTY_NOTICES.md"


def _write_third_party_notices(
  hook: "BuildHookInterface", licenses_dir: Path, distribution: str, entries: list[tuple[str, str, str, str, list[Path]]]
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


def _build_piqp(hook: "BuildHookInterface", third_party_dir: Path, out_dir: Path, vendored: Path) -> None:
  """Build libraries, headers and licenses under `out_dir` from sources cloned into the empty `third_party_dir`.

  `vendored` holds the license texts checked into the plugin."""
  lib_dir = out_dir / "lib"
  include_dir = out_dir / "include" / "piqp"
  licenses_dir = out_dir / "licenses"
  system = platform.system()
  machine = platform.machine().lower()
  lib_name = _shared_lib_name(system, "piqpc")
  configure = ["-G", os.environ.get("CMAKE_GENERATOR", "Ninja")] if system == "Windows" else []
  command = _compiler_command(os.environ.get("CC", _DEFAULT_CC))
  if len(command) >= 2 and Path(command[0]).stem == "zig" and command[1] == "cc":
    # CMake discovers only zig.exe for ASM and cannot discover its ar/ranlib subcommands.
    # A toolchain file also propagates these commands into BLASFEO's nested ISA probes.
    toolchain = third_party_dir / "zig.cmake"
    cxx = _compiler_command(os.environ.get("CXX", "")) or [command[0], "c++", *command[2:]]
    lines = [
      f"set(CMAKE_{language}_COMPILER {' '.join(json.dumps(part.replace(chr(92), '/')) for part in args)})"
      for language, args in (("C", command), ("CXX", cxx), ("ASM", command))
    ]
    zig = json.dumps(command[0].replace("\\", "/"))
    lines += [f"set(CMAKE_AR {zig})", f"set(CMAKE_RANLIB {zig})", "set(CMAKE_ASM_COMPILER_ID Clang)"]
    rules = third_party_dir / "zig-rules.cmake"
    lines.append(f"set(CMAKE_USER_MAKE_RULES_OVERRIDE {json.dumps(rules.as_posix())})")
    archive_rules = []
    for language in ("C", "CXX", "ASM"):
      archive_rules += [
        f'set(CMAKE_{language}_ARCHIVE_CREATE "<CMAKE_AR> ar qc <TARGET> <LINK_FLAGS> <OBJECTS>")',
        f'set(CMAKE_{language}_ARCHIVE_APPEND "<CMAKE_AR> ar q <TARGET> <LINK_FLAGS> <OBJECTS>")',
        f'set(CMAKE_{language}_ARCHIVE_FINISH "<CMAKE_RANLIB> ranlib <TARGET>")',
      ]
    rules.write_text("\n".join(archive_rules) + "\n")
    toolchain.write_text("\n".join(lines) + "\n")
    configure.append(f"-DCMAKE_TOOLCHAIN_FILE={toolchain.as_posix()}")

  hook.app.display_info("Building PIQP C interface...")

  eigen_dir = third_party_dir / f"eigen-{EIGEN_TAG}"
  eigen_install_dir = third_party_dir / f"eigen-{EIGEN_TAG}-install"
  eigen_cmake_dir = eigen_install_dir / "share" / "eigen3" / "cmake"
  _clone(hook, "https://gitlab.com/libeigen/eigen.git", EIGEN_TAG, eigen_dir)
  eigen_build_dir = eigen_dir / "build"
  eigen_build_dir.mkdir()
  eigen_install_dir.mkdir()
  eigen_install_path = eigen_install_dir.resolve().as_posix()
  hook.app.display_info(f"Configuring Eigen (install to {eigen_install_path})...")
  subprocess.run(
    [
      "cmake",
      "..",
      *configure,
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

  blasfeo_dir = third_party_dir / f"blasfeo-{BLASFEO_TAG}"
  _clone(hook, "https://github.com/giaf/blasfeo.git", BLASFEO_TAG, blasfeo_dir)
  blasfeo_target, blasfeo_suffix = _blasfeo_target(system, machine)
  blasfeo_install_dir = third_party_dir / f"blasfeo-{BLASFEO_TAG}-install" / blasfeo_suffix
  blasfeo_build_dir = blasfeo_dir / f"build_{blasfeo_suffix}"
  blasfeo_build_dir.mkdir()
  blasfeo_install_dir.mkdir(parents=True)
  hook.app.display_info(f"Configuring Blasfeo ({blasfeo_target})...")
  subprocess.run(
    [
      "cmake",
      "..",
      *configure,
      "-DCMAKE_BUILD_TYPE=Release",
      "-DCMAKE_POLICY_VERSION_MINIMUM=3.5",
      f"-DTARGET={blasfeo_target}",
      "-DBLAS_API=OFF",
      "-DBLASFEO_EXAMPLES=OFF",
      f"-DCMAKE_INSTALL_PREFIX={blasfeo_install_dir.as_posix()}",
    ],
    cwd=blasfeo_build_dir,
    check=True,
  )
  hook.app.display_info("Building Blasfeo...")
  subprocess.run(["cmake", "--build", "."], cwd=blasfeo_build_dir, check=True)
  hook.app.display_info("Installing Blasfeo...")
  subprocess.run(["cmake", "--install", "."], cwd=blasfeo_build_dir, check=True)

  piqp_dir = third_party_dir / f"piqp-{PIQP_TAG}"
  _clone(hook, "https://github.com/PREDICT-EPFL/piqp.git", PIQP_TAG, piqp_dir)
  if system == "Windows":
    # Zig's -lpiqpc searches piqpc.dll, whereas CMake's MinGW default adds lib.
    # Set the target prefix before linking so the import library names the same DLL.
    c_cmake = piqp_dir / "interfaces" / "c" / "CMakeLists.txt"
    c_cmake.write_text(c_cmake.read_text() + '\nset_target_properties(piqp_c PROPERTIES PREFIX "")\n')

  build_dir = piqp_dir / "build"
  build_dir.mkdir()
  hook.app.display_info("Configuring PIQP with CMake...")
  cmake_args = [
    "cmake",
    "..",
    *configure,
    "-DCMAKE_BUILD_TYPE=Release",
    "-DBUILD_C_INTERFACE=ON",
    "-DBUILD_TESTS=OFF",
    "-DBUILD_EXAMPLES=OFF",
    "-DBUILD_PYTHON_INTERFACE=OFF",
    "-DBUILD_SHARED_LIBS=ON",
    # Eigen refuses to compile any file that is not MPL-2.0 under this define, so the build itself
    # proves the shipped library carries no LGPL Eigen code; THIRD_PARTY_NOTICES.md relies on that.
    "-DCMAKE_CXX_FLAGS=-DEIGEN_MPL2_ONLY",
    f"-DEigen3_DIR={eigen_cmake_dir.as_posix()}",
    "-DFETCHCONTENT_FULLY_DISCONNECTED=ON",
    "-DBUILD_WITH_BLASFEO=ON",
    f"-Dblasfeo_DIR={blasfeo_install_dir.as_posix()}",
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
  if system == "Windows":
    # Consumers link through the import library while ctypes loads the DLL.
    for pattern in ("*piqpc*.dll.a", "*piqpc*.lib"):
      for import_lib in build_dir.rglob(pattern):
        shutil.copy2(import_lib, lib_dir / import_lib.name)
  if system == "Darwin":
    _run(["install_name_tool", "-id", "@rpath/libpiqpc.dylib", str(dst_path)], cwd=lib_dir)

  c_include_dir = piqp_dir / "interfaces" / "c" / "include"
  for header in ("piqp.h", "piqp_typedef.h"):
    src = c_include_dir / header
    dst = include_dir / header
    hook.app.display_info(f"Copying {src} to {dst}")
    shutil.copy2(src, dst)

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
