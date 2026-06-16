"""
Custom build hook that vendors PIQP and IPOPT as shared libraries inside the alloy wheel.

Outputs land under `src/alloy/lib/` (libraries) and `src/alloy/include/` (headers), and are
loaded by the Python runtime via ctypes and linked by AOT C++ consumers.

The build is deliberately cached: each component is skipped if its install artifacts already
exist on disk. Cold build is ~5-8 minutes; warm builds (CI cache hit) are near-instant.

Static-linking note: IPOPT and MUMPS are linked with `-static-libgfortran -static-libgcc
-static-libstdc++` so the resulting libipopt has no runtime dependency on the host's Fortran
toolchain, which keeps the AOT story simple for external C++ users.
"""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


# ----------------------------- platform helpers ------------------------------------


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


def _find_fortran_compiler(required: bool = True) -> str | None:
  """Locate a Fortran 90 compiler.

  Homebrew's `gcc` formula on macOS runners ships gfortran as a versioned binary
  (`gfortran-16`, `gfortran-15`, …) without an unversioned `gfortran` symlink. Autoconf's
  default FC detection only probes `gfortran`/`f95`/`f90`, so we need to hand it the
  explicit path. Linux usually ships an unversioned `gfortran` from `apt install gfortran`.
  """
  fc = os.environ.get("FC")
  if fc:
    path = shutil.which(fc) or (fc if Path(fc).exists() else None)
    if path:
      return path
    if required:
      raise RuntimeError(f"FC={fc!r} was set, but no such compiler was found.")
    return None

  candidates = ["gfortran", "gfortran-16", "gfortran-15", "gfortran-14", "gfortran-13", "gfortran-12", "gfortran-11"]
  for name in candidates:
    path = shutil.which(name)
    if path:
      return path

  versioned: list[tuple[int, str]] = []
  for entry in os.environ.get("PATH", "").split(os.pathsep):
    if not entry:
      continue
    try:
      for child in Path(entry).iterdir():
        match = re.fullmatch(r"gfortran-(\d+)", child.name)
        if match and child.is_file() and os.access(child, os.X_OK):
          versioned.append((int(match.group(1)), str(child)))
    except OSError:
      pass
  if versioned:
    return max(versioned)[1]

  if required:
    raise RuntimeError(
      "No Fortran compiler found in PATH. Install gfortran via `brew install gcc` (macOS) "
      "or `sudo apt-get install gfortran` (Linux)."
    )
  return None


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


def _missing_ipopt_tools() -> list[str]:
  missing = [cmd for cmd in ("git", "make") if shutil.which(cmd) is None]
  if shutil.which("cc") is None:
    missing.append("cc")
  if not _has_cxx_compiler():
    missing.append("c++")
  if _find_fortran_compiler(required=False) is None:
    missing.append("gfortran")
  return missing


def _static_fortran_ldflags(system: str) -> str:
  """Flags that ask gfortran to statically pull libgfortran/libgcc into the linked shared lib.

  These flags are only meaningful when gfortran acts as the link driver — they go to FCFLAGS
  rather than LDFLAGS so the C/C++ configure conftests (driven by clang/gcc) don't fail.
  Passing them as LDFLAGS to autoconf breaks the configure step on macOS with Apple Clang.
  See MIGRATION_NOTES.md ("Static gfortran linking") for the current limitation.
  """
  if system == "Linux":
    return "-static-libgfortran -static-libgcc -static-libstdc++"
  # macOS: rely on Homebrew gfortran's default linkage for v1. Achieving a fully
  # static libgfortran on macOS requires resolving the Apple Clang vs gfortran driver
  # split (likely via explicit `-Wl,-force_load <libgfortran.a>`).
  return ""


# ------------------------------ PIQP build -----------------------------------------


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

  # Eigen 3.4.1
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

  # Blasfeo
  blasfeo_dir = third_party_dir / "blasfeo"
  blasfeo_install_root = third_party_dir / "blasfeo_install"
  blasfeo_install_dir = None
  if not blasfeo_dir.exists():
    third_party_dir.mkdir(parents=True, exist_ok=True)
    hook.app.display_info(f"Cloning Blasfeo to {blasfeo_dir}")
    subprocess.run(
      ["git", "clone", "--depth=1", "https://github.com/giaf/blasfeo.git", str(blasfeo_dir)],
      check=True,
    )

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

  # PIQP v0.6.2
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
  for loc in (
    build_dir / "interfaces" / "c" / lib_name,
    build_dir / "lib" / lib_name,
    build_dir / lib_name,
  ):
    if loc.exists():
      built_lib = loc
      break
  if built_lib is None:
    for f in build_dir.rglob(lib_name):
      built_lib = f
      break
  if built_lib is None:
    raise RuntimeError(f"Could not find built library {lib_name} in {build_dir}")

  hook.app.display_info(f"Copying {built_lib} to {lib_dir}")
  shutil.copy2(built_lib, lib_dir / lib_name)

  c_include_dir = piqp_dir / "interfaces" / "c" / "include"
  for header in ("piqp.h", "piqp_typedef.h"):
    src = c_include_dir / header
    dst = include_dir / header
    hook.app.display_info(f"Copying {src} to {dst}")
    shutil.copy2(src, dst)

  hook.app.display_info("PIQP C interface build complete.")


# ------------------------------ IPOPT build ----------------------------------------


# Versions/branches pinned to known-good combinations.
IPOPT_BRANCH = "releases/3.14.19"
MUMPS_BRANCH = "releases/3.0.12"
METIS_BRANCH = "releases/2.0.1"
OPENBLAS_BRANCH = "v0.3.28"


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
  subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _build_openblas(hook: "BuildHook", third_party_dir: Path, install_dir: Path) -> Path:
  """Build OpenBLAS on Linux. Returns the install prefix."""
  marker = install_dir / "lib" / "libopenblas.a"
  if marker.exists():
    hook.app.display_info(f"Using existing OpenBLAS install at {install_dir}")
    return install_dir
  src_dir = third_party_dir / "openblas"
  if not src_dir.exists():
    hook.app.display_info(f"Cloning OpenBLAS {OPENBLAS_BRANCH} to {src_dir}")
    _run(
      ["git", "clone", "--depth=1", "--branch", OPENBLAS_BRANCH, "https://github.com/OpenMathLib/OpenBLAS.git", str(src_dir)],
      cwd=third_party_dir,
    )
  hook.app.display_info("Building OpenBLAS (this can take a few minutes)...")
  jobs = str(os.cpu_count() or 2)
  build_flags = ["NO_SHARED=1", "USE_OPENMP=0", "DYNAMIC_ARCH=1"]
  _run(["make", f"-j{jobs}", *build_flags], cwd=src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  # OpenBLAS' install target recomputes the library filename from the build flags;
  # repeat them here or it tries to install a shared libopenblas*.so even though we
  # only built the static archive.
  _run(["make", f"PREFIX={install_dir.resolve()}", *build_flags, "install"], cwd=src_dir)
  return install_dir


def _coinor_clone(hook: "BuildHook", url: str, branch: str, dest: Path) -> None:
  if dest.exists():
    return
  hook.app.display_info(f"Cloning {url}@{branch} to {dest}")
  _run(["git", "clone", "--depth=1", "--branch", branch, url, str(dest)], cwd=dest.parent)


def _build_metis(hook: "BuildHook", third_party_dir: Path, install_dir: Path) -> Path:
  marker = install_dir / "lib"
  if any(marker.glob("libcoinmetis*")) if marker.exists() else False:
    hook.app.display_info(f"Using existing METIS install at {install_dir}")
    return install_dir
  src_dir = third_party_dir / "ThirdParty-Metis"
  _coinor_clone(hook, "https://github.com/coin-or-tools/ThirdParty-Metis.git", METIS_BRANCH, src_dir)
  # Download upstream METIS tarball into the source tree. METIS 1.x/2.x bundle different
  # upstream METIS releases under varying directory names, so look for any `metis*` dir.
  if not any(src_dir.glob("metis-*")) and not (src_dir / "GKlib").exists():
    hook.app.display_info("Fetching METIS sources via get.Metis...")
    _run(["./get.Metis"], cwd=src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  jobs = str(os.cpu_count() or 2)
  hook.app.display_info("Configuring METIS...")
  # METIS 4.0.3 has K&R-style implicit declarations that modern clang rejects by default.
  # Downgrade to warnings to keep the upstream sources buildable on Apple Clang 17+ / Clang 19+.
  legacy_c_cflags = (
    "-O2 -fPIC -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-error"
  )
  _run(
    [
      "./configure",
      f"--prefix={install_dir.resolve()}",
      "--disable-shared",
      "--with-pic",
      f"CFLAGS={legacy_c_cflags}",
    ],
    cwd=src_dir,
  )
  hook.app.display_info("Building METIS...")
  _run(["make", f"-j{jobs}"], cwd=src_dir)
  _run(["make", "install"], cwd=src_dir)
  return install_dir


def _build_mumps(
  hook: "BuildHook",
  third_party_dir: Path,
  install_dir: Path,
  metis_install: Path,
  lapack_lflags: str,
  static_ldflags: str,
  fc: str,
) -> Path:
  marker = install_dir / "lib"
  if marker.exists() and any(marker.glob("libcoinmumps*")):
    hook.app.display_info(f"Using existing MUMPS install at {install_dir}")
    return install_dir
  src_dir = third_party_dir / "ThirdParty-Mumps"
  _coinor_clone(hook, "https://github.com/coin-or-tools/ThirdParty-Mumps.git", MUMPS_BRANCH, src_dir)
  if not (src_dir / "MUMPS").exists():
    hook.app.display_info("Fetching MUMPS sources via get.Mumps...")
    _run(["./get.Mumps"], cwd=src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  jobs = str(os.cpu_count() or 2)
  metis_cflags = f"-I{(metis_install / 'include' / 'coin-or' / 'metis').resolve()}"
  # ThirdParty-Metis builds a static libcoinmetis whose objects reference libm
  # (sqrtf/log/pow). Because we pass explicit user lflags to MUMPS configure,
  # autoconf does not append its probed -lm for us.
  metis_lflags = f"-L{(metis_install / 'lib').resolve()} -lcoinmetis -lm"
  hook.app.display_info(f"Configuring MUMPS (FC={fc})...")
  configure_args = [
    "./configure",
    f"--prefix={install_dir.resolve()}",
    "--disable-shared",
    "--with-pic",
    f"--with-metis-cflags={metis_cflags}",
    f"--with-metis-lflags={metis_lflags}",
    f"--with-lapack-lflags={lapack_lflags}",
    f"FC={fc}",
  ]
  if static_ldflags:
    configure_args.append(f"FCFLAGS={static_ldflags}")
  _run(configure_args, cwd=src_dir)
  hook.app.display_info("Building MUMPS (this can take a couple of minutes)...")
  _run(["make", f"-j{jobs}"], cwd=src_dir)
  _run(["make", "install"], cwd=src_dir)
  return install_dir


def _build_ipopt(
  hook: "BuildHook",
  third_party_dir: Path,
  install_dir: Path,
  mumps_install: Path,
  metis_install: Path,
  lapack_lflags: str,
  static_ldflags: str,
  fc: str,
) -> Path:
  src_dir = third_party_dir / "Ipopt"
  _coinor_clone(hook, "https://github.com/coin-or/Ipopt.git", IPOPT_BRANCH, src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  jobs = str(os.cpu_count() or 2)
  mumps_cflags = f"-I{(mumps_install / 'include' / 'coin-or' / 'mumps').resolve()}"
  mumps_lflags = (
    f"-L{(mumps_install / 'lib').resolve()} -lcoinmumps "
    f"-L{(metis_install / 'lib').resolve()} -lcoinmetis -lm"
  )
  build_dir = src_dir / "build"
  build_dir.mkdir(exist_ok=True)
  hook.app.display_info(f"Configuring IPOPT (FC={fc})...")
  configure_args = [
    "../configure",
    f"--prefix={install_dir.resolve()}",
    "--disable-java",
    "--enable-shared",
    "--with-pic",
    f"--with-mumps-cflags={mumps_cflags}",
    f"--with-mumps-lflags={mumps_lflags}",
    f"--with-lapack-lflags={lapack_lflags}",
    f"FC={fc}",
  ]
  if static_ldflags:
    configure_args.append(f"FCFLAGS={static_ldflags}")
  _run(configure_args, cwd=build_dir)
  hook.app.display_info("Building IPOPT (this can take a few minutes)...")
  _run(["make", f"-j{jobs}"], cwd=build_dir)
  _run(["make", "install"], cwd=build_dir)
  return install_dir


def _ipopt_built(system: str, lib_dir: Path, include_dir: Path) -> bool:
  lib_name = _shared_lib_name(system, "ipopt")
  has_lib = (lib_dir / lib_name).exists()
  if system == "Linux":
    has_lib = has_lib and any(lib_dir.glob("libipopt.so.*"))
  return has_lib and (include_dir / "coin-or" / "IpStdCInterface.h").exists()


def _linux_major_so_name(path: Path) -> str:
  parts = path.name.split(".")
  return ".".join(parts[:3]) if len(parts) >= 3 else path.name


def _build_ipopt_stack(hook: "BuildHook", third_party_dir: Path, lib_dir: Path, include_dir: Path) -> None:
  system = platform.system()
  lib_name = _shared_lib_name(system, "ipopt")

  if _ipopt_built(system, lib_dir, include_dir):
    hook.app.display_info(f"IPOPT already built at {lib_dir}")
    return

  hook.app.display_info("Building IPOPT stack (METIS -> MUMPS -> IPOPT)...")
  third_party_dir.mkdir(parents=True, exist_ok=True)

  if system not in {"Darwin", "Linux"}:
    raise RuntimeError(f"Unsupported platform: {system}")

  static_ldflags = _static_fortran_ldflags(system)
  fc = _find_fortran_compiler()
  hook.app.display_info(f"Using Fortran compiler: {fc}")

  # LAPACK choice per platform. Keep this after the Fortran preflight so a missing
  # compiler fails before spending minutes building OpenBLAS on a cold Linux checkout.
  if system == "Darwin":
    lapack_lflags = "-framework Accelerate"
  else:
    openblas_install = _build_openblas(hook, third_party_dir, third_party_dir / "openblas_install")
    lapack_lflags = f"-L{(openblas_install / 'lib').resolve()} -lopenblas -lm -lpthread -lgfortran"

  metis_install = _build_metis(hook, third_party_dir, third_party_dir / "metis_install")
  mumps_install = _build_mumps(
    hook,
    third_party_dir,
    third_party_dir / "mumps_install",
    metis_install,
    lapack_lflags,
    static_ldflags,
    fc,
  )
  ipopt_install = _build_ipopt(
    hook,
    third_party_dir,
    third_party_dir / "ipopt_install",
    mumps_install,
    metis_install,
    lapack_lflags,
    static_ldflags,
    fc,
  )

  # Copy the shared lib and headers out into src/alloy/.
  lib_dir.mkdir(parents=True, exist_ok=True)
  (include_dir / "coin-or").mkdir(parents=True, exist_ok=True)

  # Prefer the versioned dylib (libipopt.X.dylib) over the symlink so we get a real file
  # with the correct LC_ID. We rename it to the unversioned name in src/alloy/lib.
  if system == "Darwin":
    versioned = sorted((ipopt_install / "lib").glob("libipopt.*.dylib"))
    if not versioned:
      raise RuntimeError(f"Could not find versioned libipopt under {ipopt_install / 'lib'}")
    built_lib = versioned[-1]
  else:
    versioned = sorted((ipopt_install / "lib").glob("libipopt.so.*"))
    built_lib = versioned[-1] if versioned else ipopt_install / "lib" / lib_name

  dst_path = lib_dir / lib_name
  hook.app.display_info(f"Copying {built_lib} to {dst_path}")
  shutil.copy2(built_lib, dst_path)
  if system == "Linux" and built_lib.name != lib_name:
    soname_path = lib_dir / _linux_major_so_name(built_lib)
    hook.app.display_info(f"Copying {built_lib} to {soname_path}")
    shutil.copy2(built_lib, soname_path)

  # Rewrite install_name to @rpath so consumers can load this from any layout (the JIT
  # runtime resolves rpath at load time; AOT consumers set -rpath at link time).
  if system == "Darwin":
    _run(["install_name_tool", "-id", f"@rpath/{lib_name}", str(dst_path)], cwd=lib_dir)
    # Also rewrite the install_name embedded in PIQP's lib in case it points at build dir.
    piqp_lib = lib_dir / "libpiqpc.dylib"
    if piqp_lib.exists():
      _run(["install_name_tool", "-id", f"@rpath/libpiqpc.dylib", str(piqp_lib)], cwd=lib_dir)

  ipopt_headers_dir = ipopt_install / "include" / "coin-or"
  for header in ipopt_headers_dir.glob("*.h"):
    shutil.copy2(header, include_dir / "coin-or" / header.name)

  hook.app.display_info("IPOPT build complete.")


# ------------------------------ hatch hook -----------------------------------------


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "alloy"

  def initialize(self, version: str, build_data: dict) -> None:
    if self.target_name == "sdist":
      self.app.display_info("Skipping vendored solver build for sdist target")
      return

    root = Path(self.root)
    third_party_dir = root / "third_party"
    lib_dir = root / "src" / "alloy" / "lib"
    piqp_include_dir = root / "src" / "alloy" / "include" / "piqp"
    base_include_dir = root / "src" / "alloy" / "include"

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
      msg = "Windows IPOPT build not yet supported — track in https://github.com/PREDICT-EPFL/alloy/issues/1"
      if strict:
        raise RuntimeError(msg)
      self.app.display_info(f"Skipping vendored solver build for editable install: {msg}")
      return

    missing = [] if _piqp_built(system, lib_dir, piqp_include_dir) else _missing_piqp_tools()
    if missing:
      msg = f"missing native toolchain for PIQP build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(f"{msg}. Install CMake, git, and a C/C++ compiler.")
      self.app.display_info(f"Skipping PIQP build for editable install ({msg}); set ALLOY_BUILD_SOLVERS=required to make this fatal.")
    else:
      _build_piqp(self, third_party_dir, lib_dir, piqp_include_dir)

    missing = [] if _ipopt_built(system, lib_dir, base_include_dir) else _missing_ipopt_tools()
    if missing:
      msg = f"missing native toolchain for IPOPT build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(
          f"{msg}. Install gfortran via `brew install gcc` (macOS) or `sudo apt-get install gfortran` (Linux)."
        )
      self.app.display_info(f"Skipping IPOPT build for editable install ({msg}); set ALLOY_BUILD_SOLVERS=required to make this fatal.")
    else:
      _build_ipopt_stack(self, third_party_dir, lib_dir, base_include_dir)

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    for path in (
      root / "src" / "alloy" / "lib",
      root / "src" / "alloy" / "include",
      root / "third_party",
    ):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
