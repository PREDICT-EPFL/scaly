"""Custom build hook that vendors IPOPT as a shared library inside the alloy-ipopt wheel."""

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _shared_lib_name(system: str, base: str) -> str:
  if system == "Darwin":
    return f"lib{base}.dylib"
  if system == "Linux":
    return f"lib{base}.so"
  raise RuntimeError(f"Unsupported platform: {system}")


def _find_fortran_compiler(required: bool = True) -> str | None:
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
      "No Fortran compiler found in PATH. Install gfortran via `brew install gcc` (macOS) or `sudo apt-get install gfortran` (Linux)."
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
  if system == "Linux":
    return "-static-libgfortran -static-libgcc -static-libstdc++"
  return ""


IPOPT_BRANCH = "releases/3.14.19"
MUMPS_BRANCH = "releases/3.0.12"
METIS_BRANCH = "releases/2.0.1"
OPENBLAS_BRANCH = "v0.3.28"


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
  subprocess.run(cmd, cwd=cwd, env=env, check=True)


def _build_openblas(hook: "BuildHook", third_party_dir: Path, install_dir: Path) -> Path:
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
  if not any(src_dir.glob("metis-*")) and not (src_dir / "GKlib").exists():
    hook.app.display_info("Fetching METIS sources via get.Metis...")
    _run(["./get.Metis"], cwd=src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  jobs = str(os.cpu_count() or 2)
  hook.app.display_info("Configuring METIS...")
  legacy_c_cflags = "-O2 -fPIC -Wno-implicit-function-declaration -Wno-implicit-int -Wno-int-conversion -Wno-error"
  _run(
    ["./configure", f"--prefix={install_dir.resolve()}", "--disable-shared", "--with-pic", f"CFLAGS={legacy_c_cflags}"],
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
  mumps_lflags = f"-L{(mumps_install / 'lib').resolve()} -lcoinmumps -L{(metis_install / 'lib').resolve()} -lcoinmetis -lm"
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


def _macos_deps(path: Path) -> list[str]:
  """Install names `path` loads that are not part of the OS.

  Keeps `@rpath`/`@loader_path` entries: Homebrew's own libgfortran reaches libgcc_s that
  way, so following only absolute names would leave the bundle one library short."""
  out = subprocess.run(["otool", "-L", str(path)], check=True, capture_output=True, text=True).stdout
  deps = []
  for line in out.splitlines()[1:]:
    name = line.strip().split(" (")[0]
    if name.startswith(("/usr/lib/", "/System/")) or Path(name).name == path.name:
      continue  # OS library, or the dylib's own install id, which otool -L lists first
    deps.append(name)
  return deps


def _macos_rpaths(path: Path) -> list[str]:
  out = subprocess.run(["otool", "-l", str(path)], check=True, capture_output=True, text=True).stdout
  lines = out.splitlines()
  return [lines[i + 2].strip().split("path ")[1].split(" (offset")[0] for i, line in enumerate(lines) if line.strip() == "cmd LC_RPATH"]


def _resolve_macos_dep(dep: str, origin: Path) -> Path | None:
  """Locate `dep` as `origin` itself would, expanding `@rpath` against origin's own rpaths."""
  if dep.startswith("/"):
    return Path(dep) if Path(dep).exists() else None
  name = Path(dep).name
  for rpath in _macos_rpaths(origin):
    candidate = Path(rpath.replace("@loader_path", str(origin.parent)).replace("@executable_path", str(origin.parent))) / name
    if candidate.exists():
      return candidate
  sibling = origin.parent / name
  return sibling if sibling.exists() else None


def _bundle_macos_runtime(hook: "BuildHook", lib_dir: Path, lib_name: str) -> None:
  """Copy the Homebrew gcc runtime next to libipopt and repoint everything at `@rpath`.

  IPOPT links libgfortran/libquadmath by absolute Homebrew path, so a library built on one
  machine will not load on another that lacks that exact formula — which is how the macOS
  `solver tests` job broke once it stopped rebuilding IPOPT itself. Vendoring the runtime
  and adding an `@loader_path` rpath makes the shipped library self-contained, matching
  what `-static-libgfortran` already gives us on Linux."""
  # each entry pairs the copy under lib_dir with the original it came from, whose rpaths
  # are the ones that can still resolve its dependencies
  pending, bundled = [(lib_dir / lib_name, lib_dir / lib_name)], set()
  while pending:
    current, origin = pending.pop()
    for dep in _macos_deps(current):
      name = Path(dep).name
      if name not in bundled:
        src = _resolve_macos_dep(dep, origin)
        if src is None:
          raise RuntimeError(f"cannot vendor {dep!r} needed by {current}: not found via its rpaths")
        bundled.add(name)
        dst = lib_dir / name
        hook.app.display_info(f"Bundling {src} -> {dst}")
        shutil.copy2(src, dst)
        dst.chmod(0o755)
        _run(["install_name_tool", "-id", f"@rpath/{name}", str(dst)], cwd=lib_dir)
        pending.append((dst, src))
      if dep != f"@rpath/{name}":
        _run(["install_name_tool", "-change", dep, f"@rpath/{name}", str(current)], cwd=lib_dir)
    # `@loader_path` lets a dlopen'd libipopt find its siblings without the host adding an rpath
    subprocess.run(["install_name_tool", "-add_rpath", "@loader_path", str(current)], cwd=lib_dir, capture_output=True)
    # rewriting load commands invalidates the signature; arm64 refuses to load an unsigned image
    subprocess.run(["codesign", "--force", "--sign", "-", str(current)], cwd=lib_dir, capture_output=True)
  hook.app.display_info(f"Bundled macOS runtime: {sorted(bundled) or 'nothing to bundle'}")


def _macos_self_contained(lib_dir: Path, lib_name: str) -> bool:
  """Every non-OS dependency in the closure resolves to a sibling under `lib_dir`."""
  for path in [lib_dir / lib_name, *(p for p in lib_dir.glob("*.dylib") if p.name != lib_name)]:
    for dep in _macos_deps(path):
      if not (lib_dir / Path(dep).name).exists():
        return False
  return True


def _ipopt_built(system: str, lib_dir: Path, include_dir: Path) -> bool:
  lib_name = _shared_lib_name(system, "ipopt")
  has_lib = (lib_dir / lib_name).exists()
  if system == "Linux":
    has_lib = has_lib and any(lib_dir.glob("libipopt.so.*"))
  if system == "Darwin" and has_lib and not _macos_self_contained(lib_dir, lib_name):
    return False  # built before the runtime was vendored: still absolute-linked, rebuild
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
  if system == "Darwin":
    lapack_lflags = "-framework Accelerate"
  else:
    openblas_install = _build_openblas(hook, third_party_dir, third_party_dir / "openblas_install")
    lapack_lflags = f"-L{(openblas_install / 'lib').resolve()} -lopenblas -lm -lpthread -lgfortran"

  metis_install = _build_metis(hook, third_party_dir, third_party_dir / "metis_install")
  mumps_install = _build_mumps(hook, third_party_dir, third_party_dir / "mumps_install", metis_install, lapack_lflags, static_ldflags, fc)
  ipopt_install = _build_ipopt(
    hook, third_party_dir, third_party_dir / "ipopt_install", mumps_install, metis_install, lapack_lflags, static_ldflags, fc
  )

  lib_dir.mkdir(parents=True, exist_ok=True)
  (include_dir / "coin-or").mkdir(parents=True, exist_ok=True)
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
  if system == "Darwin":
    _run(["install_name_tool", "-id", f"@rpath/{lib_name}", str(dst_path)], cwd=lib_dir)
    _bundle_macos_runtime(hook, lib_dir, lib_name)

  ipopt_headers_dir = ipopt_install / "include" / "coin-or"
  for header in ipopt_headers_dir.glob("*.h"):
    shutil.copy2(header, include_dir / "coin-or" / header.name)
  hook.app.display_info("IPOPT build complete.")


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "alloy"

  def initialize(self, version: str, build_data: dict) -> None:
    if self.target_name == "sdist":
      self.app.display_info("Skipping vendored solver build for sdist target")
      return

    root = Path(self.root)
    third_party_dir = root / "third_party"
    lib_dir = root / "src" / "alloy_ipopt" / "lib"
    include_dir = root / "src" / "alloy_ipopt" / "include"
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

    missing = [] if _ipopt_built(system, lib_dir, include_dir) else _missing_ipopt_tools()
    if missing:
      msg = f"missing native toolchain for IPOPT build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(f"{msg}. Install gfortran via `brew install gcc` (macOS) or `sudo apt-get install gfortran` (Linux).")
      self.app.display_info(f"Skipping IPOPT build for editable install ({msg}); set ALLOY_BUILD_SOLVERS=required to make this fatal.")
    else:
      _build_ipopt_stack(self, third_party_dir, lib_dir, include_dir)

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    for path in (root / "src" / "alloy_ipopt" / "lib", root / "src" / "alloy_ipopt" / "include", root / "third_party"):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
