# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "hatchling", "packaging", "cmake",
#   "auditwheel; sys_platform == 'linux'", "patchelf; sys_platform == 'linux'",
#   "delvewheel==1.13.2; sys_platform == 'win32'", "pefile==2024.8.26; sys_platform == 'win32'",
# ]
# ///
"""Build the Ipopt native stack for the hook or a complete host-native archive."""

from __future__ import annotations

import hashlib
import ast
import io
import gzip
import json
import os
import platform
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
import tarfile
import zlib
from contextlib import contextmanager
from pathlib import Path, PurePosixPath

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


def _shared_lib_name(system: str, base: str) -> str:
  if system == "Darwin":
    return f"lib{base}.dylib"
  if system == "Linux":
    return f"lib{base}.so"
  if system == "Windows":
    return f"libscaly_{base}-3.dll"
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


def _wheel_tag() -> str:
  """`py3-none-<platform>`: the solvers load through ctypes, so one wheel serves every Python.

  The platform part is the one hatchling's `infer_tag` would pick."""
  from hatchling.builders.macos import process_macos_plat_tag
  from packaging.tags import sys_tags

  plat = next(t.platform for t in sys_tags() if "manylinux" not in t.platform and "musllinux" not in t.platform)
  if sys.platform == "darwin":
    plat = process_macos_plat_tag(plat, compat=False)
  return f"py3-none-{plat}"


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


def _missing_ipopt_tools() -> list[str]:
  missing = [cmd for cmd in ("git", "make", "cmake") if shutil.which(cmd) is None]
  if os.name == "nt":
    missing += [cmd for cmd in ("ninja", "objdump") if shutil.which(cmd) is None]
  if shutil.which("cc") is None:
    missing.append("cc")
  if not _has_cxx_compiler():
    missing.append("c++")
  fc = _find_fortran_compiler(required=False)
  if fc is None:
    missing.append("gfortran")
  elif os.name == "nt":
    try:
      versions = dict(line.split(" ", 1) for line in _windows_packages(fc).splitlines())
    except (OSError, subprocess.CalledProcessError):
      missing.append("MSYS2 UCRT64 with pacman")
    else:
      for component in ("gcc-runtime", "winpthreads"):
        expected = _BUILD_CONFIG[component]["version"]
        for name in _BUILD_CONFIG[component]["packages"]:
          package = f"mingw-w64-ucrt-x86_64-{name}"
          if versions.get(package) != expected:
            missing.append(f"{package} {expected} (found {versions.get(package, 'missing')})")
  return missing


_BUILD_CONFIG = json.loads((Path(__file__).parent / "src" / "scaly_ipopt" / "build_config.json").read_text())
IPOPT_BRANCH = _BUILD_CONFIG["ipopt"]["branch"]
MUMPS_BRANCH = _BUILD_CONFIG["mumps"]["coinor_branch"]
METIS_TAG = _BUILD_CONFIG["metis"]["tag"]
GKLIB_COMMIT = _BUILD_CONFIG["metis"]["gklib_commit"]
OPENBLAS_BRANCH = _BUILD_CONFIG["blas"]["linux"]["branch"]
_BUILD_LOG: Path | None = None
_COMMAND_LOG: Path | None = None


def _run(cmd: list[str], cwd: Path, env: dict | None = None, attempts: int = 1) -> None:
  """Run `cmd`, retrying a failure `attempts - 1` times; downloads use 3 to ride out a busy host."""
  if _COMMAND_LOG is not None:
    with _COMMAND_LOG.open("a", encoding="utf-8") as log:
      log.write(json.dumps({"cwd": str(cwd), "args": cmd}) + "\n")
  if os.name == "nt":
    fc = _find_fortran_compiler()
    bash = Path(fc).parents[2] / "usr" / "bin" / "bash.exe"
    args = [re.sub(r"([A-Za-z]):/(?!/)", lambda m: f"/{m[1].lower()}/", arg.replace("\\", "/")) for arg in cmd]
    cmd = [str(bash), "-lc", shlex.join(args)]
    env = dict(os.environ if env is None else env, MSYSTEM="UCRT64", CHERE_INVOKING="1")
  for attempt in range(1, attempts + 1):
    try:
      if _BUILD_LOG is None:
        subprocess.run(cmd, cwd=cwd, env=env, check=True)
      else:
        with _BUILD_LOG.open("a", encoding="utf-8") as log:
          subprocess.run(cmd, cwd=cwd, env=env, check=True, stdout=log, stderr=subprocess.STDOUT)
      return
    except subprocess.CalledProcessError:
      if attempt == attempts:
        raise
      time.sleep(10 * attempt)


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


def _build_openblas(hook: "BuildHook", third_party_dir: Path, install_dir: Path) -> Path:
  src_dir = third_party_dir / "openblas"
  _clone(hook, "https://github.com/OpenMathLib/OpenBLAS.git", OPENBLAS_BRANCH, src_dir)
  hook.app.display_info("Building OpenBLAS (this can take a few minutes)...")
  jobs = str(os.cpu_count() or 2)
  build_flags = ["NO_SHARED=1", "USE_OPENMP=0", "DYNAMIC_ARCH=1"]
  # Recursive BLAS/LAPACK builds update one archive. Lock writers without reducing compiler
  # parallelism or OpenBLAS's independently detected runtime thread capacity.
  for variable, tool in (("AR", "ar"), ("RANLIB", "ranlib")):
    command = [sys.executable, str(Path(__file__).resolve()), "_archive", str(third_party_dir / "openblas-archive.lock"), shutil.which(tool) or tool]
    build_flags.append(f"{variable}={shlex.join(command)}")
  # Fix the common-code baseline instead of detecting a potentially unsupported build CPU.
  # DYNAMIC_ARCH still selects optimized kernels at runtime.
  if platform.machine().lower() in {"x86_64", "amd64"}:
    build_flags.append("TARGET=PRESCOTT")
  _run(["make", f"-j{jobs}", *build_flags], cwd=src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  _run(["make", f"PREFIX={install_dir.resolve()}", *build_flags, "install"], cwd=src_dir)
  return install_dir


def _archive_native(lock: Path, command: list[str]) -> None:
  with _locked(lock):
    subprocess.run(command, check=True)


def _clone(hook: "BuildHook", url: str, branch: str, dest: Path) -> None:
  if dest.exists():
    _check_checkout(dest, branch)
    return
  hook.app.display_info(f"Cloning {url}@{branch} to {dest}")
  _run(["git", "clone", "--depth=1", "--branch", branch, url, str(dest)], cwd=dest.parent, attempts=3)


def _clone_commit(hook: "BuildHook", url: str, commit: str, dest: Path) -> None:
  """Shallow-fetch one commit; `git clone --depth=1 --branch` only accepts branches and tags."""
  if dest.exists():
    _check_checkout(dest, commit)
    return
  hook.app.display_info(f"Fetching {url}@{commit} to {dest}")
  dest.mkdir(parents=True)
  _run(["git", "init", "-q"], cwd=dest)
  _run(["git", "fetch", "-q", "--depth=1", url, commit], cwd=dest, attempts=3)
  _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


def _check_checkout(dest: Path, revision: str) -> None:
  def rev(name):
    return subprocess.check_output(["git", "-C", str(dest), "rev-parse", name + "^0"], text=True).strip()

  if rev("HEAD") != rev(revision):
    raise RuntimeError(f"Retained checkout {dest} does not match {revision}; use a new work directory")


def _cmake_build(hook: "BuildHook", src_dir: Path, build_dir: Path, install_dir: Path, defines: list[str]) -> None:
  build_dir.mkdir(parents=True, exist_ok=True)
  common = [
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DCMAKE_INSTALL_PREFIX={install_dir.resolve()}",
    "-DCMAKE_INSTALL_LIBDIR=lib",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DBUILD_SHARED_LIBS=OFF",
  ]
  if os.name == "nt":
    common += ["-G", "Ninja", "-DCMAKE_C_STANDARD=11"]
  _run(["cmake", str(src_dir.resolve()), *common, *defines], cwd=build_dir)
  _run(["cmake", "--build", ".", f"-j{os.cpu_count() or 2}"], cwd=build_dir)
  _run(["cmake", "--install", "."], cwd=build_dir)


def _build_metis(hook: "BuildHook", third_party_dir: Path, install_dir: Path) -> Path:
  """Build GKlib and METIS 5 (both Apache-2.0) as static libraries into one prefix.

  METIS 5 no longer bundles GKlib, so GKlib is built first and METIS is pointed at the same
  prefix. `make config` upstream only prepends the index and real widths to `metis.h` before
  calling CMake, so the hook does the same and MUMPS gets the 32-bit `idx_t` it requires."""
  gklib_src = third_party_dir / "GKlib"
  _clone_commit(hook, "https://github.com/KarypisLab/GKlib.git", GKLIB_COMMIT, gklib_src)
  hook.app.display_info("Building GKlib...")
  _cmake_build(hook, gklib_src, gklib_src / "build", install_dir, ["-DGKLIB_BUILD_APPS=OFF"])

  metis_src = third_party_dir / "METIS"
  _clone(hook, "https://github.com/KarypisLab/METIS.git", METIS_TAG, metis_src)
  # METIS' GCC flags hardcode -march=native, which would tie the wheel to the build host.
  gkbuild = metis_src / "conf" / "gkbuild.cmake"
  gkbuild.write_text(gkbuild.read_text().replace(" -march=native", ""))
  if os.name == "nt":
    cmake = metis_src / "CMakeLists.txt"
    cmake.write_text(cmake.read_text().replace('add_subdirectory("programs")', ""))
  xinclude = metis_src / "build" / "xinclude"
  xinclude.mkdir(parents=True, exist_ok=True)
  (xinclude / "metis.h").write_text("#define IDXTYPEWIDTH 32\n#define REALTYPEWIDTH 32\n" + (metis_src / "include" / "metis.h").read_text())
  shutil.copy2(metis_src / "include" / "CMakeLists.txt", xinclude / "CMakeLists.txt")
  hook.app.display_info("Building METIS...")
  # METIS declares cmake_minimum_required 2.8, which CMake 4 refuses without this override.
  _cmake_build(hook, metis_src, metis_src / "build", install_dir, [f"-DGKLIB_PATH={install_dir.resolve()}", "-DCMAKE_POLICY_VERSION_MINIMUM=3.5"])
  return install_dir


def _build_mumps(
  hook: "BuildHook",
  third_party_dir: Path,
  install_dir: Path,
  metis_install: Path,
  lapack_lflags: str,
  fc: str,
) -> Path:
  src_dir = third_party_dir / "ThirdParty-Mumps"
  _clone(hook, "https://github.com/coin-or-tools/ThirdParty-Mumps.git", MUMPS_BRANCH, src_dir)
  hook.app.display_info("Fetching MUMPS sources via get.Mumps...")
  if not (src_dir / "MUMPS").exists():
    _run(["./get.Mumps"], cwd=src_dir, attempts=3)
  install_dir.mkdir(parents=True, exist_ok=True)
  jobs = str(os.cpu_count() or 2)
  metis_cflags = f"-I{(metis_install / 'include').resolve()}"
  metis_lflags = f"-L{(metis_install / 'lib').resolve()} -lmetis -lGKlib -lm"
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
  if os.name == "nt":
    configure_args.append("--without-openmp")
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
  ldflags: str,
  fc: str,
) -> Path:
  src_dir = third_party_dir / "Ipopt"
  _clone(hook, "https://github.com/coin-or/Ipopt.git", IPOPT_BRANCH, src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  jobs = str(os.cpu_count() or 2)
  mumps_cflags = f"-I{(mumps_install / 'include' / 'coin-or' / 'mumps').resolve()}"
  mumps_lflags = f"-L{(mumps_install / 'lib').resolve()} -lcoinmumps -L{(metis_install / 'lib').resolve()} -lmetis -lGKlib -lm"
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
  if ldflags:
    configure_args.append(f"LDFLAGS={ldflags}")
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
  and adding an `@loader_path` rpath makes the shipped library self-contained, the same job
  `_bundle_linux_runtime` does with `$ORIGIN`."""
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


# libgcc_s and libstdc++ are part of every glibc distribution's base install, but the Fortran
# runtime only arrives with gfortran, so it is the piece a machine without a toolchain lacks.
_LINUX_VENDORED = ("libgfortran", "libquadmath")

# `$ORIGIN` has to reach the linker through two layers that both eat a `$`: configure copies
# LDFLAGS into the generated Makefiles verbatim, where make turns `$$` back into `$`, and the
# recipe shell then needs the backslash or it expands `$ORIGIN` to the empty string -- which
# leaves a bare `-Wl,-rpath` that swallows the next argument and breaks the link.
LINUX_RPATH_LDFLAGS = r"-Wl,-rpath,\$$ORIGIN"


def _linux_runtime_deps(path: Path) -> dict[str, str | None]:
  """Soname -> resolved path, over `path`'s whole closure, for the runtime we vendor.

  `ldd` reports the transitive closure with each object's own rpath applied, so a vendored
  sibling shows up here as the sibling and a missing library as None."""
  out = subprocess.run(["ldd", str(path)], check=True, capture_output=True, text=True).stdout
  deps = {}
  for line in out.splitlines():
    soname, _, resolved = line.strip().partition(" => ")
    if soname.split(" (")[0].startswith(_LINUX_VENDORED):
      deps[soname.split(" (")[0]] = None if not resolved or "not found" in resolved else resolved.split(" (")[0]
  return deps


def _bundle_linux_runtime(hook: "BuildHook", lib_dir: Path, lib_name: str) -> None:
  """Copy the Fortran runtime next to libipopt.so, where its `$ORIGIN` rpath will find it.

  Same reasoning as `_bundle_macos_runtime`: IPOPT links libgfortran dynamically, so without
  this the shipped library only loads on machines that have gfortran installed."""
  bundled = _linux_runtime_deps(lib_dir / lib_name)
  for soname, src in bundled.items():
    if src is None:
      raise RuntimeError(f"cannot vendor {soname!r} needed by {lib_dir / lib_name}: ldd could not resolve it")
    dst = lib_dir / soname
    hook.app.display_info(f"Bundling {src} -> {dst}")
    shutil.copy2(src, dst)
    dst.chmod(0o755)
  hook.app.display_info(f"Bundled Linux runtime: {sorted(bundled) or 'nothing to bundle'}")


def _linux_self_contained(lib_dir: Path, lib_name: str) -> bool:
  """Every vendored dependency in the closure resolves to a sibling under `lib_dir`."""
  return all(resolved and Path(resolved).parent == lib_dir.resolve() for resolved in _linux_runtime_deps(lib_dir.resolve() / lib_name).values())


def _linux_major_so_name(path: Path) -> str:
  parts = path.name.split(".")
  return ".".join(parts[:3]) if len(parts) >= 3 else path.name


def _windows_deps(path: Path) -> list[str]:
  import pefile

  with pefile.PE(str(path)) as pe:
    return [
      entry.dll.decode("ascii") for directory in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT") for entry in getattr(pe, directory, ())
    ]


def _bundle_windows_runtime(hook: "BuildHook", lib_dir: Path, lib_name: str, runtime_dir: Path) -> None:
  system = Path(os.environ["SystemRoot"]) / "System32"
  pending, seen = [lib_dir / lib_name], set()
  while pending:
    current = pending.pop()
    if current.name.lower() in seen:
      continue
    seen.add(current.name.lower())
    for name in _windows_deps(current):
      src = runtime_dir / name
      if name.lower().startswith(("api-ms-win-", "ext-ms-win-")) or (not src.is_file() and (system / name).is_file()):
        continue
      dst = lib_dir / name
      if not dst.exists():
        if not src.is_file():
          raise RuntimeError(f"cannot vendor {name!r} needed by {current}: not found in the toolchain")
        hook.app.display_info(f"Bundling {src} -> {dst}")
        shutil.copy2(src, dst)
      pending.append(dst)


def _namespace_windows_runtime(lib_dir: Path, lib_name: str) -> None:
  import pefile

  dlls = sorted(lib_dir.glob("*.dll"))
  digest = hashlib.sha256(b"".join(path.read_bytes() for path in dlls)).hexdigest()[:12]
  names = {path.name.lower(): f"scaly_ipopt_{digest}_{path.name}" for path in dlls if path.name != lib_name}
  for path in dlls:
    changes = [arg for dep in _windows_deps(path) if dep.lower() in names for arg in ("-change", dep, names[dep.lower()])]
    if changes:
      subprocess.run([sys.executable, "-m", "delvewheel", "replace-needed", str(path), "--strip", *changes], check=True)
  for old, new in names.items():
    (lib_dir / old).rename(lib_dir / new)
  with pefile.PE(str(lib_dir / lib_name)) as pe:
    exports = sorted(symbol.name.decode("ascii") for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols if symbol.name is not None)
  definition = lib_dir / "ipopt.def"
  definition.write_text("EXPORTS\n" + "\n".join(exports) + "\n", encoding="ascii")
  subprocess.run(["dlltool", "-m", "i386:x86-64", "-D", lib_name, "-d", str(definition), "-l", str(lib_dir / "ipopt.lib")], check=True)
  definition.unlink()


# Everything else that decides the built files, beside this hook, the pins and the license texts.
_KEY_ENV = ("CC", "CXX", "FC", "CFLAGS", "CXXFLAGS", "FFLAGS", "FCFLAGS", "CPPFLAGS", "LDFLAGS", "MACOSX_DEPLOYMENT_TARGET")
_OUTPUTS = ("lib", "include", "licenses")
_CACHE_MAX_AGE = 30 * 24 * 3600


def _cache_root() -> Path:
  if path := os.environ.get("SCALY_SOLVER_CACHE"):
    return Path(path)
  return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "scaly" / "solvers"


def _tool_identity(cmd: str) -> str:
  path = shutil.which(cmd)
  if path is None:
    return f"{cmd}: none"
  out = subprocess.run([path, "--version"], capture_output=True, text=True).stdout.strip()
  return f"{path}: {out.splitlines()[0] if out else ''}"


def _windows_packages(fc: str) -> str:
  bash = Path(fc).resolve().parent.parent.parent / "usr" / "bin" / "bash.exe"
  return subprocess.run([str(bash), "-lc", "pacman -Q"], check=True, capture_output=True, text=True).stdout


def _remove_tree(path: Path) -> None:
  def remove_readonly(function, filename, error):
    if not isinstance(error, PermissionError):
      raise error
    os.chmod(filename, stat.S_IWRITE)
    function(filename)

  if path.exists():
    shutil.rmtree(path, onexc=remove_readonly)


def _cache_key(root: Path, bundle: bytes | None = None) -> str:
  """Identify a source build as before, or an explicit archive by its full content SHA-256.

  Archive identity needs no compiler and never uses the archive's path or modification time.
  Recipe identity for producers is separate, see `_recipe_hash`.
  """
  if bundle is not None:
    return "bundle-" + hashlib.sha256(bundle).hexdigest()
  digest = hashlib.sha256()
  inputs = [Path(__file__), root / "src" / "scaly_ipopt" / "build_config.json", *(root / "licenses").rglob("*")]
  for path in sorted(p for p in inputs if p.is_file()):
    digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
  fc = _find_fortran_compiler(required=False) or "gfortran"
  tools = [_tool_identity(os.environ.get("CC", "cc")), _tool_identity(os.environ.get("CXX", "c++")), _tool_identity(fc)]
  if os.name == "nt" and Path(fc).is_file():
    from importlib.metadata import version

    try:
      tools.append(_windows_packages(fc))
    except (OSError, subprocess.CalledProcessError) as error:
      tools.append(f"pacman: {error}")
    tools += [version("delvewheel"), version("pefile")]
  env = [f"{name}={os.environ.get(name, '')}" for name in _KEY_ENV]
  for part in (platform.system(), platform.machine(), *platform.libc_ver(), platform.mac_ver()[0], *tools, *env):
    digest.update(part.encode() + b"\0")
  return digest.hexdigest()[:16]


@contextmanager
def _locked(path: Path, *, wait: bool = True):
  """Hold an exclusive lock on `path`, yielding whether it was taken. Lock files are never deleted."""
  path.parent.mkdir(parents=True, exist_ok=True)
  if os.name == "nt":
    import msvcrt

    with path.open("a+b") as f:
      if f.seek(0, os.SEEK_END) == 0:
        f.write(b"\0")
        f.flush()
      while True:
        f.seek(0)
        try:
          msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
          break
        except OSError:
          if not wait:
            yield False
            return
          time.sleep(0.1)
      try:
        yield True
      finally:
        f.seek(0)
        msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)
    return
  import fcntl

  with path.open("w") as f:
    try:
      fcntl.flock(f, fcntl.LOCK_EX if wait else fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
      yield False
      return
    yield True


def _prune_cache(cache: Path) -> None:
  """Delete builds unused for 30 days. A checkout holds its own copy, so this only ever costs a rebuild."""
  cutoff = time.time() - _CACHE_MAX_AGE
  with _locked(cache / "locks" / "prune.lock", wait=False) as pruning:
    for path in cache.iterdir() if pruning else ():
      if path.name == "locks":
        continue
      with _locked(cache / "locks" / f"{path.name.removeprefix('build-')}.lock", wait=False) as held:
        # a build may have replaced or published the path since it was listed
        if held and path.exists() and path.stat().st_mtime <= cutoff:
          _remove_tree(path)


def _install_build(
  hook: "BuildHook", root: Path, package_dir: Path, entry: Path, key: str, missing_tools, build, *, manifest: dict | None = None
) -> list[str]:
  """Copy the cached build `entry` into `package_dir`, first running `build(src_dir, out_dir)` if no checkout made it yet.

  Returns the native tools that are missing when a build is needed and cannot run. A build runs in a fresh scratch directory and is renamed into the cache only once complete, so a
  cache entry is never partial. A failed build keeps its scratch directory for inspection until the
  next attempt at the same key replaces it.

  Source cache entries retain lib/include/licenses. Archive entries use the relocatable archive
  root, including scaly_ipopt.libs. With `manifest`, validate cached and installed bytes and
  install them without native repair.
  """
  cache = entry.parent
  with _locked(cache / "locks" / f"{entry.name}.lock"):
    if not entry.exists():
      if missing := missing_tools():
        return missing
      scratch = cache / f"build-{entry.name}"
      _remove_tree(scratch)
      (scratch / "src").mkdir(parents=True)
      try:
        build(scratch / "src", scratch / "out")
      except BaseException:
        hook.app.display_error(f"The build failed; its sources and build trees are kept in {scratch}")
        raise
      (scratch / "out").rename(entry)
      _remove_tree(scratch)
    if manifest is not None:
      try:
        _verify_payload(entry, manifest)
      except (RuntimeError, OSError) as error:
        raise RuntimeError(f"Invalid native cache entry {entry}; remove it and reinstall: {error}") from error
    os.utime(entry)
    stamp = root / ".build_key"
    stamp.unlink(missing_ok=True)
    sibling = package_dir.parent / "scaly_ipopt.libs"
    shutil.rmtree(sibling, ignore_errors=True)
    (package_dir / "native_manifest.json").unlink(missing_ok=True)
    for name in _OUTPUTS:
      shutil.rmtree(package_dir / name, ignore_errors=True)
      shutil.copytree((entry / "scaly_ipopt" if manifest is not None else entry) / name, package_dir / name, symlinks=True)
    if manifest is not None:
      if (entry / "scaly_ipopt.libs").exists():
        shutil.copytree(entry / "scaly_ipopt.libs", sibling, symlinks=True)
      shutil.copy2(entry / "manifest.json", package_dir / "native_manifest.json")
      _verify_payload(package_dir.parent, manifest)
    stamp.write_text(key)
  hook.app.display_info(f"Installed the build {entry} into {package_dir}")
  _prune_cache(cache)
  return []


def _build_ipopt_stack(hook: "BuildHook", third_party_dir: Path, lib_dir: Path, include_dir: Path, licenses_dir: Path, vendored: Path) -> None:
  """The single native recipe, shared by the ordinary source hook and bundle producer.

  Build lib/include/licenses from pinned sources in `third_party_dir`. Retained checkouts must
  match their pinned revisions. Native build tools can incrementally rebuild those directories.
  `vendored` holds checked-in license texts. The producer additionally freezes the complete
  repaired layout and its manifest; the default hook's wheel repair remains with cibuildwheel.
  """
  system = platform.system()
  lib_name = _shared_lib_name(system, "ipopt")
  hook.app.display_info("Building IPOPT stack (METIS -> MUMPS -> IPOPT)...")
  if system not in {"Darwin", "Linux", "Windows"}:
    raise RuntimeError(f"Unsupported platform: {system}")
  if system == "Windows" and platform.machine().lower() not in {"x86_64", "amd64"}:
    raise RuntimeError("Windows IPOPT builds require x86-64 Python.")
  ldflags = LINUX_RPATH_LDFLAGS if system == "Linux" else ""
  fc = _find_fortran_compiler()
  hook.app.display_info(f"Using Fortran compiler: {fc}")
  if system == "Darwin":
    lapack_lflags = "-framework Accelerate"
  else:
    openblas_install = _build_openblas(hook, third_party_dir, third_party_dir / "openblas_install")
    lapack_lflags = f"-L{(openblas_install / 'lib').resolve()} -lopenblas -lm -lpthread -lgfortran"

  metis_install = _build_metis(hook, third_party_dir, third_party_dir / "metis_install")
  mumps_install = _build_mumps(hook, third_party_dir, third_party_dir / "mumps_install", metis_install, lapack_lflags, fc)
  ipopt_install = _build_ipopt(hook, third_party_dir, third_party_dir / "ipopt_install", mumps_install, metis_install, lapack_lflags, ldflags, fc)

  lib_dir.mkdir(parents=True, exist_ok=True)
  (include_dir / "coin-or").mkdir(parents=True, exist_ok=True)
  if system == "Darwin":
    versioned = sorted((ipopt_install / "lib").glob("libipopt.*.dylib"))
    if not versioned:
      raise RuntimeError(f"Could not find versioned libipopt under {ipopt_install / 'lib'}")
    built_lib = versioned[-1]
  elif system == "Windows":
    built_lib = ipopt_install / "bin" / "libipopt-3.dll"
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
  elif system == "Windows":
    _bundle_windows_runtime(hook, lib_dir, lib_name, Path(fc).parent)
    _namespace_windows_runtime(lib_dir, lib_name)
  else:
    _bundle_linux_runtime(hook, lib_dir, lib_name)
    if not _linux_self_contained(lib_dir, lib_name):
      raise RuntimeError(f"{dst_path} still resolves its Fortran runtime outside {lib_dir}: {_linux_runtime_deps(dst_path)}")

  ipopt_headers_dir = ipopt_install / "include" / "coin-or"
  for header in ipopt_headers_dir.glob("*.h"):
    shutil.copy2(header, include_dir / "coin-or" / header.name)

  mumps_src = third_party_dir / "ThirdParty-Mumps"
  gklib_src = third_party_dir / "GKlib"
  runtime = sorted(p.name for p in lib_dir.iterdir() if p.name.startswith(("libgfortran", "libquadmath", "libgcc_s", "libstdc++")))
  entries = [
    ("ipopt", _BUILD_CONFIG["ipopt"]["version"], "EPL-2.0", "https://github.com/coin-or/Ipopt", [third_party_dir / "Ipopt" / "LICENSE"]),
    ("mumps", _BUILD_CONFIG["mumps"]["version"], "CeCILL-C", "https://mumps-solver.org", [mumps_src / "MUMPS" / "LICENSE"]),
    (
      "thirdparty-mumps",
      _BUILD_CONFIG["mumps"]["coinor_version"],
      "EPL-2.0",
      "https://github.com/coin-or-tools/ThirdParty-Mumps",
      [mumps_src / "LICENSE"],
    ),
    ("metis", _BUILD_CONFIG["metis"]["version"], "Apache-2.0", "https://github.com/KarypisLab/METIS", [third_party_dir / "METIS" / "LICENSE"]),
    (
      "gklib",
      f"commit {GKLIB_COMMIT[:12]}",
      "Apache-2.0, plus the LGPL-2.1-or-later and BSD-3-Clause files listed in LICENSES.md",
      "https://github.com/KarypisLab/GKlib",
      [gklib_src / "LICENSE.txt", gklib_src / "LICENSES.md", gklib_src / "LICENSES" / "BSD-3-Clause-MT.txt", gklib_src / "LICENSES" / "LGPL-2.1.txt"],
    ),
    (
      "gcc-runtime",
      _BUILD_CONFIG["gcc-runtime"]["version"] if system == "Windows" else ", ".join(runtime),
      "GPL-3.0-or-later WITH GCC-exception-3.1",
      "https://gcc.gnu.org",
      [vendored / "gcc-runtime" / name for name in ("COPYING3", "COPYING.RUNTIME")],
    ),
  ]
  if system != "Darwin":
    openblas_src = third_party_dir / "openblas"
    entries.append(
      (
        "openblas",
        _BUILD_CONFIG["blas"]["linux"]["version"],
        "BSD-3-Clause",
        "https://github.com/OpenMathLib/OpenBLAS",
        [openblas_src / "LICENSE", openblas_src / "GotoBLAS_00License.txt"],
      )
    )
  if system == "Windows":
    entries.append(
      (
        "winpthreads",
        _BUILD_CONFIG["winpthreads"]["version"],
        "MIT and BSD-3-Clause",
        "https://www.mingw-w64.org",
        [Path(fc).parents[1] / "share" / "licenses" / "libwinpthread" / "COPYING"],
      )
    )
  _write_third_party_notices(hook, licenses_dir, "scaly-ipopt", entries)
  hook.app.display_info("IPOPT build complete.")


def _host_target() -> dict:
  system, arch = platform.system(), platform.machine().lower()
  arch = {"amd64": "x86_64", "arm64": "aarch64" if system == "Linux" else "arm64"}.get(arch, arch)
  floor = {"Windows": "Windows 10 UCRT", "Linux": "glibc " + platform.libc_ver()[1], "Darwin": "macOS " + platform.mac_ver()[0]}[system]
  return {"system": system, "arch": arch, "platform_tag": _wheel_tag().removeprefix("py3-none-"), "floor": floor}


def _safe_name(name: str) -> PurePosixPath:
  path = PurePosixPath(name)
  reserved = {"CON", "PRN", "AUX", "NUL", *(f"COM{i}" for i in range(1, 10)), *(f"LPT{i}" for i in range(1, 10))}
  if (
    not name
    or path.is_absolute()
    or "\\" in name
    or ":" in name
    or re.search(r'[<>"|?*\x00-\x1f]', name)
    or str(path) != name
    or any(p in {".", ".."} or p.endswith((".", " ")) or p.split(".")[0].upper() in reserved for p in path.parts)
  ):
    raise RuntimeError(f"Unsafe archive path: {name!r}")
  return path


def _payload_name(name: str) -> bool:
  parts = _safe_name(name).parts
  return (len(parts) >= 3 and parts[0] == "scaly_ipopt" and parts[1] in _OUTPUTS) or (len(parts) == 2 and parts[0] == "scaly_ipopt.libs")


def _system_import(system: str, name: str) -> bool:
  if system == "Windows":
    return name == Path(name).name and (
      name.lower().startswith(("api-ms-win-", "ext-ms-win-")) or (Path(os.environ["SystemRoot"]) / "System32" / name).is_file()
    )
  if system == "Darwin":
    return name.startswith(("/usr/lib/", "/System/"))
  return re.fullmatch(r"(?:ld-linux[^/]*|lib(?:c|m|pthread|dl|rt|util|resolv|z|gcc_s|stdc\+\+)\.so[^/]*)", name) is not None


def _validate_manifest(manifest: dict) -> None:
  if manifest["schema_version"] != 1:
    raise RuntimeError("Unsupported native bundle schema")
  target = manifest["target"]
  host = _host_target()
  system, arch, tag = target["system"], target["arch"], target["platform_tag"]
  if (system, arch) != (host["system"], host["arch"]):
    raise RuntimeError(f"Wrong bundle target {system}/{arch}; host is {host['system']}/{host['arch']}")
  patterns = {
    ("Windows", "x86_64"): r"win_amd64",
    ("Linux", "x86_64"): r"(?:linux|manylinux_2_\d+)_x86_64",
    ("Linux", "aarch64"): r"(?:linux|manylinux_2_\d+)_aarch64",
    ("Darwin", "x86_64"): r"macosx_\d+_\d+_x86_64",
    ("Darwin", "arm64"): r"macosx_\d+_\d+_arm64",
  }
  if not re.fullmatch(patterns.get((system, arch), r"(?!)"), tag) or not target["floor"]:
    raise RuntimeError("Invalid bundle platform tag or floor")
  if system == "Windows":
    if target["floor"] != "Windows 10 UCRT":
      raise RuntimeError("Invalid Windows bundle floor")
  else:
    prefix = "glibc " if system == "Linux" else "macOS "

    def floor_version(value):
      if not value.startswith(prefix) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value[len(prefix) :]):
        raise RuntimeError("Invalid bundle compatibility floor")
      return tuple(map(int, value[len(prefix) :].split(".")[:2]))

    floor = floor_version(target["floor"])
    if floor > floor_version(host["floor"]) or floor < ((2, 27) if system == "Linux" else (15, 0)):
      raise RuntimeError("Unsupported bundle compatibility floor")
    described = tuple(map(int, tag.split("_")[1:3])) if tag.startswith(("manylinux_", "macosx_")) else floor
    if described != floor:
      raise RuntimeError("Bundle platform tag does not describe its compatibility floor")
  if not re.fullmatch(r"[0-9a-f]{64}", manifest["recipe_sha256"]):
    raise RuntimeError("Invalid native recipe hash")
  for field in ("inputs", "configuration", "dependencies", "files"):
    if not isinstance(manifest[field], dict) or not manifest[field]:
      raise RuntimeError(f"Incomplete bundle manifest: {field}")
  for field in ("sources", "toolchain", "environment", "build_config", "diagnostics"):
    if field not in manifest["inputs"]:
      raise RuntimeError(f"Incomplete native inputs: {field}")
  if manifest["configuration"]["precision"] != "double" or manifest["configuration"]["integer_bits"] != 32:
    raise RuntimeError("Unsupported native numerical configuration")
  for field in ("blas", "cpu", "threading"):
    if not manifest["configuration"][field]:
      raise RuntimeError(f"Incomplete native configuration: {field}")
  files = manifest["files"]
  required = {
    "scaly_ipopt/lib/" + _shared_lib_name(system, "ipopt"),
    "scaly_ipopt/include/coin-or/IpStdCInterface.h",
    "scaly_ipopt/include/coin-or/IpoptConfig.h",
    "scaly_ipopt/licenses/" + NOTICES,
  }
  if system == "Windows":
    required.add("scaly_ipopt/lib/ipopt.lib")
  components = {"ipopt", "mumps", "thirdparty-mumps", "metis", "gklib", "gcc-runtime"}
  if system != "Darwin":
    components.add("openblas")
  if system == "Windows":
    components.add("winpthreads")
  if not components.issubset(manifest["components"]) or not required.issubset(files):
    raise RuntimeError("Incomplete bundle: missing native library, matching headers/import library or components")
  for component in components:
    if not any(name.startswith(f"scaly_ipopt/licenses/{component}/") for name in files):
      raise RuntimeError(f"Incomplete bundle: missing {component} license")
  for name, entry in files.items():
    if not _payload_name(name) or not re.fullmatch(r"[0-9a-f]{64}", entry["sha256"]) or entry["size"] < 0:
      raise RuntimeError(f"Invalid native file entry: {name}")
    if entry["mode"] & ~0o777:
      raise RuntimeError(f"Unsafe native file mode: {name}")
  native = {name for name in files if ".so" in name or name.endswith((".dylib", ".dll"))}
  runtimes = {
    "Windows": ("libgfortran-5.dll", "libquadmath-0.dll", "libgcc_s_seh-1.dll", "libstdc++-6.dll", "libwinpthread-1.dll"),
    "Linux": ("libgfortran.so.5",),
    "Darwin": ("libgfortran.5.dylib", "libquadmath.0.dylib"),
  }[system]
  if any(not any(name.endswith(runtime) for name in native) for runtime in runtimes):
    raise RuntimeError("Incomplete bundle: missing native language/thread runtime")
  if set(manifest["dependencies"]) != native:
    raise RuntimeError("Incomplete native dependency inventory")
  for name, deps in manifest["dependencies"].items():
    if (
      not isinstance(deps["system"], list)
      or not set(deps["bundled"]).issubset(native)
      or not all(_system_import(system, dep) for dep in deps["system"])
    ):
      raise RuntimeError(f"Incomplete bundled dependency closure: {name}")


def _read_bundle(bundle: bytes, out: Path | None = None) -> dict:
  """Validate the full archive before extraction. The caller supplies immutable, content-keyed bytes.

  Version 1 has manifest.json plus scaly_ipopt/{lib,include,licenses} and optional
  scaly_ipopt.libs. Relative symlinks may name a regular payload file. Directory links,
  hardlinks, devices, duplicate/case-colliding names and unlisted files are forbidden.
  No native tool is needed to validate or extract an archive.
  """
  try:
    with tarfile.open(fileobj=io.BytesIO(gzip.decompress(bundle)), mode="r:") as archive:
      members = {}
      folded = set()
      for member in archive.getmembers():
        name = member.name.rstrip("/") if member.isdir() else member.name
        _safe_name(name)
        if name.casefold() in folded or not (member.isfile() or member.isdir() or member.issym()):
          raise RuntimeError(f"Unsafe or duplicate archive member: {name}")
        folded.add(name.casefold())
        members[name] = member
      meta = members["manifest.json"]
      if not meta.isfile():
        raise RuntimeError("Invalid bundle manifest member")
      manifest = json.loads(archive.extractfile(meta).read())
      _validate_manifest(manifest)
      files = manifest["files"]
      if set(name for name, member in members.items() if not member.isdir()) != {"manifest.json", *files}:
        raise RuntimeError("Incomplete archive or unlisted native files")
      directories = {str(parent) for name in files for parent in PurePosixPath(name).parents if str(parent) != "."}
      if any(member.isdir() and name not in directories for name, member in members.items()):
        raise RuntimeError("Invalid native archive directory")
      if any(name in members and not members[name].isdir() for name in directories):
        raise RuntimeError("Unsafe archive parent: files and links cannot be directories")
      contents = {}
      for name, entry in files.items():
        member = members[name]
        if member.issym():
          link = member.linkname
          if "\\" in link or ":" in link or PurePosixPath(link).is_absolute():
            raise RuntimeError(f"Unsafe archive link: {name}")
          import posixpath

          destination = posixpath.normpath(str(PurePosixPath(name).parent / link))
          if destination not in files or not members[destination].isfile() or entry.get("link") != link:
            raise RuntimeError(f"Unsafe archive link: {name}")
          data = link.encode()
        else:
          if "link" in entry:
            raise RuntimeError(f"Invalid archive link metadata: {name}")
          data = archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["size"] or member.mode != entry["mode"]:
          raise RuntimeError(f"Corrupt native payload: {name}")
        contents[name] = data
      if out is not None:
        out.mkdir(parents=True)
        for name, data in contents.items():
          path = out / name
          path.parent.mkdir(parents=True, exist_ok=True)
          if "link" not in files[name]:
            path.write_bytes(data)
            path.chmod(files[name]["mode"])
        for name, entry in files.items():
          if "link" in entry:
            (out / name).symlink_to(entry["link"])
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
      return manifest
  except (KeyError, TypeError, ValueError, AttributeError, tarfile.TarError, EOFError, gzip.BadGzipFile, zlib.error) as error:
    raise RuntimeError(f"Invalid native bundle: {error}") from error


def _verify_payload(root: Path, manifest: dict) -> None:
  actual = set()
  for directory in (*(root / "scaly_ipopt" / name for name in _OUTPUTS), root / "scaly_ipopt.libs"):
    if directory.is_symlink():
      raise RuntimeError(f"Unsafe cached native directory: {directory}")
    for path in directory.rglob("*"):
      if path.is_file() or path.is_symlink():
        actual.add(path.relative_to(root).as_posix())
  if actual != set(manifest["files"]):
    raise RuntimeError("Incomplete cached native payload or unlisted files")
  for name, entry in manifest["files"].items():
    path = root / name
    if path.is_symlink() != ("link" in entry):
      raise RuntimeError(f"Corrupt cached native payload: {name}")
    data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
    if hashlib.sha256(data).hexdigest() != entry["sha256"] or len(data) != entry["size"]:
      raise RuntimeError(f"Corrupt cached native payload: {name}")


def _recipe_hash(root: Path, source: str | None = None) -> str:
  """Native recipe identity excludes wrapper code, consumer pins and archive bytes.

  Hash the native recipe definitions, native configuration and checked-in license texts.
  Resolved source revisions, toolchains and effective flags also live in each manifest.
  Equal recipes do not promise equal outputs; each archive has its own full SHA-256.
  """
  source = source if source is not None else (root / "hatch_build.py").read_text(encoding="utf-8")
  tree = ast.parse(source)
  first = next(i for i, node in enumerate(tree.body) if isinstance(node, ast.FunctionDef) and node.name == "_run")
  last = next(
    i
    for i, node in enumerate(tree.body)
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "_KEY_ENV" for t in node.targets)
  )
  nodes = tree.body[first:last] + [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in {"_shared_lib_name", "_find_fortran_compiler", "_build_ipopt_stack", "_windows_packages", "_repair_linux_payload", "_locked"}
  ]
  digest = hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode())
  config = json.loads((root / "src/scaly_ipopt/build_config.json").read_text())
  native_config = {key: config[key] for key in ("ipopt", "mumps", "metis", "gcc-runtime", "winpthreads", "blas")}
  digest.update(json.dumps(native_config, sort_keys=True).encode())
  for path in sorted(p for p in (root / "licenses").rglob("*") if p.is_file()):
    digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
  return digest.hexdigest()


def _native_dependencies(payload: Path) -> dict:
  """Inventory every shared object's direct bundled and system imports after final repair."""
  system = platform.system()
  native = {p.name.lower(): p for p in payload.rglob("*") if p.is_file() and (".so" in p.name or p.suffix in {".dll", ".dylib"})}
  graph = {}
  for path in native.values():
    if system == "Windows":
      imports = _windows_deps(path)
    elif system == "Darwin":
      output = subprocess.check_output(["otool", "-L", str(path)], text=True)
      imports = [line.strip().split(" (")[0] for line in output.splitlines()[1:]]
      imports = [name for name in imports if Path(name).name != path.name]
    else:
      output = subprocess.check_output(["readelf", "-d", str(path)], text=True)
      imports = re.findall(r"\(NEEDED\).*\[(.*?)\]", output)
    bundled, external = [], []
    for name in imports:
      dependency = native.get(Path(name).name.lower())
      if dependency is not None:
        bundled.append(dependency.relative_to(payload).as_posix())
      elif _system_import(system, name):
        external.append(name)
      else:
        raise RuntimeError(f"Incomplete native dependency closure: {path} imports {name}")
    graph[path.relative_to(payload).as_posix()] = {"bundled": sorted(bundled), "system": sorted(external)}
  return graph


def _repair_linux_payload(work: Path, payload: Path, tag: str) -> None:
  """Apply the existing auditwheel repair before freezing a requested manylinux payload."""
  import zipfile

  wheel = work / f"scaly_ipopt-0.0.0-py3-none-linux_{_host_target()['arch']}.whl"
  metadata = "scaly_ipopt-0.0.0.dist-info"
  with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
    names = []
    for path in sorted(p for p in payload.rglob("*") if p.is_file()):
      name = path.relative_to(payload).as_posix()
      archive.write(path, name)
      names.append(name)
    entries = {
      f"{metadata}/METADATA": "Metadata-Version: 2.1\nName: scaly-ipopt\nVersion: 0.0.0\n",
      f"{metadata}/WHEEL": f"Wheel-Version: 1.0\nGenerator: scaly-native\nRoot-Is-Purelib: false\nTag: py3-none-linux_{_host_target()['arch']}\n",
    }
    for name, value in entries.items():
      archive.writestr(name, value)
      names.append(name)
    archive.writestr(f"{metadata}/RECORD", "".join(f"{name},,\n" for name in [*names, f"{metadata}/RECORD"]))
  repaired = work / "repaired"
  _remove_tree(repaired)
  _run([sys.executable, "-m", "auditwheel", "repair", "--plat", tag, "--only-plat", "--wheel-dir", str(repaired), str(wheel)], cwd=work)
  with zipfile.ZipFile(next(repaired.glob("*.whl"))) as archive:
    _remove_tree(payload)
    for name in archive.namelist():
      if not name.startswith(metadata + "/") and not name.endswith("/") and _payload_name(name):
        destination = payload / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(archive.read(name))
        destination.chmod((archive.getinfo(name).external_attr >> 16) & 0o777 or 0o644)


def _bundle_manifest(root: Path, work: Path, payload: Path, target: dict) -> dict:
  sources = {}
  for name in ("Ipopt", "ThirdParty-Mumps", "METIS", "GKlib", "openblas"):
    path = work / "src" / name
    if not path.exists():
      continue
    git = ["git", "-C", str(path)]
    patch = subprocess.check_output([*git, "diff", "--binary", "HEAD"], text=True)
    sources[name] = {
      "url": subprocess.check_output([*git, "remote", "get-url", "origin"], text=True).strip()
      if name != "GKlib"
      else "https://github.com/KarypisLab/GKlib.git",
      "commit": subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip(),
      "patch": patch,
      "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
    }
  diagnostics = {}
  for pattern in ("**/config.log", "**/config.status", "**/Makefile.conf", "**/CMakeCache.txt", "openblas/config.h"):
    for path in (work / "src").glob(pattern):
      diagnostics[path.relative_to(work).as_posix()] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "text": path.read_text(errors="replace"),
      }
  # MUMPS is downloaded by the pinned wrapper rather than stored in its Git tree.
  mumps = work / "src/ThirdParty-Mumps/MUMPS"
  sources["MUMPS"] = {
    "version": _BUILD_CONFIG["mumps"]["version"],
    "files": {
      path.relative_to(mumps).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
      for path in sorted(mumps.rglob("*"))
      if path.is_file() and path.suffix.lower() in {".f", ".f90", ".c", ".h"}
    },
  }
  blas = {}
  for name in ("Makefile.conf", "config.h"):
    path = work / "src/openblas" / name
    if path.exists():
      blas[name] = path.read_text(errors="replace")
  files = {}
  for path in sorted(p for p in payload.rglob("*") if p.is_file() or p.is_symlink()):
    data = os.readlink(path).encode() if path.is_symlink() else path.read_bytes()
    entry = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "mode": stat.S_IMODE(path.lstat().st_mode)}
    if path.is_symlink():
      entry["link"] = os.readlink(path)
    files[path.relative_to(payload).as_posix()] = entry
  toolchain = {cmd: _tool_identity(os.environ.get(env, cmd)) for env, cmd in (("CC", "cc"), ("CXX", "c++"), ("FC", "gfortran"))}
  for cmd in ("git", "make", "cmake", "ninja", "ar", "ranlib", "objdump", "readelf", "patchelf", "codesign"):
    if shutil.which(cmd):
      toolchain[cmd] = _tool_identity(cmd)
  from importlib.metadata import version

  for name in ("delvewheel", "pefile") if target["system"] == "Windows" else ("auditwheel", "patchelf") if target["system"] == "Linux" else ():
    toolchain[name] = version(name)
  if target["system"] == "Windows":
    fc = _find_fortran_compiler()
    toolchain["pacman"] = _windows_packages(fc)
    bash = Path(fc).resolve().parents[2] / "usr/bin/bash.exe"
    command = 'for tool in gcc g++ gfortran make cmake ninja objdump dlltool; do command -v "$tool"; "$tool" --version | head -n 1; done'
    toolchain["native_shell_tools"] = subprocess.check_output(
      [str(bash), "-lc", command], text=True, env=dict(os.environ, MSYSTEM="UCRT64", CHERE_INVOKING="1")
    )
  components = sorted(
    {PurePosixPath(name).parts[2] for name in files if name.startswith("scaly_ipopt/licenses/") and len(PurePosixPath(name).parts) > 3}
  )
  return {
    "schema_version": 1,
    "target": target,
    "recipe_sha256": _recipe_hash(root),
    "inputs": {
      "sources": sources,
      "build_config": {k: _BUILD_CONFIG[k] for k in ("ipopt", "mumps", "metis", "gcc-runtime", "winpthreads", "blas")},
      "toolchain": toolchain,
      "environment": {
        name: os.environ.get(name, "")
        for name in (
          *_KEY_ENV,
          "MAKE_NB_JOBS",
          "PYTHON_CPU_COUNT",
          "NUM_THREADS",
          "USE_THREAD",
          "OPENBLAS_NUM_THREADS",
          "OMP_NUM_THREADS",
          "VECLIB_MAXIMUM_THREADS",
        )
      },
      "diagnostics": diagnostics,
      "commands": (work / "commands.jsonl").read_text() if (work / "commands.jsonl").exists() else "",
    },
    "configuration": {
      "precision": "double",
      "integer_bits": 32,
      "blas": {"implementation": "Accelerate" if target["system"] == "Darwin" else "OpenBLAS", "effective": blas or "OS-provided Accelerate"},
      "cpu": {
        "baseline": "PRESCOTT" if target["arch"] == "x86_64" and blas else "see effective native flags",
        "dispatch": "DYNAMIC_ARCH=1" if blas else "OS-provided Accelerate",
      },
      "threading": {"effective": blas or "OS-provided Accelerate; runtime thread count not measured"},
    },
    "components": components,
    "dependencies": _native_dependencies(payload),
    "files": files,
  }


def _macos_minimum(commands: str) -> tuple[int, int]:
  """Read deployment versions from otool load commands, excluding SDK and tool versions."""
  minimum, command = (0, 0), ""
  for line in commands.splitlines():
    fields = line.split()
    if len(fields) != 2:
      continue
    key, value = fields
    if key == "cmd":
      command = value
    elif (command, key) in {("LC_BUILD_VERSION", "minos"), ("LC_VERSION_MIN_MACOSX", "version")}:
      major, minor, *_ = map(int, value.split("."))
      minimum = max(minimum, (major, minor))
  return minimum


def _produce_bundle(work: Path, output: Path, platform_tag: str | None = None) -> dict:
  """Build only the host target, retain native work/logs, repair and freeze a complete archive."""
  from types import SimpleNamespace

  global _BUILD_LOG, _COMMAND_LOG
  root = Path(__file__).resolve().parent
  work, output = work.resolve(), output.resolve()
  work.mkdir(parents=True, exist_ok=True)
  target = _host_target()
  if (target["system"], target["arch"]) not in {
    ("Windows", "x86_64"),
    ("Linux", "x86_64"),
    ("Linux", "aarch64"),
    ("Darwin", "x86_64"),
    ("Darwin", "arm64"),
  }:
    raise RuntimeError(f"Unsupported native target: {target}")
  missing = _missing_ipopt_tools()
  if target["system"] == "Linux" and shutil.which("readelf") is None:
    missing.append("readelf")
  if missing:
    raise RuntimeError(f"Missing native tools: {', '.join(missing)}")
  if platform_tag and not (target["system"] == "Linux" and re.fullmatch(r"manylinux_2_\d+_" + target["arch"], platform_tag)):
    raise RuntimeError("--platform-tag accepts only a host-architecture manylinux tag, audited by auditwheel")
  state = {"recipe_sha256": _recipe_hash(root), "target": target, "environment": {key: os.environ.get(key, "") for key in _KEY_ENV}}
  stamp = work / "recipe.json"
  if stamp.exists() and json.loads(stamp.read_text()) != state:
    raise RuntimeError(f"Native inputs changed; use a new --work-dir instead of {work}")
  stamp.write_text(json.dumps(state, indent=2) + "\n")
  _BUILD_LOG, _COMMAND_LOG = work / "build.log", work / "commands.jsonl"
  payload = work / "payload"
  with _locked(work / "producer.lock"):
    _remove_tree(payload)
    (work / "src").mkdir(exist_ok=True)
    package = payload / "scaly_ipopt"
    hook = SimpleNamespace(app=SimpleNamespace(display_info=lambda message: print(message, flush=True)))
    print(f"Retaining native sources, build trees and diagnostics in {work}", flush=True)
    _build_ipopt_stack(hook, work / "src", package / "lib", package / "include", package / "licenses", root / "licenses")
    if platform_tag:
      _repair_linux_payload(work, payload, platform_tag)
      target |= {"platform_tag": platform_tag, "floor": "glibc 2." + platform_tag.split("_")[2]}
    elif target["system"] == "Darwin":
      minimum = (15, 0)
      for path in (package / "lib").glob("*.dylib"):
        commands = subprocess.check_output(["otool", "-l", str(path)], text=True)
        minimum = max(minimum, _macos_minimum(commands))
      target |= {"platform_tag": f"macosx_{minimum[0]}_{minimum[1]}_{target['arch']}", "floor": f"macOS {minimum[0]}.{minimum[1]}"}
    manifest = _bundle_manifest(root, work, payload, target)
    _validate_manifest(manifest)
    (payload / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    with tarfile.open(temporary, "w:gz", format=tarfile.PAX_FORMAT) as archive:
      for path in sorted(payload.iterdir()):
        archive.add(path, arcname=path.name)
    _read_bundle(temporary.read_bytes())
    temporary.replace(output)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n")
    output.with_name(output.name + ".manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"{output}\nSHA-256 {digest}", flush=True)
    return manifest


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "scaly"

  def initialize(self, version: str, build_data: dict) -> None:
    """Select explicit verified archive bytes, otherwise retain the ordinary source-build path.

    Sdists do no native work. The manifest supplies archive wheel tags and the content key;
    archive selection is strict even for editable auto mode. Skip mode keeps its precedence.
    """
    if self.target_name == "sdist":
      self.app.display_info("Skipping vendored solver build for sdist target")
      return

    root = Path(self.root)
    package_dir = root / "src" / "scaly_ipopt"
    lib_dir = package_dir / "lib"
    mode = _solver_build_mode()
    strict = mode == "require" or (mode == "auto" and version != "editable")
    system = platform.system()

    if mode != "skip" or any(lib_dir.glob("lib*")):
      build_data["pure_python"] = False
      build_data["tag"] = _wheel_tag()
    if mode == "skip":
      if (package_dir / "native_manifest.json").is_file():
        try:
          manifest = json.loads((package_dir / "native_manifest.json").read_text())
          _validate_manifest(manifest)
          _verify_payload(package_dir.parent, manifest)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, RuntimeError) as error:
          raise RuntimeError(f"Invalid installed Ipopt manifest in {package_dir}; reinstall scaly-ipopt: {error}") from error
        build_data["tag"] = "py3-none-" + manifest["target"]["platform_tag"]
      self.app.display_info("Skipping vendored solver build because SCALY_BUILD_SOLVERS=skip/0")
      return
    if archive := os.environ.get("SCALY_IPOPT_BUNDLE"):
      try:
        bundle = Path(archive).expanduser().read_bytes()
      except OSError as error:
        raise RuntimeError(f"Cannot read SCALY_IPOPT_BUNDLE={archive!r}: {error}") from error
      manifest = _read_bundle(bundle)
      build_data["pure_python"] = False
      build_data["tag"] = "py3-none-" + manifest["target"]["platform_tag"]
      key = _cache_key(root, bundle)
      _install_build(
        self,
        root,
        package_dir,
        _cache_root() / f"ipopt-{key}",
        key,
        lambda: [],
        lambda src, out: _read_bundle(bundle, out),
        manifest=manifest,
      )
      return
    key = _cache_key(root)
    stamp = root / ".build_key"
    if stamp.exists() and stamp.read_text() == key and all((package_dir / name).is_dir() for name in _OUTPUTS):
      self.app.display_info(f"IPOPT already built at {lib_dir}")
      return

    def build(src: Path, out: Path) -> None:
      _build_ipopt_stack(self, src, out / "lib", out / "include", out / "licenses", root / "licenses")

    missing = _install_build(self, root, package_dir, _cache_root() / f"ipopt-{key}", key, _missing_ipopt_tools, build)
    if missing:
      msg = f"missing native toolchain for IPOPT build: {', '.join(missing)}"
      if strict:
        hint = (
          "Run the build in MSYS2 UCRT64 with GCC, GFortran, CMake, Ninja, make and git installed."
          if system == "Windows"
          else "Install gfortran via `brew install gcc` (macOS) or `sudo apt-get install gfortran` (Linux)."
        )
        raise RuntimeError(f"{msg}. {hint}")
      self.app.display_info(f"Skipping IPOPT build for editable install ({msg}); set SCALY_BUILD_SOLVERS=required to make this fatal.")

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    (root / ".build_key").unlink(missing_ok=True)
    for path in (root / "src" / "scaly_ipopt" / "lib", root / "src" / "scaly_ipopt" / "include", root / "src" / "scaly_ipopt" / "licenses"):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
    shutil.rmtree(root / "src" / "scaly_ipopt.libs", ignore_errors=True)
    (root / "src" / "scaly_ipopt" / "native_manifest.json").unlink(missing_ok=True)


if __name__ == "__main__":
  if sys.argv[1:2] == ["_archive"]:
    _archive_native(Path(sys.argv[2]), sys.argv[3:])
    raise SystemExit(0)
  import argparse

  parser = argparse.ArgumentParser(description="Build a complete host-native Ipopt tar.gz bundle without publishing it.")
  parser.add_argument("command", choices=["bundle"])
  parser.add_argument("--work-dir", type=Path, required=True, help="Retained sources, incremental native build trees and logs")
  parser.add_argument("--output", type=Path, required=True, help="Output tar.gz archive; SHA-256 and manifest are also written beside it")
  parser.add_argument("--platform-tag", help="Linux only: auditwheel repair to this manylinux tag in a suitable compatibility environment")
  arguments = parser.parse_args()
  _produce_bundle(arguments.work_dir, arguments.output, arguments.platform_tag)
