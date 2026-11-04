"""Build and bundle the native IPOPT libraries, headers and notices."""

from __future__ import annotations

import json
import os
import platform
import re
import shlex
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
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


_BUILD_CONFIG = json.loads((Path(__file__).parent / "src" / "scaly_ipopt" / "build_config.json").read_text())
IPOPT_BRANCH = _BUILD_CONFIG["ipopt"]["branch"]
MUMPS_BRANCH = _BUILD_CONFIG["mumps"]["coinor_branch"]
METIS_TAG = _BUILD_CONFIG["metis"]["tag"]
GKLIB_COMMIT = _BUILD_CONFIG["metis"]["gklib_commit"]
OPENBLAS_BRANCH = _BUILD_CONFIG["blas"]["linux"]["branch"]


def _msys2_bash(fc: str) -> Path:
  return Path(fc).resolve().parents[2] / "usr" / "bin" / "bash.exe"


def _run(cmd: list[str], cwd: Path, env: dict | None = None, attempts: int = 1) -> None:
  """Run `cmd`, retrying a failure `attempts - 1` times; downloads use 3 to ride out a busy host."""
  if os.name == "nt":
    bash = _msys2_bash(_find_fortran_compiler())
    args = [re.sub(r"([A-Za-z]):/(?!/)", lambda m: f"/{m[1].lower()}/", arg.replace("\\", "/")) for arg in cmd]
    cmd = [str(bash), "-lc", shlex.join(args)]
    env = dict(os.environ if env is None else env, MSYSTEM="UCRT64", CHERE_INVOKING="1")
  for attempt in range(1, attempts + 1):
    try:
      subprocess.run(cmd, cwd=cwd, env=env, check=True)
      return
    except subprocess.CalledProcessError:
      if attempt == attempts:
        raise
      time.sleep(10 * attempt)


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


def _build_openblas(hook: "BuildHookInterface", third_party_dir: Path, install_dir: Path) -> Path:
  src_dir = third_party_dir / "openblas"
  _clone(hook, "https://github.com/OpenMathLib/OpenBLAS.git", OPENBLAS_BRANCH, src_dir)
  hook.app.display_info("Building OpenBLAS (this can take a few minutes)...")
  jobs = str(os.cpu_count() or 2)
  build_flags = ["NO_SHARED=1", "USE_OPENMP=0", "DYNAMIC_ARCH=1"]
  # The BLAS and LAPACK sub-makes add members to one static archive concurrently, which corrupts it
  # on many-core machines. Serialise only the archive writes, so compilation stays parallel.
  for variable, tool in (("AR", "ar"), ("RANLIB", "ranlib")):
    command = [sys.executable, str(Path(__file__).resolve()), str(third_party_dir / "openblas-archive.lock"), shutil.which(tool) or tool]
    build_flags.append(f"{variable}={shlex.join(command)}")
  # Fix the common-code baseline instead of detecting a potentially unsupported build CPU.
  # DYNAMIC_ARCH still selects optimized kernels at runtime.
  if platform.machine().lower() in {"x86_64", "amd64"}:
    build_flags.append("TARGET=PRESCOTT")
  _run(["make", f"-j{jobs}", *build_flags], cwd=src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  _run(["make", f"PREFIX={install_dir.resolve()}", *build_flags, "install"], cwd=src_dir)
  return install_dir


def _archive(lock: Path, command: list[str]) -> None:
  with _locked(lock):
    subprocess.run(command, check=True)


def _clone(hook: "BuildHookInterface", url: str, branch: str, dest: Path) -> None:
  hook.app.display_info(f"Cloning {url}@{branch} to {dest}")
  _run(["git", "clone", "--depth=1", "--branch", branch, url, str(dest)], cwd=dest.parent, attempts=3)


def _clone_commit(hook: "BuildHookInterface", url: str, commit: str, dest: Path) -> None:
  """Shallow-fetch one commit; `git clone --depth=1 --branch` only accepts branches and tags."""
  hook.app.display_info(f"Fetching {url}@{commit} to {dest}")
  dest.mkdir(parents=True)
  _run(["git", "init", "-q"], cwd=dest)
  _run(["git", "fetch", "-q", "--depth=1", url, commit], cwd=dest, attempts=3)
  _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


def _cmake_build(hook: "BuildHookInterface", src_dir: Path, build_dir: Path, install_dir: Path, defines: list[str]) -> None:
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


def _build_metis(hook: "BuildHookInterface", third_party_dir: Path, install_dir: Path) -> Path:
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
  hook: "BuildHookInterface",
  third_party_dir: Path,
  install_dir: Path,
  metis_install: Path,
  lapack_lflags: str,
  fc: str,
) -> Path:
  src_dir = third_party_dir / "ThirdParty-Mumps"
  _clone(hook, "https://github.com/coin-or-tools/ThirdParty-Mumps.git", MUMPS_BRANCH, src_dir)
  hook.app.display_info("Fetching MUMPS sources via get.Mumps...")
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
  hook: "BuildHookInterface",
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


def _bundle_macos_runtime(hook: "BuildHookInterface", lib_dir: Path, lib_name: str) -> None:
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


def _bundle_linux_runtime(hook: "BuildHookInterface", lib_dir: Path, lib_name: str) -> None:
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


# The GCC runtime that the MinGW-built Ipopt and MUMPS DLLs import, all from the toolchain's `bin`.
_WINDOWS_VENDORED = ("libgcc_s_seh-1.dll", "libgfortran-5.dll", "libquadmath-0.dll", "libstdc++-6.dll", "libwinpthread-1.dll")


def _namespace_windows_runtime(lib_dir: Path, lib_name: str) -> None:
  import pefile

  dlls = sorted(lib_dir.glob("*.dll"))
  names = {path.name.lower(): f"scaly_ipopt_{path.name}" for path in dlls if path.name != lib_name}
  for path in dlls:
    with pefile.PE(str(path)) as pe:
      deps = [entry.dll.decode("ascii") for entry in pe.DIRECTORY_ENTRY_IMPORT]
    changes = [arg for dep in deps if dep.lower() in names for arg in ("-change", dep, names[dep.lower()])]
    if changes:
      subprocess.run([sys.executable, "-m", "delvewheel", "replace-needed", str(path), "--strip", *changes], check=True)
  for old, new in names.items():
    (lib_dir / old).rename(lib_dir / new)
  with pefile.PE(str(lib_dir / lib_name)) as pe:
    exports = sorted(symbol.name.decode("ascii") for symbol in pe.DIRECTORY_ENTRY_EXPORT.symbols if symbol.name is not None)
  definition = lib_dir / "ipopt.def"
  definition.write_text("EXPORTS\n" + "\n".join(exports) + "\n", encoding="ascii")
  subprocess.run(["dlltool", "-D", lib_name, "-d", str(definition), "-l", str(lib_dir / "ipopt.lib")], check=True)
  definition.unlink()


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


def _build_ipopt_stack(hook: "BuildHookInterface", third_party_dir: Path, out_dir: Path, vendored: Path) -> None:
  """Build libraries, headers and licenses under `out_dir` from sources cloned into the empty `third_party_dir`.

  `vendored` holds the license texts checked into the plugin."""
  lib_dir = out_dir / "lib"
  include_dir = out_dir / "include"
  licenses_dir = out_dir / "licenses"
  system = platform.system()
  lib_name = _shared_lib_name(system, "ipopt")
  hook.app.display_info("Building IPOPT stack (METIS -> MUMPS -> IPOPT)...")
  if system not in {"Darwin", "Linux", "Windows"}:
    raise RuntimeError(f"Unsupported platform: {system}")
  if system == "Windows" and platform.machine().lower() != "amd64":
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
    for name in _WINDOWS_VENDORED:
      src, dst = Path(fc).resolve().parent / name, lib_dir / name
      hook.app.display_info(f"Bundling {src} -> {dst}")
      shutil.copy2(src, dst)
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
  names = (p.name.removeprefix("scaly_ipopt_") for p in lib_dir.iterdir())
  runtime = sorted(name for name in names if name.startswith(("libgfortran", "libquadmath", "libgcc_s", "libstdc++")))
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
      ", ".join(runtime),
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
        "libwinpthread-1.dll",
        "MIT and BSD-3-Clause",
        "https://www.mingw-w64.org",
        [Path(fc).resolve().parents[1] / "share" / "licenses" / "libwinpthread" / "COPYING"],
      )
    )
  _write_third_party_notices(hook, licenses_dir, "scaly-ipopt", entries)
  hook.app.display_info("IPOPT build complete.")


if __name__ == "__main__":
  _archive(Path(sys.argv[1]), sys.argv[2:])
