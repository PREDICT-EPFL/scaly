"""Custom build hook that vendors IPOPT as a shared library inside the scaly-ipopt wheel."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from contextlib import contextmanager
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
  if shutil.which("cc") is None:
    missing.append("cc")
  if not _has_cxx_compiler():
    missing.append("c++")
  if _find_fortran_compiler(required=False) is None:
    missing.append("gfortran")
  return missing


_BUILD_CONFIG = json.loads((Path(__file__).parent / "src" / "scaly_ipopt" / "build_config.json").read_text())
IPOPT_BRANCH = _BUILD_CONFIG["ipopt"]["branch"]
MUMPS_BRANCH = _BUILD_CONFIG["mumps"]["coinor_branch"]
METIS_TAG = _BUILD_CONFIG["metis"]["tag"]
GKLIB_COMMIT = _BUILD_CONFIG["metis"]["gklib_commit"]
OPENBLAS_BRANCH = _BUILD_CONFIG["blas"]["linux"]["branch"]


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
  # Fix the common-code baseline instead of detecting a potentially unsupported build CPU.
  # DYNAMIC_ARCH still selects optimized kernels at runtime.
  if platform.machine() == "x86_64":
    build_flags.append("TARGET=PRESCOTT")
  _run(["make", f"-j{jobs}", *build_flags], cwd=src_dir)
  install_dir.mkdir(parents=True, exist_ok=True)
  _run(["make", f"PREFIX={install_dir.resolve()}", *build_flags, "install"], cwd=src_dir)
  return install_dir


def _clone(hook: "BuildHook", url: str, branch: str, dest: Path) -> None:
  hook.app.display_info(f"Cloning {url}@{branch} to {dest}")
  _run(["git", "clone", "--depth=1", "--branch", branch, url, str(dest)], cwd=dest.parent, attempts=3)


def _clone_commit(hook: "BuildHook", url: str, commit: str, dest: Path) -> None:
  """Shallow-fetch one commit; `git clone --depth=1 --branch` only accepts branches and tags."""
  hook.app.display_info(f"Fetching {url}@{commit} to {dest}")
  dest.mkdir(parents=True)
  _run(["git", "init", "-q"], cwd=dest)
  _run(["git", "fetch", "-q", "--depth=1", url, commit], cwd=dest, attempts=3)
  _run(["git", "checkout", "-q", "FETCH_HEAD"], cwd=dest)


def _cmake_build(hook: "BuildHook", src_dir: Path, build_dir: Path, install_dir: Path, defines: list[str]) -> None:
  build_dir.mkdir(parents=True, exist_ok=True)
  common = [
    "-DCMAKE_BUILD_TYPE=Release",
    f"-DCMAKE_INSTALL_PREFIX={install_dir.resolve()}",
    "-DCMAKE_INSTALL_LIBDIR=lib",
    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON",
    "-DBUILD_SHARED_LIBS=OFF",
  ]
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


# Everything else that decides the built files, beside this hook, the pins and the license texts.
_KEY_ENV = ("CC", "CXX", "FC", "CFLAGS", "CXXFLAGS", "FFLAGS", "FCFLAGS", "LDFLAGS", "MACOSX_DEPLOYMENT_TARGET")
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


def _cache_key(root: Path) -> str:
  """Hash every input of the build, so two checkouts share a build exactly when it would come out the same."""
  digest = hashlib.sha256()
  inputs = [Path(__file__), root / "src" / "scaly_ipopt" / "build_config.json", *(root / "licenses").rglob("*")]
  for path in sorted(p for p in inputs if p.is_file()):
    digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
  fc = _find_fortran_compiler(required=False) or "gfortran"
  tools = [_tool_identity(os.environ.get("CC", "cc")), _tool_identity(os.environ.get("CXX", "c++")), _tool_identity(fc)]
  env = [f"{name}={os.environ.get(name, '')}" for name in _KEY_ENV]
  for part in (platform.system(), platform.machine(), *platform.libc_ver(), platform.mac_ver()[0], *tools, *env):
    digest.update(part.encode() + b"\0")
  return digest.hexdigest()[:16]


@contextmanager
def _locked(path: Path, *, wait: bool = True):
  """Hold an exclusive lock on `path`, yielding whether it was taken. Lock files are never deleted."""
  import fcntl

  path.parent.mkdir(parents=True, exist_ok=True)
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
      if path.name == "locks" or path.stat().st_mtime > cutoff:
        continue
      with _locked(cache / "locks" / f"{path.name.removeprefix('build-')}.lock", wait=False) as held:
        if held and path.stat().st_mtime <= cutoff:
          shutil.rmtree(path)


def _install_build(hook: "BuildHook", root: Path, package_dir: Path, entry: Path, key: str, build) -> None:
  """Copy the cached build `entry` into `package_dir`, first running `build(src_dir, out_dir)` if no checkout made it yet.

  A build runs in a fresh scratch directory and is renamed into the cache only once complete, so a
  cache entry is never partial. A failed build keeps its scratch directory for inspection until the
  next attempt at the same key replaces it."""
  cache = entry.parent
  with _locked(cache / "locks" / f"{entry.name}.lock"):
    if not entry.exists():
      scratch = cache / f"build-{entry.name}"
      shutil.rmtree(scratch, ignore_errors=True)
      (scratch / "src").mkdir(parents=True)
      try:
        build(scratch / "src", scratch / "out")
      except BaseException:
        hook.app.display_error(f"The build failed; its sources and build trees are kept in {scratch}")
        raise
      (scratch / "out").rename(entry)
      shutil.rmtree(scratch)
    os.utime(entry)
    stamp = root / ".build_key"
    stamp.unlink(missing_ok=True)
    for name in _OUTPUTS:
      shutil.rmtree(package_dir / name, ignore_errors=True)
      shutil.copytree(entry / name, package_dir / name, symlinks=True)
    stamp.write_text(key)
  hook.app.display_info(f"Installed the build {entry} into {package_dir}")
  _prune_cache(cache)


def _build_ipopt_stack(hook: "BuildHook", third_party_dir: Path, lib_dir: Path, include_dir: Path, licenses_dir: Path, vendored: Path) -> None:
  """Build into `lib_dir`, `include_dir` and `licenses_dir` from sources cloned into the empty `third_party_dir`.

  `vendored` holds the license texts checked into the plugin."""
  system = platform.system()
  lib_name = _shared_lib_name(system, "ipopt")
  hook.app.display_info("Building IPOPT stack (METIS -> MUMPS -> IPOPT)...")
  if system not in {"Darwin", "Linux"}:
    raise RuntimeError(f"Unsupported platform: {system}")
  ldflags = "" if system == "Darwin" else LINUX_RPATH_LDFLAGS
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
  else:
    _bundle_linux_runtime(hook, lib_dir, lib_name)
    if not _linux_self_contained(lib_dir, lib_name):
      raise RuntimeError(f"{dst_path} still resolves its Fortran runtime outside {lib_dir}: {_linux_runtime_deps(dst_path)}")

  ipopt_headers_dir = ipopt_install / "include" / "coin-or"
  for header in ipopt_headers_dir.glob("*.h"):
    shutil.copy2(header, include_dir / "coin-or" / header.name)

  mumps_src = third_party_dir / "ThirdParty-Mumps"
  gklib_src = third_party_dir / "GKlib"
  runtime = sorted(p.name for p in lib_dir.iterdir() if p.name.startswith(("libgfortran", "libquadmath", "libgcc_s")))
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
  if system == "Linux":
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
  _write_third_party_notices(hook, licenses_dir, "scaly-ipopt", entries)
  hook.app.display_info("IPOPT build complete.")


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "scaly"

  def initialize(self, version: str, build_data: dict) -> None:
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
      self.app.display_info("Skipping vendored solver build because SCALY_BUILD_SOLVERS=skip/0")
      return
    if system == "Windows":
      msg = "Windows IPOPT build not yet supported — track in https://github.com/PREDICT-EPFL/scaly/issues/1"
      if strict:
        raise RuntimeError(msg)
      self.app.display_info(f"Skipping vendored solver build for editable install: {msg}")
      return

    key = _cache_key(root)
    stamp = root / ".build_key"
    if stamp.exists() and stamp.read_text() == key and all((package_dir / name).is_dir() for name in _OUTPUTS):
      self.app.display_info(f"IPOPT already built at {lib_dir}")
      return
    entry = _cache_root() / f"ipopt-{key}"
    missing = [] if entry.exists() else _missing_ipopt_tools()
    if missing:
      msg = f"missing native toolchain for IPOPT build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(f"{msg}. Install gfortran via `brew install gcc` (macOS) or `sudo apt-get install gfortran` (Linux).")
      self.app.display_info(f"Skipping IPOPT build for editable install ({msg}); set SCALY_BUILD_SOLVERS=required to make this fatal.")
      return

    def build(src: Path, out: Path) -> None:
      _build_ipopt_stack(self, src, out / "lib", out / "include", out / "licenses", root / "licenses")

    _install_build(self, root, package_dir, entry, key, build)

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    (root / ".build_key").unlink(missing_ok=True)
    for path in (root / "src" / "scaly_ipopt" / "lib", root / "src" / "scaly_ipopt" / "include", root / "src" / "scaly_ipopt" / "licenses"):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
