"""Custom build hook that vendors PIQP as a shared library inside the scaly-piqp wheel."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import shlex
import shutil
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
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
  if system == "Windows":
    return f"{base}.dll"
  raise RuntimeError(f"Unsupported platform: {system}")


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


# Windows has no system C compiler, so the build requirements bring the zig that the JIT also uses.
_DEFAULT_CC, _DEFAULT_CXX = ("zig cc", ("zig c++",)) if platform.system() == "Windows" else ("cc", ("c++", "g++", "clang++"))


def _compiler_command(value: str) -> list[str]:
  if path := shutil.which(value):
    return [path]
  command = [part.strip('"') for part in shlex.split(value, posix=os.name != "nt")]
  if command and (path := shutil.which(command[0])):
    return [path, *command[1:]]
  return []


def _missing_piqp_tools() -> list[str]:
  missing = [cmd for cmd in ("git", "cmake") if shutil.which(cmd) is None]
  if not _compiler_command(os.environ.get("CC", _DEFAULT_CC)):
    missing.append("cc")
  if not any(_compiler_command(cmd) for cmd in ([os.environ["CXX"]] if "CXX" in os.environ else _DEFAULT_CXX)):
    missing.append("c++")
  if platform.system() == "Windows" and shutil.which("ninja") is None:
    missing.append("ninja")
  return missing


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


def _clone(hook: "BuildHook", url: str, tag: str, dest: Path) -> None:
  hook.app.display_info(f"Cloning {url}@{tag} to {dest}")
  _run(["git", "clone", "--depth=1", "--branch", tag, url, str(dest)], cwd=dest.parent, attempts=3)


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


# Everything else that decides the built files, beside this hook, the pins and the license texts.
_KEY_ENV = ("CC", "CXX", "CFLAGS", "CXXFLAGS", "LDFLAGS", "MACOSX_DEPLOYMENT_TARGET", "CMAKE_GENERATOR")
_OUTPUTS = ("lib", "include", "licenses")
_CACHE_MAX_AGE = 30 * 24 * 3600


def _remove_tree(path: Path) -> None:
  def remove_readonly(function, filename, error):
    if not isinstance(error, PermissionError):
      raise error
    # Git marks pack files read-only on Windows, including in our disposable clones.
    os.chmod(filename, stat.S_IWRITE)
    function(filename)

  if path.exists():
    shutil.rmtree(path, onexc=remove_readonly)


def _cache_root() -> Path:
  if path := os.environ.get("SCALY_SOLVER_CACHE"):
    return Path(path)
  return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "scaly" / "solvers"


def _tool_identity(cmd: str) -> str:
  command = _compiler_command(cmd)
  if not command:
    return f"{cmd}: none"
  out = subprocess.run([*command, "--version"], capture_output=True, text=True).stdout.strip()
  # The name, not the path: an isolated build environment's zig lives in a new temporary directory each time.
  return f"{[Path(command[0]).name, *command[1:]]}: {out.splitlines()[0] if out else ''}"


def _cache_key(root: Path) -> str:
  """Hash every input of the build, so two checkouts share a build exactly when it would come out the same."""
  digest = hashlib.sha256()
  inputs = [Path(__file__), root / "src" / "scaly_piqp" / "build_config.json", *(root / "licenses").rglob("*")]
  for path in sorted(p for p in inputs if p.is_file()):
    digest.update(path.relative_to(root).as_posix().encode() + b"\0" + path.read_bytes())
  tools = [_tool_identity(os.environ.get("CC", _DEFAULT_CC)), _tool_identity(os.environ.get("CXX", _DEFAULT_CXX[0]))]
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


def _install_build(hook: "BuildHook", root: Path, package_dir: Path, entry: Path, key: str, missing_tools, build) -> list[str]:
  """Copy the cached build `entry` into `package_dir`, first running `build(src_dir, out_dir)` if no checkout made it yet.

  Returns the native tools that are missing when a build is needed and cannot run. A build runs in a fresh scratch directory and is renamed into the cache only once complete, so a
  cache entry is never partial. A failed build keeps its scratch directory for inspection until the
  next attempt at the same key replaces it."""
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
    os.utime(entry)
    stamp = root / ".build_key"
    stamp.unlink(missing_ok=True)
    for name in _OUTPUTS:
      shutil.rmtree(package_dir / name, ignore_errors=True)
      shutil.copytree(entry / name, package_dir / name, symlinks=True)
    stamp.write_text(key)
  hook.app.display_info(f"Installed the build {entry} into {package_dir}")
  _prune_cache(cache)
  return []


def _build_piqp(hook: "BuildHook", third_party_dir: Path, lib_dir: Path, include_dir: Path, licenses_dir: Path, vendored: Path) -> None:
  """Build into `lib_dir`, `include_dir` and `licenses_dir` from sources cloned into the empty `third_party_dir`.

  `vendored` holds the license texts checked into the plugin."""
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


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "scaly"

  def initialize(self, version: str, build_data: dict) -> None:
    if self.target_name == "sdist":
      self.app.display_info("Skipping vendored solver build for sdist target")
      return

    root = Path(self.root)
    package_dir = root / "src" / "scaly_piqp"
    lib_dir = package_dir / "lib"
    mode = _solver_build_mode()
    strict = mode == "require" or (mode == "auto" and version != "editable")
    if mode != "skip" or any(lib_dir.iterdir() if lib_dir.exists() else ()):
      build_data["pure_python"] = False
      build_data["tag"] = _wheel_tag()
    if mode == "skip":
      self.app.display_info("Skipping vendored solver build because SCALY_BUILD_SOLVERS=skip/0")
      return
    key = _cache_key(root)
    stamp = root / ".build_key"
    if stamp.exists() and stamp.read_text() == key and all((package_dir / name).is_dir() for name in _OUTPUTS):
      self.app.display_info(f"PIQP C interface already built at {lib_dir}")
      return

    def build(src: Path, out: Path) -> None:
      _build_piqp(self, src, out / "lib", out / "include" / "piqp", out / "licenses", root / "licenses")

    missing = _install_build(self, root, package_dir, _cache_root() / f"piqp-{key}", key, _missing_piqp_tools, build)
    if missing:
      msg = f"missing native toolchain for PIQP build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(f"{msg}. Install CMake, git, and a C/C++ compiler.")
      self.app.display_info(f"Skipping PIQP build for editable install ({msg}); set SCALY_BUILD_SOLVERS=required to make this fatal.")

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    package_dir = root / "src" / "scaly_piqp"
    (root / ".build_key").unlink(missing_ok=True)
    for path in (package_dir / "lib", package_dir / "include", package_dir / "licenses"):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
