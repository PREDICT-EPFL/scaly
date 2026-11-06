"""Custom build hook that vendors Fatrop as a shared library inside the scaly-fatrop wheel."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import runpy
import shutil
import stat
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_HELPER = Path(__file__).parent / "solver_build.py"
if not _HELPER.exists():
  _HELPER = _HELPER.parent.parent / "solver_build.py"
_SHARED = runpy.run_path(str(_HELPER))
_NATIVE = runpy.run_path(str(Path(__file__).with_name("native_build.py")))
_compiler_command = _NATIVE["_compiler_command"]
_DEFAULT_CC = _NATIVE["_DEFAULT_CC"]
_DEFAULT_CXX = _NATIVE["_DEFAULT_CXX"]
_build_fatrop = _NATIVE["_build_fatrop"]


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
  if raw in {"source", "force"}:
    return "source"
  if raw in _BUILD_SOLVER_SKIP:
    return "skip"
  if raw in _BUILD_SOLVER_REQUIRE:
    return "require"
  if raw in {"", "auto"}:
    return "auto"
  raise RuntimeError("SCALY_BUILD_SOLVERS must be one of auto, required/1/true, source/force, or skip/0/false")


def _missing_fatrop_tools() -> list[str]:
  missing = [cmd for cmd in ("git", "cmake") if shutil.which(cmd) is None]
  if not _compiler_command(os.environ.get("CC", _DEFAULT_CC)):
    missing.append("cc")
  if not any(_compiler_command(cmd) for cmd in ([os.environ["CXX"]] if "CXX" in os.environ else _DEFAULT_CXX)):
    missing.append("c++")
  if platform.system() == "Windows" and shutil.which("ninja") is None:
    missing.append("ninja")
  return missing


# Everything else that decides the built files, beside this hook, the pins and the license texts.
_KEY_ENV = ("CC", "CXX", "CFLAGS", "CXXFLAGS", "ASMFLAGS", "LDFLAGS", "MACOSX_DEPLOYMENT_TARGET", "CMAKE_GENERATOR")
_OUTPUTS = ("lib", "include", "licenses")
_CACHE_MAX_AGE = 30 * 24 * 3600


def _remove_tree(path: Path) -> None:
  def remove_readonly(function, filename, error):
    if not isinstance(error, PermissionError):
      raise error
    # Disposable Git clones contain read-only pack files on Windows.
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
  return f"{[Path(command[0]).name, *command[1:]]}: {out.splitlines()[0] if out else ''}"


def _cache_key(root: Path) -> str:
  """Hash every input of the build, so two checkouts share a build exactly when it would come out the same."""
  digest = hashlib.sha256()
  digest.update(_SHARED["native_hash"](root, "scaly_fatrop").encode())
  tools = [_tool_identity(os.environ.get("CC", _DEFAULT_CC)), _tool_identity(os.environ.get("CXX", _DEFAULT_CXX[0]))]
  env = [f"{name}={os.environ.get(name, '')}" for name in _KEY_ENV]
  for part in (platform.system(), platform.machine(), *platform.libc_ver(), platform.mac_ver()[0], *tools, *env):
    digest.update(part.encode() + b"\0")
  return digest.hexdigest()[:16]


def _native_context() -> dict:
  tools = {name: _tool_identity(os.environ.get(name, default)) for name, default in (("CC", _DEFAULT_CC), ("CXX", _DEFAULT_CXX[0]))}
  return {
    "platform": [platform.system(), platform.machine(), *platform.libc_ver(), platform.mac_ver()[0]],
    "tools": tools,
    "env": {name: os.environ.get(name, "") for name in _KEY_ENV},
  }


def _prepare_native(root: Path) -> tuple[Path, str, str]:
  digest = _SHARED["native_hash"](root, "scaly_fatrop")
  cache = _cache_root()
  key = _cache_key(root)
  entry = cache / f"fatrop-{key}"
  if _solver_build_mode() == "source":
    return entry, key, digest
  if entry.exists():
    return entry, key, digest
  local = _SHARED["find_local"](cache, "fatrop", digest, _native_context())
  if local is not None:
    return local, local.name.removeprefix("fatrop-"), digest
  release = cache / _SHARED["release_key"]("scaly_fatrop", digest)
  with _locked(cache / "locks" / f"{release.name}.lock"):
    if release.exists() or _SHARED["download_release"]("scaly_fatrop", digest, release):
      return release, release.name, digest
  return entry, key, digest


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
    if _solver_build_mode() == "source":
      _remove_tree(entry)
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
      metadata = {"hash": _SHARED["native_hash"](root, "scaly_fatrop"), **_native_context()}
      (scratch / "out" / "native-build.json").write_text(json.dumps(metadata))
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


class BuildHook(BuildHookInterface):
  PLUGIN_NAME = "scaly"

  def initialize(self, version: str, build_data: dict) -> None:
    if self.target_name == "sdist":
      self.app.display_info("Skipping vendored solver build for sdist target")
      return

    root = Path(self.root)
    package_dir = root / "src" / "scaly_fatrop"
    lib_dir = package_dir / "lib"
    mode = _solver_build_mode()
    strict = mode in {"require", "source"} or (mode == "auto" and version != "editable")
    if mode != "skip" or any(lib_dir.iterdir() if lib_dir.exists() else ()):
      build_data["pure_python"] = False
      build_data["tag"] = _wheel_tag()
    if mode == "skip":
      self.app.display_info("Skipping vendored solver build because SCALY_BUILD_SOLVERS=skip/0")
      return
    entry, key, digest = _prepare_native(root)
    marker = package_dir / "native-inputs.sha256"
    marker.write_text(digest + "\n")
    build_data.setdefault("force_include", {})[str(marker)] = "scaly_fatrop/native-inputs.sha256"
    if (entry / "wheel-libs").is_dir():
      build_data["force_include"][str(entry / "wheel-libs")] = "scaly_fatrop.libs"
    stamp = root / ".build_key"
    if mode != "source" and stamp.exists() and stamp.read_text() == key and all((package_dir / name).is_dir() for name in _OUTPUTS):
      self.app.display_info(f"Fatrop C interface already built at {lib_dir}")
      return

    def build(src: Path, out: Path) -> None:
      _build_fatrop(self, src, out)

    missing = _install_build(self, root, package_dir, entry, key, _missing_fatrop_tools, build)
    if missing:
      msg = f"missing native toolchain for Fatrop build: {', '.join(missing)}"
      if strict:
        raise RuntimeError(f"{msg}. Install CMake, git, and a C/C++ compiler.")
      self.app.display_info(f"Skipping Fatrop build for editable install ({msg}); set SCALY_BUILD_SOLVERS=required to make this fatal.")

  def clean(self, versions: list[str]) -> None:
    root = Path(self.root)
    package_dir = root / "src" / "scaly_fatrop"
    (root / ".build_key").unlink(missing_ok=True)
    for path in (package_dir / "lib", package_dir / "include", package_dir / "licenses"):
      if path.exists():
        self.app.display_info(f"Removing {path}")
        shutil.rmtree(path)
