"""Custom build hook that vendors IPOPT as a shared library inside the scaly-ipopt wheel."""

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
from pathlib import Path

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_HELPER = Path(__file__).parent / "solver_build.py"
if not _HELPER.exists():
  _HELPER = _HELPER.parent.parent / "solver_build.py"
_SHARED = runpy.run_path(str(_HELPER))
_NATIVE = runpy.run_path(str(Path(__file__).with_name("native_build.py")))
_find_fortran_compiler = _NATIVE["_find_fortran_compiler"]
_msys2_bash = _NATIVE["_msys2_bash"]
_locked = _NATIVE["_locked"]
_build_ipopt_stack = _NATIVE["_build_ipopt_stack"]


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


def _has_cxx_compiler() -> bool:
  return any(shutil.which(cmd) for cmd in ("c++", "g++", "clang++"))


def _missing_ipopt_tools() -> list[str]:
  missing = [cmd for cmd in ("git", "make", "cmake") if shutil.which(cmd) is None]
  if os.name == "nt":
    missing += [cmd for cmd in ("ninja", "dlltool") if shutil.which(cmd) is None]
  if shutil.which("cc") is None:
    missing.append("cc")
  if not _has_cxx_compiler():
    missing.append("c++")
  fc = _find_fortran_compiler(required=False)
  if fc is None:
    missing.append("gfortran")
  elif os.name == "nt" and not _msys2_bash(fc).is_file():
    missing.append("MSYS2 UCRT64 GFortran")
  return missing


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


def _remove_tree(path: Path) -> None:
  def remove_readonly(function, filename, error):
    if not isinstance(error, PermissionError):
      raise error
    os.chmod(filename, stat.S_IWRITE)
    function(filename)

  if path.exists():
    shutil.rmtree(path, onexc=remove_readonly)


def _cache_key(root: Path) -> str:
  """Hash every input of the build, so two checkouts share a build exactly when it would come out the same."""
  digest = hashlib.sha256()
  digest.update(_SHARED["native_hash"](root, "scaly_ipopt").encode())
  fc = _find_fortran_compiler(required=False) or "gfortran"
  tools = [_tool_identity(os.environ.get("CC", "cc")), _tool_identity(os.environ.get("CXX", "c++")), _tool_identity(fc)]
  env = [f"{name}={os.environ.get(name, '')}" for name in _KEY_ENV]
  for part in (platform.system(), platform.machine(), *platform.libc_ver(), platform.mac_ver()[0], *tools, *env):
    digest.update(part.encode() + b"\0")
  return digest.hexdigest()[:16]


def _native_context() -> dict:
  tools = {name: _tool_identity(os.environ.get(name, default)) for name, default in (("CC", "cc"), ("CXX", "c++"))}
  tools["FC"] = _tool_identity(_find_fortran_compiler(required=False) or "gfortran")
  return {
    "platform": [platform.system(), platform.machine(), *platform.libc_ver(), platform.mac_ver()[0]],
    "tools": tools,
    "env": {name: os.environ.get(name, "") for name in _KEY_ENV},
  }


def _prepare_native(root: Path) -> tuple[Path, str, str]:
  digest = _SHARED["native_hash"](root, "scaly_ipopt")
  cache = _cache_root()
  key = _cache_key(root)
  entry = cache / f"ipopt-{key}"
  if _solver_build_mode() == "source":
    return entry, key, digest
  if entry.exists():
    return entry, key, digest
  local = _SHARED["find_local"](cache, "ipopt", digest, _native_context())
  if local is not None:
    return local, local.name.removeprefix("ipopt-"), digest
  release = cache / _SHARED["release_key"]("scaly_ipopt", digest)
  with _locked(cache / "locks" / f"{release.name}.lock"):
    if release.exists() or _SHARED["download_release"]("scaly_ipopt", digest, release):
      return release, release.name, digest
  return entry, key, digest


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
      metadata = {"hash": _SHARED["native_hash"](root, "scaly_ipopt"), **_native_context()}
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
    package_dir = root / "src" / "scaly_ipopt"
    lib_dir = package_dir / "lib"
    mode = _solver_build_mode()
    strict = mode in {"require", "source"} or (mode == "auto" and version != "editable")
    system = platform.system()

    if mode != "skip" or any(lib_dir.glob("lib*")):
      build_data["pure_python"] = False
      build_data["tag"] = _wheel_tag()
    if mode == "skip":
      self.app.display_info("Skipping vendored solver build because SCALY_BUILD_SOLVERS=skip/0")
      return
    entry, key, digest = _prepare_native(root)
    marker = package_dir / "native-inputs.sha256"
    marker.write_text(digest + "\n")
    build_data.setdefault("force_include", {})[str(marker)] = "scaly_ipopt/native-inputs.sha256"
    if (entry / "wheel-libs").is_dir():
      build_data["force_include"][str(entry / "wheel-libs")] = "scaly_ipopt.libs"
    stamp = root / ".build_key"
    if mode != "source" and stamp.exists() and stamp.read_text() == key and all((package_dir / name).is_dir() for name in _OUTPUTS):
      self.app.display_info(f"IPOPT already built at {lib_dir}")
      return

    def build(src: Path, out: Path) -> None:
      _build_ipopt_stack(self, src, out, root / "licenses")

    missing = _install_build(self, root, package_dir, entry, key, _missing_ipopt_tools, build)
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


if __name__ == "__main__":
  entry, _, _ = _prepare_native(Path(__file__).parent)
  print(f"source={str(not entry.exists() or _solver_build_mode() == 'source').lower()}")
