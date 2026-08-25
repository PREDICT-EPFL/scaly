from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess


def _run(args: list[str], root: Path) -> str:
  return subprocess.run(args, cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def _runtime_library(link_library: Path) -> Path:
  if link_library.suffix != ".so":
    return link_library
  digest = hashlib.sha256(link_library.read_bytes()).digest()
  matches = [path for path in link_library.parent.glob(f"{link_library.name}.*") if hashlib.sha256(path.read_bytes()).digest() == digest]
  return min(matches, key=lambda path: len(path.name)) if matches else link_library


def collect(root: Path, compiler: str, cli_args: list[str]) -> dict[str, object]:
  try:
    commit = _run(["git", "rev-parse", "HEAD"], root)
    dirty = bool(_run(["git", "status", "--porcelain"], root))
  except (FileNotFoundError, subprocess.CalledProcessError):
    commit, dirty = "unknown", None
  try:
    compiler_version = subprocess.run([compiler, "--version"], check=True, text=True, capture_output=True).stdout.splitlines()[0]
  except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
    compiler_version = "unknown"
  try:
    casadi_version = importlib.metadata.version("casadi")
  except importlib.metadata.PackageNotFoundError:
    casadi_version = "not installed"
  from alloy.solvers.paths import solver_paths
  from alloy.solvers.registry import loaded_backends

  paths = solver_paths()
  ipopt = loaded_backends().get("ipopt")
  ipopt_library = paths.loads.get("ipopt")
  native_solvers = {}
  if ipopt is not None and ipopt_library is not None:
    link_library = Path(ipopt_library).resolve()
    native_solvers["ipopt"] = {
      "library": str(_runtime_library(link_library)),
      "link_library": str(link_library),
      "build": getattr(ipopt, "build_config", {}),
    }
  return {
    "git_commit": commit,
    "git_dirty": dirty,
    "alloy_version": importlib.metadata.version("alloy"),
    "casadi_version": casadi_version,
    "native_solvers": native_solvers,
    "compiler": compiler_version,
    "platform": platform.platform(),
    "python_version": platform.python_version(),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "cli_args": cli_args,
  }


def write(path: Path, data: dict[str, object]) -> Path:
  sidecar = Path(f"{path}.provenance.json")
  sidecar.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
  return sidecar
