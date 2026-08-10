from __future__ import annotations

from datetime import datetime, timezone
import importlib.metadata
import json
import platform
from pathlib import Path
import subprocess


def _run(args: list[str], root: Path) -> str:
  return subprocess.run(args, cwd=root, check=True, text=True, capture_output=True).stdout.strip()


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
  return {
    "git_commit": commit,
    "git_dirty": dirty,
    "alloy_version": importlib.metadata.version("alloy"),
    "casadi_version": casadi_version,
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
