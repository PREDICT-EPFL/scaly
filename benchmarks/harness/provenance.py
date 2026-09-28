from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess

from benchmarks.harness import NATIVE_CFLAGS


def _run(args: list[str], root: Path) -> str:
  return subprocess.run(args, cwd=root, check=True, text=True, capture_output=True).stdout.strip()


def _runtime_library(link_library: Path) -> Path:
  if link_library.suffix != ".so":
    return link_library
  digest = hashlib.sha256(link_library.read_bytes()).digest()
  matches = [path for path in link_library.parent.glob(f"{link_library.name}.*") if hashlib.sha256(path.read_bytes()).digest() == digest]
  return min(matches, key=lambda path: len(path.name)) if matches else link_library


def cpu_settings(root: Path = Path("/sys/devices/system/cpu")) -> dict[str, object]:
  """Read frequency policy and boost settings for reproducible timings."""
  policies = {}
  for policy in sorted((root / "cpufreq").glob("policy*")):
    policies[policy.name] = {
      name: (policy / name).read_text().strip()
      for name in ("scaling_driver", "scaling_governor", "energy_performance_preference", "scaling_min_freq", "scaling_max_freq")
      if (policy / name).exists()
    }
  boost = root / "cpufreq/boost"
  no_turbo = root / "intel_pstate/no_turbo"
  enabled = bool(int(boost.read_text())) if boost.exists() else (not bool(int(no_turbo.read_text())) if no_turbo.exists() else None)
  return {"policies": policies, "boost_enabled": enabled, "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None}


def require_headline_settings(expected_boost: bool) -> dict[str, object]:
  """Reject headline timings without the requested frequency policy."""
  settings = cpu_settings()
  policies = settings["policies"]
  if not policies:
    raise RuntimeError("--headline requires Linux cpufreq controls; on macOS or unsupported hosts, run without --headline")
  if any(policy.get("scaling_governor") != "performance" for policy in policies.values()):
    raise RuntimeError("headline runs require the performance governor on every CPU policy")
  if settings["boost_enabled"] != expected_boost:
    raise RuntimeError(f"headline runs require boost {'enabled' if expected_boost else 'disabled'}")
  return settings


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
  from scaly.opt.external.paths import solver_paths
  from scaly.opt.method import external_methods

  paths = solver_paths()
  ipopt = external_methods().get("ipopt")
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
    "scaly_version": importlib.metadata.version("scaly"),
    "casadi_version": casadi_version,
    "native_solvers": native_solvers,
    "compiler": compiler_version,
    "native_cflags": list(NATIVE_CFLAGS),
    "platform": platform.platform(),
    "cpu_settings": cpu_settings(),
    "compilation_caches": {name: os.environ.get(name) for name in ("SCALY_CACHE_DIR", "SCALY_CASADI_IPOPT_CACHE")},
    "python_version": platform.python_version(),
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "cli_args": cli_args,
  }


def write(path: Path, data: dict[str, object]) -> Path:
  sidecar = Path(f"{path}.provenance.json")
  sidecar.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
  return sidecar
