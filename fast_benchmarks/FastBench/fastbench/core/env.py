"""Capture the execution environment for reproducibility."""
from __future__ import annotations
import os
import platform
import subprocess
from datetime import datetime, timezone


def _git_hash() -> str:
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             cwd=here, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "n/a"
    except Exception:
        return "n/a"


def _pkg_versions() -> dict:
    versions = {}
    for mod in ["numpy", "scipy", "casadi", "osqp", "piqp", "acados_template",
                "do_mpc", "pygrampc"]:
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = None
    return versions


def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            with open("/proc/cpuinfo") as fh:
                for line in fh:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            out = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"],
                                 capture_output=True, text=True, timeout=5)
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor() or "unknown"


def capture_environment() -> dict:
    """Return a dict describing the host, used to stamp every result set."""
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_hash": _git_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_model(),
        "cpu_count": os.cpu_count(),
        "omp_num_threads": os.environ.get("OMP_NUM_THREADS"),
        "openblas_num_threads": os.environ.get("OPENBLAS_NUM_THREADS"),
        "package_versions": _pkg_versions(),
    }
