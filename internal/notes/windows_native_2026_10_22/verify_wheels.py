"""Install built Windows wheels in a clean environment and solve with PIQP and SQP from there.

Run with `uv run internal/notes/windows_native_2026_10_22/verify_wheels.py DIST [PYTHON ...]`, where
DIST holds the scaly, scaly-piqp and scaly-sqp wheels. For each Python version (3.12 by default) the
script creates a virtual environment outside the checkout, installs `scaly[toolchain]` and the
plugins from DIST, and runs the checks below in isolated mode, from an empty directory, with a PATH
of only that environment and the Windows system directories.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

CHECK = r'''
import ctypes
import sys
import sysconfig
from pathlib import Path

import numpy as np
import scaly as sc
import scaly_piqp
import scaly_sqp

site = Path(sysconfig.get_path("purelib")).resolve()
for module in (sc, scaly_piqp, scaly_sqp):
  assert site in Path(module.__file__).resolve().parents, module.__file__

# Direct: a solver Function called from Python. min 0.5 |x - (1, 2)|^2  s.t.  x0 + x1 = 3 - p, x >= 0.
@sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", ()), name="direct_qp")
def direct(x, p):
  return sc.ProblemSpec(minimize=0.5 * sc.sumsqr(x - sc.const(np.array([1.0, 2.0]))), eq=(x.sum() - (3.0 - p),), lb=sc.const(0.0))

qp = sc.solver(direct, "piqp")
for p in (0.0, 1.0):
  x = qp(np.array(p))[0]
  np.testing.assert_allclose(x, [1.0 - p / 2, 2.0 - p / 2], atol=1e-7)
  stats = qp.stats()
  assert stats.to_solver_status().ok and stats.iter > 0, stats

# Generated: the same QP inside a Scaly function, whose compiled module calls PIQP.
@sc.function(sc.arg("p", ()), outputs=sc.arg("y"), name="nested_qp")
def nested(p):
  return qp(p)[0].sum() * 2.0

np.testing.assert_allclose(nested(np.array(1.0)), 4.0, atol=1e-7)
assert nested.solver_stats().to_solver_status().ok

# SQP over PIQP: the point of the disk of radius r nearest to (2, 1) is (2, 1) r / sqrt(5).
@sc.problem(vars=sc.arg("x", 2), params=sc.arg("r", ()), name="disk")
def disk(x, r):
  return sc.ProblemSpec(minimize=sc.sumsqr(x - sc.const(np.array([2.0, 1.0]))), ineq=(sc.bounded(sc.sumsqr(x) - r * r, hi=0.0),))

sqp = sc.solver(disk, "sqp")
for r in (1.0, 0.5):
  x = sqp(np.array(r), x0=np.array([0.1, 0.1]))[0]
  np.testing.assert_allclose(x, np.array([2.0, 1.0]) * r / np.sqrt(5.0), atol=1e-6)
  stats = sqp.stats()
  assert stats.to_solver_status().ok and stats.iter > 0 and stats.qp_iter > 0, stats

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetModuleHandleW.restype = ctypes.c_void_p
handle = kernel32.GetModuleHandleW("piqpc.dll")
assert handle, "piqpc.dll is not loaded"
buffer = ctypes.create_unicode_buffer(32768)
kernel32.GetModuleFileNameW(ctypes.c_void_p(handle), buffer, len(buffer))
assert site in Path(buffer.value).resolve().parents, buffer.value
print(f"python {sys.version.split()[0]}: direct, nested and SQP solves passed; piqpc.dll from {buffer.value}")
'''


def main() -> None:
  dist = Path(sys.argv[1]).resolve()
  wheels = {name: next(dist.glob(f"{name}-*.whl")) for name in ("scaly", "scaly_piqp", "scaly_sqp")}
  piqp = zipfile.ZipFile(wheels["scaly_piqp"])
  print(wheels["scaly_piqp"].name)
  for name in sorted(piqp.namelist()):
    if any(part in name for part in ("/lib/", "/include/", "/licenses/")):
      print(f"  {name} ({piqp.getinfo(name).file_size} bytes)", flush=True)
  for python in sys.argv[2:] or ["3.12"]:
    with tempfile.TemporaryDirectory(prefix="scaly wheel check ") as root:
      venv = Path(root) / "venv"
      subprocess.run(["uv", "venv", "--quiet", "--python", python, str(venv)], check=True)
      requirements = [f"scaly[toolchain] @ {wheels['scaly'].as_uri()}", str(wheels["scaly_piqp"]), str(wheels["scaly_sqp"])]
      subprocess.run(["uv", "pip", "install", "--quiet", "--python", str(venv), *requirements], check=True)
      system = os.environ["SystemRoot"]
      keep = ("SystemRoot", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP")
      env = {name: os.environ[name] for name in keep if name in os.environ}
      env |= {"PATH": os.pathsep.join([str(venv / "Scripts"), f"{system}\\System32", system]), "SCALY_CACHE_DIR": str(Path(root) / "jit")}
      work = Path(root) / "work"
      work.mkdir()
      subprocess.run([str(venv / "Scripts" / "python.exe"), "-I", "-c", CHECK], cwd=work, env=env, check=True)


if __name__ == "__main__":
  main()
