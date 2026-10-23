"""Verify native Windows wheels in a fresh environment without the toolchain extra."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

CHECK = r"""
import ctypes
import importlib.metadata
import sys
import sysconfig
from pathlib import Path

import numpy as np
import scaly as sc
import scaly_ipopt
import scaly_piqp
import scaly_sqp

from scaly.codegen.toolchain import find_c_compiler

assert sys.platform == "win32"
assert importlib.metadata.version("zig-bin").startswith("0.17.")
compiler = find_c_compiler()
assert compiler is not None and compiler.source == "zig-bin", compiler
assert Path(sys.prefix).resolve() in Path(compiler.command[0]).resolve().parents, compiler

@sc.function(sc.arg("x", 2), outputs=sc.arg("y", 2), name="installed_core")
def core(x):
  return x.sin() + x * x

values = np.array([0.25, -0.5])
np.testing.assert_allclose(core(values), np.sin(values) + values * values, atol=1e-12)

site = Path(sysconfig.get_path("purelib")).resolve()
for module in (sc, scaly_ipopt, scaly_piqp, scaly_sqp):
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
  assert np.isfinite(stats.t_total) and stats.t_total >= 0, stats

# Generated: the same QP inside a Scaly function, whose compiled module calls PIQP.
@sc.function(sc.arg("p", ()), outputs=sc.arg("y"), name="nested_qp")
def nested(p):
  return qp(p)[0].sum() * 2.0

np.testing.assert_allclose(nested(np.array(1.0)), 4.0, atol=1e-7)
nested_stats = nested.solver_stats()
assert nested_stats.to_solver_status().ok and nested_stats.iter > 0, nested_stats

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
  assert np.isfinite(stats.t_total) and stats.t_total >= 0, stats
  np.testing.assert_allclose(stats.obj, (np.sqrt(5.0) - r) ** 2, atol=1e-6)
  assert stats.primal_viol < 1e-6 and stats.n_eval_f > 0, stats

ipopt = sc.solver(disk, "ipopt")
for r in (1.0, 0.5):
  x = ipopt(np.array(r), x0=np.array([0.1, 0.1]))[0]
  np.testing.assert_allclose(x, np.array([2.0, 1.0]) * r / np.sqrt(5.0), atol=1e-6)
  stats = ipopt.stats()
  assert stats.to_solver_status().ok and stats.iter > 0, stats
  np.testing.assert_allclose(stats.obj, (np.sqrt(5.0) - r) ** 2, atol=1e-6)
  assert stats.primal_viol < 1e-6 and stats.n_eval_f > 0, stats
  assert np.isfinite(stats.t_total) and stats.t_total >= 0, stats

@sc.function(sc.arg("r", ()), outputs=sc.arg("y"), name="nested_nlp")
def nested_nlp(r):
  return ipopt(r)[0].sum()

np.testing.assert_allclose(nested_nlp(np.array(1.0)), 3.0 / np.sqrt(5.0), atol=1e-6)
stats = nested_nlp.solver_stats()
assert stats.to_solver_status().ok and stats.iter > 0, stats

kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
kernel32.GetModuleHandleW.restype = ctypes.c_void_p
handle = kernel32.GetModuleHandleW("piqpc.dll")
assert handle, "piqpc.dll is not loaded"
buffer = ctypes.create_unicode_buffer(32768)
kernel32.GetModuleFileNameW(ctypes.c_void_p(handle), buffer, len(buffer))
assert site in Path(buffer.value).resolve().parents, buffer.value
for dll in scaly_ipopt.lib_dir().glob("*.dll"):
  handle = kernel32.GetModuleHandleW(dll.name)
  assert handle, f"{dll.name} is not loaded"
  kernel32.GetModuleFileNameW(ctypes.c_void_p(handle), buffer, len(buffer))
  assert Path(buffer.value).resolve() == dll.resolve(), buffer.value
print(f"python {sys.version.split()[0]}: direct and nested PIQP/IPOPT and SQP solves passed; all solver DLLs from installed wheels")
"""


def main() -> None:
  with tempfile.TemporaryDirectory(prefix="scaly installed wheels ") as root:
    root = Path(root)
    if sys.argv[1:] and sys.argv[1] != "--installed":
      dist = Path(sys.argv[1]).resolve()
      wheels = [next(dist.glob(f"{name}-*.whl")) for name in ("scaly", "scaly_ipopt", "scaly_piqp", "scaly_sqp")]
      venv = root / "venv"
      subprocess.run(["uv", "venv", "--python", sys.argv[2] if len(sys.argv) > 2 else "3.12", str(venv)], check=True)
      python = venv / "Scripts" / "python.exe"
      subprocess.run(["uv", "pip", "install", "--only-binary", ":all:", "--python", str(python), *map(str, wheels)], check=True)
    else:
      python = Path(sys.executable)
      venv = Path(sys.prefix)
    system = os.environ["SystemRoot"]
    keep = ("SystemRoot", "USERPROFILE", "LOCALAPPDATA", "APPDATA", "TEMP", "TMP")
    env = {name: os.environ[name] for name in keep if name in os.environ}
    env |= {"PATH": os.pathsep.join([str(venv / "Scripts"), f"{system}\\System32", system]), "SCALY_CACHE_DIR": str(root / "jit")}
    work = root / "work"
    work.mkdir()
    subprocess.run([str(python), "-I", "-c", CHECK], cwd=work, env=env, check=True)


if __name__ == "__main__":
  main()
