"""Probe installed Ipopt solves, coexistence and the actual loaded native identities in a fresh process."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys


def digest(path):
  with path.open("rb") as stream:
    return hashlib.file_digest(stream, "sha256").hexdigest()


def module_path(address):
  if sys.platform == "win32":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    handle = ctypes.c_void_p()
    assert kernel.GetModuleHandleExW(6, ctypes.c_void_p(address), ctypes.byref(handle))
    buffer = ctypes.create_unicode_buffer(32768)
    assert kernel.GetModuleFileNameW(handle, buffer, len(buffer))
    return Path(buffer.value).resolve()

  class Info(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("base", ctypes.c_void_p), ("symbol", ctypes.c_char_p), ("address", ctypes.c_void_p)]

  info = Info()
  assert ctypes.CDLL(None).dladdr(ctypes.c_void_p(address), ctypes.byref(info))
  return Path(os.fsdecode(info.name)).resolve()


def loaded(root, manifest, address):
  solver = module_path(address)
  expected = {
    str((root / name).resolve()): entry["sha256"] for name, entry in manifest["files"].items() if ".so" in name or name.endswith((".dll", ".dylib"))
  }
  assert str(solver) in expected, (solver, expected)
  actual = {}
  if sys.platform == "darwin":
    dyld = ctypes.CDLL(None)
    dyld._dyld_get_image_name.restype = ctypes.c_char_p
    paths = [os.fsdecode(dyld._dyld_get_image_name(i)) for i in range(dyld._dyld_image_count())]
  else:
    import psutil

    paths = [mapping.path for mapping in psutil.Process().memory_maps()]
  for name in paths:
    path = Path(name)
    if path.is_file():
      key = str(path.resolve())
      if key in expected:
        actual[key] = digest(path)
        assert actual[key] == expected[key], key
  assert str(solver) in actual, (solver, actual)
  for name in manifest.get("dependencies", {}):
    for dependency in manifest["dependencies"][name]["bundled"]:
      path = root / dependency
      assert str(path.resolve()) in actual, ("bundled runtime not loaded from candidate", path, actual)
  return {"symbol_library": str(solver), "loaded_payload": actual}


def other_libraries():
  import casadi as ca
  import numpy as np
  import scipy.linalg

  a = np.eye(96) * 2 + np.ones((96, 96)) * 0.01
  np.testing.assert_allclose(a @ scipy.linalg.solve(a, np.ones(96)), np.ones(96), atol=1e-10)
  x = ca.MX.sym("x")
  solver = ca.nlpsol("coexistence", "ipopt", {"x": x, "f": (x - 2) ** 2}, {"ipopt.print_level": 0, "print_time": False})
  np.testing.assert_allclose(float(solver(x0=0)["x"]), 2, atol=1e-7)
  assert solver.stats()["success"]


def python_solves():
  import numpy as np
  import scaly as sc
  import scaly_ipopt
  from scaly.solvers.paths import solver_paths

  assert "site-packages" in sc.__file__ and "site-packages" in scaly_ipopt.__file__
  library = Path(solver_paths(required=True).loads["ipopt"] or "").resolve()

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("r", ()), name="candidate_disk")
  def disk(x, r):
    return sc.ProblemSpec(minimize=sc.sumsqr(x - sc.const(np.array([2.0, 1.0]))), ineq=(sc.bounded(sc.sumsqr(x) - r * r, hi=0.0),))

  solve = sc.solver(disk, "ipopt", options={"linear_solver": "mumps"})
  cases = {}
  for r in (1.0, 0.5):
    x = solve(np.array(r), x0=np.array([0.1, 0.1]))[0]
    stats = solve.stats()
    np.testing.assert_allclose(x, np.array([2, 1]) * r / np.sqrt(5), atol=1e-6)
    assert stats.to_solver_status().ok and stats.iter > 0 and stats.n_eval_f > 0 and stats.n_eval_h > 0
    assert stats.primal_viol < 1e-6 and np.isfinite(stats.t_total) and stats.t_total >= 0
    cases[f"direct_{r}"] = {"status": int(stats.native_status), "iter": int(stats.iter), "x": x.tolist(), "obj": float(stats.obj)}

  @sc.function(sc.arg("r", ()), outputs=sc.arg("y"), name="candidate_nested")
  def nested(r):
    return solve(r)[0].sum()

  value = nested(np.array(1.0))
  stats = nested.solver_stats()
  np.testing.assert_allclose(value, 3 / np.sqrt(5), atol=1e-6)
  assert stats.to_solver_status().ok and stats.iter > 0
  cases["nested"] = {
    "status": int(stats.native_status),
    "iter": int(stats.iter),
    "x": np.asarray(value).reshape(-1).tolist(),
    "obj": float(stats.obj),
  }
  native = ctypes.CDLL(str(library))
  return cases, ctypes.cast(native.CreateIpoptProblem, ctypes.c_void_p).value


def main():
  parser = argparse.ArgumentParser(__doc__)
  parser.add_argument("--root", type=Path, required=True)
  parser.add_argument("--manifest", type=Path, required=True)
  parser.add_argument("--output", type=Path, required=True)
  parser.add_argument("--consumer", type=Path)
  parser.add_argument("--order", choices=("scaly", "others"), default="scaly")
  args = parser.parse_args()
  manifest = json.loads(args.manifest.read_text())
  handles = []
  if sys.platform == "win32":
    handles.append(os.add_dll_directory(str(args.root / "scaly_ipopt/lib")))
  if args.order == "scaly":
    solver = next(name for name in manifest["files"] if Path(name).name in {"libscaly_ipopt-3.dll", "libipopt.so", "libipopt.dylib"})
    handles.append(ctypes.CDLL(str(args.root / solver)))
  if args.consumer:
    import numpy as np

    consumer = ctypes.CDLL(str(args.consumer))
    consumer.probe.argtypes = [ctypes.POINTER(ctypes.c_double)]
    consumer.probe.restype = ctypes.c_int
    consumer.probe_ipopt_address.restype = ctypes.c_void_p
    out = (ctypes.c_double * 514)()
    status = consumer.probe(out)
    assert status == 0 and out[513] > 0, (status, list(out)[-2:])
    np.testing.assert_allclose(list(out)[:512], np.full(512, 5.0), atol=1e-7)
    assert abs(out[512]) < 1e-10
    cases = {"sparse_mumps_512": {"status": status, "iter": int(out[513]), "x": list(out)[:512], "obj": out[512]}}
    identity = loaded(args.root, manifest, consumer.probe_ipopt_address())
  else:
    if args.order == "others":
      other_libraries()
    cases, address = python_solves()
    identity = loaded(args.root, manifest, address)
    other_libraries()
    again, address = python_solves()
    assert cases == again, "Coexistence changed numerical solves"
    identity = loaded(args.root, manifest, address)
  args.output.write_text(json.dumps({"cases": cases, "identities": identity, "order": args.order}, indent=2) + "\n")


if __name__ == "__main__":
  main()
