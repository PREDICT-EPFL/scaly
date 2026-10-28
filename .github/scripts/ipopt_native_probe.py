"""Probe installed Ipopt solves, coexistence and the actual loaded native identities in a fresh process."""

from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
from importlib import metadata
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


def bound_address(library, path, symbol="CreateIpoptProblem"):
  if sys.platform != "win32":
    return ctypes.cast(getattr(library, symbol), ctypes.c_void_p).value
  import pefile

  with pefile.PE(str(path)) as pe:
    imports = [entry for table in ("DIRECTORY_ENTRY_IMPORT", "DIRECTORY_ENTRY_DELAY_IMPORT") for entry in getattr(pe, table, [])]
    matches = [item for entry in imports if b"ipopt" in entry.dll.lower() for item in entry.imports if symbol is None or item.name == symbol.encode()]
    assert matches, (path, symbol)
    return ctypes.c_void_p.from_address(library._handle + matches[0].address - pe.OPTIONAL_HEADER.ImageBase).value


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
  elif sys.platform == "win32":
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    modules = (ctypes.c_void_p * 4096)()
    required = ctypes.c_ulong()
    assert kernel.K32EnumProcessModules(ctypes.c_void_p(-1), modules, ctypes.sizeof(modules), ctypes.byref(required))
    assert required.value <= ctypes.sizeof(modules)
    paths = [str(module_path(address)) for address in modules[: required.value // ctypes.sizeof(ctypes.c_void_p)]]
  else:
    paths = [parts[5] for line in Path("/proc/self/maps").read_text().splitlines() if len(parts := line.split(maxsplit=5)) == 6]
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


def casadi_identity(address, package):
  actual = module_path(address)
  assert actual.parent == package.resolve() and actual.name.startswith("libipopt"), (actual, package)
  distribution = metadata.distribution("casadi")
  record = next(entry for entry in distribution.files or [] if entry.as_posix() == f"casadi/{actual.name}")
  expected = record.hash
  assert Path(str(distribution.locate_file(record))).resolve() == actual and expected and expected.mode == "sha256", actual
  sha = digest(actual)
  assert base64.urlsafe_b64encode(bytes.fromhex(sha)).decode().rstrip("=") == expected.value, actual
  return {"symbol_library": str(actual), "sha256": sha}


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
  extension = ".dll" if sys.platform == "win32" else ".dylib" if sys.platform == "darwin" else ".so"
  package = Path(ca.__file__).parent
  plugin = package / ("libcasadi_nlpsol_ipopt" + extension)
  native = ctypes.CDLL(str(plugin))
  address = bound_address(native, plugin, None if sys.platform == "win32" else "CreateIpoptProblem")
  return casadi_identity(address, package)


def python_solves():
  import numpy as np
  import scaly as sc
  import scaly_ipopt
  from scaly.function.model import as_concrete

  assert "site-packages" in sc.__file__ and "site-packages" in scaly_ipopt.__file__

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
  compiled = {"direct": as_concrete(solve.function)._compile(), "nested": as_concrete(nested)._compile()}
  return cases, {name: bound_address(function._lib, function._artifact.lib_path) for name, function in compiled.items()}


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
    cases, addresses = python_solves()
    identity = {name: loaded(args.root, manifest, address) for name, address in addresses.items()}
    casadi_identity = other_libraries()
    again, addresses = python_solves()
    assert cases == again, "Coexistence changed numerical solves"
    identity = {name: loaded(args.root, manifest, address) for name, address in addresses.items()}
    identity["interpreted_casadi"] = casadi_identity
  args.output.write_text(json.dumps({"cases": cases, "identities": identity, "order": args.order}, indent=2) + "\n")


if __name__ == "__main__":
  main()
