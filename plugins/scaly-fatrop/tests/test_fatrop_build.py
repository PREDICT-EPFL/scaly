"""The Fatrop package supplies an isolated native C solver and its headers and notices."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

from scaly.codegen.toolchain import find_c_compiler

ROOT = Path(__file__).parents[1]
native = pytest.mark.solver("fatrop", package="scaly_fatrop", header="fatrop/ocp/OCPCInterface.h")


@pytest.mark.parametrize(
  "system,machine,expected",
  [
    ("Linux", "x86_64", "X64_INTEL_HASWELL"),
    ("Windows", "amd64", "X64_INTEL_HASWELL"),
    ("Darwin", "arm64", "ARMV8A_APPLE_M1"),
    ("Linux", "aarch64", "ARMV8A_ARM_CORTEX_A76"),
  ],
)
def test_blasfeo_target(system, machine, expected):
  recipe = runpy.run_path(str(ROOT / "native_build.py"))
  assert recipe["_blasfeo_target"](system, machine) == expected


@native
def test_pinned_licenses():
  package = importlib.import_module("scaly_fatrop")
  root = Path(package.__file__).parent
  pins = json.loads((root / "build_config.json").read_text())
  notices = (root / "licenses/THIRD_PARTY_NOTICES.md").read_text()
  for name, pin in pins.items():
    assert (root / "licenses" / name).is_dir()
    assert f"| {name} | {pin['version']} |" in notices
  assert (root / "licenses/fatrop/LICENSE.txt").is_file()
  assert (root / "licenses/fatrop/LICENSE-EPL-2.0.txt").is_file()
  assert (root / "licenses/blasfeo/LICENSE.txt").is_file()


@native
def test_only_c_interface_is_exported():
  package = importlib.import_module("scaly_fatrop")
  if sys.platform == "win32":
    import pefile

    with pefile.PE(str(package.lib_dir() / "fatrop.dll")) as library:
      symbols = {entry.name.decode() for entry in library.DIRECTORY_ENTRY_EXPORT.symbols}
      imports = {entry.dll.decode().lower() for entry in library.DIRECTORY_ENTRY_IMPORT}
    assert not any("blasfeo" in name for name in imports)
  else:
    library = next(package.lib_dir().glob("libfatrop.*"))
    command = ["nm", "-gU"] if sys.platform == "darwin" else ["nm", "-D", "--defined-only"]
    result = subprocess.run([*command, str(library)], capture_output=True, text=True, check=True)
    symbols = {line.split()[-1].removeprefix("_") if sys.platform == "darwin" else line.split()[-1] for line in result.stdout.splitlines() if line}
  assert "fatrop_ocp_c_solve" in symbols
  assert all(name.startswith("fatrop_ocp_c_") for name in symbols), symbols


@pytest.fixture
def c_probe(tmp_path):
  package = importlib.import_module("scaly_fatrop")
  compiler = find_c_compiler()
  assert compiler is not None
  source = ROOT / "tests" / "solve.c"
  flags = [f"-I{package.include_dir()}", f"-L{package.lib_dir()}", f"-Wl,-rpath,{package.lib_dir()}", "-lfatrop", "-lm"]
  if sys.platform == "win32":
    flags = [f"-I{package.include_dir()}", f"-L{package.lib_dir()}", "-DFATROP_SYMBOL_EXPORT=__declspec(dllimport)", "-lfatrop"]
  executable = tmp_path / ("solve.exe" if sys.platform == "win32" else "solve")
  subprocess.run([*compiler.command, "-std=c99", str(source), *flags, "-o", str(executable)], check=True, capture_output=True, text=True)
  shared = tmp_path / ("probe.dll" if sys.platform == "win32" else "probe.dylib" if sys.platform == "darwin" else "probe.so")
  subprocess.run(
    [*compiler.command, "-std=c99", "-shared", "-fPIC", "-DPROBE_LIBRARY", str(source), *flags, "-o", str(shared)],
    check=True,
    capture_output=True,
    text=True,
  )
  return executable, shared


@native
def test_standalone_c_solve(c_probe):
  env = os.environ.copy()
  if sys.platform == "win32":
    package = importlib.import_module("scaly_fatrop")
    env["PATH"] = str(package.lib_dir()) + os.pathsep + env["PATH"]
  subprocess.run([str(c_probe[0])], env=env, check=True, capture_output=True, text=True)


@native
@pytest.mark.solver("piqp")
@pytest.mark.parametrize("piqp_first", [True, False])
def test_piqp_and_fatrop_solve_in_one_process(c_probe, piqp_first):
  subprocess.run(
    [
      sys.executable,
      "-c",
      """
import ctypes, os, sys
import numpy as np
import scaly_fatrop, scaly_piqp
from tests.solvers.problem_helpers import build_qp, solve_qp

if sys.platform == 'win32':
  dll_dirs = [os.add_dll_directory(str(package.lib_dir())) for package in (scaly_fatrop, scaly_piqp)]
mode = getattr(ctypes, 'RTLD_GLOBAL', 0)

def load_piqp():
  library = scaly_piqp.lib_dir() / 'piqpc.dll' if sys.platform == 'win32' else next(scaly_piqp.lib_dir().glob('libpiqpc.*'))
  return ctypes.CDLL(str(library), mode=mode)

if sys.argv[2] == 'True':
  piqp = load_piqp()
probe = ctypes.CDLL(sys.argv[1], mode=mode)
if sys.argv[2] == 'False':
  piqp = load_piqp()
assert probe.run_solve() == 0
qp = build_qp(P=np.eye(2), c=np.array([-1., -2.]), x_lb=np.zeros(2))
out = solve_qp(qp, np.zeros(2), np.zeros(0), np.zeros(0))
np.testing.assert_allclose(out['x'], [1., 2.], atol=1e-6)
assert probe.run_solve() == 0
""",
      str(c_probe[1]),
      str(piqp_first),
    ],
    cwd=ROOT.parents[1],
    check=True,
    capture_output=True,
    text=True,
  )
