"""External library builds remain separate and their exported manifests suffice to build them."""

import ctypes
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

import scaly as sc
from scaly.codegen.aot import write_module
from scaly.codegen.jit import get_compiled
from scaly.codegen.toolchain import find_c_compiler, resolve_library
from scaly.utils.env import shared_lib_ext, shared_lib_flag

SOURCE = """#ifndef LOCAL_ONLY
#error library flag missing
#endif
void external_double(const double* x, double* y, double* w) { (void)w; y[0] = LOCAL_ONLY*x[0]; }
void external_triple(const double* x, double* y, double* w) { (void)w; y[0] = (LOCAL_ONLY+1)*x[0]; }
"""


def library_host(library):
  double = sc.extern(library, "external_double", inputs=(sc.arg("x", ()),), outputs=sc.arg("y", ()))
  triple = sc.extern(library, "external_triple", inputs=(sc.arg("x", ()),), outputs=sc.arg("y", ()))

  @sc.function(sc.arg("x", ()), outputs=sc.arg("y", ()), name="external_build_host")
  def host(x):
    return double(x) + triple(x)

  return host


def test_library_compiles_once_for_two_symbols_and_multiple_graphs(isolated_cache, monkeypatch):
  commands = []
  run = subprocess.run

  def record(command, **kwargs):
    commands.append(command)
    return run(command, **kwargs)

  monkeypatch.setattr(subprocess, "run", record)
  library = sc.CLibrary("shared", sources={"shared.c": SOURCE}, defines={"LOCAL_ONLY": 2})
  np.testing.assert_allclose(library_host(library)(np.array(3.0)), 15)
  np.testing.assert_allclose(library_host(library)(np.array(4.0)), 20)
  compiles = [c for c in commands if "-c" in c and any(Path(str(arg)).name == "shared.c" for arg in c)]
  assert len(compiles) == 1
  assert len(write_module(library_host(library), isolated_cache / "export").libraries) == 1


def test_manifest_consumer_compiles_each_translation_unit_and_links(tmp_path):
  library = sc.CLibrary(
    "shared",
    sources={"shared.c": SOURCE},
    defines={"LOCAL_ONLY": 2},
    compile_flags=("-O1",),
    link=("m",) if sys.platform != "win32" else (),
    versions={"m": "system"},
  )
  module = write_module(library_host(library), tmp_path, lanes=1)
  manifest = json.loads((tmp_path / "external_build_host.build.json").read_text())
  compiler = find_c_compiler()
  assert compiler is not None
  objects = []
  for index, record in enumerate(manifest["libraries"]):
    flags = [
      *("-I" + str(tmp_path / path) for path in record["include_dirs"]),
      *(f"-D{k}={v}" for k, v in record["defines"].items()),
      *record["compile_flags"],
    ]
    for source in record["sources"]:
      target = tmp_path / f"lib{index}.o"
      subprocess.run([*compiler.command, "-fPIC", *flags, "-c", str(tmp_path / source), "-o", str(target)], check=True, capture_output=True)
      objects.append(str(target))
  links = [flag for record in manifest["libraries"] for name in record["link"] for flag in resolve_library(name, compiler.command).flags]
  output = tmp_path / ("libmanifest" + shared_lib_ext())
  exports = []
  if sys.platform == "win32":
    definitions = tmp_path / "exports.def"
    definitions.write_text(f"EXPORTS\n{module.fun.name}\n")
    exports.append(str(definitions))
  subprocess.run(
    [
      *compiler.command,
      "-fPIC",
      shared_lib_flag(),
      *manifest["compile_flags"],
      str(tmp_path / manifest["source"]),
      *objects,
      *exports,
      *links,
      "-o",
      str(output),
    ],
    check=True,
    capture_output=True,
  )
  entry = getattr(ctypes.CDLL(str(output)), module.fun.name)
  pointer = ctypes.POINTER(ctypes.c_double)
  entry.argtypes = [ctypes.POINTER(pointer), ctypes.POINTER(pointer), ctypes.POINTER(ctypes.c_int), pointer, ctypes.c_int]
  x, y = ctypes.c_double(3), ctypes.c_double()
  workspace = (ctypes.c_double * max(module.workspace_size, 1))()
  assert entry((pointer * 1)(ctypes.pointer(x)), (pointer * 1)(ctypes.pointer(y)), None, workspace, 0) == 0
  assert y.value == 15


def test_an_included_header_changes_library_and_jit_keys(isolated_cache):
  header = isolated_cache / "value.h"
  header.write_text("#define LOCAL_ONLY 2\n")
  library = sc.CLibrary("included", sources={"included.c": '#include "value.h"\n' + SOURCE}, include_dirs=(str(isolated_cache),))
  first = get_compiled(library_host(library))
  header.write_text("#define LOCAL_ONLY 3\n")
  second = get_compiled(library_host(library))
  assert first.cache_key != second.cache_key
  np.testing.assert_allclose(first.run([np.array(3.0)]), [15])
  np.testing.assert_allclose(second.run([np.array(3.0)]), [21])


def test_changed_linked_binary_changes_key_and_linux_namespace(isolated_cache, monkeypatch):
  import pytest
  import shlex

  if sys.platform != "linux":
    pytest.skip("Linux linker namespaces")
  compiler = find_c_compiler()
  assert compiler is not None
  source = isolated_cache / "dependency.c"
  source.write_text(SOURCE)
  target = isolated_cache / "libexternal_dependency.so"
  temporary = isolated_cache / "new.so"

  def build(scale):
    subprocess.run(
      [*compiler.command, "-shared", "-fPIC", f"-DLOCAL_ONLY={scale}", str(source), "-o", str(temporary)], check=True, capture_output=True
    )
    temporary.replace(target)

  monkeypatch.setenv("SCALY_CC", shlex.join((*compiler.command, "-L" + str(isolated_cache))))
  library = sc.CLibrary("linked", link=("external_dependency",))
  build(2)
  first = get_compiled(library_host(library))
  build(3)
  second = get_compiled(library_host(library))
  assert first.cache_key != second.cache_key
  np.testing.assert_allclose(first.run([np.array(3.0)]), [15])
  np.testing.assert_allclose(second.run([np.array(3.0)]), [21])
