"""External library builds remain separate and their exported manifests suffice to build them."""

import ctypes
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

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
  assert len(write_module(library_host(library), isolated_cache / "export")._libraries) == 1


def test_manifest_consumer_compiles_each_translation_unit_and_links(tmp_path):
  library = sc.CLibrary(
    "shared",
    sources={"shared.c": SOURCE},
    defines={"LOCAL_ONLY": 2, "NO_VALUE": None},
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
      *(f"-D{k}" + (f"={v}" if v is not None else "") for k, v in record["defines"].items()),
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
  include = isolated_cache / "include"
  include.mkdir()
  header = include / "value.h"
  header.write_text("#define LOCAL_ONLY 2\n")
  library = sc.CLibrary("included", sources={"included.c": '#include "value.h"\n' + SOURCE}, include_dirs=(str(include),))
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


def test_external_and_generated_prints_survive_redirected_stdout():
  script = r'''
import numpy as np
import scaly as sc
library = sc.CLibrary("printing", sources={"printing.c": r"""
#include <stdio.h>
void external_printing(const double* x, double* y, double* w) {
  (void)w; printf("external %.0f\n", x[0]); y[0] = x[0];
}
"""})
external = sc.extern(library, "external_printing", inputs=(sc.arg("x", ()),), outputs=sc.arg("y", ()))
@sc.function(sc.arg("x", ()))
def host(x):
  return sc.print("generated {}", external(x))
print("before")
host(np.array(3.))
print("between")
external(np.array(4.))
print("after")
'''
  output = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
  assert output == "before\nexternal 3\ngenerated 3\nbetween\nexternal 4\nafter\n"


def test_missing_external_symbol_names_loader_error():
  import pytest

  if sys.platform != "linux":
    pytest.skip("dlmopen diagnostics")
  missing = sc.extern(sc.CLibrary("missing"), "external_symbol_does_not_exist", inputs=(), outputs=sc.arg("y", ()))
  with pytest.raises(OSError, match="undefined symbol: external_symbol_does_not_exist"):
    missing()


@pytest.mark.parametrize("backends", [(), ("test_solver",)])
def test_aot_cli_exports_without_a_compiler_or_jit_cache(tmp_path, monkeypatch, capsys, backends):
  from dataclasses import replace
  from types import SimpleNamespace
  from scaly.codegen import aot

  library = sc.CLibrary("portable", sources={"shared.c": SOURCE}, defines={"LOCAL_ONLY": 2}, link=("m",))
  kernel = library_host(library)
  render = aot.render_c_module
  monkeypatch.setattr(aot, "render_c_module", lambda *args, **kwargs: replace(render(*args, **kwargs), backends=backends))
  monkeypatch.setattr(aot, "backend_compile_flags", lambda names: ("-ltest_solver",) if names else ())
  monkeypatch.setattr(aot.importlib, "import_module", lambda name: SimpleNamespace(kernel=kernel))
  monkeypatch.setattr(aot, "find_c_compiler", lambda: None)
  cache = tmp_path / "cache"
  monkeypatch.setenv("SCALY_CACHE_DIR", str(cache))
  aot.main(["model:kernel", "--out-dir", str(tmp_path / "export"), "--lanes", "1"])
  output = capsys.readouterr().out
  assert "build manifest: " in output
  assert ("link flags: -ltest_solver" in output) == bool(backends)
  assert not cache.exists()
  manifest = json.loads((tmp_path / "export" / "external_build_host.build.json").read_text())
  assert manifest["libraries"][0]["link"] == ["m"]


_EXTERNAL_INDEX_PROCESS = r"""
import json
import sys
from pathlib import Path
import numpy as np
import scaly as sc
import scaly.codegen.aot as aot
import scaly.codegen.jit as jit
from scaly.codegen.toolchain import find_c_compiler
from scaly.function.model import as_concrete
from tests.codegen.test_external import SOURCE, library_host

lowered = []
lower = aot.lower_function
setattr(aot, "lower_function", lambda *args, **kwargs: (lowered.append(1), lower(*args, **kwargs))[1])
root, kind = Path(sys.argv[1]), sys.argv[2]
library = (
  sc.CLibrary("indexed_header", sources={"kernel.c": '#include "value.h"\n' + SOURCE}, include_dirs=(str(root / "include"),))
  if kind == "header" else sc.CLibrary("indexed_link", link=("external_dependency",))
)
fun = as_concrete(library_host(library))
compiler = find_c_compiler()
assert compiler is not None
index = jit._index_key(fun, compiler)
assert index is not None, "the extern graph must have a trusted index key"
hit = jit._index_read(index) is not None
compiled = jit.get_compiled(fun)
print(json.dumps({"hit": hit, "index": index, "artifact": compiled.cache_key, "lowered": len(lowered), "value": float(compiled.run([np.array(3.)])[0])}))
"""


def _external_process(root, kind, *, mode="", compiler=None):
  import os
  import shlex

  environment = {**os.environ, "SCALY_CACHE_DIR": str(root / "cache"), "SCALY_JIT_KEY": mode}
  if compiler is not None:
    environment["SCALY_CC"] = shlex.join((*compiler, "-L" + str(root / "lib")))
  result = subprocess.run(
    [sys.executable, "-c", _EXTERNAL_INDEX_PROCESS, str(root), kind], env=environment, capture_output=True, text=True, timeout=120
  )
  assert result.returncode == 0, result.stderr
  return json.loads(result.stdout)


@pytest.mark.parametrize("mode", ["", "verify"], ids=["index", "verify"])
@pytest.mark.parametrize("edited", ["value.h", "extra.h"])
def test_include_header_edit_misses_the_index_in_another_process(tmp_path, mode, edited):
  include = tmp_path / "include"
  include.mkdir()
  header = include / "value.h"
  header.write_text("#define LOCAL_ONLY 2\n")
  (include / "extra.h").write_text("#define EXTRA 2\n")
  first = _external_process(tmp_path, "header")
  warm = _external_process(tmp_path, "header")
  assert not first["hit"] and warm["hit"]
  assert (first["lowered"], warm["lowered"]) == (1, 0)
  assert first["index"] == warm["index"]
  assert first["value"] == warm["value"] == 15
  (include / edited).write_text("#define LOCAL_ONLY 3\n")
  changed = _external_process(tmp_path, "header", mode=mode)
  assert not changed["hit"]
  assert changed["lowered"] == 1
  assert changed["index"] != first["index"]
  assert changed["artifact"] != first["artifact"]
  assert changed["value"] == (21 if edited == "value.h" else 15)


@pytest.mark.skipif(sys.platform != "linux", reason="swapping a named ELF dependency")
@pytest.mark.parametrize("mode", ["", "verify"], ids=["index", "verify"])
def test_linked_library_swap_misses_the_index_in_another_process(tmp_path, mode):
  compiler = find_c_compiler()
  assert compiler is not None
  directory = tmp_path / "lib"
  directory.mkdir()
  source = tmp_path / "dependency.c"
  source.write_text(SOURCE)
  target, temporary = directory / "libexternal_dependency.so", directory / "new.so"

  def build(scale):
    subprocess.run(
      [*compiler.command, "-shared", "-fPIC", f"-DLOCAL_ONLY={scale}", str(source), "-o", str(temporary)], capture_output=True, check=True
    )
    temporary.replace(target)

  build(2)
  first = _external_process(tmp_path, "link", compiler=compiler.command)
  warm = _external_process(tmp_path, "link", compiler=compiler.command)
  assert not first["hit"] and warm["hit"]
  assert (first["lowered"], warm["lowered"]) == (1, 0)
  assert first["index"] == warm["index"]
  assert first["value"] == warm["value"] == 15
  build(3)
  changed = _external_process(tmp_path, "link", mode=mode, compiler=compiler.command)
  assert not changed["hit"]
  assert changed["lowered"] == 1
  assert changed["index"] != first["index"]
  assert changed["artifact"] != first["artifact"]
  assert changed["value"] == 21
