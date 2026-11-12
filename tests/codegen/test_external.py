"""External library builds remain separate and their exported manifests suffice to build them."""

import ctypes
import json
import os
from pathlib import Path
import shlex
import shutil
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


def _namespace_libraries(root, kind):
  compiler = find_c_compiler()
  assert compiler is not None
  source = root / "dependency.c"
  source.write_text(SOURCE)

  def build(directory, name, scale, *, soname=None, nested=False):
    directory.mkdir(exist_ok=True)
    target = directory / f"lib{name}.so"
    temporary = directory / "new.so"
    inputs = [str(source), f"-DLOCAL_ONLY={scale}"]
    if nested:
      wrapper = directory / "wrapper.c"
      wrapper.write_text(
        "double nested_scale(void);\n"
        "void external_double(const double* x, double* y, double* w) { y[0] = nested_scale()*x[0]; }\n"
        "void external_triple(const double* x, double* y, double* w) { y[0] = (nested_scale()+1)*x[0]; }\n"
      )
      dependency = directory / "nested.c"
      dependency.write_text(f"double nested_scale(void) {{ return {scale}; }}\n")
      subprocess.run(
        [*compiler.command, "-shared", "-fPIC", str(dependency), "-Wl,-soname,libnested.so", "-o", str(directory / "libnested.so")],
        capture_output=True,
        check=True,
      )
      inputs = [str(wrapper), "-L" + str(directory), "-lnested", "-Wl,-rpath," + str(directory)]
    subprocess.run(
      [*compiler.command, "-shared", "-fPIC", *inputs, "-Wl,-soname," + (soname or target.name), "-o", str(temporary)],
      capture_output=True,
      check=True,
    )
    temporary.replace(target)
    os.environ["SCALY_CC"] = shlex.join((*compiler.command, "-L" + str(directory)))
    host = library_host(sc.CLibrary(name, link=(name,)))
    handle = get_compiled(host)
    cached = get_compiled(host)
    assert handle.cache_key == cached.cache_key
    np.testing.assert_allclose(cached.run([np.array(3.0)]), [(2 * scale + 1) * 3])
    return handle

  handles = []
  if kind != "exhaustion":
    for i in range(24):
      handles.append((build(root, f"independent{i}", i + 2), (2 * i + 5) * 3))
  if kind in ("path", "soname", "transitive"):
    for i in range(2):
      directory = root if kind == "path" else root / f"version{i}"
      name = f"versioned{i}" if kind == "transitive" else "versioned"
      soname = None if kind == "transitive" else "libversioned.so"
      handles.append((build(directory, name, i + 2, soname=soname, nested=kind == "transitive"), (2 * i + 5) * 3))
  if kind == "exhaustion":
    for i in range(32):
      try:
        handle = build(root, "versioned", i + 2)
      except OSError as exc:
        message = str(exc)
        assert "dlmopen failed" in message
        assert "no more namespaces" in message or "static TLS" in message, message
        assert "fresh process" in message and "versions" in message, message
        break
      handles.append((handle, (2 * i + 5) * 3))
    else:
      raise AssertionError("32 incompatible live dependencies unexpectedly fit")
  for handle, expected in reversed(handles):
    np.testing.assert_allclose(handle.run([np.array(3.0)]), [expected])


@pytest.mark.skipif(sys.platform != "linux", reason="glibc linker namespaces")
@pytest.mark.parametrize("kind", ["path", "soname", "transitive", "exhaustion"])
def test_compatible_linker_namespaces(tmp_path, kind):
  script = "from pathlib import Path; import sys; from tests.codegen.test_external import _namespace_libraries; _namespace_libraries(Path(sys.argv[1]), sys.argv[2])"
  result = subprocess.run(
    [sys.executable, "-c", script, str(tmp_path), kind],
    env={**os.environ, "SCALY_CACHE_DIR": str(tmp_path / "cache")},
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert result.returncode == 0, result.stderr


def _namespace_allocators(root, order):
  compiler = find_c_compiler()
  assert compiler is not None
  allocator = root / "allocator.c"
  allocator.write_text("""#include <stddef.h>
#include <stdlib.h>
#include <sys/mman.h>
typedef union { struct { size_t size, tag; } info; max_align_t alignment; } Header;
void* malloc(size_t n) {
  Header* h = mmap(NULL, n + sizeof(Header), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
  if (h == MAP_FAILED) return NULL;
  h->info.size = n + sizeof(Header); h->info.tag = 0xfeedbeef;
  return h + 1;
}
void free(void* p) {
  if (!p) return;
  Header* h = (Header*)p - 1;
  if (h->info.tag != 0xfeedbeef) abort();
  munmap(h, h->info.size);
}
void allocator_marker(void) {}
""")
  subprocess.run([*compiler.command, "-shared", "-fPIC", str(allocator), "-o", str(root / "liballocator.so")], capture_output=True, check=True)
  os.environ["SCALY_CC"] = shlex.join((*compiler.command, "-L" + str(root)))
  handles = []
  for kind in order.split(","):
    marker = "void allocator_marker(void);" if kind == "allocator" else ""
    call = "allocator_marker();" if kind == "allocator" else ""
    library = sc.CLibrary(
      kind,
      sources={
        "allocating.c": "#include <stdlib.h>\n#include <string.h>\n" + marker + "\n"
        "void external_allocating(const double* x, double* y, double* w) { " + call + 'char* s = strdup("hello"); y[0] = x[0] + strlen(s); free(s); }'
      },
      compile_flags=("-fno-builtin",),
      link=("allocator",) if kind == "allocator" else (),
    )
    external = sc.extern(library, "external_allocating", inputs=(sc.arg("x", ()),), outputs=sc.arg("y", ()))
    handles.append(get_compiled(external._function))
  for handle in handles:
    np.testing.assert_allclose(handle.run([np.array(3.0)]), [8])


@pytest.mark.skipif(sys.platform != "linux", reason="glibc allocator binding")
@pytest.mark.parametrize("order", ["plain,allocator", "allocator,plain"])
def test_external_namespace_keeps_libc_allocation_and_free_together(tmp_path, order):
  script = (
    "import resource, sys; from pathlib import Path; resource.setrlimit(resource.RLIMIT_CORE, (0, 0)); "
    "from tests.codegen.test_external import _namespace_allocators; _namespace_allocators(Path(sys.argv[1]), sys.argv[2])"
  )
  result = subprocess.run(
    [sys.executable, "-c", script, str(tmp_path), order],
    env={**os.environ, "SCALY_CACHE_DIR": str(tmp_path / "cache")},
    capture_output=True,
    text=True,
    timeout=120,
  )
  assert result.returncode == 0, result.stderr


def _namespace_origins(root):
  compiler = find_c_compiler()
  assert compiler is not None
  launcher = root / "launcher.c"
  launcher.write_text("""#include <dlfcn.h>
double relative_value(void) {
  void* handle = dlopen("libplugin.so", RTLD_NOW | RTLD_LOCAL);
  if (!handle) return -1;
  double (*value)(void) = dlsym(handle, "value");
  double result = value ? value() : -2;
  dlclose(handle);
  return result;
}
""")
  handles = []
  for i in range(2):
    directory = root / f"version{i}"
    directory.mkdir()
    plugin = directory / "plugin.c"
    plugin.write_text(f"double value(void) {{ return {i + 7}; }}\n")
    subprocess.run([*compiler.command, "-shared", "-fPIC", str(plugin), "-o", str(directory / "libplugin.so")], capture_output=True, check=True)
    if i == 0:
      subprocess.run(
        [
          *compiler.command,
          "-shared",
          "-fPIC",
          str(launcher),
          "-Wl,-soname,liblauncher.so",
          "-Wl,-rpath,$ORIGIN",
          "-ldl",
          "-o",
          str(directory / "liblauncher.so"),
        ],
        capture_output=True,
        check=True,
      )
    else:
      shutil.copyfile(root / "version0" / "liblauncher.so", directory / "liblauncher.so")
    os.environ["SCALY_CC"] = shlex.join((*compiler.command, "-L" + str(directory)))
    library = sc.CLibrary(
      f"origin{i}",
      sources={
        "wrapper.c": "double relative_value(void);\n"
        "void external_double(const double* x, double* y, double* w) { y[0] = relative_value()*x[0]; }\n"
        "void external_triple(const double* x, double* y, double* w) { y[0] = (relative_value()+1)*x[0]; }\n"
      },
      link=("launcher",),
    )
    handles.append(get_compiled(library_host(library)))
  for i, handle in enumerate(handles):
    np.testing.assert_allclose(handle.run([np.array(3.0)]), [(2 * i + 15) * 3])


@pytest.mark.skipif(sys.platform != "linux", reason="glibc origin-relative plugin lookup")
def test_identical_dependencies_at_different_paths_keep_their_own_plugins(tmp_path):
  script = "from pathlib import Path; import sys; from tests.codegen.test_external import _namespace_origins; _namespace_origins(Path(sys.argv[1]))"
  result = subprocess.run(
    [sys.executable, "-c", script, str(tmp_path)],
    env={**os.environ, "SCALY_CACHE_DIR": str(tmp_path / "cache")},
    capture_output=True,
    text=True,
    timeout=120,
  )
  assert result.returncode == 0, result.stderr


@pytest.mark.skipif(sys.platform != "linux", reason="glibc dependency inspection")
def test_external_dependency_inspection_does_not_need_shell_tools(isolated_cache, monkeypatch):
  from scaly.codegen import jit

  library = sc.CLibrary("without_shell", sources={"shared.c": SOURCE}, defines={"LOCAL_ONLY": 2})
  compiled = get_compiled(library_host(library))
  monkeypatch.setenv("PATH", str(isolated_cache / "empty"))
  assert str(compiled.lib_path.resolve()) in jit._namespace_dependencies(compiled.lib_path)


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


def test_aot_cli_exports_without_a_compiler_or_jit_cache(tmp_path, monkeypatch, capsys):
  from types import SimpleNamespace
  from scaly.codegen import aot

  library = sc.CLibrary("portable", sources={"shared.c": SOURCE}, defines={"LOCAL_ONLY": 2}, link=("m",))
  kernel = library_host(library)
  monkeypatch.setattr(aot.importlib, "import_module", lambda name: SimpleNamespace(kernel=kernel))
  monkeypatch.setattr(aot, "find_c_compiler", lambda: None)
  cache = tmp_path / "cache"
  monkeypatch.setenv("SCALY_CACHE_DIR", str(cache))
  aot.main(["model:kernel", "--out-dir", str(tmp_path / "export"), "--lanes", "1"])
  output = capsys.readouterr().out
  assert "build manifest: " in output
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


def _external_process(root, kind, *, compiler=None):
  import os
  import shlex

  environment = {**os.environ, "SCALY_CACHE_DIR": str(root / "cache"), "SCALY_JIT_KEY": ""}
  if compiler is not None:
    environment["SCALY_CC"] = shlex.join((*compiler, "-L" + str(root / "lib")))
  result = subprocess.run(
    [sys.executable, "-c", _EXTERNAL_INDEX_PROCESS, str(root), kind], env=environment, capture_output=True, text=True, timeout=120
  )
  assert result.returncode == 0, result.stderr
  return json.loads(result.stdout)


@pytest.mark.parametrize("edited", ["value.h", "extra.h"])
def test_include_header_edit_misses_the_index_in_another_process(tmp_path, edited):
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
  changed = _external_process(tmp_path, "header")
  assert not changed["hit"]
  assert changed["lowered"] == 1
  assert changed["index"] != first["index"]
  assert changed["artifact"] != first["artifact"]
  assert changed["value"] == (21 if edited == "value.h" else 15)


@pytest.mark.skipif(sys.platform != "linux", reason="swapping a named ELF dependency")
def test_linked_library_swap_misses_the_index_in_another_process(tmp_path):
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
  changed = _external_process(tmp_path, "link", compiler=compiler.command)
  assert not changed["hit"]
  assert changed["lowered"] == 1
  assert changed["index"] != first["index"]
  assert changed["artifact"] != first["artifact"]
  assert changed["value"] == 21
