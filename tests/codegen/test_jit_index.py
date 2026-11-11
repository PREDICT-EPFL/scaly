"""The JIT index: a warm process finds a Function's library from a digest of its graph, without lowering."""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scaly as sc
import scaly.codegen.aot as aot
import scaly.codegen.jit as jit
from scaly.codegen.toolchain import find_c_compiler
from scaly.function import digest
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.passes import program

pytestmark = pytest.mark.skipif(find_c_compiler() is None, reason="cc is required for JIT tests")

WARM_START = Path(__file__).with_name("jit_warm_start.py")


@pytest.fixture(autouse=True)
def index_trusted(monkeypatch: pytest.MonkeyPatch) -> None:
  """These tests count renders, so they trust the index unless one asks for verify mode, which CI sets for the whole suite."""
  monkeypatch.delenv("SCALY_JIT_KEY", raising=False)


def _fn(scale: float = 2.0, name: str = "index_fn") -> ConcreteFunction:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name=name)
  def fn(x: sc.Expr) -> sc.Expr:
    return (x.sin() * scale + x * x).sum()

  return as_concrete(fn)


def _lowered(monkeypatch: pytest.MonkeyPatch) -> list[str]:
  """The names of the Functions lowered from now on."""
  lowered: list[str] = []
  lower = aot.lower_function
  monkeypatch.setattr(aot, "lower_function", lambda fun, **kwargs: (lowered.append(fun.name), lower(fun, **kwargs))[1])
  return lowered


def _fresh() -> None:
  """Forget what this process built, as a new process would."""
  jit._artifact_cache.clear()


def _index_key(fun: ConcreteFunction) -> str | None:
  compiler = find_c_compiler()
  assert compiler is not None
  return jit._index_key(fun, compiler)


def _warm_start(cache: Path, seed: int, **env: str) -> dict[str, Any]:
  done = subprocess.run(
    [sys.executable, str(WARM_START)],
    env={**os.environ, "SCALY_CACHE_DIR": str(cache), "PYTHONHASHSEED": str(seed), **env},
    capture_output=True,
    text=True,
    timeout=300,
  )
  assert done.returncode == 0, done.stderr
  return json.loads(done.stdout)


def test_a_process_with_another_hash_seed_loads_the_library_without_lowering(tmp_path: Path) -> None:
  first = _warm_start(tmp_path, 1)
  second = _warm_start(tmp_path, 2)
  verified = _warm_start(tmp_path, 3, SCALY_JIT_KEY="verify")
  assert (first["lowered"], second["lowered"], verified["lowered"]) == (1, 0, 1)
  assert first["result"] == second["result"] == verified["result"]
  assert first["library"] == second["library"] == verified["library"]


def test_a_library_in_the_index_is_loaded_without_lowering(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  built = jit._build_artifact(fun)
  _fresh()
  lowered = _lowered(monkeypatch)
  assert jit._build_artifact(fun) == built
  assert lowered == []


def test_a_missing_library_behind_an_index_entry_is_rebuilt(isolated_cache: Path) -> None:
  fun = _fn()
  built = jit._build_artifact(fun)
  built.lib_path.unlink()
  _fresh()
  assert jit._build_artifact(fun) == built
  assert built.lib_path.exists()
  np.testing.assert_allclose(jit.get_compiled(fun).run([np.ones(3)])[0], 3 * (2 * np.sin(1.0) + 1))


def test_an_entry_names_its_library_inside_the_cache_so_the_cache_can_move(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  built = jit._build_artifact(fun)
  library = built.lib_path.relative_to(jit.cache_root())  # the root as the JIT spells it: an extended path on Windows
  (entry,) = (isolated_cache / "index").iterdir()
  assert json.loads(entry.read_text())["library"] == library.as_posix()
  moved = isolated_cache.with_name(isolated_cache.name + "_moved")
  isolated_cache.rename(moved)
  monkeypatch.setenv("SCALY_CACHE_DIR", str(moved))
  _fresh()
  lowered = _lowered(monkeypatch)
  assert jit._build_artifact(fun).lib_path == jit.cache_root() / library
  assert lowered == []


def test_an_entry_that_cannot_be_read_is_a_miss_and_is_written_again(isolated_cache: Path) -> None:
  fun = _fn()
  built = jit._build_artifact(fun)
  (entry,) = (isolated_cache / "index").iterdir()
  written = entry.read_text()
  entry.write_text(written[: len(written) // 2])
  _fresh()
  assert jit._build_artifact(fun) == built
  assert entry.read_text() == written
  assert [path.name for path in entry.parent.iterdir()] == [entry.name]  # entries are written beside it and renamed over it


def test_invalidation_builds_apart_from_the_entry_other_processes_use(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  built = jit._build_artifact(fun)
  (entry,) = (isolated_cache / "index").iterdir()
  written = entry.read_text()
  jit.invalidate_cache(fun)
  lowered = _lowered(monkeypatch)
  rebuilt = jit._build_artifact(fun)
  assert lowered == [fun.name] and rebuilt.lib_path != built.lib_path
  assert entry.read_text() == written and list(entry.parent.iterdir()) == [entry]
  monkeypatch.setattr(jit, "_cache_epochs", {})  # another process
  _fresh()
  assert jit._build_artifact(fun) == built
  assert lowered == [fun.name]


def test_verify_mode_renders_on_a_hit_and_raises_when_the_entry_differs(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  jit._build_artifact(fun)
  monkeypatch.setenv("SCALY_JIT_KEY", "verify")
  _fresh()
  lowered = _lowered(monkeypatch)
  jit._build_artifact(fun)
  assert lowered == [fun.name]
  (entry,) = (isolated_cache / "index").iterdir()
  # A key that misses something rendering reads names a library built from other C.
  entry.write_text(json.dumps({**json.loads(entry.read_text()), "workspace_size": 7}))
  with pytest.raises(jit.JitError, match="built from other C"):
    jit._build_artifact(fun)
  monkeypatch.setenv("SCALY_JIT_KEY", "source")
  with pytest.raises(jit.JitError, match="SCALY_JIT_KEY"):
    jit._build_artifact(fun)


def test_a_graph_without_a_digest_is_rendered(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  jit._build_artifact(fun)
  monkeypatch.setattr(jit, "graph_digest", lambda fun: None)
  _fresh()
  lowered = _lowered(monkeypatch)
  jit._build_artifact(fun)
  assert lowered == [fun.name]


class _Watching:
  def add_normalized_expr(self, name: str, fun: Any) -> None: ...
  def add_program(self, name: str, root: Any) -> None: ...
  def add_code(self, source: str) -> None: ...
  def finish(self, *, error: str | None = None) -> None: ...


def test_a_function_a_render_observer_watches_is_rendered_and_others_are_not(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  watched, other = _fn(name="index_watched"), _fn(name="index_other")
  jit._build_artifact(watched)
  jit._build_artifact(other)
  observers = [*aot._RENDER_OBSERVERS, lambda fun: _Watching() if fun is watched else None]
  monkeypatch.setattr(aot, "_RENDER_OBSERVERS", observers)
  monkeypatch.setattr(jit, "_RENDER_OBSERVERS", observers)
  _fresh()
  lowered = _lowered(monkeypatch)
  jit._build_artifact(watched)
  jit._build_artifact(other)
  assert lowered == [watched.name]


def test_the_key_changes_with_the_compile_options(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  default = _index_key(fun)
  monkeypatch.setenv("SCALY_CC_OPT", "-O1")
  assert _index_key(fun) not in (None, default)


def _rename(prog: Any) -> Any:
  return prog


def test_a_pass_put_in_the_pipeline_gives_no_key(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  assert _index_key(fun) is not None
  monkeypatch.setattr(program, "PASS_PIPELINE", (*program.PASS_PIPELINE, ("rename", _rename)))
  assert _index_key(fun) is None


def test_a_wrapper_carrying_a_scaly_passs_name_gives_no_key(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  (name, first), *rest = program.PASS_PIPELINE

  @functools.wraps(first)
  def wrapper(prog: Any) -> Any:
    return first(prog)

  monkeypatch.setattr(program, "PASS_PIPELINE", ((name, wrapper), *rest))
  assert _index_key(_fn()) is None


def test_an_option_read_while_rendering_is_part_of_the_key(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  monkeypatch.setenv("SCALY_VECTOR_LIBM", "none")
  plain = _index_key(fun)
  monkeypatch.setenv("SCALY_VECTOR_LIBM", "glibc")
  assert _index_key(fun) not in (None, plain)


_AXES_CHILD = """
import enum, json, os
import numpy as np
import scaly as sc
import scaly.codegen.jit as jit
from scaly.function.model import as_concrete

class Axis(enum.IntEnum):
  A = int(os.environ["FIRST_AXIS"])
  B = 1 - A

@sc.function(sc.arg("x", (2, 2)), outputs=sc.arg("y"), name="enum_axes")
def fn(x):
  return x.transpose((Axis.A, Axis.B))

print(json.dumps(jit.get_compiled(as_concrete(fn)).run([np.arange(4.0).reshape(2, 2)])[0].tolist()))
"""


def test_an_enum_enters_by_its_value_not_its_name(tmp_path: Path) -> None:
  results = []
  for first in ("0", "1"):
    env = {**os.environ, "SCALY_CACHE_DIR": str(tmp_path), "FIRST_AXIS": first}
    done = subprocess.run([sys.executable, "-c", _AXES_CHILD], env=env, capture_output=True, text=True, timeout=300)
    assert done.returncode == 0, done.stderr
    results.append(json.loads(done.stdout))
  assert results == [[[0.0, 1.0], [2.0, 3.0]], [[0.0, 2.0], [1.0, 3.0]]]


@pytest.mark.parametrize("across", [False, True], ids=["in one caller", "across two callers"])
def test_one_callee_called_twice_is_not_two_equal_callees(isolated_cache: Path, monkeypatch: pytest.MonkeyPatch, across: bool) -> None:
  one = _fn(name="index_shared")
  two = dataclasses.replace(one)

  def caller(second: ConcreteFunction) -> ConcreteFunction:
    @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="index_middle")
    def middle(x: sc.Expr) -> sc.Expr:
      return second(x + 1) * 2.0

    @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="index_sharing")
    def outer(x: sc.Expr) -> sc.Expr:
      return one(x) + (as_concrete(middle)(x) if across else second(x + 1))

    return as_concrete(outer)

  shared, distinct = caller(one), caller(two)
  assert digest.graph_digest(shared) not in (None, digest.graph_digest(distinct))
  jit._build_artifact(shared)
  monkeypatch.setenv("SCALY_JIT_KEY", "verify")
  jit._build_artifact(distinct)


def test_a_library_built_under_another_cache_root_is_not_indexed_in_this_one(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  fun = _fn()
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path / "first"))
  _fresh()
  built = jit._build_artifact(fun)
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path / "second"))
  assert jit._build_artifact(fun) == built
  assert not (tmp_path / "second" / "index").exists()


@pytest.mark.solver("piqp")
def test_solver_warm_start_never_lowers_at_construction_or_on_an_index_hit(tmp_path: Path) -> None:
  first = _warm_start(tmp_path, 1, TEST_SOLVER="1")
  second = _warm_start(tmp_path, 2, TEST_SOLVER="1")
  verified = _warm_start(tmp_path, 3, TEST_SOLVER="1", SCALY_JIT_KEY="verify")
  assert (first["construction"], second["construction"], verified["construction"]) == (0, 0, 0)
  assert (first["lowered"], second["lowered"], verified["lowered"]) == (2, 0, 2)
  np.testing.assert_allclose(first["result"], [0.5, -1.0, 2.0], atol=1e-6)
  assert first["result"] == second["result"] == verified["result"]
  assert first["library"] == second["library"] == verified["library"]

  moved = tmp_path.with_name(tmp_path.name + "_moved")
  tmp_path.rename(moved)
  relocated = _warm_start(moved, 4, TEST_SOLVER="1")
  assert (relocated["construction"], relocated["lowered"]) == (0, 0)
  assert relocated["result"] == first["result"]
  assert Path(relocated["library"]).parent.name == Path(first["library"]).parent.name
  assert Path(relocated["library"]).is_file()
