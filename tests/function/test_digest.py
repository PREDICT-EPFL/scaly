"""The digests the JIT index is keyed on: a Function's graph and the installed source that renders it.

Most tests here are one way devrush's reviews of its index found for one key to cover two renderings:
the digest changes, or there is none and the Function is rendered.
"""

from __future__ import annotations

import dataclasses
import enum
import functools
import importlib.util
import os
import py_compile
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly.function import digest
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.ir.expr import Expr, ExprOp
from scaly.ir.types import CanonicalPattern, DType, dtypes

# Windows keeps a file's creation time where other systems keep the time of its last change.
no_change_time = pytest.mark.skipif(sys.platform == "win32", reason="Windows has no change time in os.stat")


def _fn(scale: float = 2.0, name: str = "index_fn") -> ConcreteFunction:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name=name)
  def fn(x: sc.Expr) -> sc.Expr:
    return (x.sin() * scale + x * x).sum()

  return as_concrete(fn)


def _attr(value: Any, *, dtype: DType = dtypes.float64, symbol: str = "x") -> ConcreteFunction:
  """A Function with ``value`` as an attribute of one node, for the digest alone: it is never rendered."""
  x = Expr.sym(symbol, 3, dtype=dtype)
  return ConcreteFunction._from_exprs("attr_fn", [x], [Expr(ExprOp.NEG, (x,), x.type, attrs={"note": value})], ["x"], ["y"])


def test_every_field_of_a_function_is_read_or_named_unread() -> None:
  assert set(digest._FIELDS) | set(digest._NOT_READ) == {field.name for field in dataclasses.fields(ConcreteFunction)}


def test_everything_a_tree_holds_is_read_or_named_unread() -> None:
  held = set(vars(sc.group(sc.arg("x", 2), sc.arg("y"))))
  assert set(digest._TREE_READ) | set(digest._TREE_NOT_READ) == held | set(vars(sc.arg("z")))


@pytest.mark.parametrize(
  "one, other",
  [
    pytest.param(lambda: _fn(2.0), lambda: _fn(3.0), id="a constant"),
    pytest.param(lambda: _fn(0.0), lambda: _fn(-0.0), id="the sign of zero"),
    pytest.param(lambda: _fn(name="index_a"), lambda: _fn(name="index_b"), id="the name"),
    pytest.param(lambda: _attr(1), lambda: _attr(True), id="an int or a flag"),
    pytest.param(lambda: _attr(1), lambda: _attr(1.0), id="an int or a float"),
    pytest.param(lambda: _attr((1, 2)), lambda: _attr((2, 1)), id="a tuple's order"),
    pytest.param(lambda: _attr(np.arange(3)), lambda: _attr(np.arange(3.0)), id="an array's dtype"),
    pytest.param(lambda: _attr(slice(0, 2)), lambda: _attr(slice(0, 3)), id="a slice"),
  ],
)
def test_the_digest_changes_with(one, other) -> None:
  assert digest.graph_digest(one()) not in (None, digest.graph_digest(other()))


def test_a_callee_enters_by_its_digest() -> None:
  def caller(scale: float) -> ConcreteFunction:
    inner = _fn(scale, name="index_inner")

    @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="index_outer")
    def outer(x: sc.Expr) -> sc.Expr:
      return inner(x) + 1.0

    return as_concrete(outer)

  assert digest.graph_digest(caller(2.0)) not in (None, digest.graph_digest(caller(3.0)))


def _closure() -> Any:
  scale = 2.0
  return lambda x: x * scale


class _Bound:
  def rule(self, x: Any) -> Any:
    return x


def _plain(x: Any) -> Any:
  return x


@pytest.mark.parametrize(
  "rule",
  [
    pytest.param(_closure(), id="a closure"),
    pytest.param(_Bound().rule, id="a bound method"),
    pytest.param(functools.partial(max, 1), id="a partial"),
    pytest.param(_plain, id="a module-level function"),
  ],
)
def test_a_callable_gives_no_digest_so_neither_does_a_rule_that_carries_state(rule) -> None:
  assert digest.graph_digest(_attr(rule)) is None


def test_a_dtype_named_like_another_with_another_c_type_gives_another_digest() -> None:
  impostor = DType("float64", 64, "float", is_floating=True)
  assert digest.graph_digest(_attr(impostor)) not in (None, digest.graph_digest(_attr(dtypes.float64)))


@dataclasses.dataclass(frozen=True)
class _Note:
  text: str


def _local_values() -> tuple[Any, Any]:
  @dataclasses.dataclass(frozen=True)
  class _Note:
    text: str

  class _Shade(enum.Enum):
    DARK = 1

  return _Note("a"), _Shade.DARK


def test_a_class_defined_inside_a_function_gives_no_digest() -> None:
  assert digest.graph_digest(_attr(_Note("a"))) is not None
  for value in _local_values():
    assert digest.graph_digest(_attr(value)) is None


def test_nodes_are_numbered_in_walk_order_not_by_address() -> None:
  first = digest.graph_digest(_fn(name="index_order"))
  held = [Expr.sym(f"held{i}", 3) for i in range(100)]  # other objects take the freed addresses
  digest._DIGESTS.clear()
  assert held and digest.graph_digest(_fn(name="index_order")) == first
  # Two inputs swapped give another graph, though they are the same nodes.
  x, y = Expr.sym("x", 3), Expr.sym("y", 3)
  pair = [ConcreteFunction._from_exprs("pair", inputs, [x - y], ["a", "b"], ["out"]) for inputs in ([x, y], [y, x])]
  assert digest.graph_digest(pair[0]) != digest.graph_digest(pair[1])


@pytest.fixture
def package(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
  """A package standing for scaly's installed source, loaded once its files are written."""
  root = tmp_path / "pkg"
  root.mkdir()
  (root / "__init__.py").write_text("")
  (root / "module.py").write_text("SCALE = 2.0\n")
  (root / "sub").mkdir()
  (root / "sub" / "rules.py").write_text("RULES = ()\n")
  monkeypatch.setattr(digest, "_packages", lambda: (("pkg", (str(root),)),))
  _loaded(monkeypatch)
  return root


def _loaded(monkeypatch: pytest.MonkeyPatch) -> None:
  """Load the package now: what changes from here on runs as old code. Times are kept coarsely."""
  time.sleep(0.05)
  monkeypatch.setattr(digest, "LOADED_AT_NS", time.time_ns() + digest._CLOCK_SLACK_NS)
  time.sleep(0.05)


def test_the_source_digest_reads_the_files_contents(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  first = digest.source_digest()
  (package / "module.py").write_text("SCALE = 3.0\n")
  _loaded(monkeypatch)
  assert digest.source_digest() not in (None, first)


@no_change_time
def test_a_source_replaced_with_its_modification_time_kept_gives_no_digest(package: Path) -> None:
  assert digest.source_digest() is not None
  module = package / "module.py"
  kept = module.stat()
  module.write_text("SCALE = 3.0\n")
  os.utime(module, ns=(kept.st_atime_ns, kept.st_mtime_ns))
  assert digest.source_digest() is None


def test_a_source_edited_just_before_scaly_loaded_gives_no_digest(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  (package / "module.py").write_text("SCALE = 3.0\n")
  monkeypatch.setattr(digest, "LOADED_AT_NS", time.time_ns())  # within the clock's slack of the edit
  assert digest.source_digest() is None


@no_change_time
def test_stale_bytecode_python_would_load_gives_no_digest(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  module = package / "module.py"
  py_compile.compile(str(module), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
  _loaded(monkeypatch)
  assert digest.source_digest() is not None
  kept = module.stat()
  module.write_text("SCALE = 3.0\n")  # the same size, so with the time kept the bytecode still loads
  os.utime(module, ns=(kept.st_atime_ns, kept.st_mtime_ns))
  _loaded(monkeypatch)
  assert digest.source_digest() is None
  py_compile.compile(str(module), invalidation_mode=py_compile.PycInvalidationMode.TIMESTAMP)
  _loaded(monkeypatch)
  assert digest.source_digest() is not None


def _bytecode_written(directory: Path) -> None:
  """Python writes bytecode into ``directory``, as it does when a module there is first imported."""
  (directory / "__pycache__").mkdir(exist_ok=True)
  (directory / "__pycache__" / f"late.{time.time_ns()}.pyc").write_bytes(b"")


def test_a_module_imported_for_the_first_time_after_the_digest_keeps_it(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  first = digest.source_digest()
  monkeypatch.syspath_prepend(str(package.parent))
  monkeypatch.setattr(sys, "dont_write_bytecode", False)
  monkeypatch.delitem(sys.modules, "pkg", raising=False)
  importlib.import_module("pkg.sub.rules")  # creates both __pycache__ directories
  assert (package / "sub" / "__pycache__").is_dir()
  assert digest.source_digest() == first
  importlib.import_module("pkg.module")  # writes into one that exists
  assert digest.source_digest() == first
  for name in ("pkg.sub.rules", "pkg.sub", "pkg.module", "pkg"):
    sys.modules.pop(name, None)


def test_a_file_deleted_under_a_live_process_gives_no_digest(package: Path) -> None:
  _bytecode_written(package)
  time.sleep(0.05)
  (package / "module.py").unlink()
  assert digest.source_digest() is None


@pytest.mark.skipif(sys.platform == "win32", reason="links need privileges on Windows")
def test_a_link_pointed_elsewhere_under_a_live_process_gives_no_digest(package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  for version in ("one", "two"):
    (tmp_path / f"{version}.py").write_text(f"VERSION = {version!r}\n")
  link = package / "linked.py"
  link.symlink_to(tmp_path / "one.py")
  _loaded(monkeypatch)
  assert digest.source_digest() is not None
  swap = package / "swap.tmp"
  swap.symlink_to(tmp_path / "two.py")  # the link moves, not the files: both are as old as the load
  swap.replace(link)
  _bytecode_written(package)  # so the directory's own change is not what catches it
  assert digest.source_digest() is None


@no_change_time
def test_a_directory_swapped_under_a_live_process_gives_no_digest(package: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  other = tmp_path / "other"
  other.mkdir()
  (other / "rules.py").write_text("RULES = (1,)\n")
  _bytecode_written(other)
  _loaded(monkeypatch)
  assert digest.source_digest() is not None
  (package / "sub").rename(tmp_path / "old")
  other.rename(package / "sub")
  _bytecode_written(package)
  _bytecode_written(package / "sub")
  assert digest.source_digest() is None


class _Flag(enum.IntFlag):
  ONE = 1


def test_an_enum_value_without_a_name_has_a_digest() -> None:
  assert digest.graph_digest(_attr(_Flag(0))) not in (None, digest.graph_digest(_attr(_Flag.ONE)))


class _Scalar(np.float64):
  pass


def test_a_numpy_scalar_of_a_subclass_gives_no_digest() -> None:
  assert digest.graph_digest(_attr(np.float64(1.0))) is not None
  assert digest.graph_digest(_attr(_Scalar(1.0))) is None


def test_unchecked_bytecode_of_another_source_gives_no_digest(package: Path, monkeypatch: pytest.MonkeyPatch) -> None:
  module = package / "module.py"
  py_compile.compile(str(module), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
  _loaded(monkeypatch)
  assert digest.source_digest() is not None
  module.write_text("SCALE = 3.0\n")
  py_compile.compile(str(module), invalidation_mode=py_compile.PycInvalidationMode.CHECKED_HASH)
  _loaded(monkeypatch)
  assert digest.source_digest() is not None  # Python checks this one against the source
  stale = package / "stale.pyc"
  py_compile.compile(str(module), cfile=str(stale), invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH)
  module.write_text("SCALE = 4.0\n")
  stale.replace(importlib.util.cache_from_source(str(module)))  # restored after the source, so newer than it
  _loaded(monkeypatch)
  assert digest.source_digest() is None


def test_a_sparse_leafs_layout_is_part_of_the_digest() -> None:
  pattern = CanonicalPattern((3, 2), (0, 2, 1), (0, 0, 1))

  def identity(layout: str) -> ConcreteFunction:
    leaf = sc.arg("x", sc.TensorType(pattern.shape, pattern=pattern), layout=layout)
    return as_concrete(sc.function(leaf, outputs=leaf.relabel("out_"), name="digest_layout")(lambda x: x))

  csc, csr = identity("csc"), identity("csr")
  assert csc.inputs[0] is csr.inputs[0]  # one graph, two layouts at its boundary
  assert digest.graph_digest(csc) not in (None, digest.graph_digest(csr))


def test_a_dense_leafs_layout_is_part_of_the_digest() -> None:
  def identity(layout: str) -> ConcreteFunction:
    leaf = sc.arg("x", (2, 3), layout=layout)
    return as_concrete(sc.function(leaf, outputs=leaf.relabel("out_"), name="digest_dense_layout")(lambda x: x))

  rows, columns = identity("row-major"), identity("column-major")
  assert rows.inputs[0] is columns.inputs[0]
  assert digest.graph_digest(rows) not in (None, digest.graph_digest(columns))
