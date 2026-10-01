"""The JIT's structural key: a library built before is found from the Function's graph, with nothing
rendered, and the key changes with everything the rendering reads."""

from __future__ import annotations

import gc
import json
import os
import shutil
import sys
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
import scaly.codegen.aot as aot
import scaly.codegen.jit as jit
import scaly.codegen.structure as structure
from scaly.codegen.structure import code_digest, graph_digest
from scaly.function.extern import BuildRequirements, ExternRenderCtx, ExternSource, ExternState, extern_function
from scaly.ir.expr import Expr, ExprOp, Lowering
from scaly.ir.target import Target, resolve_target
from scaly.ir.types import TensorType, dtypes

pytestmark = pytest.mark.skipif(shutil.which(os.environ.get("SCALY_CC", "cc")) is None, reason="the JIT needs a C compiler")


@pytest.fixture
def cache(tmp_path, monkeypatch):
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))
  monkeypatch.delenv("SCALY_JIT_KEY", raising=False)
  jit._artifact_cache.clear()  # what a new process starts with
  yield tmp_path
  jit._artifact_cache.clear()


def _fn(scale: float = 2.0, *, name: str = "structure_fn", n: int = 3, x: str = "x", y: str = "y") -> sc.ConcreteFunction:
  v = sc.sym(x, n)
  return sc.Function.from_exprs(name, [v], [(v.sin() * scale + v[::-1]).sum()], [x], [y])


def _entries(root: Path) -> list[Path]:
  return sorted((root / "structure").glob("*/*.json"))


def _identity() -> tuple[str, str]:
  compiler = jit.find_c_compiler()
  assert compiler is not None
  return jit.compiler_identity(compiler.cc)


def _key(
  fun: sc.ConcreteFunction, *, target: Target | None = None, compiler: tuple[str, ...] | None = None, flags: tuple[str, ...] | None = None
) -> tuple[str, str] | None:
  return jit._structure_key(
    fun,
    resolve_target(None) if target is None else target,
    _identity() if compiler is None else compiler,
    jit.compile_flags() if flags is None else flags,
  )


def _graph(fun: sc.ConcreteFunction) -> tuple[str, frozenset[str]]:
  found = graph_digest(fun)
  assert found is not None
  return found


def _no_render(monkeypatch) -> None:
  def refuse(*args, **kwargs):
    raise AssertionError("the Function was rendered")

  monkeypatch.setattr(jit, "render_c_module", refuse)


def test_a_library_built_before_is_loaded_without_rendering(cache, monkeypatch) -> None:
  x = np.array([0.1, -0.7, 2.5])
  first = _fn()(x)
  assert len(_entries(cache)) == 1
  jit._artifact_cache.clear()
  gc.collect()
  _no_render(monkeypatch)
  again = _fn()
  np.testing.assert_array_equal(again(x), first)
  np.testing.assert_allclose(first, (np.sin(x) * 2.0 + x[::-1]).sum())
  assert again._compiled.cache_key == json.loads(_entries(cache)[0].read_text())["key"]


def test_a_function_built_twice_in_one_process_renders_once(cache, monkeypatch) -> None:
  _fn()(np.zeros(3))
  _no_render(monkeypatch)
  _fn()(np.zeros(3))


def _attr(start: int) -> sc.ConcreteFunction:
  a = sc.sym("a", 4)
  return sc.Function.from_exprs("structure_fn", [a], [a[start : start + 2]], ["a"], ["y"])


def _const(value: float) -> sc.ConcreteFunction:
  v = sc.sym("x", 3)
  return sc.Function.from_exprs("structure_fn", [v], [v + np.array([1.0, value, 3.0])], ["x"], ["y"])


def _calling(scale: float) -> sc.ConcreteFunction:
  inner = _fn(scale, name="structure_inner")
  v = sc.sym("x", 3)
  return sc.Function.from_exprs("structure_fn", [v], [inner(v) + 1.0], ["x"], ["y"])


def _mapped(scale: float) -> sc.ConcreteFunction:
  inner = _fn(scale, name="structure_inner")
  v = sc.sym("x", 12)
  return sc.Function.from_exprs("structure_fn", [v], [sc.vmap(inner, 4, [(v, 0, 3)])], ["x"], ["y"])


def _unary(op: str) -> sc.ConcreteFunction:
  v = sc.sym("x", 3)
  return sc.Function.from_exprs("structure_fn", [v], [getattr(v, op)()], ["x"], ["y"])


def _binary(swap: bool, *, declared: tuple[str, str] = ("u", "v"), symbols: tuple[str, str] = ("u", "v"), order: bool = False) -> sc.ConcreteFunction:
  u, v = sc.sym(symbols[0], 3), sc.sym(symbols[1], 3)
  return sc.Function.from_exprs("structure_fn", [v, u] if order else [u, v], [v - u if swap else u - v], list(declared), ["y"])


def _hinted(lowering: Lowering) -> sc.ConcreteFunction:
  v = Expr.sym("x", 3, lowering=lowering)
  return sc.Function.from_exprs("structure_fn", [v], [v * 2.0], ["x"], ["y"])


def _typed(dtype: str) -> sc.ConcreteFunction:
  v = sc.sym("x", 3, dtype=dtype)
  return sc.Function.from_exprs("structure_fn", [v], [v + v], ["x"], ["y"])


@pytest.mark.parametrize(
  ("one", "other"),
  [
    pytest.param(lambda: _fn(2.0), lambda: _fn(2.5), id="a scalar constant"),
    pytest.param(lambda: _const(2.0), lambda: _const(2.5), id="an entry of an array constant"),
    pytest.param(lambda: _const(0.0), lambda: _const(-0.0), id="the sign of a zero"),
    pytest.param(lambda: _fn(n=3), lambda: _fn(n=4), id="a shape"),
    pytest.param(lambda: _typed("float64"), lambda: _typed("float32"), id="a dtype"),
    pytest.param(lambda: _fn(name="structure_fn"), lambda: _fn(name="structure_fn2"), id="the Function's name"),
    pytest.param(lambda: _fn(x="x"), lambda: _fn(x="u"), id="an input's name"),
    pytest.param(lambda: _fn(y="y"), lambda: _fn(y="z"), id="an output's name"),
    pytest.param(lambda: _attr(0), lambda: _attr(1), id="an attribute"),
    pytest.param(lambda: _calling(2.0), lambda: _calling(2.5), id="the body of a Function it calls"),
    pytest.param(lambda: _mapped(2.0), lambda: _mapped(2.5), id="the body of a Function it maps"),
    pytest.param(lambda: _hinted("auto"), lambda: _hinted("scalar"), id="a lowering hint"),
    pytest.param(lambda: _unary("sin"), lambda: _unary("cos"), id="an op"),
    pytest.param(lambda: _binary(False), lambda: _binary(True), id="which argument is which"),
    pytest.param(lambda: _binary(False), lambda: _binary(False, order=True), id="the order of the inputs under the same names"),
    pytest.param(lambda: _binary(False), lambda: _binary(False, declared=("a", "v")), id="the name an input is declared under"),
    pytest.param(
      lambda: _binary(False, declared=("a", "b")), lambda: _binary(False, declared=("a", "b"), symbols=("p", "v")), id="a symbol's own name"
    ),
  ],
)
def test_the_graphs_digest_changes_with(one, other) -> None:
  a, b, again = graph_digest(one()), graph_digest(other()), graph_digest(one())
  assert a is not None and b is not None
  assert a[0] != b[0]
  assert a == again


def _sharing(*, swap_inputs: bool = False, swap_outputs: bool = False) -> sc.ConcreteFunction:
  """A Function calling one built on its own symbols, so that the callee's inputs and outputs are
  nodes the walk has numbered before it reaches the callee."""
  u, v = sc.sym("u", 3), sc.sym("v", 3)
  inner = sc.Function.from_exprs(
    "structure_inner", [v, u] if swap_inputs else [u, v], [v, u - v] if swap_outputs else [u - v, v], ["a", "b"], ["p", "q"]
  )
  p, q = inner._flat_symbolic_call([u, v])
  return sc.Function.from_exprs("structure_fn", [u, v], [p * 2.0 + q], ["u", "v"], ["y"])


def test_the_order_of_a_callees_inputs_and_outputs_is_part_of_the_digest_when_its_nodes_are_shared(cache) -> None:
  base, inputs, outputs = _sharing(), _sharing(swap_inputs=True), _sharing(swap_outputs=True)
  assert len({_graph(base)[0], _graph(inputs)[0], _graph(outputs)[0]}) == 3
  u, v = np.array([1.0, 2.0, 3.0]), np.array([0.5, 0.25, 0.125])
  np.testing.assert_allclose(base._flat_numerical_call(u, v)[0], (u - v) * 2.0 + v)
  np.testing.assert_allclose(inputs._flat_numerical_call(u, v)[0], (v - u) * 2.0 + u)
  np.testing.assert_allclose(outputs._flat_numerical_call(u, v)[0], v * 2.0 + (u - v))


def test_the_order_of_two_inputs_is_part_of_the_digest() -> None:
  u, v = sc.sym("u", 3), sc.sym("v", 3)
  a = sc.Function.from_exprs("structure_fn", [u, v], [u - v], ["u", "v"], ["y"])
  b = sc.Function.from_exprs("structure_fn", [v, u], [u - v], ["v", "u"], ["y"])
  assert _graph(a)[0] != _graph(b)[0]


def test_the_digest_does_not_depend_on_which_of_two_equal_types_a_node_holds() -> None:
  # Interning compares a node's type by value, so a node met again keeps the type object it was
  # first built with: whether it is its argument's own object depends on what ran before.
  x = Expr.sym("x", 3)

  def digest(shared: bool) -> str:
    y = Expr(ExprOp.NEG, (x,), x.type if shared else TensorType((3,)))
    assert (y.type is x.type) == shared
    total = Expr(ExprOp.CAST, (y,), TensorType((3,), dtype=dtypes.float32))
    found = graph_digest(sc.Function.from_exprs("structure_types", [x], [total], ["x"], ["y"]))
    assert found is not None
    return found[0]

  one = digest(True)
  gc.collect()  # the node is gone, so the next one is built anew, with its own type object
  assert one == digest(False)


@dataclass(frozen=True)
class _Half:
  """A value whose class leaves a field out of its equality."""

  seen: int = 0
  unseen: int = field(default=0, compare=False)


@dataclass(frozen=True)
class _Whole:
  rows: tuple[int, ...] = (0, 2)
  weight: float = 0.5


class _Opaque:
  pass


def _with_attr(value) -> sc.ConcreteFunction:
  x = sc.sym("x", 3)
  y = Expr(ExprOp.SIN, (x,), x.type, attrs={"note": value})
  return sc.Function.from_exprs("structure_noted", [x], [y], ["x"], ["y"])


_OPAQUE = _Opaque()


@pytest.mark.parametrize(
  "value",
  [
    pytest.param(_OPAQUE, id="an object of an unknown class"),
    pytest.param(len, id="a function"),
    pytest.param(_Half(), id="a dataclass with a field outside its equality"),
    pytest.param(_Whole, id="a class"),
    pytest.param((1, _OPAQUE), id="an unknown object inside a tuple"),
    pytest.param(frozenset({1.5}), id="a set of floats"),
    pytest.param(2 + 1j, id="a complex number"),
  ],
)
def test_a_graph_holding_a_value_the_digest_does_not_know_has_no_key_and_is_rendered(cache, value) -> None:
  fn = _with_attr(value)
  assert graph_digest(fn) is None
  assert _key(fn) is None
  x = np.array([0.1, 0.2, 0.3])
  np.testing.assert_allclose(fn(x), np.sin(x))
  assert not (cache / "structure").exists()


@pytest.mark.parametrize(
  ("one", "other"),
  [
    pytest.param(1, True, id="an integer and a flag"),
    pytest.param(1, 1.0, id="an integer and a float"),
    pytest.param(1, np.int64(1), id="an integer and a NumPy integer"),
    pytest.param(np.int64(0), np.float64(0.0), id="two NumPy scalars of the same bytes"),
    pytest.param(0.0, -0.0, id="the sign of a zero"),
    pytest.param(((1,), 2), ((1, 2),), id="where a nested tuple ends"),
    pytest.param(("aSb", "c"), ("a", "bSc"), id="where a string ends, when it holds a tag's byte"),
    pytest.param((1, 2), [1, 2], id="a tuple and a list"),
    pytest.param("12", ("1", "2"), id="a string and its pieces"),
    pytest.param(("ab", "c"), ("a", "bc"), id="two splits of one string"),
    pytest.param(np.arange(6).reshape(2, 3), np.arange(6).reshape(3, 2), id="an array's shape"),
    pytest.param(np.zeros(3, dtype=np.int64), np.zeros(3), id="an array's dtype"),
    pytest.param(slice(0, 3), slice(0, 3, 1), id="a slice's step"),
    pytest.param(slice(0, 3, 1), range(0, 3, 1), id="a slice and a range"),
    pytest.param(_Whole(), _Whole(weight=0.25), id="a dataclass's field"),
    pytest.param({"a": 1, "b": 2}, {"a": 2, "b": 1}, id="a dict's values"),
    pytest.param(None, 0, id="none and zero"),
    pytest.param(frozenset({1, 2}), frozenset({1, 3}), id="a set's members"),
    pytest.param(dtypes.float64, dtypes.float32, id="a dtype"),
    pytest.param(b"ab", "ab", id="bytes and a string"),
  ],
)
def test_two_values_of_an_attribute_give_two_digests(one, other) -> None:
  def digest(value) -> str:
    walk = structure._Walk()
    walk.value(value, None)
    return walk.hash.hexdigest()

  assert digest(one) != digest(other)
  assert digest(one) == digest(one)


def test_a_dicts_and_a_sets_order_are_not_part_of_the_digest() -> None:
  def digest(value) -> str:
    walk = structure._Walk()
    walk.value(value, None)
    return walk.hash.hexdigest()

  assert digest({"a": 1, "b": 2}) == digest({"b": 2, "a": 1})
  assert digest(frozenset(["b", "a", 3])) == digest(frozenset([3, "a", "b"]))


def test_the_key_changes_with_the_target_the_compiler_and_its_flags(cache) -> None:
  fn = _fn()
  base = _key(fn)
  assert base is not None and base == _key(_fn())
  target = resolve_target(None)
  compiler = _identity()
  others = [
    _key(fn, target=replace(target, l1d_bytes=target.l1d_bytes * 2)),
    _key(fn, target=replace(target, rounding="portable")),
    _key(fn, compiler=(compiler[0] + "-other", compiler[1])),
    _key(fn, compiler=(compiler[0], compiler[1] + " (patched)")),
    _key(fn, flags=(*jit.compile_flags(), "-ffast-math")),
    _key(fn, flags=jit.compile_flags()[1:]),
  ]
  assert None not in others
  assert {other[0] for other in others if other is not None} == {base[0]}  # the same code renders them all
  assert len({base[1], *(other[1] for other in others if other is not None)}) == 1 + len(others)


def test_the_key_changes_with_the_optimization_level(cache, monkeypatch) -> None:
  fn = _fn()
  monkeypatch.setenv("SCALY_CC_OPT", "-O1")
  low = jit._structure_key(fn, resolve_target(None), ("cc", "banner"), jit.compile_flags())
  monkeypatch.setenv("SCALY_CC_OPT", "-O3")
  assert low != jit._structure_key(fn, resolve_target(None), ("cc", "banner"), jit.compile_flags())


@dataclass(frozen=True)
class _Callee:
  """An extern body that scales a generated Function's output; ``runtime`` is outside its equality,
  as a solver descriptor's state is."""

  scale: str = "2.0"
  helper: str = "structure_helper_raw"
  helper_body: str = "out0[0] = in0[0];"
  factor: float = 2.0
  version: str = "1.0"
  include: str = "#include <string.h>"
  runtime: dict = field(default_factory=dict, compare=False)

  def dependencies(self) -> tuple[sc.ConcreteFunction, ...]:
    return (_fn(self.factor, name="structure_dependency"),)

  def extern_sources(self) -> tuple[ExternSource, ...]:
    return (ExternSource(self.helper, f"static void {self.helper}(const double* in0, double* out0, double* w) {{ (void)w; {self.helper_body} }}"),)

  def render(self, fun: sc.ConcreteFunction, ctx: ExternRenderCtx) -> list[str]:
    return [
      f"static void {ctx.raw_symbol}(const double* in0, double* out0, double* w) {{",
      "  double tmp[1];",
      "  structure_dependency_raw(in0, tmp, w);",
      f"  {self.helper}(tmp, out0, w);",
      f"  out0[0] *= {self.scale};",
      "}",
    ]

  def build_requirements(self, fun: sc.ConcreteFunction) -> BuildRequirements:
    return BuildRequirements(includes=(self.include,), versions=(("structure-test", self.version),))

  def state(self, fun: sc.ConcreteFunction) -> ExternState | None:
    return None


def _extern(**changes) -> sc.ConcreteFunction:
  return extern_function("structure_extern", _Callee(**changes), [("x", (3,))], [("y", ())])


@pytest.mark.parametrize(
  "change",
  [
    pytest.param(dict(scale="3.0"), id="the C it renders"),
    pytest.param(dict(helper="structure_other_raw"), id="the name of a source it adds"),
    pytest.param(dict(helper_body="out0[0] = 1.0 * in0[0];"), id="the text of a source it adds"),
    pytest.param(dict(factor=2.5), id="the body of a Function its C calls"),
    pytest.param(dict(version="1.1"), id="the version of what it links"),
    pytest.param(dict(include="#include <stdlib.h>"), id="an include"),
  ],
)
def test_an_extern_bodys_key_changes_with(cache, change) -> None:
  base, other = _key(_extern()), _key(_extern(**change))
  assert base is not None and other is not None
  assert base != other
  assert base == _key(_extern())


def test_an_extern_body_is_found_without_rendering_and_inside_a_generated_function(cache, monkeypatch) -> None:
  def host() -> sc.ConcreteFunction:
    inner = _extern()
    v = sc.sym("x", 3)
    return sc.Function.from_exprs("structure_host", [v], [inner(v) + 1.0], ["x"], ["y"])

  x = np.array([0.1, -0.7, 2.5])
  expected = 2.0 * (np.sin(x) * 2.0 + x[::-1]).sum()
  np.testing.assert_allclose(_extern()(x), expected)
  np.testing.assert_allclose(host()(x), expected + 1.0)
  assert _key(host()) != _key(_extern())
  jit._artifact_cache.clear()
  _no_render(monkeypatch)
  np.testing.assert_allclose(_extern()(x), expected)
  np.testing.assert_allclose(host()(x), expected + 1.0)


def test_an_extern_callee_in_a_graph_that_is_not_its_own_has_no_key() -> None:
  x = sc.sym("x", 3)
  call = Expr(
    ExprOp.EXTERN_CALL, (x,), TensorType((), diff=False), attrs={"extern": _Callee(), "name": "structure_extern", "output": 0, "output_name": "y"}
  )
  assert graph_digest(sc.Function.from_exprs("structure_stray", [x], [call], ["x"], ["y"])) is None


def test_an_extern_body_whose_libraries_cannot_be_found_has_no_key(cache) -> None:
  def missing(names):
    raise RuntimeError(f"no such library: {names}")

  @dataclass(frozen=True)
  class Linked(_Callee):
    def build_requirements(self, fun: sc.ConcreteFunction) -> BuildRequirements:
      return BuildRequirements(libraries=("structure_absent",), link_flags=missing)

  assert _key(extern_function("structure_extern", Linked(), [("x", (3,))], [("y", ())])) is None


def _package(root: Path, name: str, monkeypatch, *, age: int = 3600) -> Path:
  """A package on disk with two modules and a data file, imported, its files written ``age`` seconds ago."""
  package = root / name
  (package / "sub").mkdir(parents=True)
  (package / "__init__.py").write_text("VALUE = 1\n")
  (package / "sub" / "__init__.py").write_text("")
  (package / "sub" / "rules.py").write_text("def rule():\n  return 1\n")
  (package / "template.c").write_text("double f(void);\n")
  (package / "__pycache__").mkdir()
  (package / "__pycache__" / "stale.pyc").write_bytes(b"\0")
  _age(package, age)
  monkeypatch.syspath_prepend(str(root))
  __import__(name)
  monkeypatch.setitem(sys.modules, name, sys.modules[name])  # dropped again when the test ends
  return package


def _age(root: Path, seconds: int) -> None:
  then = structure.LOADED_AT_NS - seconds * 1_000_000_000
  for path in [root, *root.rglob("*")]:
    os.utime(path, ns=(then, then))


def test_the_codes_digest_changes_with_a_files_contents_and_name_not_its_time(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_a", monkeypatch)
  base = code_digest({"structure_pkg_a"})
  assert base is not None and base == code_digest({"structure_pkg_a"})
  assert base != code_digest(())  # scaly's own code alone
  seen = {base}
  rules = package / "sub" / "rules.py"
  rules.write_text("def rule():\n  return 2\n")  # the same size, and below the same time: what a rebuilt wheel can be
  _age(package, 3600)
  monkeypatch.setattr(structure, "_CONTENTS", {})  # a new process: nothing read yet
  seen.add(code_digest({"structure_pkg_a"}))
  (package / "template.c").write_text("double g(void);\n")  # a file that is not Python
  _age(package, 3600)
  monkeypatch.setattr(structure, "_CONTENTS", {})
  seen.add(code_digest({"structure_pkg_a"}))
  (package / "template.c").rename(package / "template2.c")
  _age(package, 3600)
  seen.add(code_digest({"structure_pkg_a"}))
  (package / "sub" / "more.py").write_text("")
  _age(package, 3600)
  seen.add(code_digest({"structure_pkg_a"}))
  assert None not in seen and len(seen) == 5
  before = code_digest({"structure_pkg_a"})
  (package / "__pycache__" / "other.pyc").write_bytes(b"\0\0")  # bytecode is not source
  assert code_digest({"structure_pkg_a"}) == before
  _age(package, 1800)  # the same contents at another time: another checkout of the same code
  assert code_digest({"structure_pkg_a"}) == before


def test_a_file_is_read_once_and_a_large_one_is_taken_by_its_size_and_time(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_e", monkeypatch)
  library = package / "lib" / "libbig.so"
  library.parent.mkdir()
  library.write_bytes(b"\1" * (structure._READ_WHOLE + 1))
  edge = package / "edge.bin"
  edge.write_bytes(b"\1" * structure._READ_WHOLE)
  _age(package, 3600)
  reads = []
  opened = open
  monkeypatch.setattr("builtins.open", lambda path, *args, **kwargs: reads.append(str(path)) or opened(path, *args, **kwargs))
  monkeypatch.setattr(structure, "_CONTENTS", {})
  base = code_digest({"structure_pkg_e"})
  first = [path for path in reads if "structure_pkg_e" in path]
  assert base is not None and str(edge) in first and str(library) not in first
  assert code_digest({"structure_pkg_e"}) == base
  assert [path for path in reads if "structure_pkg_e" in path] == first  # nothing read twice
  library.write_bytes(b"\2" * (structure._READ_WHOLE + 1))  # other contents, the same size
  _age(package, 3600)
  assert code_digest({"structure_pkg_e"}) == base  # not read: only its size and time
  _age(package, 1800)
  timed = code_digest({"structure_pkg_e"})
  assert timed not in (None, base)
  library.write_bytes(b"\2" * (structure._READ_WHOLE + 2))
  _age(package, 1800)
  assert code_digest({"structure_pkg_e"}) not in (None, base, timed)
  sized = code_digest({"structure_pkg_e"})
  edge.write_bytes(b"\3" * structure._READ_WHOLE)  # read again: what was remembered was the file at another time
  then = structure.LOADED_AT_NS - 900 * 1_000_000_000
  os.utime(edge, ns=(then, then))
  assert code_digest({"structure_pkg_e"}) not in (None, sized)


def test_code_written_after_scaly_was_loaded_has_no_digest(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_b", monkeypatch)
  assert code_digest({"structure_pkg_b"}) is not None
  (package / "sub" / "rules.py").write_text("def rule():\n  return 2\n")  # now: the module loaded runs the old code
  assert code_digest({"structure_pkg_b"}) is None
  now = structure.LOADED_AT_NS
  os.utime(package / "sub" / "rules.py", ns=(now, now))  # at the very instant counts as after
  assert code_digest({"structure_pkg_b"}) is None
  os.utime(package / "sub" / "rules.py", ns=(now - 1, now - 1))
  assert code_digest({"structure_pkg_b"}) is not None


def test_a_package_that_is_not_loaded_or_has_no_files_has_no_digest(tmp_path, monkeypatch) -> None:
  assert code_digest({"structure_pkg_never_imported"}) is None
  empty = tmp_path / "structure_pkg_c"
  empty.mkdir()
  monkeypatch.syspath_prepend(str(tmp_path))
  __import__("structure_pkg_c")  # a namespace package with nothing in it
  monkeypatch.setitem(sys.modules, "structure_pkg_c", sys.modules["structure_pkg_c"])
  assert code_digest({"structure_pkg_c"}) is None


def test_the_interpreters_and_numpys_code_is_named_by_version(monkeypatch) -> None:
  base = code_digest(())
  assert base is not None
  assert code_digest({"numpy", "math", "functools"}) == base  # no files of theirs are read
  monkeypatch.setattr(np, "__version__", np.__version__ + ".post1")
  assert code_digest(()) != base


def test_a_single_file_module_is_digested_by_its_file(tmp_path, monkeypatch) -> None:
  module = tmp_path / "structure_single.py"
  module.write_text("X = 1\n")
  _age(tmp_path, 3600)
  monkeypatch.syspath_prepend(str(tmp_path))
  __import__("structure_single")
  monkeypatch.setitem(sys.modules, "structure_single", sys.modules["structure_single"])
  base = code_digest({"structure_single"})
  assert base is not None and base != code_digest(())
  module.write_text("X = 2\n")
  _age(tmp_path, 3600)
  monkeypatch.setattr(structure, "_CONTENTS", {})
  assert code_digest({"structure_single"}) not in (None, base)


def test_the_code_behind_an_ops_rules_is_part_of_the_key(tmp_path, monkeypatch) -> None:
  from scaly.ir.expr import op_def

  package = _package(tmp_path, "structure_pkg_d", monkeypatch)
  rule = __import__("structure_pkg_d.sub.rules", fromlist=["rule"]).rule
  fn = _fn()
  assert _graph(fn)[1] == {"scaly", "numpy"}
  monkeypatch.setattr(op_def(ExprOp.SIN), "fold", rule)
  digest, packages = _graph(fn)
  assert packages == {"scaly", "numpy", "structure_pkg_d"}
  assert digest == _graph(_fn())[0]
  before = _key(fn)
  (package / "sub" / "rules.py").write_text("def rule():\n  return 2\n")
  _age(package, 3600)
  monkeypatch.setattr(structure, "_CONTENTS", {})
  after = _key(fn)
  assert before is not None and after is not None and before[0] != after[0] and before[1] == after[1]
  monkeypatch.setattr(op_def(ExprOp.SIN), "traits", {**op_def(ExprOp.SIN).traits, "structure_test": rule})
  monkeypatch.setattr(op_def(ExprOp.SIN), "fold", None)
  assert _graph(fn)[1] == {"scaly", "numpy", "structure_pkg_d"}


def test_code_edited_under_a_live_process_stops_the_key_and_the_function_is_rendered(cache, monkeypatch) -> None:
  monkeypatch.setattr(structure, "LOADED_AT_NS", 0)  # every file is newer than the load
  fn = _fn()
  assert _key(fn) is None
  np.testing.assert_allclose(fn(np.zeros(3)), 0.0)
  assert not (cache / "structure").exists()


@pytest.mark.parametrize("mode", ["source", "verify"])
def test_the_other_key_modes_render_every_time(cache, monkeypatch, mode: str) -> None:
  monkeypatch.setenv("SCALY_JIT_KEY", mode)
  renders = []
  render = jit.render_c_module
  monkeypatch.setattr(jit, "render_c_module", lambda fun, **kwargs: renders.append(fun.name) or render(fun, **kwargs))
  for _ in range(2):
    _fn()(np.zeros(3))
    jit._artifact_cache.clear()
  assert renders == ["structure_fn", "structure_fn"]
  assert len(_entries(cache)) == (0 if mode == "source" else 1)


def test_an_unknown_key_mode_is_refused(cache, monkeypatch) -> None:
  monkeypatch.setenv("SCALY_JIT_KEY", "graph")
  with pytest.raises(jit.JitError, match="SCALY_JIT_KEY must be one of structure, source, verify"):
    _fn()(np.zeros(3))


def test_verify_mode_catches_a_key_that_misses_what_the_rendering_reads(cache, monkeypatch) -> None:
  # A digest blind to the constant: two Functions that render other C share one key.
  monkeypatch.setattr(jit, "graph_digest", lambda fun: ("0" * 64, frozenset()))
  x = np.array([0.1, -0.7, 2.5])
  monkeypatch.setenv("SCALY_JIT_KEY", "verify")
  np.testing.assert_allclose(_fn(2.0)(x), (np.sin(x) * 2.0 + x[::-1]).sum())
  np.testing.assert_allclose(_fn(2.0)(x), (np.sin(x) * 2.0 + x[::-1]).sum())  # the same C: no complaint
  with pytest.raises(jit.JitError, match="structural key of 'structure_fn' names a library built from other C"):
    _fn(2.5)(x)
  monkeypatch.setenv("SCALY_JIT_KEY", "source")
  np.testing.assert_allclose(_fn(2.5)(x), (np.sin(x) * 2.5 + x[::-1]).sum())


def test_an_index_entry_cut_short_or_naming_a_missing_library_is_rebuilt(cache) -> None:
  x = np.array([0.1, -0.7, 2.5])
  expected = (np.sin(x) * 2.0 + x[::-1]).sum()
  np.testing.assert_allclose(_fn()(x), expected)
  (entry,) = _entries(cache)
  whole = entry.read_text()
  for broken in (whole[: len(whole) // 2], "{}", "[]", '{"key": 3}', ""):
    entry.write_text(broken)
    jit._artifact_cache.clear()
    np.testing.assert_allclose(_fn()(x), expected)
    assert json.loads(entry.read_text()) == json.loads(whole)
  shutil.rmtree(cache / json.loads(whole)["key"])
  jit._artifact_cache.clear()
  np.testing.assert_allclose(_fn()(x), expected)
  assert (cache / json.loads(whole)["key"] / json.loads(whole)["library"]).exists()


def test_a_library_removed_under_a_live_process_is_rebuilt(cache) -> None:
  x = np.array([0.1, -0.7, 2.5])
  first = _fn()
  first(x)
  shutil.rmtree(cache / first._compiled.cache_key)
  np.testing.assert_allclose(_fn()(x), (np.sin(x) * 2.0 + x[::-1]).sum())  # the entry in memory names a library that is gone
  assert Path(first._compiled.lib_path).exists()


def test_recompiling_drops_the_index_entry_and_builds_again(cache) -> None:
  fn = _fn()
  fn(np.zeros(3))
  (entry,) = _entries(cache)
  jit.invalidate_cache(fn)
  assert not entry.exists() and not (cache / fn._compiled.cache_key).exists()
  fn.recompile()
  fn(np.zeros(3))
  assert entry.exists()


def test_the_index_entry_holds_what_the_handle_needs(cache) -> None:
  def big() -> sc.ConcreteFunction:
    a = sc.sym("a", (40, 40))
    return sc.Function.from_exprs("structure_big", [a], [(a @ a) @ a], ["a"], ["y"])

  first = big()
  first(np.eye(40))
  (entry,) = _entries(cache)
  stored = json.loads(entry.read_text())
  assert stored == {
    "key": first._compiled.cache_key,
    "library": Path(first._compiled.lib_path).name,
    "flags": [],
    "workspace_size": first._compiled._sz_w,
    "isolated": False,
  }
  assert stored["workspace_size"] >= 1600
  jit._artifact_cache.clear()
  again = big()
  np.testing.assert_allclose(again(np.eye(40) * 2.0), np.eye(40) * 8.0)
  assert again._compiled._sz_w == stored["workspace_size"]


def test_a_watched_function_is_rendered_although_its_library_is_there(cache, monkeypatch) -> None:
  _fn()(np.zeros(3))
  jit._artifact_cache.clear()
  seen = []

  class Observer:
    def add_normalized_expr(self, name, fun) -> None: ...
    def add_program(self, name, root) -> None: ...
    def add_code(self, source) -> None:
      seen.append(source)

    def finish(self, *, error=None) -> None: ...

  watched = _fn()
  monkeypatch.setattr(aot, "_RENDER_OBSERVERS", [lambda fun: Observer() if fun is watched else None])
  monkeypatch.setattr(aot, "_RENDER_WATCHES", [lambda fun: fun is watched])
  assert aot.render_watched(watched) and not aot.render_watched(_fn())
  assert _key(watched) is None and _key(_fn()) is not None
  watched(np.zeros(3))
  assert len(seen) == 1 and "structure_fn" in seen[0]
  monkeypatch.setattr(aot, "_RENDER_WATCHES", [None])  # an observer that does not say: every function counts
  assert aot.render_watched(_fn())


def test_a_visualized_function_is_watched() -> None:
  from scaly.viz import unvisualize_function, visualize_function

  fn = _fn()
  assert not aot.render_watched(fn)
  visualize_function(fn)
  try:
    assert aot.render_watched(fn) and _key(fn) is None
  finally:
    unvisualize_function(fn)
  assert not aot.render_watched(fn)


def test_the_index_keeps_the_states_of_the_code_written_last(cache, monkeypatch) -> None:
  index = cache / "structure"
  for age in range(12):
    state = index / f"{age:032x}"
    state.mkdir(parents=True)
    (state / "entry.json").write_text("{}")
    then = structure.LOADED_AT_NS - (age + 1) * 1_000_000_000
    os.utime(state, ns=(then, then))
  _fn()(np.zeros(3))
  kept = sorted(path.name for path in index.iterdir())
  (current,) = [path.parent.name for path in _entries(cache) if path.name != "entry.json"]
  assert kept == sorted([current, *(f"{age:032x}" for age in range(jit._INDEX_STATES_KEPT - 1))])
  _fn(3.0)(np.zeros(3))  # a second entry in a state that is there already: nothing is pruned
  (index / f"{99:032x}").mkdir()
  _fn(4.0)(np.zeros(3))
  assert len(list(index.iterdir())) == jit._INDEX_STATES_KEPT + 1
