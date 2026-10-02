"""The JIT's structural key: a library built before is found from the Function's graph, with nothing
rendered, and the key changes with everything the rendering reads."""

from __future__ import annotations

import enum
import functools
import gc
import importlib.util
import json
import math
import os
import shutil
import struct
import sys
import time
import types
import weakref
from dataclasses import dataclass, field, replace
from dataclasses import fields as dataclass_fields
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
import scaly.codegen.aot as aot
import scaly.codegen.jit as jit
import scaly.codegen.structure as structure
import scaly.passes.lowering as lowering
import scaly.passes.program as program_passes
import scaly.utils.env as env
import scaly.utils.options as options_module
from scaly.codegen.structure import code_digest, graph_digest
from scaly.function.extern import BuildRequirements, ExternRenderCtx, ExternSource, ExternState, extern_function
from scaly.ir.expr import Expr, ExprOp, Lowering, op_def
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.target import Target, resolve_target
from scaly.ir.types import DType, SparsityType, TensorType, dtypes
from scaly.utils.options import default_options, get_options

pytestmark = pytest.mark.skipif(shutil.which(os.environ.get("SCALY_CC", "cc")) is None, reason="the JIT needs a C compiler")


@pytest.fixture(autouse=True)
def loaded_now(monkeypatch):
  """As if scaly were loaded now: a file written in the two seconds before the load stops the key,
  and a test must not depend on how long ago a file it reads was saved."""
  monkeypatch.setattr(env, "LOADED_AT_NS", time.time_ns() + 3_000_000_000)
  # A package other than scaly is judged by when the process started: here, when scaly was loaded.
  monkeypatch.setattr(env, "process_started_ns", lambda: env.LOADED_AT_NS)


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
  return found.digest, found.packages


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


def _two_ops(first: str, second: str) -> sc.ConcreteFunction:
  u, v = sc.sym("u", 3), sc.sym("v", 3)
  return sc.Function.from_exprs("structure_fn", [u, v], [getattr(u, first)() - getattr(v, second)()], ["u", "v"], ["y"])


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
    pytest.param(lambda: _two_ops("sin", "cos"), lambda: _two_ops("cos", "sin"), id="which node has which op"),
    pytest.param(lambda: _with_attr(1, "note"), lambda: _with_attr(1, "mark"), id="an attribute's name"),
    pytest.param(lambda: _binary(False), lambda: _binary(True), id="which argument is which"),
    pytest.param(lambda: _binary(False), lambda: _binary(False, order=True), id="the order of the inputs under the same names"),
    pytest.param(lambda: _binary(False), lambda: _binary(False, declared=("a", "v")), id="the name an input is declared under"),
    pytest.param(
      lambda: _binary(False, declared=("a", "b")), lambda: _binary(False, declared=("a", "b"), symbols=("p", "v")), id="a symbol's own name"
    ),
  ],
)
def test_the_graphs_digest_changes_with(one, other) -> None:
  a, b, again = _graph(one()), _graph(other()), _graph(one())
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
    return _graph(sc.Function.from_exprs("structure_types", [x], [total], ["x"], ["y"]))[0]

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


@dataclass(frozen=True)
class _Whole2:
  rows: tuple[int, ...] = (0, 2)
  weight: float = 0.5


class _Colour(enum.Enum):
  RED = 1
  BLUE = 2


class _Shade(enum.Enum):
  RED = 1


class _Opaque:
  pass


class _Tagged(np.ndarray):
  pass


def _local_class():
  @dataclass(frozen=True)
  class Local:
    n: int = 1

  return Local


def _with_attr(value, name: str = "note") -> sc.ConcreteFunction:
  x = sc.sym("x", 3)
  y = Expr(ExprOp.SIN, (x,), x.type, attrs={name: value})
  return sc.Function.from_exprs("structure_noted", [x], [y], ["x"], ["y"])


def _digest(value) -> str | None:
  """The digest of one attribute value, or None if the walk refuses it."""
  walk = structure._Walk()
  try:
    walk.value(value, None)
  except structure.Unkeyed:
    return None
  return walk.hash.hexdigest()


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
    pytest.param(np.ma.masked_array([1.0, 2.0], mask=[False, True]), id="a masked array"),
    pytest.param(np.zeros(2, dtype=[("a", "<f8"), ("b", "<i4")]), id="a structured array"),
    pytest.param(np.array(["a", "b"]), id="an array of strings"),
    pytest.param(np.zeros(2).view(_Tagged), id="an array of a subclass, which may hold more than its bytes"),
    pytest.param(np.datetime64("2026-01-01"), id="a NumPy date"),
    pytest.param({1: "a"}, id="a dict keyed by an integer"),
    pytest.param(_local_class()(), id="an object of a class defined in a function"),
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
    pytest.param({"a": 1}, {"b": 1}, id="a dict's keys"),
    pytest.param(None, 0, id="none and zero"),
    pytest.param(frozenset({1, 2}), frozenset({1, 3}), id="a set's members"),
    pytest.param(dtypes.float64, dtypes.float32, id="a dtype"),
    pytest.param(b"ab", "ab", id="bytes and a string"),
    pytest.param(np.float64(1.5), 1.5, id="a NumPy float and a float"),
    pytest.param(10**5000, 10**5000 + 1, id="integers of five thousand digits"),
    pytest.param("\udc80", "\udc81", id="strings that are not valid text"),
    pytest.param(_Whole(), _Whole2(), id="two classes with the same fields"),
    pytest.param(_Colour.RED, _Shade.RED, id="two enums with the same member"),
    pytest.param(_Colour.RED, _Colour.BLUE, id="two members of an enum"),
    pytest.param(np.arange(6.0).reshape(2, 3), np.arange(6.0).reshape(2, 3).T.copy().reshape(2, 3), id="an array's entries in another order"),
  ],
)
def test_two_values_of_an_attribute_give_two_digests(one, other) -> None:
  def digest(value) -> str:
    walk = structure._Walk()
    walk.value(value, None)
    return walk.hash.hexdigest()

  assert digest(one) != digest(other)
  assert digest(one) == digest(one)


def test_a_dicts_order_is_part_of_the_digest_and_a_sets_is_not() -> None:
  def digest(value) -> str:
    walk = structure._Walk()
    walk.value(value, None)
    return walk.hash.hexdigest()

  assert digest({"a": 1, "b": 2}) != digest({"b": 2, "a": 1})  # code that reads it may follow its order
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
  dependency: str = "structure_dependency"
  block: str = "#define STRUCTURE_ONE 1.0"
  helper_workspace: int = 0
  libraries: tuple[str, ...] = ()
  isolated: bool = False
  runtime: dict = field(default_factory=dict, compare=False)

  def dependencies(self) -> tuple[sc.ConcreteFunction, ...]:
    return (_fn(self.factor, name=self.dependency),)  # built anew at every call, as a callee may

  def extern_sources(self) -> tuple[ExternSource, ...]:
    source = f"static void {self.helper}(const double* in0, double* out0, double* w) {{ (void)w; {self.helper_body} }}"
    return (ExternSource(self.helper, source, self.helper_workspace),)

  def render(self, fun: sc.ConcreteFunction, ctx: ExternRenderCtx) -> list[str]:
    return [
      f"static void {ctx.raw_symbol}(const double* in0, double* out0, double* w) {{",
      "  double tmp[1];",
      f"  {self.dependency}_raw(in0, tmp, w);",
      f"  {self.helper}(tmp, out0, w);",
      f"  out0[0] *= {self.scale} * STRUCTURE_ONE;",
      "}",
    ]

  def build_requirements(self, fun: sc.ConcreteFunction) -> BuildRequirements:
    return BuildRequirements(
      includes=(self.include,),
      source_blocks=((self.block,),),
      libraries=self.libraries,
      link_flags=(lambda names: [f"-l{name}" for name in names]) if self.libraries else None,
      isolated=self.isolated,
      versions=(("structure-test", self.version),),
    )

  def state(self, fun: sc.ConcreteFunction) -> ExternState | None:
    return None


def _extern(name: str = "structure_extern", **changes) -> sc.ConcreteFunction:
  return extern_function(name, _Callee(**changes), [("x", (3,))], [("y", ())])


@pytest.mark.parametrize(
  "change",
  [
    pytest.param(dict(scale="3.0"), id="the C it renders"),
    pytest.param(dict(helper="structure_other_raw"), id="the name of a source it adds"),
    pytest.param(dict(helper_body="out0[0] = 1.0 * in0[0];"), id="the text of a source it adds"),
    pytest.param(dict(factor=2.5), id="the body of a Function its C calls"),
    pytest.param(dict(version="1.1"), id="the version of what it links"),
    pytest.param(dict(include="#include <stdlib.h>"), id="an include"),
    pytest.param(dict(block="#define STRUCTURE_ONE (1.0)"), id="a block of its source"),
    pytest.param(dict(helper_workspace=8), id="the workspace a source of its needs"),
    pytest.param(dict(libraries=("m",)), id="a library it links, through the flags it resolves to"),
    pytest.param(dict(isolated=True), id="whether it loads in a namespace of its own"),
  ],
)
def test_an_extern_bodys_key_changes_with(cache, change) -> None:
  base, other = _key(_extern()), _key(_extern(**change))
  assert base is not None and other is not None
  assert base != other
  assert base == _key(_extern())


def test_extern_functions_with_equal_bodies_have_one_key(cache) -> None:
  """Equal bodies intern to one ``EXTERN_CALL`` node, which holds the body built first. Each
  Function is keyed by its own body, and neither is refused for holding the other's."""
  first, second = _extern(version="equal bodies"), _extern(version="equal bodies")  # a body no other test builds
  assert first.extern is not second.extern and first.outputs[0] is second.outputs[0]
  assert first.outputs[0].attrs["extern"] is first.extern
  assert _key(first) is not None and _key(first) == _key(second)


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


def _missing(names):
  raise RuntimeError(f"no such library: {names}")


@dataclass(frozen=True)
class _Unlinked(_Callee):
  def build_requirements(self, fun: sc.ConcreteFunction) -> BuildRequirements:
    return BuildRequirements(libraries=("structure_absent",), link_flags=_missing)


def test_an_extern_body_whose_libraries_cannot_be_found_has_no_key(cache) -> None:
  fn = extern_function("structure_extern", _Unlinked(), [("x", (3,))], [("y", ())])
  assert graph_digest(fn) is not None  # the graph is digested: it is the libraries that stop the key
  assert _key(fn) is None


SECOND = 1_000_000_000


def _loaded_now(monkeypatch, *, ahead: int = 3 * SECOND) -> None:
  """Let scaly have been loaded after every file written so far: a file counts as older than the
  load only two seconds before it, and a test cannot wait or set a file's change time."""
  monkeypatch.setattr(env, "LOADED_AT_NS", time.time_ns() + ahead)


def _package(root: Path, name: str, monkeypatch) -> Path:
  """A package on disk with two modules and a data file, imported, and scaly loaded after it."""
  package = root / name
  (package / "sub").mkdir(parents=True)
  (package / "__init__.py").write_text("VALUE = 1\n")
  (package / "sub" / "__init__.py").write_text("")
  (package / "sub" / "rules.py").write_text("def rule():\n  return 1\n")
  (package / "template.c").write_text("double f(void);\n")
  (package / "__pycache__").mkdir()
  (package / "__pycache__" / "stale.pyc").write_bytes(b"\0")
  monkeypatch.syspath_prepend(str(root))
  __import__(name)
  monkeypatch.setitem(sys.modules, name, sys.modules[name])  # dropped again when the test ends
  _loaded_now(monkeypatch)
  return package


def test_the_codes_digest_changes_with_a_files_contents_and_name_not_its_time(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_a", monkeypatch)
  base = code_digest({"structure_pkg_a"})
  assert base is not None and base == code_digest({"structure_pkg_a"})
  assert base != code_digest(())  # scaly's own code alone
  seen = {base}
  rules = package / "sub" / "rules.py"
  stamp = rules.stat()
  rules.write_text("def rule():\n  return 2\n")  # the same size and, below, the same time: what a rebuilt wheel can be
  os.utime(rules, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
  _loaded_now(monkeypatch)
  seen.add(code_digest({"structure_pkg_a"}))
  (package / "template.c").write_text("double g(void);\n")  # a file that is not Python
  _loaded_now(monkeypatch)
  seen.add(code_digest({"structure_pkg_a"}))
  (package / "template.c").rename(package / "template2.c")
  _loaded_now(monkeypatch)
  seen.add(code_digest({"structure_pkg_a"}))
  (package / "sub" / "more.py").write_text("")
  _loaded_now(monkeypatch)
  seen.add(code_digest({"structure_pkg_a"}))
  assert None not in seen and len(seen) == 5
  before = code_digest({"structure_pkg_a"})
  (package / "__pycache__" / "other.pyc").write_bytes(b"\0\0")  # bytecode is not source
  for path in package.rglob("*.py"):  # the same contents at another time: another checkout of the same code
    os.utime(path, ns=(stamp.st_atime_ns - 1800 * SECOND, stamp.st_mtime_ns - 1800 * SECOND))
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_a"}) == before


def test_hidden_files_and_links_to_nothing_are_not_code_and_a_linked_directory_is(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_f", monkeypatch)
  base = code_digest({"structure_pkg_f"})
  assert base is not None
  (package / ".DS_Store").write_bytes(b"\1")
  (package / "sub" / ".rules.py.swp").write_bytes(b"\2")
  (package / ".ipynb_checkpoints").mkdir()
  (package / ".ipynb_checkpoints" / "rules-checkpoint.py").write_text("def rule():\n  return 3\n")
  (package / "sub" / ".#rules.py").symlink_to("someone@host.1234")  # an editor's lock: a link to nothing
  (package / "dangling.py").symlink_to(tmp_path / "nowhere.py")
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_f"}) == base
  elsewhere = tmp_path / "elsewhere"
  elsewhere.mkdir()
  (elsewhere / "helper.py").write_text("X = 1\n")
  (package / "linked").symlink_to(elsewhere, target_is_directory=True)
  (package / "sub" / "back").symlink_to(package, target_is_directory=True)  # a loop: walked once
  _loaded_now(monkeypatch)
  linked = code_digest({"structure_pkg_f"})
  assert linked not in (None, base)
  (elsewhere / "helper.py").write_text("X = 2\n")
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_f"}) not in (None, base, linked)


def test_a_file_is_read_once_and_a_large_one_is_taken_by_its_size_and_time(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_e", monkeypatch)
  library = package / "lib" / "libbig.so"
  library.parent.mkdir()
  library.write_bytes(b"\1" * (structure._READ_WHOLE + 1))
  edge = package / "edge.bin"
  edge.write_bytes(b"\1" * structure._READ_WHOLE)
  _loaded_now(monkeypatch)
  reads = []
  opened = open
  monkeypatch.setattr("builtins.open", lambda path, *args, **kwargs: reads.append(str(path)) or opened(path, *args, **kwargs))
  monkeypatch.setattr(structure, "_CONTENTS", {})
  base = code_digest({"structure_pkg_e"})
  first = [path for path in reads if "structure_pkg_e" in path]
  assert base is not None and str(edge) in first and str(library) not in first
  assert code_digest({"structure_pkg_e"}) == base
  assert [path for path in reads if "structure_pkg_e" in path] == first  # nothing read twice
  stamp = library.stat()
  os.utime(library, ns=(stamp.st_atime_ns, stamp.st_mtime_ns - SECOND))  # the same bytes at another time
  _loaded_now(monkeypatch)
  timed = code_digest({"structure_pkg_e"})
  assert timed not in (None, base)
  library.write_bytes(b"\2" * (structure._READ_WHOLE + 1))  # other contents, the size and the modification time kept
  os.utime(library, ns=(stamp.st_atime_ns, stamp.st_mtime_ns - SECOND))
  _loaded_now(monkeypatch)
  replaced = code_digest({"structure_pkg_e"})
  assert replaced not in (None, base, timed)  # told by its change time: it is not read
  assert str(library) not in reads
  library.write_bytes(b"\2" * (structure._READ_WHOLE + 2))
  os.utime(library, ns=(stamp.st_atime_ns, stamp.st_mtime_ns - SECOND))
  _loaded_now(monkeypatch)
  sized = code_digest({"structure_pkg_e"})
  assert sized not in (None, base, timed, replaced)
  stamp = edge.stat()
  edge.write_bytes(b"\3" * structure._READ_WHOLE)  # read again, although its size and modification time are the same
  os.utime(edge, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_e"}) not in (None, sized)


def test_code_written_or_replaced_after_scaly_was_loaded_has_no_digest(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_b", monkeypatch)
  rules = package / "sub" / "rules.py"
  assert code_digest({"structure_pkg_b"}) is not None
  slack = structure._CLOCK_SLACK_NS
  scanned = [package, *(path for path in package.rglob("*") if "__pycache__" not in path.parts)]
  changed = max(max(path.stat().st_ctime_ns, path.stat().st_mtime_ns) for path in scanned)
  monkeypatch.setattr(env, "LOADED_AT_NS", changed + slack)  # loaded two seconds after the last file was written: too close to tell
  assert code_digest({"structure_pkg_b"}) is None
  monkeypatch.setattr(env, "LOADED_AT_NS", changed + slack + 1)
  assert code_digest({"structure_pkg_b"}) is not None
  # A file written after the load: the module that was imported runs the old code.
  monkeypatch.setattr(env, "LOADED_AT_NS", time.time_ns())
  rules.write_text("def rule():\n  return 2\n")
  assert code_digest({"structure_pkg_b"}) is None
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_b"}) is not None
  # A file replaced with its modification time kept, as a copy from an archive is: its change time
  # moves. Every file's modification time is an hour old here, so only the change time can tell.
  old = time.time_ns() - 3600 * SECOND
  for path in package.rglob("*"):
    os.utime(path, ns=(old, old))
  monkeypatch.setattr(env, "LOADED_AT_NS", time.time_ns() + slack)  # loaded now: everything so far is older
  assert code_digest({"structure_pkg_b"}) is not None
  rules.write_text("def rule():\n  return 3\n")
  os.utime(rules, ns=(old, old))
  assert max(path.stat().st_mtime_ns for path in package.rglob("*.py")) == old
  assert code_digest({"structure_pkg_b"}) is None
  # A modification time set ahead of the load, the change time behind it.
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_b"}) is not None
  os.utime(rules, ns=(old, time.time_ns() + 60 * SECOND))
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_b"}) is None


def test_a_package_that_is_not_loaded_or_has_no_files_has_no_digest(tmp_path, monkeypatch) -> None:
  assert code_digest({"structure_pkg_never_imported"}) is None
  empty = tmp_path / "structure_pkg_c"
  empty.mkdir()
  (empty / ".hidden").write_text("")
  monkeypatch.syspath_prepend(str(tmp_path))
  __import__("structure_pkg_c")  # a namespace package with nothing in it
  monkeypatch.setitem(sys.modules, "structure_pkg_c", sys.modules["structure_pkg_c"])
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_c"}) is None


def test_the_interpreters_and_numpys_code_is_named_by_version(monkeypatch) -> None:
  base = code_digest(())
  assert base is not None
  assert code_digest({"numpy", "math", "functools"}) == base  # no files of theirs are read
  import scipy

  seen = {base}
  for module, name in ((np, "__version__"), (scipy, "__version__"), (sys, "version")):
    monkeypatch.setattr(module, name, getattr(module, name) + ".post1")
    seen.add(code_digest(()))
  assert None not in seen and len(seen) == 4


def test_two_packages_with_the_same_files_have_two_digests(tmp_path, monkeypatch) -> None:
  _package(tmp_path, "structure_pkg_g", monkeypatch)
  _package(tmp_path, "structure_pkg_h", monkeypatch)
  assert code_digest({"structure_pkg_g"}) != code_digest({"structure_pkg_h"})


def test_a_single_file_module_is_digested_by_its_file(tmp_path, monkeypatch) -> None:
  module = tmp_path / "structure_single.py"
  module.write_text("X = 1\n")
  monkeypatch.syspath_prepend(str(tmp_path))
  __import__("structure_single")
  monkeypatch.setitem(sys.modules, "structure_single", sys.modules["structure_single"])
  _loaded_now(monkeypatch)
  base = code_digest({"structure_single"})
  assert base is not None and base != code_digest(())
  module.write_text("X = 2\n")
  _loaded_now(monkeypatch)
  assert code_digest({"structure_single"}) not in (None, base)


def test_the_code_behind_an_ops_rules_is_part_of_the_key(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_d", monkeypatch)
  rule = __import__("structure_pkg_d.sub.rules", fromlist=["rule"]).rule
  fn = _fn()
  base, packages = _graph(fn)
  assert packages == {"scaly", "numpy"}
  monkeypatch.setattr(op_def(ExprOp.SIN), "fold", rule)
  digest, packages = _graph(fn)
  assert packages == {"scaly", "numpy", "structure_pkg_d"}
  assert digest != base  # another rule is another definition of the op
  assert digest == _graph(_fn())[0]
  before = _key(fn)
  (package / "sub" / "rules.py").write_text("def rule():\n  return 2\n")
  _loaded_now(monkeypatch)
  after = _key(fn)
  assert before is not None and after is not None and before[0] != after[0] and before[1] == after[1]
  monkeypatch.setattr(op_def(ExprOp.SIN), "fold", None)
  assert _graph(fn) == (base, {"scaly", "numpy"})
  monkeypatch.setattr(op_def(ExprOp.SIN), "traits", {**op_def(ExprOp.SIN).traits, "structure_test": rule})
  assert _graph(fn)[1] == {"scaly", "numpy", "structure_pkg_d"}


def _closure(x):
  return lambda *args: x


def _plain(*args):
  return None


def _wrapped(fn):
  """A decorator's wrapper: a closure that takes the name of what it wraps."""

  @functools.wraps(fn)
  def wrapper(*args):
    return fn(*args)

  return wrapper


class _Bound:
  def rule(self, *args):
    return None


@pytest.mark.parametrize(
  ("rule", "keyed"),
  [
    pytest.param(math.sin, True, id="a builtin, whose module is no state"),
    pytest.param(np.cos, True, id="a NumPy ufunc"),
    pytest.param(_closure(1), False, id="a closure"),
    pytest.param(_wrapped(_plain), False, id="a closure under a module-level name, as a decorator leaves"),
    pytest.param(functools.partial(max, 1), False, id="a partial application"),
    pytest.param(_Bound().rule, False, id="a bound method"),
    pytest.param(lambda *args: None, True, id="a lambda at module level"),
  ],
)
def test_a_rule_that_carries_state_outside_scaly_gives_no_key(monkeypatch, rule, keyed: bool) -> None:
  fn = _fn()
  base = _graph(fn)[0]
  monkeypatch.setattr(op_def(ExprOp.SIN), "numpy", rule)
  found = graph_digest(fn)
  assert (found is not None) == keyed
  assert found is None or found.digest != base


def test_two_rules_of_one_package_are_two_definitions(monkeypatch) -> None:
  fn = _fn()
  monkeypatch.setattr(op_def(ExprOp.SIN), "numpy", math.sin)
  one = _graph(fn)
  monkeypatch.setattr(op_def(ExprOp.SIN), "numpy", math.cos)
  assert _graph(fn)[0] != one[0] and _graph(fn)[1] == one[1]
  monkeypatch.setattr(op_def(ExprOp.SIN), "numpy", np.sin)
  ufunc = _graph(fn)[0]
  monkeypatch.setattr(op_def(ExprOp.SIN), "numpy", np.cos)
  assert _graph(fn)[0] != ufunc


def test_an_ops_arity_and_the_values_of_its_traits_are_part_of_the_digest(monkeypatch) -> None:
  fn = _fn()
  seen = {_graph(fn)[0]}
  sin = op_def(ExprOp.SIN)
  monkeypatch.setattr(sin, "traits", {**sin.traits, "elementwise": ProgramOp.COS})  # the op it lowers to
  seen.add(_graph(fn)[0])
  monkeypatch.setattr(sin, "traits", {**sin.traits, "expensive": False})
  seen.add(_graph(fn)[0])
  monkeypatch.setattr(sin, "arity", 2)
  seen.add(_graph(fn)[0])
  monkeypatch.setattr(sin, "differentiable", False)
  seen.add(_graph(fn)[0])
  monkeypatch.setattr(op_def(ExprOp.COS), "arity", 2)  # an op the graph does not hold
  seen.add(_graph(fn)[0])
  assert len(seen) == 5


def _rename_pass(prog: ProgramNode) -> ProgramNode:
  return prog


def test_an_inserted_pass_and_the_in_place_switch_are_part_of_the_key(cache, monkeypatch) -> None:
  fn = _fn()
  base, packages = _graph(fn)
  monkeypatch.setattr(program_passes, "_INSERTED", [(True, "fold_arith", "structure_pass", _rename_pass)])
  after, after_packages = _graph(fn)
  assert after != base and after_packages == packages | {_rename_pass.__module__.partition(".")[0]}
  monkeypatch.setattr(program_passes, "_INSERTED", [(False, "fold_arith", "structure_pass", _rename_pass)])
  assert _graph(fn)[0] not in (base, after)  # the slot counts
  monkeypatch.setattr(program_passes, "_INSERTED", [(True, "fold_arith", "structure_pass", _closure(1))])
  assert graph_digest(fn) is None  # a pass that carries state
  monkeypatch.setattr(program_passes, "_INSERTED", [])
  assert _graph(fn)[0] == base
  monkeypatch.setattr(lowering, "DONATE_CARRIES", False)
  assert _graph(fn)[0] != base


def test_rendering_reads_the_default_options_whatever_is_in_force(cache, monkeypatch) -> None:
  seen = []
  matmul = op_def(ExprOp.MATMUL)
  rule = matmul.lower
  assert rule is not None

  def watching(ctx, node):
    seen.append(get_options().nonsmooth)
    return rule(ctx, node)

  monkeypatch.setattr(matmul, "lower", watching)
  x = sc.sym("x", (2, 3))
  fn = sc.Function.from_exprs("structure_options", [x], [x @ x.T], ["x"], ["y"])
  with sc.options(nonsmooth="error"):
    sc.codegen.render_c_source(fn)
    assert get_options().nonsmooth == "error"  # in force again once the render is done
    with default_options():
      assert get_options().nonsmooth == "split"
    assert get_options().nonsmooth == "error"
  assert seen and set(seen) == {"split"}
  # The built-in defaults, not the process's own (``sc.set_options``).
  monkeypatch.setattr(options_module, "_default", replace(options_module.Options(), nonsmooth="first"))
  assert get_options().nonsmooth == "first"
  seen.clear()
  sc.codegen.render_c_source(sc.Function.from_exprs("structure_options2", [x], [x @ x.T * 2.0], ["x"], ["y"]))
  assert seen and set(seen) == {"split"} and get_options().nonsmooth == "first"


def test_an_extern_bodys_c_is_rendered_under_the_default_options(cache) -> None:
  fn = extern_function("structure_extern", _Optioned(), [("x", (3,))], [("y", ())])
  with sc.options(nonsmooth="first"):
    inside, source = _key(fn), sc.codegen.render_c_source(fn)
  assert inside is not None and inside == _key(fn)
  assert "// split" in source and source == sc.codegen.render_c_source(fn)


def test_code_edited_under_a_live_process_stops_the_key_and_the_function_is_rendered(cache, monkeypatch) -> None:
  monkeypatch.setattr(env, "LOADED_AT_NS", 0)  # every file is newer than the load
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
  monkeypatch.setattr(jit, "graph_digest", lambda fun: structure.GraphDigest("0" * 64, frozenset(), ()))
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
    then = env.LOADED_AT_NS - (age + 1) * 1_000_000_000
    os.utime(state, ns=(then, then))
  _fn()(np.zeros(3))
  kept = sorted(path.name for path in index.iterdir())
  (current,) = [path.parent.name for path in _entries(cache) if path.name != "entry.json"]
  assert kept == sorted([current, *(f"{age:032x}" for age in range(jit._INDEX_STATES_KEPT - 1))])
  _fn(3.0)(np.zeros(3))  # a second entry in a state that is there already: nothing is pruned
  (index / f"{99:032x}").mkdir()
  _fn(4.0)(np.zeros(3))
  assert len(list(index.iterdir())) == jit._INDEX_STATES_KEPT + 1


_RESOLVERS = {flag: (lambda names, flag=flag: [flag, *(f"-l{name}" for name in names)]) for flag in ("-L/one", "-L/other")}


@dataclass(frozen=True)
class _Linked(_Callee):
  flag: str = "-L/one"

  def build_requirements(self, fun: sc.ConcreteFunction) -> BuildRequirements:
    return replace(_Callee.build_requirements(self, fun), libraries=("m",), link_flags=_RESOLVERS[self.flag])


@dataclass(frozen=True)
class _Optioned(_Callee):
  def render(self, fun: sc.ConcreteFunction, ctx: ExternRenderCtx) -> list[str]:
    return [*super().render(fun, ctx), f"// {get_options().nonsmooth}"]


def test_an_extern_bodys_link_flags_are_part_of_the_key(cache) -> None:
  one, other = (_key(extern_function("structure_extern", _Linked(flag=flag), [("x", (3,))], [("y", ())])) for flag in ("-L/one", "-L/other"))
  assert one is not None and other is not None and one[0] == other[0] and one[1] != other[1]


def test_the_functions_an_extern_body_builds_when_asked_are_each_digested(cache) -> None:
  # Two extern bodies in one graph, each building the Function it calls when asked: the first one's
  # is freed before the second's is built, at the same address as often as not.
  def host(factor: float) -> sc.ConcreteFunction:
    first = _extern("structure_first", dependency="structure_dep_a", helper="structure_help_a")
    second = _extern("structure_second", dependency="structure_dep_b", helper="structure_help_b", factor=factor)
    v = sc.sym("x", 3)
    return sc.Function.from_exprs("structure_two", [v], [first(v) + second(v)], ["x"], ["y"])

  digests = set()
  for factor in (2.0, 2.5, 3.0, 3.5):
    for _ in range(3):
      gc.collect()
      digests.add((factor, _graph(host(factor))[0]))
  assert len(digests) == 4 and len({digest for _, digest in digests}) == 4
  x = np.array([0.1, -0.7, 2.5])
  base = (np.sin(x) * 2.0 + x[::-1]).sum()
  for factor in (2.5, 3.0):
    np.testing.assert_allclose(host(factor)(x), 2.0 * base + 2.0 * (np.sin(x) * factor + x[::-1]).sum())


@dataclass(frozen=True)
class _Raising(_Callee):
  def render(self, fun: sc.ConcreteFunction, ctx: ExternRenderCtx) -> list[str]:
    raise NotImplementedError("no C for this one")


def test_an_extern_body_that_raises_when_asked_has_no_key_and_rendering_says_why(cache) -> None:
  fn = extern_function("structure_extern", _Raising(), [("x", (3,))], [("y", ())])
  assert graph_digest(fn) is None
  with pytest.raises(jit.JitUnavailable, match="no C for this one"):
    fn(np.zeros(3))


def test_an_index_entry_keeps_the_link_flags(cache) -> None:
  x = np.array([0.1, -0.7, 2.5])
  first = extern_function("structure_extern", _Linked(), [("x", (3,))], [("y", ())])
  np.testing.assert_allclose(first(x), 2.0 * (np.sin(x) * 2.0 + x[::-1]).sum())
  (entry,) = _entries(cache)
  assert json.loads(entry.read_text())["flags"] == ["-L/one", "-lm"]
  jit._artifact_cache.clear()
  key = _key(extern_function("structure_extern", _Linked(), [("x", (3,))], [("y", ())]))
  assert key is not None
  found = jit._index_lookup(key)
  assert found is not None and found.flags == ("-L/one", "-lm")


def test_an_isolated_library_is_loaded_isolated_from_its_index_entry(cache) -> None:
  x = np.array([0.1, -0.7, 2.5])
  first = _extern(isolated=True)
  first(x)
  (entry,) = _entries(cache)
  assert json.loads(entry.read_text())["isolated"] is True
  jit._artifact_cache.clear()
  key = _key(_extern(isolated=True))
  assert key is not None
  found = jit._index_lookup(key)
  assert found is not None and found.isolated and found.flags == () and found.workspace_size == first._compiled._sz_w


def test_types_with_one_name_and_other_fields_are_two_types() -> None:
  # A dtype's own equality is its name; its C type is what the code reads.
  def typed(dtype) -> str:
    x = Expr.sym("x", 3)
    y = Expr(ExprOp.CAST, (x,), TensorType((3,), dtype=dtype))
    return _graph(sc.Function.from_exprs("structure_types", [x], [y], ["x"], ["y"]))[0]

  odd = replace(dtypes.float32, c_type="double")
  assert odd == dtypes.float32
  assert typed(dtypes.float32) != typed(odd)
  assert typed(dtypes.float32) == typed(replace(dtypes.float32))  # an equal copy is the same type


_PATTERN = SparsityType((2, 2), (0, 1), (0, 1))


@pytest.mark.parametrize(
  ("first", "second"),
  [
    pytest.param(TensorType((3,)), TensorType((4,)), id="shape"),
    pytest.param(TensorType((3,)), TensorType((3,), dtype=dtypes.float32), id="dtype"),
    pytest.param(
      TensorType((3,), dtype=dtypes.float32), TensorType((3,), dtype=replace(dtypes.float32, c_type="double")), id="a dtype's C type under one name"
    ),
    pytest.param(TensorType((2, 2), sparsity=_PATTERN), TensorType((2, 2), sparsity=SparsityType((2, 2), (0, 1), (1, 0))), id="sparsity pattern"),
    pytest.param(TensorType((2, 2), sparsity=_PATTERN), TensorType((2, 2)), id="having a pattern"),
    pytest.param(TensorType((3,)), TensorType((3,), diff=False), id="differentiability"),
  ],
)
def test_two_types_in_one_graph_that_differ_in_their(first: TensorType, second: TensorType) -> None:
  # The second type must be told from the first where both are in the graph: a type is written
  # once and then referred to by number.
  def digest(types: tuple[TensorType, TensorType]) -> str:
    # Interning compares a dtype by its name, so an input of the call before, if a cycle still
    # holds it, would be handed back here with the type it had there.
    gc.collect()
    a, b = (Expr(ExprOp.INPUT, type=t, name=name) for t, name in zip(types, ("a", "b"), strict=True))
    assert (a.type, b.type) == types and a.type.dtype.c_type == types[0].dtype.c_type and b.type.dtype.c_type == types[1].dtype.c_type
    return _graph(sc.Function.from_exprs("structure_two_types", [a, b], [a, b], ["a", "b"], ["p", "q"]))[0]

  assert digest((first, second)) != digest((first, first))
  assert digest((first, second)) != digest((second, first))
  assert digest((first, first)) == digest((first, replace(first)))  # an equal copy is the same type


def test_an_arrays_digest_is_of_its_entries_however_it_is_laid_out() -> None:
  a = np.arange(12.0).reshape(3, 4)
  assert _digest(a) == _digest(np.asfortranarray(a)) == _digest(np.arange(24.0).reshape(3, 8)[:, ::2] / 2.0)
  assert _digest(a) != _digest(a.T.copy().reshape(3, 4))
  assert _digest(np.zeros(())) != _digest(np.zeros(1)) != _digest(np.zeros((1, 1)))
  assert _digest(np.zeros(0)) != _digest(np.zeros((0, 3)))
  assert _digest(np.arange(3, dtype=">i4")) != _digest(np.arange(3, dtype="<i4"))


def test_a_symbol_without_a_name_is_not_one_named_none() -> None:
  def named(name) -> str:
    x = Expr(ExprOp.INPUT, type=TensorType((3,)), name=name)
    return _graph(sc.Function.from_exprs("structure_named", [x], [x * 2.0], ["x"], ["y"]))[0]

  assert named(None) != named("None")


class _Sub(sc.ConcreteFunction):
  pass


def test_a_functions_class_is_part_of_the_digest() -> None:
  plain = _fn()
  sub = _Sub.__new__(_Sub)
  sub.__dict__.update(plain.__dict__)
  base, packages = _graph(plain)
  other, other_packages = _graph(sub)
  assert other != base and other_packages == packages | {_Sub.__module__.partition(".")[0]}


def test_verify_mode_compares_what_the_index_entry_holds(cache, monkeypatch) -> None:
  fn = _fn()
  fn(np.zeros(3))
  monkeypatch.setenv("SCALY_JIT_KEY", "verify")
  lookup = jit._index_lookup
  for change in (dict(workspace_size=7), dict(flags=("-lm",)), dict(isolated=True), dict(key="0" * 64)):
    jit._artifact_cache.clear()

    def changed(key, change=change):
      found = lookup(key)
      assert found is not None
      return replace(found, **change)

    monkeypatch.setattr(jit, "_index_lookup", changed)
    with pytest.raises(jit.JitError, match="structural key"):
      _fn()(np.zeros(3))
  monkeypatch.setattr(jit, "_index_lookup", lookup)
  _fn()(np.zeros(3))


def test_a_state_of_the_index_that_is_read_is_kept_like_one_that_is_written(cache, monkeypatch) -> None:
  _fn()(np.zeros(3))
  (entry,) = _entries(cache)
  old = env.LOADED_AT_NS - 3600 * SECOND
  os.utime(entry.parent, ns=(old, old))
  jit._artifact_cache.clear()
  monkeypatch.setattr(jit, "_index_states_read", set())  # a new process
  _no_render(monkeypatch)
  _fn()(np.zeros(3))
  assert entry.parent.stat().st_mtime_ns > old
  touched = entry.parent.stat().st_mtime_ns
  jit._artifact_cache.clear()
  _fn()(np.zeros(3))
  assert entry.parent.stat().st_mtime_ns == touched  # once per process


def test_pruning_survives_a_state_that_vanishes_and_an_index_that_is_not_there(cache, monkeypatch) -> None:
  jit._prune_index(cache / "structure" / "nowhere")  # no index yet
  index = cache / "structure"
  states = [index / f"{k:032x}" for k in range(12)]
  for state in states:
    state.mkdir(parents=True)
  real, calls = os.scandir, []

  def scandir(path):
    """The listing the pruning takes, during which another process removes one state."""
    calls.append(path)
    if len(calls) > 1:
      return real(path)
    entries = list(real(path))
    shutil.rmtree(states[3])
    return entries

  monkeypatch.setattr(os, "scandir", scandir)
  jit._prune_index(states[0])
  monkeypatch.setattr(os, "scandir", real)
  assert states[0].exists() and not states[3].exists() and len(list(index.iterdir())) == jit._INDEX_STATES_KEPT


def _looped() -> sc.ConcreteFunction:
  c = sc.sym("c", 4)
  body = sc.Function.from_exprs(
    "structure_body", [c], [sc.put_add(c, sc.const(np.array([0, 1]), dtype="int64"), c[2:].sin() * 0.5, in_range=True)], ["c"], ["n"]
  )
  cond = sc.Function.from_exprs("structure_cond", [c], [sc.norm_inf(c[:2]) < 3.0], ["c"], ["go"])
  init = sc.sym("init", 4)
  out, count = sc.while_loop(cond, body, init, max_iter=10)
  (scanned,) = sc.scan(body, init, length=3)
  return sc.Function.from_exprs("structure_loops", [init], [out, count, scanned], ["init"], ["out", "count", "scanned"])


def _derived() -> sc.ConcreteFunction:
  x = sc.sym("x", 5)
  y = (x[1:] * x[:-1]).sin().sum() + (x**3).sum()
  return sc.Function.from_exprs("structure_derived", [x], [sc.gradient(y, x), sc.hessian(y, x)], ["x"], ["g", "h"])


@pytest.mark.parametrize(
  "build",
  [_looped, _derived, lambda: _mapped(2.0), lambda: _calling(2.0), _extern],
  ids=["loops", "derivatives", "a map", "a call", "an extern body"],
)
def test_graphs_of_every_kind_have_a_key_and_are_found_by_it(cache, monkeypatch, build) -> None:
  first = build()
  assert _key(first) is not None  # a walk that refused one of these would turn the index off unseen
  rng = np.random.default_rng(0)
  args = [rng.standard_normal(shape) for shape in first.input_shapes]
  expected = first._flat_numerical_call(*args)
  jit._artifact_cache.clear()
  _no_render(monkeypatch)
  for got, want in zip(build()._flat_numerical_call(*args), expected, strict=True):
    np.testing.assert_array_equal(got, want)


@pytest.mark.method("opt.ipopt")
def test_a_solver_has_a_key_and_is_found_by_it(cache, monkeypatch) -> None:
  from scaly.testing.helpers import build_nlp

  def solver() -> sc.ConcreteFunction:
    x, param = sc.sym("x", 3), sc.sym("param", 1)
    return build_nlp(x=x, f=((x - param.exp()) ** 2).sum(), p=param, name="structure_solver").concrete

  args = (np.zeros(3), np.zeros(3), np.zeros(0), np.zeros(0), np.array([0.2]))
  assert _key(solver()) is not None and _key(solver()) == _key(solver())
  first = solver()(*args)[0]
  np.testing.assert_allclose(first, np.full(3, np.exp(0.2)), atol=1e-7)
  jit._artifact_cache.clear()
  _no_render(monkeypatch)
  np.testing.assert_array_equal(solver()(*args)[0], first)


# --- what a second review proved: each of these was one key over two renderings ------------------


def test_a_package_other_than_scaly_is_judged_by_when_the_process_started(tmp_path, monkeypatch) -> None:
  # It may have been imported before scaly was: a file of its written between the two is not what runs.
  package = _package(tmp_path, "structure_pkg_i", monkeypatch)
  assert code_digest({"structure_pkg_i"}) is not None
  written = max(path.stat().st_ctime_ns for path in [package, *package.rglob("*")])
  monkeypatch.setattr(env, "process_started_ns", lambda: written - SECOND)  # the process started before the files were written
  assert code_digest({"structure_pkg_i"}) is None
  assert code_digest(()) is not None  # scaly's own code loads when scaly does
  monkeypatch.setattr(env, "process_started_ns", lambda: None)  # a platform that does not say, or a forked process
  assert code_digest({"structure_pkg_i"}) is None and code_digest(()) is not None


def test_the_process_start_is_before_scaly_was_loaded_and_unknown_in_a_fork(monkeypatch) -> None:
  monkeypatch.undo()  # the real clock, not the fixture's
  started = env.process_started_ns()
  if started is None:
    pytest.skip("this platform does not say when a process started")
  assert 0 < env.LOADED_AT_NS - started < 3600 * SECOND
  monkeypatch.setattr(env, "_started", [])
  monkeypatch.setattr(env, "_process_start", lambda: env.LOADED_AT_NS + 1)  # a start after the load is no start
  assert env.process_started_ns() is None
  monkeypatch.setattr(env, "_started", [])
  monkeypatch.setattr(env, "_process_start", lambda: env.LOADED_AT_NS)
  assert env.process_started_ns() == env.LOADED_AT_NS
  monkeypatch.setattr(env, "_LOADED_BY", os.getpid() + 1)  # as in a child forked from the process that loaded scaly
  assert env.process_started_ns() is None


def test_a_link_pointed_elsewhere_or_a_directory_changed_under_a_live_process_has_no_digest(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_j", monkeypatch)
  for version in ("v1", "v2"):
    (tmp_path / version).mkdir()
    (tmp_path / version / "impl.py").write_text(f"VERSION = '{version}'\n")
  (package / "current").symlink_to(tmp_path / "v1", target_is_directory=True)
  (package / "one.py").symlink_to(tmp_path / "v1" / "impl.py")
  _loaded_now(monkeypatch)
  assert code_digest({"structure_pkg_j"}) is not None
  loaded = time.time_ns() + structure._CLOCK_SLACK_NS  # loaded now: every time so far is older
  monkeypatch.setattr(env, "LOADED_AT_NS", loaded)
  assert code_digest({"structure_pkg_j"}) is not None

  def relink(name: str, target: Path, directory: bool) -> None:
    """Point the link elsewhere by renaming a new link over it, as an upgrade in place does."""
    fresh = package / f"{name}.new"
    fresh.symlink_to(target, target_is_directory=directory)
    fresh.rename(package / name)

  for change in (
    lambda: relink("current", tmp_path / "v2", True),
    lambda: relink("one.py", tmp_path / "v2" / "impl.py", False),
    lambda: (package / "sub" / "rules.py").unlink(),
    lambda: (package / "sub" / "added.py").write_text(""),
    lambda: (package / "template.c").rename(package / "renamed.c"),
  ):
    monkeypatch.setattr(env, "LOADED_AT_NS", time.time_ns() + structure._CLOCK_SLACK_NS)
    assert code_digest({"structure_pkg_j"}) is not None
    change()
    assert code_digest({"structure_pkg_j"}) is None


def _bytecode(source: Path, body: bytes, *, mtime: int | None = None, size: int | None = None, flags: int = 0) -> bytes:
  """A bytecode file's bytes for ``source``: the header Python checks (the source's modification
  time and size, unless given others), then ``body``."""
  stat = source.stat()
  recorded = int(stat.st_mtime) if mtime is None else mtime
  return importlib.util.MAGIC_NUMBER + struct.pack("<III", flags, recorded & 0xFFFFFFFF, (stat.st_size if size is None else size) & 0xFFFFFFFF) + body


def test_bytecode_python_would_run_in_place_of_a_changed_source_is_part_of_the_digest(tmp_path, monkeypatch) -> None:
  # Python loads the bytecode it kept when the source's size and modification time match what the
  # bytecode recorded: a source replaced with both kept runs as the old code, and only the
  # bytecode says so.
  package = _package(tmp_path, "structure_pkg_k", monkeypatch)
  source = package / "__init__.py"
  cached = Path(importlib.util.cache_from_source(str(source)))
  cached.unlink(missing_ok=True)

  def digest(kept: bytes | None, *, older: bool) -> str | None:
    if kept is not None:
      cached.write_bytes(kept)
      when = source.stat().st_ctime_ns - SECOND if older else source.stat().st_ctime_ns
      os.utime(cached, ns=(when, when))
    _loaded_now(monkeypatch)
    return code_digest({"structure_pkg_k"})

  base = digest(None, older=False)
  assert base is not None
  # Written at or after the source's last change: the source's own, whatever it holds. Bytecode
  # that one worker writes while another runs moves no key.
  assert digest(_bytecode(source, b"one"), older=False) == base
  assert digest(_bytecode(source, b"two"), older=False) == base
  # Older than the source's last change, and for a source of this size and time: what runs.
  stale = digest(_bytecode(source, b"of the source that was here before"), older=True)
  other = digest(_bytecode(source, b"of yet another"), older=True)
  assert len({base, stale, other}) == 3 and None not in (stale, other)
  # Older, but for a source of another time or size: Python compiles the source again.
  assert digest(_bytecode(source, b"x", mtime=int(source.stat().st_mtime) - 5), older=True) == base
  assert digest(_bytecode(source, b"x", size=source.stat().st_size + 1), older=True) == base
  assert digest(b"cut", older=True) == base
  # Bytecode checked by a hash of the source is not checked here: taken as accepted.
  assert digest(_bytecode(source, b"hashed", mtime=0, size=0, flags=1), older=True) not in (None, base, stale, other)


def _passing(prog: ProgramNode) -> ProgramNode:
  return prog


def _named_like(fn):
  """A function of this module carrying the name of another, as a decorator's wrapper does."""
  wrapper = types.FunctionType(_passing.__code__, _passing.__globals__, getattr(fn, "__name__"))
  wrapper.__module__, wrapper.__qualname__ = fn.__module__, getattr(fn, "__qualname__")
  return wrapper


def test_a_function_is_told_by_its_code_and_not_by_the_name_it_carries(monkeypatch) -> None:
  fn = _fn()
  base, packages = _graph(fn)
  fold = dict(program_passes.PASS_PIPELINE)["fold_arith"]
  disguised = _named_like(fold)
  assert (disguised.__module__, disguised.__qualname__) == (fold.__module__, getattr(fold, "__qualname__")) and disguised.__closure__ is None
  monkeypatch.setattr(
    program_passes, "PASS_PIPELINE", tuple((name, disguised if name == "fold_arith" else pass_) for name, pass_ in program_passes.PASS_PIPELINE)
  )
  other, other_packages = _graph(fn)
  assert other != base and other_packages == packages | {__name__.partition(".")[0]}
  monkeypatch.setattr(
    program_passes, "PASS_PIPELINE", tuple((name, _wrapped(fold) if name == "fold_arith" else pass_) for name, pass_ in program_passes.PASS_PIPELINE)
  )
  assert graph_digest(fn) is None  # a real wrapper is a closure


_FIRST = lambda *args: None  # noqa: E731
_SECOND = lambda *args: None  # noqa: E731


def test_two_functions_of_one_name_are_told_by_their_line_and_their_defaults(monkeypatch) -> None:
  fn = _fn()
  sin = op_def(ExprOp.SIN)
  digests = []
  for rule in (_FIRST, _SECOND):  # two lambdas of one module: one qualified name, two lines
    monkeypatch.setattr(sin, "fold", rule)
    digests.append(_graph(fn)[0])
  for scale in (1.0, 2.0):  # one code, two defaults, as a loop at module level makes them
    made = types.FunctionType(_plain.__code__, _plain.__globals__, "_plain", (scale,))
    monkeypatch.setattr(sin, "fold", made)
    digests.append(_graph(fn)[0])
  made = types.FunctionType(_plain.__code__, _plain.__globals__, "_plain")
  made.__kwdefaults__ = {"scale": 3.0}
  monkeypatch.setattr(sin, "fold", made)
  digests.append(_graph(fn)[0])
  assert len(set(digests)) == 5
  made.__kwdefaults__ = {"scale": _OPAQUE}  # a default the walk cannot write
  assert graph_digest(fn) is None
  for scope in ({"__name__": "structure_no_such_module"}, {"__name__": __name__}):  # globals that are no module's own
    monkeypatch.setattr(sin, "fold", types.FunctionType(_plain.__code__, scope, "_plain"))
    assert graph_digest(fn) is None


def _local():
  def inner(*args):
    return None

  return inner


def test_a_function_defined_in_another_is_keyed_when_it_captures_nothing(monkeypatch) -> None:
  # Every call of the outer function gives the same code with nothing of its own, so one stands for all.
  fn = _fn()
  monkeypatch.setattr(op_def(ExprOp.SIN), "fold", _local())
  one = graph_digest(fn)
  monkeypatch.setattr(op_def(ExprOp.SIN), "fold", _local())
  assert one is not None and graph_digest(fn) == one


class _Posing:
  pass


def test_a_class_whose_name_leads_to_another_gives_no_key() -> None:
  assert _digest(_Whole()) is not None
  posing = dataclass(frozen=True)(type("Posing", (), {"__annotations__": {"n": int}, "n": 1, "__module__": __name__, "__qualname__": "_Whole"}))
  assert _digest(posing()) is None
  assert _Posing.__qualname__ == "_Posing"


def test_a_ufunc_that_is_not_numpys_gives_no_key(monkeypatch) -> None:
  fn = _fn()
  assert graph_digest(fn) is not None
  made = np.frompyfunc(math.cos, 1, 1)  # as a compiled extension's ufunc is: one NumPy's name does not lead to
  monkeypatch.setattr(op_def(ExprOp.SIN), "numpy", made)
  assert graph_digest(fn) is None


def test_a_target_of_a_subclass_has_no_key(cache) -> None:
  class Tuned(Target):
    @property
    def sum_lanes(self) -> int:
      return 16

  fn = _fn()
  host = resolve_target(None)
  assert _key(fn, target=host) is not None
  assert _key(fn, target=Tuned(**{f.name: getattr(host, f.name) for f in dataclass_fields(host)})) is None


def test_a_module_named_like_one_of_the_interpreters_is_digested_by_its_files(tmp_path, monkeypatch) -> None:
  assert structure._versioned("math") and structure._versioned("functools") and structure._versioned("json") and structure._versioned("numpy")
  assert not structure._versioned("scaly") and not structure._versioned("structure_never_imported")
  # What is installed beside the interpreter's own modules is not the interpreter's.
  installed = os.path.realpath(pytest.__file__)
  with monkeypatch.context() as patched:
    patched.setattr(structure, "_STDLIB", (installed[: installed.index("site-packages")],))
    assert not structure._versioned("pytest")
  module = tmp_path / "trace.py"
  module.write_text("X = 1\n")
  spec = importlib.util.spec_from_file_location("trace", module)
  assert spec is not None and spec.loader is not None
  shadow = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(shadow)
  monkeypatch.setitem(sys.modules, "trace", shadow)
  assert not structure._versioned("trace")
  _loaded_now(monkeypatch)
  base = code_digest({"trace"})
  assert base not in (None, code_digest(()))
  module.write_text("X = 2\n")
  _loaded_now(monkeypatch)
  assert code_digest({"trace"}) not in (None, base)


def test_what_fusion_reads_of_the_whole_registry_is_part_of_the_digest(monkeypatch) -> None:
  fn = _fn()  # no abs in it
  base = _graph(fn)[0]
  absolute = op_def(ExprOp.ABS)
  monkeypatch.setattr(absolute, "traits", {**absolute.traits, "expensive": True})
  assert _graph(fn)[0] != base


def test_a_shape_of_numpy_integers_is_the_shape() -> None:
  def digest(shape) -> str:
    x = Expr(ExprOp.INPUT, type=TensorType(shape), name="x")
    return _graph(sc.Function.from_exprs("structure_shape", [x], [x], ["x"], ["y"]))[0]

  plain = digest((3, 4))
  gc.collect()
  assert digest((np.int64(3), np.int64(4))) == plain


def test_a_type_of_a_subclass_gives_no_key() -> None:
  class Tagged(TensorType):
    pass

  class Float(DType):
    pass

  for kind in (Tagged((3,)), TensorType((3,), dtype=Float(**{f.name: getattr(dtypes.float64, f.name) for f in dataclass_fields(dtypes.float64)}))):
    x = Expr(ExprOp.INPUT, type=kind, name="structure_subtyped")
    assert graph_digest(sc.Function.from_exprs("structure_subtype", [x], [x], ["x"], ["y"])) is None


def test_every_rule_of_an_op_and_every_traits_name_is_part_of_the_digest(monkeypatch) -> None:
  fn = _fn()
  sin = op_def(ExprOp.SIN)
  seen = {_graph(fn)[0]}
  for kind in ("lower", "jvp", "vjp", "sparsity", "fold"):
    monkeypatch.setattr(sin, kind, _plain)
    seen.add(_graph(fn)[0])
  assert len(seen) == 6
  monkeypatch.setattr(sin, "traits", {**sin.traits, "exact_reads": True, "structure_one": True})
  one = _graph(fn)[0]
  monkeypatch.setattr(sin, "traits", {**{k: v for k, v in sin.traits.items() if k != "structure_one"}, "structure_two": True})
  assert _graph(fn)[0] != one


def test_a_file_written_under_two_seconds_before_the_load_may_have_been_written_after_it(tmp_path, monkeypatch) -> None:
  package = _package(tmp_path, "structure_pkg_l", monkeypatch)
  written = max(max(path.stat().st_ctime_ns, path.stat().st_mtime_ns) for path in [package, *package.rglob("*")] if "__pycache__" not in path.parts)
  monkeypatch.setattr(env, "LOADED_AT_NS", written + 1_500_000_000)
  assert code_digest({"structure_pkg_l"}) is None
  monkeypatch.setattr(env, "LOADED_AT_NS", written + 2_500_000_000)
  assert code_digest({"structure_pkg_l"}) is not None


def test_a_graph_nested_too_deep_for_the_walk_has_no_key(monkeypatch) -> None:
  def deep(self, fun):
    raise RecursionError

  monkeypatch.setattr(structure._Walk, "function", deep)
  assert graph_digest(_fn()) is None


_BUILT: list[weakref.ref] = []


@dataclass(frozen=True)
class _Building(_Callee):
  def dependencies(self) -> tuple[sc.ConcreteFunction, ...]:
    made = _Callee.dependencies(self)
    _BUILT.extend(weakref.ref(f) for f in made)
    return made


def test_the_walk_keeps_alive_what_it_numbers_by_address() -> None:
  # An extern body may build the Functions it calls when asked; freed, their addresses are reused,
  # and the next Function at one would pass for the last.
  _BUILT.clear()
  walk = structure._Walk()
  walk.function(extern_function("structure_extern", _Building(), [("x", (3,))], [("y", ())]))
  gc.collect()
  assert _BUILT and all(ref() is not None for ref in _BUILT)
  del walk
  gc.collect()
  assert all(ref() is None for ref in _BUILT)
