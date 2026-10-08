"""Unit tests for the Program IR optimization passes (``scaly.passes.program``).

These exercise the passes directly on lowered Program IR (structural assertions) plus a couple of
end-to-end numeric checks under the Program IR renderer. See ``docs/how_it_works/lowering.md``
for the pipeline order and pass contracts.
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.codegen.c import render_program_c_source
from scaly.codegen.toolchain import find_c_compiler
from scaly.passes.lowering import lower_function, main_proc
from scaly.passes.program.combine_scatter_sums import combine_scatter_sums
from scaly.passes.program.fold_arith import fold_arith
from scaly.passes.program.fuse_elementwise import fuse_elementwise
from scaly.passes.program.hoist_invariant import hoist_invariant
from scaly.passes.program._common import allocated_name, prune_procedures
from scaly.ir import program as p
from scaly.ir.program_spec import verify_program
from scaly.passes.program.pack_workspace import WORKSPACE_SPILL_THRESHOLD
from scaly.passes.program.unroll_unit_loops import unroll_unit_loops
from scaly.ir.program import (
  ProgramNode,
  ProgramOp,
  add,
  buffer,
  const_float,
  const_int,
  for_,
  load,
  mul,
  proc as proc_,
  program,
  range_,
  store,
  var,
  view,
  walk_program,
)
from scaly.ir.types import Lowering, dtypes

_HAVE_CC = find_c_compiler() is not None


def _main_body(fn: sc.Function) -> list:
  proc = main_proc(lower_function(fn))
  pc = int(proc.attrs["param_count"])
  return list(proc.args[pc:])


def _classify(body: list) -> tuple[list, list, list]:
  """(compute_buffers, alias_buffers, for_loops) among the proc's top-level statements."""
  compute = [s for s in body if s.op == ProgramOp.BUFFER and s.attrs.get("address_space") == "private" and "alias_of" not in s.attrs]
  aliases = [s for s in body if s.op == ProgramOp.BUFFER and "alias_of" in s.attrs]
  loops = [s for s in body if s.op == ProgramOp.FOR]
  return compute, aliases, loops


def _sz_w(fn: sc.Function) -> int:
  return int(main_proc(lower_function(fn)).attrs.get("sz_w", 0))


# --- fusion ---------------------------------------------------------------------


def test_fusion_collapses_elementwise_chain() -> None:
  """A same-shape elementwise chain fuses into a single loop with no intermediate buffers."""

  @sc.function(sc.group(sc.arg("x", 8), sc.arg("y", 8)), outputs=sc.arg("out0"), name="chain")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return ((x.sin() + y) * y - x).tanh()

  compute, aliases, loops = _classify(_main_body(f))
  assert compute == []  # every intermediate inlined into the output loop
  assert len(loops) == 1


def _long_scalar_chain(length: int, *, scalar: bool = False) -> sc.Function:
  """``x[0]*1 + x[1]*2 + ...`` accumulated as a left fold, so the expression is ``length`` deep."""

  @sc.function(sc.arg("x", 4), outputs=sc.arg("y"), name=f"fold{length}")
  def fn(x: sc.Expr) -> sc.Expr:
    acc = sc.const(0.0)
    for i in range(length):
      acc = acc + x[i % 4] * float(i + 1)
    out = acc.reshape((1,))
    return out.scalar() if scalar else out

  return fn


def _long_scalar_chain_numpy(length: int, x: np.ndarray) -> float:
  return sum(x[i % 4] * float(i + 1) for i in range(length))


def _flat_stage_fold(stages: int, n: int = 4) -> sc.Function:
  """Per-stage ``slice * weights`` products summed flat, one term at a time, as a left fold."""

  @sc.function(sc.group(sc.arg("x", stages * n), sc.arg("w", n)), outputs=sc.arg("y"), name=f"stages{stages}")
  def fn(inputs):
    x, w = inputs
    terms = [(x[i * n : (i + 1) * n] * w)[k] for i in range(stages) for k in range(n)]
    return sum(terms, sc.const(0.0)).reshape((1,))

  return fn


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the deep-fold witnesses")
@pytest.mark.parametrize("length", [400, 3000])
def test_deep_scalar_fold_lowers_renders_and_runs(length: int) -> None:
  """Depth in the expression must not become depth on the Python stack or in the C source: the
  passes and renderer are iterative, and the renderer splits trees at ``MAX_SCALAR_DEPTH``."""
  f = _long_scalar_chain(length)
  assert re.search(r"double v\d+ = ", render_program_c_source(f))
  x = np.arange(1.0, 5.0)
  np.testing.assert_allclose(f(x), _long_scalar_chain_numpy(length, x), rtol=1e-12)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the deep-fold witnesses")
def test_deep_scalar_fold_through_scalarize_runs() -> None:
  f = _long_scalar_chain(3000, scalar=True)
  assert main_proc(lower_function(f)).attrs.get("scalarized")
  x = np.arange(1.0, 5.0)
  np.testing.assert_allclose(f(x), _long_scalar_chain_numpy(3000, x), rtol=1e-12)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the deep-fold witnesses")
def test_flat_per_stage_fold_at_a_hundred_stages_runs() -> None:
  """The NPMPC-shaped objective written as a flat Python fold over per-stage products."""
  f = _flat_stage_fold(100)
  x = np.linspace(-1.0, 1.0, 400)
  w = np.arange(1.0, 5.0)
  np.testing.assert_allclose(f((x, w)), (x.reshape(100, 4) * w).sum(), rtol=1e-12)


def test_fusion_skips_matmul_operand() -> None:
  """A matmul operand must NOT be inlined: a contraction reads each operand element m*k times,
  so inlining the producer's expression there multiplies compute. Regression guard for the
  blow-up that an unguarded fusion introduces."""

  @sc.function(sc.group(sc.arg("A", (4, 4)), sc.arg("x", 4)), outputs=sc.arg("out0"), name="mm_operand")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    A, x = inputs
    return A @ x.sin()  # sin(x) is the matvec operand — must stay materialized

  compute, _aliases, _loops = _classify(_main_body(f))
  assert len(compute) == 1  # exactly the materialized sin(x); the A@x result aliases the output
  assert compute[0].attrs["shape"] == (4,)


def test_fusion_into_reduction() -> None:
  """An elementwise producer feeding a SUM fuses into the reduce loop (each element read once)."""

  @sc.function(sc.arg("x", 8), outputs=sc.arg("out0"), name="sumf")
  def f(x: sc.Expr) -> sc.Expr:
    return (x.sin() + x).sum()

  compute, _aliases, loops = _classify(_main_body(f))
  # Only the scalar accumulator survives; the elementwise buffer is gone (inlined into the reduce).
  assert all(s.attrs["shape"] == (1,) for s in compute)
  assert len(loops) == 1  # the single reduce loop


def test_fusion_counts_expensive_work_through_producer_chains() -> None:
  x, y = (buffer(name, dtypes.float64, (4,)) for name in ("x", "y"))
  a, b = (buffer(name, dtypes.float64, (4,), address_space="private") for name in ("a", "b"))
  i = var("i")
  at = lambda buf: view(buf, [i])
  sin = ProgramNode(ProgramOp.SIN, (load(at(x)),), dtype=dtypes.float64)
  body = [
    a,
    b,
    for_(range_("i", 0, 4), [store(at(a), sin)]),
    for_(range_("i", 0, 4), [store(at(b), add(load(at(a)), const_float(1.0)))]),
    for_(range_("i", 0, 4), [store(at(y), mul(load(at(b)), load(at(b))))]),
  ]
  result = fuse_elementwise(program([proc_("chain_cost", [x, y], body)])).args[0]
  result_body = result.args[int(result.attrs["param_count"]) :]
  assert sum(stmt.op == ProgramOp.FOR for stmt in result_body) == 2
  assert sum(node.op == ProgramOp.SIN for stmt in result_body for node in walk_program(stmt)) == 1


def test_fusion_does_not_move_reads_into_a_consumer_that_overwrites_them() -> None:
  x = buffer("x", dtypes.float64, (4,))
  a = buffer("a", dtypes.float64, (4,), address_space="private")
  i = var("i")
  body = [
    a,
    for_(range_("i", 0, 4), [store(view(a, [i]), add(load(view(x, [i])), const_float(1.0)))]),
    for_(range_("i", 0, 4), [store(view(x, [i]), load(view(a, [p.sub(const_int(3), i)])))]),
  ]
  prog = program([proc_("read_motion_consumer", [x], body)])
  verify_program(prog)
  assert fuse_elementwise(prog) is prog


def test_buffer_analysis_keeps_arguments_with_unspecified_access_modes() -> None:
  from scaly.passes.program._common import buffer_refs, prune_dead_buffers

  a = buffer("a", dtypes.float64, (4,), address_space="private")
  stmt = p.call("external", [a])
  proc = proc_("invoke", [], [a, stmt])
  verify_program(proc)
  refs = buffer_refs(stmt)
  assert refs.reads == refs.writes == frozenset({"a"})
  assert prune_dead_buffers(proc) is proc
  assert fuse_elementwise(program([proc])).args[0] is proc


def test_fusion_does_not_move_a_read_past_a_write() -> None:
  x, y = (buffer(name, dtypes.float64, (1,)) for name in ("x", "y"))
  a = buffer("a", dtypes.float64, (1,), address_space="private")
  i = var("i")
  at = lambda buf: view(buf, [i])
  body = [
    a,
    for_(range_("i", 0, 1), [store(at(a), mul(load(at(x)), const_float(2.0)))]),
    store(view(x, [const_int(0)]), const_float(3.0)),
    for_(range_("i", 0, 1), [store(at(y), add(load(at(a)), const_float(1.0)))]),
  ]
  result = fuse_elementwise(program([proc_("read_motion", [x, y], body)])).args[0]
  assert any(stmt.op == ProgramOp.BUFFER and stmt.attrs["name"] == "a" for stmt in result.args)


# --- unit-loop unrolling ---------------------------------------------------------


def test_unit_loop_unrolls_scalar_elementwise_output() -> None:
  @sc.function(sc.group(sc.arg("x", 1), sc.arg("y", 1)), outputs=sc.arg("out0"), name="scalar_add")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return x + y

  _compute, _aliases, loops = _classify(_main_body(f))
  assert loops == []


def test_unit_loop_pass_removes_empty_and_substitutes_only_value() -> None:
  x = buffer("x", dtypes.float64, (1,))
  y = buffer("y", dtypes.float64, (5,))
  i = var("i")
  j = var("j")
  empty = for_(range_("j", 4, 4), [store(view(y, [j]), const_float(2.0))])
  one = for_(range_("i", 3, 4), [store(view(y, [i]), const_float(1.0)), empty])
  prog = program([proc_("unit", [x, y], [one])])

  lowered_proc = unroll_unit_loops(prog).args[0]
  body = list(lowered_proc.args[int(lowered_proc.attrs["param_count"]) :])
  assert len(body) == 1
  assert body[0].op == ProgramOp.STORE
  assert body[0].args[0].args[0].op == ProgramOp.CONST_INT
  assert body[0].args[0].args[0].attrs["value"] == 3


def test_fold_after_unit_unroll_reads_nonuniform_constant_table() -> None:
  table = ProgramNode(
    ProgramOp.BUFFER,
    (),
    {"name": "table", "shape": (2,), "address_space": "constant", "values": (2.0, 7.0)},
    dtypes.float64,
  )
  y = buffer("y", dtypes.float64, (1,))
  i = var("i")
  loop = for_(range_("i", 1, 2), [store(view(y, [const_int(0)]), load(view(table, [i])))])
  result = fold_arith(unroll_unit_loops(program([proc_("constant_unit", [y], [table, loop])]))).args[0]
  stores = [node for node in result.args if node.op == ProgramOp.STORE]
  assert stores[0].args[1].op == ProgramOp.CONST_FLOAT
  assert stores[0].args[1].attrs["value"] == 7.0


def test_procedure_pruning_keeps_entry_calls_and_solver_oracles_in_order() -> None:
  leaf = proc_("leaf", [], [])
  dead = proc_("dead", [], [])
  oracle = proc_("oracle", [], [])
  call = ProgramNode(ProgramOp.CALL, (), {"callee": "leaf", "n_in": 0, "n_out": 0, "returns": ()})
  entry = proc_("entry", [], [call])
  prog = ProgramNode(ProgramOp.PROGRAM, (leaf, dead, oracle, entry), {"proc_count": 4, "solver_oracles": {"solver": ("oracle",)}})
  result = prune_procedures(prog)
  assert [proc.attrs["name"] for proc in result.args] == ["leaf", "oracle", "entry"]


def test_generated_name_reserves_raw_and_c_identifier_collisions() -> None:
  from scaly.utils.names import c_ident

  used = {c_ident(name) for name in ("split-name", "split_name_2")}
  assert allocated_name("split_name", used) == "split_name_3"


def test_generated_names_do_not_rescan_the_namespace(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.passes.program import _common

  calls = 0
  c_ident = _common.c_ident

  def counted(name: str) -> str:
    nonlocal calls
    calls += 1
    return c_ident(name)

  monkeypatch.setattr(_common, "c_ident", counted)
  spellings = {f"input_{i}" for i in range(1000)}
  for i in range(100):
    assert allocated_name(f"output_{i}", spellings) == f"output_{i}"
  assert calls <= 200


def test_dead_buffer_removal_analyzes_each_statement_once(monkeypatch: pytest.MonkeyPatch) -> None:
  from scaly.passes.program import _common

  a = buffer("a", dtypes.float64, (4,), address_space="private")
  stmt = store(view(a, [const_int(0)]), const_float(1.0))
  calls: list[ProgramNode] = []
  buffer_refs = _common.buffer_refs

  def counted(node: ProgramNode, aliases: dict[str, str] | None = None):
    calls.append(node)
    return buffer_refs(node, aliases)

  monkeypatch.setattr(_common, "buffer_refs", counted)
  _common.prune_dead_buffers(proc_("references", [], [a, stmt]))
  assert calls == [stmt]


def test_hoist_reserves_exported_names_across_repeated_loop_variables() -> None:
  a, x, y = (buffer(name, dtypes.float64, (1,)) for name in ("a", "x", "y"))
  temp = buffer("temp", dtypes.float64, (1,), address_space="private")
  zero = const_int(0)
  sin_a = ProgramNode(ProgramOp.SIN, (load(view(a, [zero])),), dtype=dtypes.float64)
  callee_body = [temp, store(view(temp, [zero]), sin_a), store(view(y, [zero]), mul(load(view(temp, [zero])), load(view(x, [zero]))))]
  callee = ProgramNode(
    ProgramOp.PROC,
    (a, x, y, *callee_body),
    {**proc_("mapped", [a, x, y], callee_body).attrs, "input_count": 2, "scalarize_mode": "disabled"},
  )
  za, zx, zy = (buffer(name, dtypes.float64, (2,)) for name in ("za", "zx", "zy"))
  call = lambda: ProgramNode(
    ProgramOp.CALL,
    (view(za, [zero]), view(zx, [var("it")]), view(zy, [var("it")])),
    {"callee": "mapped", "n_in": 2, "n_out": 1, "returns": ()},
  )
  root_body = [for_(range_("it", 0, 2), [call()]), for_(range_("it", 0, 2), [call()])]
  root_proc = ProgramNode(ProgramOp.PROC, (za, zx, zy, *root_body), {**proc_("root", [za, zx, zy], root_body).attrs, "input_count": 2})
  result = hoist_invariant(program([callee, root_proc]))
  root_result = result.args[-1]
  names = [node.attrs["name"] for node in root_result.args if node.op == ProgramOp.BUFFER and node.attrs.get("address_space") == "private"]
  assert names == ["it_temp", "it_temp_2"]


# --- contiguous-slice aliasing --------------------------------------------------


def test_contiguous_slice_aliases_source() -> None:
  """A contiguous slice becomes a zero-copy pointer alias (no copy loop), unlike a strided one."""

  @sc.function(sc.arg("x", 8), outputs=sc.arg("out0"), name="slc")
  def f(x: sc.Expr) -> sc.Expr:
    s = x[2:6]
    return (s * s).sin()  # use s twice so it stays materialized (not inlined) -> visible alias

  _compute, aliases, _loops = _classify(_main_body(f))
  assert len(aliases) == 1
  assert aliases[0].attrs["alias_of"] == "x"
  assert aliases[0].attrs["alias_offset"] == 2


def test_strided_slice_is_not_aliased() -> None:
  @sc.function(sc.arg("x", 8), outputs=sc.arg("out0"), name="strided")
  def f(x: sc.Expr) -> sc.Expr:
    return x[::2] + x[1::2]  # strided -> no contiguous offset -> no alias

  _compute, aliases, _loops = _classify(_main_body(f))
  assert aliases == []


# --- workspace packing / spilling -----------------------------------------------


def test_no_spill_when_temps_small() -> None:
  @sc.function(sc.group(sc.arg("x", 8), sc.arg("y", 8)), outputs=sc.arg("out0"), name="small")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, y = inputs
    return (x.sin() + y).tanh()

  assert _sz_w(f) == 0  # everything stays on the stack


def test_packing_reuses_slots_and_spills() -> None:
  """Two disjoint-lifetime 1600-element matmul temps share ONE spilled slot, so sz_w is 1600
  (not 3200) — proving both lifetime slot-reuse and the >= 1024 spill-to-w[] threshold."""

  @sc.function(sc.group(sc.arg("A", (40, 40)), sc.arg("B", (40, 40))), outputs=sc.arg("out0"), name="spill")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    A, B = inputs
    c = (A @ B).sum()  # C (1600) lives only until this reduce
    d = (B @ A).sum()  # D (1600) is born after C is dead -> reuses C's slot
    return c + d

  assert 1600 >= WORKSPACE_SPILL_THRESHOLD
  assert _sz_w(f) == 1600


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the numeric workspace collision check")
def test_workspace_slots_do_not_collide_with_output_names() -> None:
  @sc.function(sc.arg("x", 4), outputs=sc.group(sc.arg("s1"), sc.arg("s2")), name="slot_collision")
  def f(x):
    matrix = x.reshape((2, 2))
    square = matrix @ matrix
    cube = square @ matrix
    return square + cube, square - cube

  data = np.arange(1.0, 5.0)
  square_ref = data.reshape((2, 2)) @ data.reshape((2, 2))
  cube_ref = square_ref @ data.reshape((2, 2))
  got_sum, got_difference = f(data)
  np.testing.assert_allclose(got_sum, square_ref + cube_ref)
  np.testing.assert_allclose(got_difference, square_ref - cube_ref)


def test_call_output_does_not_reuse_slot_that_produced_input() -> None:
  """Regression for the macOS Apple-clang inline-callee miscompile.

  The bad packed shape was ``inner_fwd2_y_x_raw(s0, s2, s1)``: ``s2`` was computed from
  ``s1``, then the call output reused ``s1``. That lifetime reuse is legal IR, but an older
  Apple clang rematerialized through the clobbered slot after inlining. The packer now keeps
  call-input producer closures live through the call, so the output lands in a distinct slot
  while raw callees can remain inlineable.
  """

  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="inner")
  def inner(x):
    return x.sin() + x * x

  @sc.function(sc.arg("z", 2), outputs=sc.arg("y"), name="outer")
  def outer(z):
    return inner(z * z)

  jf = outer.factory("J", ["z"], [sc.factory.Jac("y", "z")])

  source = render_program_c_source(jf)
  declaration = re.search(r"static __attribute__\(\(noinline\)\) void (inner_fwd2\w+_raw)\(", source)
  assert declaration is not None
  match = re.search(re.escape(declaration.group(1)) + r"\(([^)]*)\);", source)
  assert match is not None
  args = [a.strip() for a in match.group(1).split(",")]
  assert args[:3] == ["s0", "s3", "s4"]


def test_regular_raw_callees_stay_inline() -> None:
  @sc.function(sc.arg("x", 2), outputs=sc.arg("y"), name="inner")
  def inner(x):
    return x.sin()

  @sc.function(sc.arg("z", 2), outputs=sc.arg("out"), name="outer")
  def outer(z):
    return inner(z) + 1.0

  source = render_program_c_source(outer)
  assert "static inline void inner_raw" in source
  assert "noinline" not in source


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler for the numeric spill check")
def test_spilled_function_matches_numpy() -> None:
  """End-to-end: a function whose temporaries spill to w[] still computes correctly (the JIT
  allocates w from the rendered sz_w and passes it through)."""

  @sc.function(sc.group(sc.arg("A", (40, 40)), sc.arg("B", (40, 40))), outputs=sc.arg("out0"), name="spill_num")
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    A, B = inputs
    return ((A @ B) + (B @ A)).sum()

  assert _sz_w(f) >= WORKSPACE_SPILL_THRESHOLD  # the (40,40) matmul temp spills
  render_program_c_source(f)  # loud coverage gate
  f.recompile()
  rng = np.random.default_rng(0)
  a, b = rng.standard_normal((40, 40)), rng.standard_normal((40, 40))
  got = f((a, b))
  ref = ((a @ b) + (b @ a)).sum()
  np.testing.assert_allclose(np.asarray(got).reshape(-1), np.asarray(ref).reshape(-1), rtol=1e-9, atol=1e-10)


def test_optimized_program_still_verifies() -> None:
  """The pass pipeline output must pass the Program IR verifier (lower_function asserts this)."""

  @sc.function(sc.arg("x", 6), outputs=sc.arg("out0"), name="verif")
  def f(x: sc.Expr) -> sc.Expr:
    m = x.reshape((2, 3))
    return (m @ x[:3]).sin() + x[3:5]  # (2,3)@(3,) -> (2,), + x[3:5] (2,)

  prog = lower_function(f)  # raises VerifyError on any malformed node
  assert prog.op == ProgramOp.PROGRAM


@pytest.mark.parametrize("count", [2, 8])
def test_slice_gradient_combines_pads(count: int) -> None:
  @sc.function(sc.arg("x", 4 * count), outputs=sc.arg("cost"), name="slice_cost")
  def f(x):
    return sum(((x[2 * i : 2 * i + 3] ** 2).sum() for i in range(count)), start=sc.const(0.0))

  grad = f.factory("slice_grad", ["x"], [sc.factory.Grad("cost", "x")])
  stages = {}
  lower_function(grad, observe=lambda name, prog: stages.__setitem__(name, prog))
  combined = main_proc(stages["pass:combine_scatter_sums"])
  buffers, _, _ = _classify(list(combined.args[int(combined.attrs["param_count"]) :]))
  assert not [b for b in buffers if b.attrs["shape"] == (4 * count,)]
  combined_loops = [
    loop
    for loop in combined.args[int(combined.attrs["param_count"]) :]
    if loop.op == ProgramOp.FOR and loop.args[1].op == ProgramOp.STORE and loop.args[1].args[0].attrs["buffer"] == "grad_cost_x"
  ]
  assert len(combined_loops) == count + 1
  compute, _, loops = _classify(_main_body(grad))
  assert not [b for b in compute if b.attrs["shape"] == (4 * count,)]
  output_loops = [loop for loop in loops if loop.args[1].op == ProgramOp.STORE and loop.args[1].args[0].attrs["buffer"] == "grad_cost_x"]
  assert output_loops
  assert sum(loop.args[1].args[1].op == ProgramOp.CONST_FLOAT for loop in loops if len(loop.args) == 2 and loop.args[1].op == ProgramOp.STORE) == 1
  data = np.linspace(-1.0, 2.0, 4 * count)
  expected = np.zeros_like(data)
  for i in range(count):
    expected[2 * i : 2 * i + 3] += 2 * data[2 * i : 2 * i + 3]
  np.testing.assert_allclose(grad(data), expected)


@pytest.mark.parametrize("shared", [False, True])
def test_scatter_sum_preserves_overlaps_and_shared_outputs(shared: bool) -> None:
  from scaly.ir.expr import scatter

  output_tree = sc.group(sc.arg("sum"), sc.arg("a")) if shared else sc.arg("sum")

  @sc.function(sc.arg("x", 3), outputs=output_tree, name="scatter_sum")
  def f(x):
    a = scatter(x, [1, 3, 5], 8)
    b = scatter(2 * x, [3, 5, 7], 8)
    c = scatter(-x, [0, 3, 7], 8)
    return (a + (b + c), a) if shared else a + (b + c)

  data = np.array([-1.5, 2.0, 0.25])
  expected = np.zeros(8)
  expected[[1, 3, 5]] += data
  expected[[3, 5, 7]] += 2 * data
  expected[[0, 3, 7]] -= data
  result = f(data)
  np.testing.assert_allclose(result[0] if shared else result, expected)
  if shared:
    original = np.zeros(8)
    original[[1, 3, 5]] = data
    np.testing.assert_array_equal(result[1], original)


@pytest.mark.parametrize("alias", [False, True])
def test_scatter_sum_does_not_move_source_reads_past_writes(alias: bool) -> None:
  from scaly.ir.expr import scatter
  from scaly.ir.program import ProgramNode, const_int

  @sc.function(sc.arg("x", 4), outputs=sc.arg("sum"), name="mutated_scatter")
  def f(x):
    a = scatter(x[:2] if alias else x, [0, 2] if alias else [0, 2, 4, 6], 8)
    b = scatter(x[2:] if alias else 2 * x, [1, 3] if alias else [1, 3, 5, 7], 8)
    return a + b

  stages = {}
  lower_function(f, observe=lambda name, prog: stages.__setitem__(name, prog))
  lowered = stages["lowered"]
  original = main_proc(lowered)
  params = list(original.args[: int(original.attrs["param_count"])])
  body = list(original.args[len(params) :])
  body.insert(-1, store(view(params[0], [const_int(0)]), const_float(42.0)))
  mutated = ProgramNode(ProgramOp.PROC, (*params, *body), original.attrs, original.dtype)
  prog = program([mutated])
  assert combine_scatter_sums(prog) is prog


@pytest.mark.parametrize("left_associated", [False, True])
def test_scatter_sum_preserves_addition_grouping(left_associated: bool) -> None:
  from scaly.ir.expr import scatter

  @sc.function(sc.arg("x", 3), outputs=sc.arg("sum"), name="grouped_scatter")
  def f(x):
    a, b, c = (scatter(x[i : i + 1], [1], 4) for i in range(3))
    return (a + b) + c if left_associated else a + (b + c)

  np.testing.assert_array_equal(f(np.array([1e16, -1e16, 1.0])), [0, 1 if left_associated else 0, 0, 0])


def test_scatter_sum_combines_reshaped_scatters() -> None:
  """The chain Hessian sums flat scatters through a reshape: same size, different declared shape."""
  from scaly.ir.expr import scatter

  index_sets = [[0, 3, 6], [1, 4, 7], [2, 5, 6], [0, 1, 2]]

  @sc.function(sc.arg("x", 3), outputs=sc.arg("sum"), name="reshaped_scatter_sum")
  def f(x):
    terms = [scatter((i + 1) * x, idx, 8).reshape((2, 4)) for i, idx in enumerate(index_sets)]
    return sum(terms[1:], start=terms[0])

  _, _, loops = _classify(_main_body(f))
  zero_fills = [
    loop for loop in loops if len(loop.args) == 2 and loop.args[1].op == ProgramOp.STORE and loop.args[1].args[1].op == ProgramOp.CONST_FLOAT
  ]
  assert len(zero_fills) == 1
  data = np.array([-1.5, 2.0, 0.25])
  expected = np.zeros(8)
  for i, idx in enumerate(index_sets):
    expected[idx] += (i + 1) * data
  np.testing.assert_allclose(f(data), expected.reshape(2, 4))


@pytest.mark.parametrize("shared", [False, True])
def test_scatter_sum_combines_accumulate_loops(shared: bool) -> None:
  output_tree = sc.group(sc.arg("sum"), sc.arg("a")) if shared else sc.arg("sum")

  @sc.function(sc.arg("x", 5), outputs=output_tree)
  def f(x):
    a = sc.scatter(x, [1, 3, 1, 1, 0], 8)
    b = sc.scatter(2 * x, [2, 4, 2, 4, 6], 8)
    c = sc.scatter(-x, [0, 3, 7, 5, 2], 8)
    return (a + b + c, a) if shared else a + b + c

  stages = {}
  lower_function(f, observe=lambda name, prog: stages.__setitem__(name, prog))
  before = main_proc(stages["lowered"])
  after = main_proc(stages["pass:combine_scatter_sums"])
  if not shared:
    before_buffers, _, _ = _classify(list(before.args[int(before.attrs["param_count"]) :]))
    after_buffers, _, _ = _classify(list(after.args[int(after.attrs["param_count"]) :]))
    assert len(after_buffers) < len(before_buffers)
    loops = [s for s in after.args[int(after.attrs["param_count"]) :] if s.op == ProgramOp.FOR]
    assert sum(loop.args[1].args[1].op == ProgramOp.CONST_FLOAT for loop in loops) == 1

  data = np.array([-1.5, 2.0, 0.25, 0.5, 3.0])
  terms = []
  for ids, values in (([1, 3, 1, 1, 0], data), ([2, 4, 2, 4, 6], 2 * data), ([0, 3, 7, 5, 2], -data)):
    term = np.zeros(8)
    np.add.at(term, ids, values)
    terms.append(term)
  result = f(data)
  np.testing.assert_array_equal(result[0] if shared else result, terms[0] + terms[1] + terms[2])
  if shared:
    np.testing.assert_array_equal(result[1], terms[0])


def test_scatter_lowering_marks_repeats_as_reductions() -> None:
  from scaly.ir.program import RangeKind

  for ids, kind in (([3, 1, 7, 0], RangeKind.GLOBAL), ([3, 1, 3, 0], RangeKind.REDUCE)):

    @sc.function(sc.arg("x", 4), outputs=sc.arg("y"))
    def f(x):
      return sc.scatter(x, ids, 8)

    stages = {}
    lower_function(f, observe=lambda name, prog: stages.__setitem__(name, prog))
    proc = main_proc(stages["lowered"])
    loops = [s for s in proc.args[int(proc.attrs["param_count"]) :] if s.op == ProgramOp.FOR]
    assert loops[-1].args[0].attrs["kind"] == kind
    assert loops[-1].args[1].args[1].op == (ProgramOp.ADD if kind == RangeKind.REDUCE else ProgramOp.LOAD)


# --- loop-invariant hoisting ---------------------------------------------------------


def _mapped_mlp(stages: int, *, broadcast: bool) -> tuple[sc.Function, np.ndarray, np.ndarray]:
  """``sin(exp(W) @ x_i)`` per stage; ``W`` is one broadcast matrix or a fresh one per stage."""

  @sc.function(sc.group(sc.arg("x", 3), sc.arg("w", 9)), outputs=sc.arg("y"), name="hoist_stage")
  def stage(inputs):
    x, w = inputs
    return (w.reshape((3, 3)).exp() @ x).sin()

  @sc.function(
    sc.group(sc.arg("z", 3 * stages), sc.arg("weights", 9 if broadcast else 9 * stages)), outputs=sc.arg("y"), name=f"hoist_map_{broadcast}"
  )
  def fn(inputs):
    z, weights = inputs
    return _mapped_call(stage, stages, {"x": (z, 0, 3), "w": (weights, 0, 0 if broadcast else 9)})

  weights = as_concrete(fn).inputs[1]
  rng = np.random.default_rng(3)
  return fn, rng.normal(size=3 * stages), rng.normal(size=weights.shape[0])


def _proc_names(fn: sc.Function) -> list[str]:
  prog = lower_function(fn)
  return [pr.attrs["name"] for pr in prog.args[: int(prog.attrs["proc_count"])]]


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_hoist_moves_broadcast_argument_work_before_the_mapped_loop() -> None:
  fn, z, w = _mapped_mlp(5, broadcast=True)
  assert _proc_names(fn) == ["hoist_stage_hoist_1", "hoist_map_True"]
  body = [s for s in _main_body(fn) if s.op != ProgramOp.BUFFER]
  calls = [s for s in body if s.op == ProgramOp.CALL]
  loops = [s for s in body if s.op == ProgramOp.FOR]
  assert [c.attrs["callee"] for c in calls] == ["hoist_stage_hoist_1"] and body.index(calls[0]) < body.index(loops[0])
  assert not any(n.op == ProgramOp.CALL for n in walk_program(loops[0]))
  prologue, hoisted = lower_function(fn).args[:2]
  ops = lambda proc: {n.op for stmt in proc.args[int(proc.attrs["param_count"]) :] for n in walk_program(stmt)}
  assert ProgramOp.EXP in ops(prologue) and ProgramOp.EXP not in ops(hoisted)
  assert ProgramOp.SIN in ops(hoisted) and ProgramOp.SIN not in ops(prologue)
  expected = np.sin(np.exp(w.reshape(3, 3)) @ z.reshape(5, 3).T).T.reshape(-1)
  np.testing.assert_allclose(fn((z, w)), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_hoist_leaves_per_trip_arguments_in_the_loop() -> None:
  fn, z, w = _mapped_mlp(5, broadcast=False)
  assert _proc_names(fn) == ["hoist_map_False"]
  expected = np.sin(np.einsum("sij,sj->si", np.exp(w.reshape(5, 3, 3)), z.reshape(5, 3))).reshape(-1)
  np.testing.assert_allclose(fn((z, w)), expected, rtol=1e-14, atol=1e-14)


@pytest.mark.skipif(not _HAVE_CC, reason="no C compiler")
def test_hoist_names_each_invariant_position_set_of_one_callee() -> None:
  """The same callee mapped once with ``w`` broadcast and once with ``x`` broadcast gets two distinct splits."""

  @sc.function(sc.group(sc.arg("x", 3), sc.arg("w", 9)), outputs=sc.arg("y"), name="hoist_two")
  def stage(inputs):
    x, w = inputs
    return (w.reshape((3, 3)).exp() @ x.exp()).sin()

  @sc.function(sc.group(sc.arg("z", 12), sc.arg("weights", 36)), outputs=sc.arg("y"), name="hoist_two_map")
  def fn(inputs):
    z, weights = inputs
    first = _mapped_call(stage, 4, {"x": (z, 0, 3), "w": (weights, 0, 0)})
    second = _mapped_call(stage, 4, {"x": (z, 0, 0), "w": (weights, 0, 9)})
    return first + second

  assert _proc_names(fn) == ["hoist_two_hoist_1", "hoist_two_hoist_0", "hoist_two_map"]
  zv, wv = np.random.default_rng(5).normal(size=12), np.random.default_rng(6).normal(size=36)
  ew, ex = np.exp(wv.reshape(4, 3, 3)), np.exp(zv.reshape(4, 3))
  expected = np.sin(ew[0] @ ex.T).T + np.sin(np.einsum("sij,j->si", ew, ex[0]))
  np.testing.assert_allclose(fn((zv, wv)), expected.reshape(-1), rtol=1e-14, atol=1e-14)


def test_hoist_refuses_a_buffer_read_between_two_invariant_writes() -> None:
  """``t = a; y = t * x; t = 2a; y += t``: both writes of ``t`` are invariant, but the first read must see the first."""
  a, x, y = (buffer(n, dtypes.float64, (1,)) for n in ("a", "x", "y"))
  t = buffer("t", dtypes.float64, (1,), address_space="private")
  at = lambda b: view(b, [const_int(0)])
  body = [
    t,
    store(at(t), load(at(a))),
    store(at(y), mul(load(at(t)), load(at(x)))),
    store(at(t), mul(const_float(2.0), load(at(a)))),
    store(at(y), add(load(at(y)), load(at(t)))),
  ]
  callee = ProgramNode(
    ProgramOp.PROC,
    (a, x, y, *body),
    {**proc_("twice", [a, x, y], body).attrs, "input_count": 2, "lowering": "auto", "scalarize_mode": "inline"},
  )
  za, zx, zy = (buffer(n, dtypes.float64, (4,)) for n in ("za", "zx", "zy"))
  rng = range_("it", 0, 4)
  call = ProgramNode(
    ProgramOp.CALL,
    (view(za, [const_int(0)]), view(zx, [var("it")]), view(zy, [var("it")])),
    {"callee": "twice", "n_in": 2, "n_out": 1, "returns": ()},
  )
  root = ProgramNode(ProgramOp.PROC, (za, zx, zy, for_(rng, [call])), {**proc_("root", [za, zx, zy], []).attrs, "input_count": 2})
  prog = program([callee, root])
  assert hoist_invariant(prog) is prog


@pytest.mark.parametrize("hint,mode", [("auto", "inline"), ("scalar", "procedure"), ("block", "disabled"), ("opaque", "disabled")])
def test_hoist_prunes_before_scalarization_and_preserves_prologue_policy(hint: Lowering, mode: str) -> None:
  @sc.function(sc.group(sc.arg("x", 1), sc.arg("w", 1)), outputs=sc.arg("y"), name="policy_stage")
  def stage(inputs):
    x, w = inputs
    return (x * w.sin()).with_lowering(hint)

  @sc.function(sc.group(sc.arg("z", 3), sc.arg("w", 1)), outputs=sc.arg("y"), name="policy_root")
  def root(inputs):
    z, w = inputs
    return _mapped_call(stage, 3, [(z, 0, 1), (w, 0, 0)])

  stages = {}
  lower_function(root, observe=lambda name, prog: stages.__setitem__(name, prog))
  procs = stages["pass:hoist_invariant"].args
  assert stage.name not in {proc.attrs["name"] for proc in procs}
  prologue = next(proc for proc in procs if proc.attrs["name"].startswith("policy_stage_hoist_"))
  assert prologue.attrs["scalarize_mode"] == mode


def test_hoist_recognizes_generated_pure_callees_in_nested_maps() -> None:
  @sc.function(sc.group(sc.arg("x", 1), sc.arg("w", 1)), outputs=sc.arg("y"), name="nested_inner")
  def inner(inputs):
    x, w = inputs
    return x * w.sin()

  @sc.function(sc.group(sc.arg("z", 2), sc.arg("w", 1)), outputs=sc.arg("y"), name="nested_middle")
  def middle(inputs):
    z, w = inputs
    return _mapped_call(inner, 2, [(z, 0, 1), (w, 0, 0)]).block()

  @sc.function(sc.group(sc.arg("z", 6), sc.arg("w", 1)), outputs=sc.arg("y"), name="nested_root")
  def root(inputs):
    z, w = inputs
    return _mapped_call(middle, 3, [(z, 0, 2), (w, 0, 0)])

  stages = {}
  lower_function(root, observe=lambda name, prog: stages.__setitem__(name, prog))
  assert any(proc.attrs.get("hoisted_from") == middle.name for proc in stages["pass:hoist_invariant"].args)
  values = np.arange(6.0)
  np.testing.assert_allclose(root((values, np.array([0.3]))), values * np.sin(0.3))


def test_hoist_positions_and_existing_procedure_names_do_not_collide() -> None:
  @sc.function(
    sc.group(sc.group(*(sc.arg(f"x{i}", 1) for i in range(8))), sc.group(*(sc.arg(f"x{i}", 1) for i in range(8, 13)))),
    outputs=sc.arg("y"),
    name="position_stage",
  )
  def stage(inputs):
    return sum((x.sin() for group in inputs for x in group), sc.const(0.0)).block()

  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"), name="position_stage_hoist_1_2")
  def other(x):
    return x.cos().block()

  @sc.function(
    sc.group(sc.group(*(sc.arg(f"z{i}", 3) for i in range(8))), sc.group(*(sc.arg(f"z{i}", 3) for i in range(8, 13)))),
    outputs=sc.arg("y"),
    name="position_root",
  )
  def root(inputs):
    outer = (*inputs[0], *inputs[1])
    mapped = [_mapped_call(stage, 3, [(z, 0, 0 if i in fixed else 1) for i, z in enumerate(outer)]) for fixed in [(1, 2), (12,)]]
    return other(outer[0][:1]) + mapped[0] + mapped[1]

  procs = lower_function(root).args
  names = [proc.attrs["name"] for proc in procs]
  assert len(names) == len(set(names))
  assert other.name in names
  assert "position_stage_hoist_1_2_2" in names
  assert "position_stage_hoisted_1_2" in names and "position_stage_hoisted_12" in names
  values = tuple(np.arange(3.0) / 4 + i / 10 for i in range(13))
  expected = np.full(3, np.cos(values[0][0]))
  for fixed in [(1, 2), (12,)]:
    expected += sum(np.sin(z[0] if i in fixed else z) for i, z in enumerate(values))
  np.testing.assert_allclose(root((values[:8], values[8:])), expected)


def test_hoist_keeps_unknown_call_outputs_inside_the_map() -> None:
  a, x, y = (buffer(name, dtypes.float64, (1,)) for name in ("a", "x", "y"))
  temp = buffer("temp", dtypes.float64, (1,), address_space="private")
  at = lambda b: view(b, [const_int(0)])
  opaque = ProgramNode(ProgramOp.CALL, (a, temp), {"callee": "external_solver", "n_in": 1, "n_out": 1, "returns": ()})
  raw = proc_("opaque_stage", [a, x, y], [temp, opaque, store(at(y), mul(load(at(temp)), load(at(x))))])
  stage = ProgramNode(raw.op, raw.args, {**raw.attrs, "input_count": 2, "lowering": "block", "scalarize_mode": "disabled"}, raw.dtype)
  z, out = (buffer(name, dtypes.float64, (3,)) for name in ("z", "out"))
  call = ProgramNode(
    ProgramOp.CALL, (a, view(z, [var("i")]), view(out, [var("i")])), {"callee": "opaque_stage", "n_in": 2, "n_out": 1, "returns": ()}
  )
  root = proc_("opaque_root", [a, z, out], [for_(range_("i", 0, 3), [call])])
  prog = program([stage, root])
  assert hoist_invariant(prog) is prog


def test_scatter_sum_preserves_repeated_leaf_grouping() -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("sum"))
  def f(x):
    return sc.scatter(x[:1], [1], 4) + sc.scatter(x[1:], [1, 1], 4)

  np.testing.assert_array_equal(f(np.array([1e16, -1e16, 1.0])), [0, 0, 0, 0])
