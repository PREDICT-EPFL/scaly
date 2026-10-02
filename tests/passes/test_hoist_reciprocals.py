"""Reciprocal hoisting respects loop execution and mutable divisor dependencies."""

import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.types import dtypes
from scaly.ir.program import walk_program
from scaly.passes.program.hoist_reciprocals import hoist_reciprocals


def _program(stop=6, *, mutation=None, dtype=dtypes.float64):
  x = p.buffer("x", dtype, (6,))
  y = p.buffer("y", dtype, (1,))
  out = p.buffer("out", dtype, (6,))
  divisor = p.load(p.view(y, [p.const_int(0)]))
  statements = [p.store(p.view(out, [p.var("i")]), p.div(p.load(p.view(x, [p.var("i")])), divisor))]
  if mutation == "store":
    statements.append(p.store(p.view(y, [p.const_int(0)]), p.load(p.view(x, [p.var("i")]))))
  elif mutation == "call":
    statements.append(p.call("mutates", [y]))
  loop = p.for_(p.range_("i", 0, stop), statements)
  return p.program([p.proc("reciprocals", [x, y, out], [loop])])


def test_shared_reciprocal_precedes_loop():
  result = hoist_reciprocals(_program())
  proc = result.args[0]
  assignment, loop = proc.args[3:]
  assert assignment.op == ProgramOp.ASSIGN
  assert assignment.args[0].op == ProgramOp.DIV
  assert not any(n.op == ProgramOp.DIV for n in walk_program(loop))
  assert any(n.op == ProgramOp.MUL for n in walk_program(loop))


@pytest.mark.parametrize("stop", [0, p.var("dynamic_stop")])
def test_potentially_empty_loop_does_not_evaluate_reciprocal(stop):
  prog = _program(stop)
  assert hoist_reciprocals(prog) is prog


@pytest.mark.parametrize("mutation", ["store", "call"])
def test_mutated_divisor_stays_in_loop(mutation):
  prog = _program(mutation=mutation)
  assert hoist_reciprocals(prog) is prog


def test_integer_division_is_unchanged():
  prog = _program(dtype=dtypes.int64)
  assert hoist_reciprocals(prog) is prog


def test_alias_store_blocks_hoisting():
  proc = _program().args[0]
  x, y, out, loop = proc.args
  alias = ProgramNode(ProgramOp.BUFFER, (), {**y.attrs, "name": "alias", "alias_of": "y"}, y.dtype)
  write = p.store(p.view(alias, [p.const_int(0)]), p.const_float(2))
  prog = p.program([p.proc("alias_write", [x, y, out], [alias, p.for_(loop.args[0], [*loop.args[1:], write])])])
  assert hoist_reciprocals(prog) is prog


def test_immutable_scalar_divisor_is_resolved_before_loop():
  proc = _program().args[0]
  x, y, out, loop = proc.args
  divisor = p.var("d", dtypes.float64)
  store = p.store(p.view(out, [p.var("i")]), p.div(p.load(p.view(x, [p.var("i")])), divisor))
  body = [p.assign("d", p.const_float(2), declare=True), store]
  prog = p.program([p.proc("scalar_write", [x, y, out], [p.for_(loop.args[0], body)])])
  result = hoist_reciprocals(prog).args[0]
  prologue, loop = result.args[3:]
  assert prologue.args[0].args[1] is p.const_float(2)
  assert not any(n.op == ProgramOp.DIV for n in walk_program(loop))


def test_empty_nested_loop_does_not_escape_outer_loop():
  proc = _program(0).args[0]
  x, y, out, inner = proc.args
  prog = p.program([p.proc("nested", [x, y, out], [p.for_(p.range_("outer", 0, 3), [inner])])])
  assert hoist_reciprocals(prog) is prog


@pytest.mark.parametrize("reciprocal", [False, True])
@pytest.mark.parametrize("mapped", [False, True])
def test_compiled_reciprocal_policy(tmp_path, reciprocal, mapped):
  import ctypes
  import shutil
  import subprocess

  import numpy as np
  import scaly as sc
  from scaly.codegen.c import render_program_c
  from scaly.passes.lowering import lower_function
  from scaly.utils.env import shared_lib_ext, shared_lib_flag

  compiler = shutil.which("cc")
  if compiler is None:
    pytest.skip("C compiler required")

  @sc.function(sc.arg("x", 2), sc.arg("y", 1), outputs=sc.arg("out", 2), name="reciprocal_stage")
  def stage(x: sc.Expr, y: sc.Expr) -> sc.Expr:
    return x / y

  @sc.function(sc.arg("x", 6), sc.arg("y", 1), outputs=sc.arg("out", 6), name="reciprocal_policy")
  def fun(x: sc.Expr, y: sc.Expr) -> sc.Expr:
    return sc.vmap(stage, 3)(x, sc.broadcast(y)).vec() if mapped else x / y

  program = lower_function(fun, reciprocal=reciprocal)
  if mapped and reciprocal:
    loops = [n for n in program.args[-1].args if n.op == ProgramOp.FOR]
    assert loops and not any(n.op == ProgramOp.DIV for loop in loops for n in walk_program(loop))
  source = tmp_path / "policy.c"
  library = tmp_path / ("policy" + shared_lib_ext())
  source.write_text(render_program_c(program, fun.instantiate()))
  subprocess.run([compiler, "-O3", "-fPIC", shared_lib_flag(), str(source), "-lm", "-o", str(library)], check=True)
  entry = ctypes.CDLL(str(library)).reciprocal_policy
  ptr = ctypes.POINTER(ctypes.c_double)
  entry.argtypes = [ctypes.POINTER(ptr), ctypes.POINTER(ptr), ctypes.POINTER(ctypes.c_int), ptr, ctypes.c_int]
  entry.restype = ctypes.c_int
  values = np.array([0.25, 1, -2, 4, 7, 20.0])
  denominator, out = np.array([3.0]), np.empty(6)
  args = (ptr * 2)(values.ctypes.data_as(ptr), denominator.ctypes.data_as(ptr))
  results = (ptr * 1)(out.ctypes.data_as(ptr))
  assert entry(args, results, None, None, 0) == 0
  expected = values * (1 / denominator) if reciprocal else values / denominator
  np.testing.assert_array_equal(out, expected)
  for magnitude in (1e-310, 1e308):
    values.fill(magnitude)
    denominator.fill(magnitude)
    assert entry(args, results, None, None, 0) == 0
    with np.errstate(over="ignore"):
      expected = values * (1 / denominator) if reciprocal else values / denominator
    np.testing.assert_array_equal(out, expected)


def _scalar_program(definitions, after=(), *, before=(), stop=6):
  original = _program(stop).args[0]
  x, y, out, loop = original.args
  divisor = p.var("d", dtypes.float64)
  store = p.store(p.view(out, [p.var("i")]), p.div(p.load(p.view(x, [p.var("i")])), divisor))
  return p.program([p.proc("scalar_chain", [x, y, out], [*before, p.for_(loop.args[0], [*definitions(x, y), store, *after])])])


def test_transitive_definitions_and_index_offsets_are_resolved():
  def definitions(_x, y):
    return [
      p.assign("offset", p.const_int(0), declare=True),
      p.assign("loaded", p.load(p.view(y, [p.var("offset")])), declare=True),
      p.assign("d", p.mul(p.var("loaded", dtypes.float64), p.const_float(2)), declare=True),
    ]

  result = hoist_reciprocals(_scalar_program(definitions)).args[0]
  prologue, loop = result.args[3:]
  assert prologue.op == ProgramOp.ASSIGN
  assert not any(n.op == ProgramOp.VAR for n in walk_program(prologue))
  assert not any(n.op == ProgramOp.DIV for n in walk_program(loop))


@pytest.mark.parametrize("nested", [False, True])
def test_reassignment_anywhere_in_loop_blocks_scalar_resolution(nested):
  reassigned = p.assign("d", p.const_float(4))
  if nested:
    reassigned = p.for_(p.range_("j", 0, 1), [reassigned])
  prog = _scalar_program(lambda _x, _y: [p.assign("d", p.const_float(2), declare=True)], [reassigned])
  assert hoist_reciprocals(prog) is prog


def test_identical_repeated_assignments_are_counted_separately():
  declaration = p.assign("d", p.const_float(2), declare=True)
  prog = _scalar_program(lambda _x, _y: [declaration, declaration])
  assert hoist_reciprocals(prog) is prog


def test_definition_must_dominate_division():
  prog = _scalar_program(lambda _x, _y: [], [p.assign("d", p.const_float(2))], before=[p.assign("d", p.const_float(3), declare=True)])
  assert hoist_reciprocals(prog) is prog


def test_transitive_loop_varying_definition_is_not_invariant():
  prog = _scalar_program(
    lambda x, _y: [
      p.assign("loaded", p.load(p.view(x, [p.var("i")])), declare=True),
      p.assign("d", p.mul(p.var("loaded", dtypes.float64), p.const_float(2)), declare=True),
    ]
  )
  assert hoist_reciprocals(prog) is prog


def test_mutated_buffer_behind_scalar_definition_is_not_invariant():
  original = _program().args[0]
  y = original.args[1]
  prog = _scalar_program(
    lambda _x, y: [p.assign("d", p.load(p.view(y, [p.const_int(0)])), declare=True)], [p.store(p.view(y, [p.const_int(0)]), p.const_float(2))]
  )
  assert hoist_reciprocals(prog) is prog


def test_empty_loop_does_not_hoist_scalar_definition():
  prog = _scalar_program(lambda _x, y: [p.assign("d", p.load(p.view(y, [p.const_int(0)])), declare=True)], stop=0)
  assert hoist_reciprocals(prog) is prog


def test_definition_inside_nested_scope_is_not_visible_outside():
  prog = _scalar_program(
    lambda _x, _y: [p.for_(p.range_("j", 0, 1), [p.assign("d", p.const_float(2))])], before=[p.assign("d", p.const_float(3), declare=True)]
  )
  assert hoist_reciprocals(prog) is prog
