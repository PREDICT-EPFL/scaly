"""Hand-built carried ranges survive the program optimizations that assume independent trips."""

import pytest

from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.program_spec import verify_program
from scaly.ir.types import dtypes
from scaly.passes.program.fuse_ranges import fuse_ranges
from scaly.passes.program.fuse_elementwise import fuse_elementwise
from scaly.passes.program.fold_arith import fold_arith
from scaly.passes.program.combine_scatter_sums import combine_scatter_sums
from scaly.passes.program.hoist_invariant import hoist_invariant
from scaly.passes.program.scalarize import scalarize_program
from scaly.passes.program.unroll_unit_loops import unroll_unit_loops
from scaly.passes.program.widen_ranges import widen_ranges


def _range(name, count):
  rng = p.range_(name, 0, count)
  return ProgramNode(rng.op, rng.args, {**rng.attrs, "carried": True}, rng.dtype)


def _exit_loop(rng, body):
  loop = p.for_(rng, body)
  return ProgramNode(loop.op, loop.args, {**loop.attrs, "exit_var": True}, loop.dtype)


def _proc(name, params, body, **attrs):
  raw = p.proc(name, params, body)
  return ProgramNode(raw.op, raw.args, {**raw.attrs, **attrs}, raw.dtype)


@pytest.mark.parametrize("hint", ["auto", "scalar"])
@pytest.mark.parametrize("mode", ["procedure", "inline"])
def test_scalarization_refuses_carried_work_and_its_callers(hint, mode):
  x, y = (p.buffer(name, dtypes.float64, (1,)) for name in ("x", "y"))
  at = lambda buf: p.view(buf, [p.const_int(0)])
  loop = p.for_(_range("i", 3), [p.store(at(y), p.mul(p.load(at(y)), p.load(at(x))))])
  callee = _proc("step", [x, y], [p.store(at(y), p.const_float(1)), loop], input_count=1, scalarize_mode=mode, lowering=hint)
  call = ProgramNode(ProgramOp.CALL, (x, y), {"callee": "step", "n_in": 1, "n_out": 1})
  root = _proc("root", [x, y], [call], input_count=1, scalarize_mode="procedure", lowering=hint)
  prog = p.program([callee, root])
  assert scalarize_program(prog) is prog


@pytest.mark.parametrize("count", [0, 1])
def test_unrolling_keeps_exit_variable_even_without_a_break(count):
  out = p.buffer("out", dtypes.int64, (1,))
  loop = _exit_loop(_range("i", count), [])
  prog = p.program([p.proc("root", [out], [loop, p.store(p.view(out, [p.const_int(0)]), p.var("i"))])])
  assert unroll_unit_loops(prog) is prog


@pytest.mark.parametrize("nested", [False, True])
def test_widening_refuses_carried_trips_and_enclosing_ranges(nested):
  x, y = (p.buffer(name, dtypes.float64, (16,)) for name in ("x", "y"))
  i = p.var("i")
  store = p.store(p.view(y, [i]), p.load(p.view(x, [i])))
  loop = p.for_(p.range_("i", 0, 16), [p.for_(_range("j", 1), [store])]) if nested else p.for_(_range("i", 16), [store])
  prog = p.program([p.proc("root", [x, y], [loop])])
  result = widen_ranges(prog, lanes=2)
  assert result is prog
  assert not any(n.op == ProgramOp.RANGE and "lanes" in n.attrs for n in walk_program(result))


@pytest.mark.parametrize("carried", ["producer", "consumer"])
def test_range_fusion_refuses_carried_seeds_and_consumers(carried):
  x, y = (p.buffer(name, dtypes.float64, (5,)) for name in ("x", "y"))
  tmp = p.buffer("tmp", dtypes.float64, (5,), address_space="private")
  a, b = (p.buffer(name, dtypes.float64, (1,)) for name in ("a", "b"))
  stage = _proc("stage", [a, b], [p.store(p.view(b, [p.const_int(0)]), p.load(p.view(a, [p.const_int(0)])))], input_count=1, scalarized=True)
  i = p.var("i")
  call = ProgramNode(ProgramOp.CALL, (p.view(x, [i]), p.view(tmp, [i])), {"callee": "stage", "n_in": 1, "n_out": 1})
  producer = p.for_(_range("i", 5) if carried == "producer" else p.range_("i", 0, 5), [call])
  consumer = p.for_(_range("i", 5) if carried == "consumer" else p.range_("i", 0, 5), [p.store(p.view(y, [i]), p.load(p.view(tmp, [i])))])
  prog = p.program([stage, _proc("root", [x, y], [tmp, producer, consumer], input_count=1)])
  assert fuse_ranges(prog) is prog


def _hoist_program(*, nested=False, effect=False, scalar=False):
  x, weights, out = (p.buffer(name, dtypes.float64, (1,)) for name in ("x", "weights", "out"))
  temp = p.buffer("temp", dtypes.float64, (1,), address_space="private")
  at = lambda buf: p.view(buf, [p.const_int(0)])
  fixed = p.store(at(temp), p.mul(p.load(at(weights)), p.load(at(weights))))
  result = p.store(at(out), p.mul(p.load(at(temp)), p.load(at(x))))
  body = [temp, fixed, result]
  if nested:
    state = p.buffer("state", dtypes.float64, (1,), address_space="private")
    count = p.buffer("count", dtypes.int64, (1,), address_space="private")
    loop = _exit_loop(_range("j", 3), [p.store(at(state), p.mul(p.load(at(state)), p.load(at(weights))))])
    # The exit variable is also a scalar dependency, invisible to buffer-only invariant analysis.
    result = p.store(p.view(out, [p.sub(p.load(at(count)), p.const_int(3))]), p.mul(p.load(at(state)), p.load(at(x))))
    body = [temp, state, count, fixed, p.store(at(state), p.load(at(temp))), loop, p.store(at(count), p.var("j")), result]
    if scalar:
      carry = p.var("carry", dtypes.float64)
      loop = _exit_loop(_range("j", 3), [p.assign("carry", p.mul(carry, p.load(at(weights))))])
      result = p.store(p.view(out, [p.sub(p.load(at(count)), p.const_int(3))]), p.mul(carry, p.load(at(x))))
      body = [temp, count, fixed, p.assign("carry", p.load(at(temp)), declare=True), loop, p.store(at(count), p.var("j")), result]
  if effect:
    body.insert(1, p.print_(["weight=", "\n"], [p.load(at(weights))]))
  stage = _proc("stage", [x, weights, out], body, input_count=2)
  xs, ys = (p.buffer(name, dtypes.float64, (4,)) for name in ("xs", "ys"))
  call = ProgramNode(ProgramOp.CALL, (p.view(xs, [p.var("i")]), weights, p.view(ys, [p.var("i")])), {"callee": "stage", "n_in": 2, "n_out": 1})
  loop = _exit_loop(_range("i", 4), [call])
  root = _proc("root", [xs, weights, ys], [loop], input_count=2)
  return p.program([stage, root]), loop


def test_hoisting_a_pure_prologue_preserves_the_carried_loop_attributes():
  prog, loop = _hoist_program()
  result = hoist_invariant(prog)
  verify_program(result)
  final_loop = next(n for n in walk_program(result.args[-1]) if n.op == ProgramOp.FOR)
  assert result is not prog
  assert final_loop.attrs == loop.attrs and final_loop.args[0] is loop.args[0]
  prologue = next(pr for pr in result.args if "_hoist_" in pr.attrs["name"])
  assert any(n.op == ProgramOp.MUL for n in walk_program(prologue))


@pytest.mark.parametrize("scalar", [False, True])
def test_hoisting_keeps_carried_work_and_exit_variable_users_in_the_body(scalar):
  prog, _ = _hoist_program(nested=True, scalar=scalar)
  result = hoist_invariant(prog)
  verify_program(result)
  assert result is not prog
  prologue = next(pr for pr in result.args if "_hoist_" in pr.attrs["name"])
  variant = next(pr for pr in result.args if "_hoisted_" in pr.attrs["name"])
  assert not any(n.op == ProgramOp.RANGE and n.attrs.get("carried") for n in walk_program(prologue))
  assert not any(n.op == ProgramOp.VAR and n.attrs["name"] == "j" for n in walk_program(prologue))
  assert not any(n.op == ProgramOp.ASSIGN and n.attrs["target"] == "carry" for n in walk_program(prologue))
  assert any(n.op == ProgramOp.RANGE and n.attrs.get("carried") for n in walk_program(variant))


def test_hoisting_keeps_effects_in_the_carried_body():
  prog, _ = _hoist_program(effect=True)
  assert hoist_invariant(prog) is prog


@pytest.mark.parametrize("transform", [fuse_elementwise, fold_arith])
def test_producer_optimizations_keep_carried_fills_and_the_exit_count(transform):
  out = p.buffer("out", dtypes.float64, (4,))
  count = p.buffer("count", dtypes.int64, (1,))
  temp = p.buffer("temp", dtypes.float64, (4,), address_space="private")
  i = p.var("i")
  fill = _exit_loop(_range("i", 4), [p.store(p.view(temp, [i]), p.const_float(2))])
  copy = p.for_(p.range_("j", 0, 4), [p.store(p.view(out, [p.var("j")]), p.load(p.view(temp, [p.var("j")])))])
  prog = p.program([p.proc("root", [out, count], [temp, fill, p.store(p.view(count, [p.const_int(0)]), i), copy])])
  assert transform(prog) is prog


def test_index_folding_keeps_carried_coordinates_unsplit():
  out = p.buffer("out", dtypes.int64, (16,))
  i, j = p.var("i"), p.var("j")
  value = p.div(j, p.const_int(2))
  store = p.store(p.view(out, [p.add(p.mul(i, p.const_int(8)), j)]), value)
  carried = _exit_loop(_range("j", 8), [store])
  outer = p.for_(p.range_("i", 0, 2), [carried])
  prog = p.program([p.proc("root", [out], [outer])])
  assert fold_arith(prog) is prog


@pytest.mark.parametrize("which", ["fill", "scatter", "sum"])
def test_scatter_combination_keeps_carried_ranges(which):
  x = p.buffer("x", dtypes.float64, (2,))
  out = p.buffer("out", dtypes.float64, (4,))
  a, b = (p.buffer(name, dtypes.float64, (4,), address_space="private") for name in ("a", "b"))
  i = p.var("i")
  body = [a, b]
  for index, buf in enumerate((a, b)):
    fill_range = _range("i", 4) if which == "fill" else p.range_("i", 0, 4)
    body.append(p.for_(fill_range, [p.store(p.view(buf, [i]), p.const_float(0))]))
    scatter_range = _range("i", 2) if which == "scatter" else p.range_("i", 0, 2)
    body.append(p.for_(scatter_range, [p.store(p.view(buf, [p.add(p.mul(i, p.const_int(2)), p.const_int(index))]), p.load(p.view(x, [i])))]))
  sum_range = _range("i", 4) if which == "sum" else p.range_("i", 0, 4)
  body.append(p.for_(sum_range, [p.store(p.view(out, [i]), p.add(p.load(p.view(a, [i])), p.load(p.view(b, [i]))))]))
  prog = p.program([p.proc("root", [x, out], body)])
  assert combine_scatter_sums(prog) is prog
