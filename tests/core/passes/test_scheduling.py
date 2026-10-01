"""``passes.program.scheduling``: in a loop, what a C expression would compute only under a condition
is computed before it when it loads, so the C compiler sees a select and not a branch around a load."""

from __future__ import annotations

import re

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source
from scaly.ir import program as p
from scaly.ir.program import ProgramNode, ProgramOp
from scaly.ir.types import dtypes
from scaly.passes.program.prepare_scalar import prepare_scalar_expressions
from scaly.passes.program.scheduling import ScalarNameAllocator, schedule_values

X = p.buffer("x", dtypes.float64, (16,))
Y = p.buffer("y", dtypes.float64, (16,))
N = p.buffer("n", dtypes.int64, (16,))
MASK = p.buffer("k", dtypes.bool_, (16,))
OUT = p.buffer("o", dtypes.float64, (16,))
Z = p.buffer("z", dtypes.float64, (16,))
INDEX = p.var("i")
ZERO = p.const_float(0.0)


def _at(buf: ProgramNode, index: ProgramNode = INDEX) -> ProgramNode:
  return p.load(p.view(buf, [index]))


def _scheduled(value: ProgramNode) -> tuple[list[ProgramNode], ProgramNode]:
  declarations, (root,) = schedule_values((value,), ScalarNameAllocator(set()), ahead=True)
  return [d.args[0] for d in declarations], root


def _call(op: ProgramOp, *args: ProgramNode) -> ProgramNode:
  return ProgramNode(op, args, dtype=dtypes.float64)


def test_a_selects_loading_branch_is_computed_before_it() -> None:
  branch = p.mul(_at(X), p.sub(_at(Y), p.const_float(1.0)))
  named, root = _scheduled(p.select(_at(MASK), branch, ZERO))
  assert named == [branch]  # the mask is the condition, loaded where it stands; the constant needs no name
  assert root.op == ProgramOp.SELECT and root.args[1].op == ProgramOp.VAR and root.args[2] is ZERO
  named, root = _scheduled(p.select(_at(MASK), ZERO, branch))  # the other side
  assert named == [branch] and root.args[1] is ZERO and root.args[2].op == ProgramOp.VAR
  # A select inside a branch: once its own branch has a name it holds no load, and the outer one follows.
  doubled = p.mul(_at(Y), p.const_float(2.0))
  inner = p.select(p.compare(ProgramOp.LT, p.var("t", dtypes.float64), ZERO), doubled, ZERO)
  named, root = _scheduled(p.select(_at(MASK), _at(X), inner))
  assert named == [doubled, _at(X)]
  assert root.args[1].op == ProgramOp.VAR and root.args[2].op == ProgramOp.SELECT and root.args[2].args[1].op == ProgramOp.VAR


def test_a_select_between_two_working_branches_keeps_them() -> None:
  """Both computed for every element is twice the work, and more for a piecewise function: one with
  a division in each of three pieces ran 1.4x slower that way on data the processor predicts, and
  one with a single loading piece 1.1x."""
  first, second = p.mul(_at(X), p.const_float(2.0)), p.mul(_at(Y), p.const_float(3.0))
  named, root = _scheduled(p.select(_at(MASK), first, second))
  assert named == [] and root.args[1] is first and root.args[2] is second
  x = _at(X)
  below = lambda bound: p.compare(ProgramOp.LT, x, p.const_float(bound))  # noqa: E731
  piece = lambda buf, shift: p.div(_at(buf), p.add(p.const_float(shift), x))  # noqa: E731
  named, root = _scheduled(p.select(below(0.0), piece(Y, 1.0), p.select(below(1.0), piece(OUT, 2.0), piece(Z, 3.0))))
  assert named == [x] and root.args[1].op == ProgramOp.DIV  # x is shared; no piece is computed ahead
  # One piece loads and the other is arithmetic on values already at hand: work all the same.
  named, root = _scheduled(p.select(below(0.0), piece(Y, 1.0), p.mul(x, x)))
  assert named == [x] and root.args[1].op == ProgramOp.DIV


def test_work_is_counted_in_the_branches_as_they_will_be_computed() -> None:
  """A condition is evaluated whatever the branches cost, and a value with a name is computed
  already: neither is work that keeps the other branch from being computed ahead."""
  t = p.var("t", dtypes.float64)
  branch = p.div(_at(X), _at(Y))
  busy = p.compare(ProgramOp.LT, p.mul(t, t), p.sub(t, p.const_float(1.0)))  # arithmetic, in a condition
  named, root = _scheduled(p.select(_at(MASK), branch, p.select(busy, t, ZERO)))
  assert named == [branch] and root.args[2].op == ProgramOp.SELECT
  shared = p.mul(t, p.const_float(2.0))  # used twice: named, so the select's other branch is a variable
  named, root = _scheduled(p.add(p.select(_at(MASK), branch, shared), shared))
  assert named == [shared, branch] and all(arg.op == ProgramOp.VAR for arg in root.args[0].args[1:])


def test_two_loads_under_a_condition_that_varies_are_both_computed_first() -> None:
  named, root = _scheduled(p.select(_at(MASK), p.mul(_at(X), _at(Z)), _at(Y)))
  assert [n.op for n in named] == [ProgramOp.MUL, ProgramOp.LOAD] and all(arg.op == ProgramOp.VAR for arg in root.args[1:])


def test_a_select_on_a_flag_keeps_its_branches() -> None:
  """No variable enters the condition: the C compiler tests it once outside the loop and copies
  one side, which a select of both sides' loads ran 1.1x slower than."""
  flag = p.load(p.view(MASK, [p.const_int(3)]))
  named, root = _scheduled(p.select(flag, _at(X), _at(Y)))
  assert named == [] and root.args[1].op == ProgramOp.LOAD
  named, root = _scheduled(p.select(flag, p.div(_at(X), _at(Y)), ZERO))
  assert named == [] and root.args[1].op == ProgramOp.DIV


def test_a_branch_without_a_load_stays_in_the_select() -> None:
  t = p.var("t", dtypes.float64)
  branch = p.mul(t, p.const_float(2.0))
  named, root = _scheduled(p.select(p.compare(ProgramOp.LT, t, ZERO), branch, ZERO))
  assert named == [] and root.args[1] is branch


def test_an_integer_branch_stays_in_the_select() -> None:
  """An index chosen under a condition (a scatter's entry or its scratch slot) is integer
  arithmetic, which overflows where a float only rounds."""
  entry = p.add(p.mul(_at(N), p.const_int(16)), INDEX)
  named, root = _scheduled(p.select(_at(MASK), entry, p.const_int(0)))
  assert named == [] and root.args[1] is entry


def test_the_right_operand_of_a_short_circuit_is_computed_first() -> None:
  below = p.compare(ProgramOp.LT, _at(X), p.const_float(1e-16))
  for op in (ProgramOp.AND, ProgramOp.OR):
    named, root = _scheduled(ProgramNode(op, (_at(MASK), below), dtype=dtypes.bool_))
    assert named == [below] and root.args[0].op == ProgramOp.LOAD and root.args[1].op == ProgramOp.VAR
    # The left operand is always evaluated: nothing to name.
    named, root = _scheduled(ProgramNode(op, (below, p.var("flag", dtypes.bool_)), dtype=dtypes.bool_))
    assert named == [] and root.args[0] is below


WORK = {
  "a libm call": lambda: _call(ProgramOp.SIN, _at(X)),
  "a square root": lambda: _call(ProgramOp.SQRT, _at(X)),
  "a floor": lambda: _call(ProgramOp.FLOOR, _at(X)),
  "a minimum": lambda: _call(ProgramOp.MINIMUM, _at(X), ZERO),
  "an integer division": lambda: p.cast(p.div(p.const_int(7), _at(N)), dtypes.float64),
  "an integer remainder": lambda: p.cast(p.mod(p.const_int(7), _at(N)), dtypes.float64),
  "a float to an integer": lambda: p.cast(p.cast(_at(X), dtypes.int64), dtypes.float64),
}


@pytest.mark.parametrize("why", WORK)
def test_what_could_fault_or_cost_a_call_stays_under_its_condition(why: str) -> None:
  """Skipped by C where the condition fails: a libm call is as slow as the branch (``sqrt``,
  ``floor`` and ``fmin`` are calls under some compilers and flags), a zero divisor traps on x86,
  and a float outside an integer's range has no defined conversion."""
  branch = WORK[why]()
  named, root = _scheduled(p.select(_at(MASK), branch, ZERO))
  assert named == [] and root.args[1] is branch
  # Around it too: the product holds the work that must stay, and a load of its own.
  product = p.mul(branch, _at(Y))
  named, root = _scheduled(p.select(_at(MASK), product, ZERO))
  assert named == [] and root.args[1] is product


def test_a_float_division_is_computed_first() -> None:
  branch = p.div(p.const_float(1.0), p.add(_call(ProgramOp.ABS, _at(X)), p.neg(_at(Y))))
  named, root = _scheduled(p.select(_at(MASK), branch, ZERO))
  assert named == [branch] and root.args[1].op == ProgramOp.VAR


def test_a_shared_libm_value_already_named_does_not_hold_its_branch_back() -> None:
  """The sine is used twice, so it has a name and is computed anyway: what is left of the branch
  is a product with a load."""
  sine = _call(ProgramOp.SIN, _at(X))
  branch = p.mul(sine, _at(Y))
  named, root = _scheduled(p.add(p.select(_at(MASK), branch, ZERO), sine))
  assert named[0] is sine and named[-1].op == ProgramOp.MUL and root.args[0].args[1].op == ProgramOp.VAR


def _masked(index: ProgramNode) -> ProgramNode:
  return p.select(_at(MASK, index), p.div(_at(X, index), _at(Y, index)), ZERO)


@pytest.mark.parametrize("kind", ["store", "store_pair", "assign"])
def test_straight_line_code_keeps_its_branches(kind: str) -> None:
  """Outside a loop there is nothing to vectorize, and the branch skips the work; in a loop every
  kind of statement has its select's branch computed ahead."""

  def statement(i: ProgramNode) -> ProgramNode:
    if kind == "store":
      return p.store(p.view(OUT, [i]), _masked(i))
    if kind == "store_pair":
      return p.store_pair(p.view(OUT, [p.mul(i, p.const_int(2))]), _masked(i), ZERO)
    return p.assign("t", _masked(i), dtypes.float64, declare=True)

  body = [statement(p.const_int(0)), p.for_(p.range_("i", 1, 4), [statement(INDEX)])]
  straight, loop = prepare_scalar_expressions(p.program([p.proc("both", [X, Y, MASK, OUT], body)])).args[0].args[4:]
  selected = lambda stmt: stmt.args[0] if kind == "assign" else stmt.args[1]  # noqa: E731
  assert straight.op != ProgramOp.FOR and selected(straight).args[1].op == ProgramOp.DIV
  declared, looped = loop.args[1:]
  assert declared.op == ProgramOp.ASSIGN and declared.args[0].op == ProgramOp.DIV and selected(looped).args[1].op == ProgramOp.VAR
  assert schedule_values((_masked(INDEX),), ScalarNameAllocator(set()))[0] == []  # the scalarizer's call


@pytest.mark.parametrize("op", [ProgramOp.ADD, ProgramOp.SUB])
def test_a_running_sum_keeps_its_branch(op: ProgramOp) -> None:
  """``s = s + (mask ? x / y : 0)`` is a chain through ``s`` whatever becomes of the select: the
  branch computed ahead buys nothing and costs what a predicted branch would have skipped. A store
  that reads another element of its target is no such chain."""
  total = p.view(OUT, [p.const_int(0)])
  step = lambda read: p.store(total, ProgramNode(op, (p.load(read), _masked(INDEX)), dtype=dtypes.float64))  # noqa: E731
  loops = [p.for_(p.range_("i", 1, 4), [step(read)]) for read in (total, p.view(OUT, [p.const_int(5)]))]
  summed, other = prepare_scalar_expressions(p.program([p.proc("sums", [X, Y, MASK, OUT], loops)])).args[0].args[4:]
  assert [stmt.op for stmt in summed.args[1:]] == [ProgramOp.STORE]
  assert [stmt.op for stmt in other.args[1:]] == [ProgramOp.ASSIGN, ProgramOp.STORE]


def test_a_masked_reciprocal_renders_as_a_value_and_a_select() -> None:
  """``where(mask, 1 / x, 0)``: the reciprocal of every element, a zero's too, then the select;
  the infinities are discarded as NumPy discards them. Its sum keeps the branch."""
  keep = np.arange(64) % 3 != 0
  x = sc.sym("x", 64)
  masked = sc.where(keep, 1.0 / x, 0.0)
  fn = sc.Function.from_exprs("masked_reciprocal", [x], [masked.block()], ["x"], ["y"])
  assert re.search(r"double (v\d+) = \(1\.0 / [^;?]+\);\s+[^;?]+ = \([^;?]+ \? \1 : 0\.0\);", render_c_source(fn))
  total = sc.Function.from_exprs("masked_reciprocal_sum", [x], [masked.sum().block()], ["x"], ["total"])
  source = render_c_source(total)
  assert re.search(r"\? \(1\.0 / [^;?]+\) : 0\.0\)", source) and not re.search(r"= \(1\.0 / [^;?]+\);", source), source
  values = np.where(keep, np.linspace(0.5, 2.0, 64), 0.0)
  want = np.where(keep, 1.0 / np.where(keep, values, 1.0), 0.0)
  np.testing.assert_array_equal(np.asarray(fn(values)), want)
  np.testing.assert_allclose(float(np.asarray(total(values))), want.sum(), rtol=1e-14)
