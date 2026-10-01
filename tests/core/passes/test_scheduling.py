"""``passes.program.scheduling``: what a C expression would compute only under a condition is computed
before it when it loads, so the C compiler sees a select and not a branch around a load."""

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
INDEX = p.var("i")
ZERO = p.const_float(0.0)


def _at(buf: ProgramNode) -> ProgramNode:
  return p.load(p.view(buf, [INDEX]))


def _scheduled(value: ProgramNode) -> tuple[list[ProgramNode], ProgramNode]:
  declarations, (root,) = schedule_values((value,), ScalarNameAllocator(set()), ahead=True)
  return [d.args[0] for d in declarations], root


def test_a_selects_branches_are_computed_before_it() -> None:
  branch = p.mul(_at(X), p.sub(_at(Y), p.const_float(1.0)))
  named, root = _scheduled(p.select(_at(MASK), branch, ZERO))
  assert named == [branch]  # the mask is the condition, loaded where it stands; the constant needs no name
  assert root.op == ProgramOp.SELECT and root.args[1].op == ProgramOp.VAR and root.args[2] is ZERO
  named, root = _scheduled(p.select(_at(MASK), ZERO, branch))  # the other side
  assert named == [branch] and root.args[1] is ZERO and root.args[2].op == ProgramOp.VAR
  # Both branches, and a select inside a branch: once its own branch has a name it holds no load.
  doubled = p.mul(_at(Y), p.const_float(2.0))
  inner = p.select(p.compare(ProgramOp.LT, p.var("t", dtypes.float64), ZERO), doubled, ZERO)
  named, root = _scheduled(p.select(_at(MASK), _at(X), inner))
  assert named == [_at(X), doubled]
  assert root.args[1].op == ProgramOp.VAR and root.args[2].op == ProgramOp.SELECT and root.args[2].args[1].op == ProgramOp.VAR


@pytest.mark.parametrize("kind", ["store", "store_pair", "assign"])
def test_straight_line_code_keeps_its_branches(kind: str) -> None:
  """Outside a loop there is nothing to vectorize, and the branch skips the work; in a loop every
  kind of statement has its select's branch computed ahead."""
  x, y, mask, out = (
    p.buffer(name, dtype, (8,)) for name, dtype in (("x", dtypes.float64), ("y", dtypes.float64), ("k", dtypes.bool_), ("o", dtypes.float64))
  )
  at = lambda buf, i: p.load(p.view(buf, [i]))  # noqa: E731
  value = lambda i: p.select(at(mask, i), p.div(at(x, i), at(y, i)), ZERO)  # noqa: E731

  def statement(i: ProgramNode) -> ProgramNode:
    if kind == "store":
      return p.store(p.view(out, [i]), value(i))
    if kind == "store_pair":
      return p.store_pair(p.view(out, [p.mul(i, p.const_int(2))]), value(i), ZERO)
    return p.assign("t", value(i), dtypes.float64, declare=True)

  body = [statement(p.const_int(0)), p.for_(p.range_("i", 1, 4), [statement(INDEX)])]
  straight, loop = prepare_scalar_expressions(p.program([p.proc("both", [x, y, mask, out], body)])).args[0].args[4:]
  selected = lambda stmt: stmt.args[0] if kind == "assign" else stmt.args[1]  # noqa: E731
  assert straight.op != ProgramOp.FOR and selected(straight).args[1].op == ProgramOp.DIV
  declared, looped = loop.args[1:]
  assert declared.op == ProgramOp.ASSIGN and declared.args[0].op == ProgramOp.DIV and selected(looped).args[1].op == ProgramOp.VAR
  assert schedule_values((value(INDEX),), ScalarNameAllocator(set()))[0] == []  # the scalarizer's call


def test_a_branch_without_a_load_stays_in_the_select() -> None:
  t = p.var("t", dtypes.float64)
  branch = p.mul(t, p.const_float(2.0))
  named, root = _scheduled(p.select(p.compare(ProgramOp.LT, t, ZERO), branch, ZERO))
  assert named == [] and root.args[1] is branch


def test_the_right_operand_of_a_short_circuit_is_computed_first() -> None:
  below = p.compare(ProgramOp.LT, _at(X), p.const_float(1e-16))
  for op in (ProgramOp.AND, ProgramOp.OR):
    named, root = _scheduled(ProgramNode(op, (_at(MASK), below), dtype=dtypes.bool_))
    assert named == [below] and root.args[0].op == ProgramOp.LOAD and root.args[1].op == ProgramOp.VAR
    # The left operand is always evaluated: nothing to name.
    named, root = _scheduled(ProgramNode(op, (below, p.var("flag", dtypes.bool_)), dtype=dtypes.bool_))
    assert named == [] and root.args[0] is below


@pytest.mark.parametrize("why", ["a libm call", "an integer division", "an integer remainder", "a float to an integer"])
def test_what_could_fault_or_cost_stays_under_its_condition(why: str) -> None:
  """Skipped by C where the condition fails: a libm call is as slow as the branch, a zero divisor
  traps on x86, and a float outside an integer's range has no defined conversion."""
  if why == "a libm call":
    branch, other = ProgramNode(ProgramOp.SIN, (_at(X),), dtype=dtypes.float64), ZERO
  elif why == "an integer division":
    branch, other = p.div(p.const_int(7), _at(N)), p.const_int(0)
  elif why == "an integer remainder":
    branch, other = p.mod(p.const_int(7), _at(N)), p.const_int(0)
  else:
    branch, other = p.cast(_at(X), dtypes.int64), p.const_int(0)
  named, root = _scheduled(p.select(_at(MASK), branch, other))
  assert named == [] and root.args[1] is branch
  # Around it too: the sum holds the branch that must stay.
  named, root = _scheduled(p.select(_at(MASK), p.add(branch, branch.args[0] if why == "a libm call" else other), other))
  assert all(n.op == ProgramOp.LOAD for n in named) and root.args[1].op != ProgramOp.VAR


def test_a_float_division_is_computed_first() -> None:
  branch = p.div(p.const_float(1.0), _at(X))
  named, root = _scheduled(p.select(_at(MASK), branch, ZERO))
  assert named == [branch] and root.args[1].op == ProgramOp.VAR


def test_a_shared_libm_value_already_named_does_not_hold_its_branch_back() -> None:
  """The sine is used twice, so it has a name and is computed anyway: what is left of the branch
  is a product with a load."""
  sine = ProgramNode(ProgramOp.SIN, (_at(X),), dtype=dtypes.float64)
  branch = p.mul(sine, _at(Y))
  named, root = _scheduled(p.add(p.select(_at(MASK), branch, ZERO), sine))
  assert named[0] is sine and named[-1].op == ProgramOp.MUL and root.args[0].args[1].op == ProgramOp.VAR


def test_a_masked_reciprocal_renders_as_a_value_and_a_select() -> None:
  """``where(mask, 1 / x, 0)``: the reciprocal of every element, a zero's too, then the select;
  the infinities are discarded as NumPy discards them."""
  keep = np.arange(64) % 3 != 0
  x = sc.sym("x", 64)
  fn = sc.Function.from_exprs("masked_reciprocal", [x], [sc.where(keep, 1.0 / x, 0.0).block()], ["x"], ["y"])
  source = render_c_source(fn)
  assert re.search(r"double (v\d+) = \(1\.0 / [^;?]+\);\s+[^;?]+ = \([^;?]+ \? \1 : 0\.0\);", source), source
  values = np.where(keep, np.linspace(0.5, 2.0, 64), 0.0)
  np.testing.assert_array_equal(np.asarray(fn(values)), np.where(keep, 1.0 / np.where(keep, values, 1.0), 0.0))
