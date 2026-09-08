"""The shared rewrite driver in ``alloy.ir.match``: sharing, fixpoint, revisit, termination."""

from __future__ import annotations

import pytest

import alloy as al
from alloy.ir.expr import ExprOp
from alloy.ir.match import Pattern, rewrite
from alloy.ir.program import ProgramNode, ProgramOp, add, buffer, const_float, load, var, view
from alloy.ir.types import dtypes
from alloy.passes.program._common import rebuild_program


def test_shared_subgraph_is_rewritten_once_and_stays_shared() -> None:
  x = al.sym("x", 2)
  shared = x * 1.0
  root = shared + shared
  calls: list[al.Expr] = []

  def drop_one(e: al.Expr) -> al.Expr:
    calls.append(e)
    return e.args[0]

  out = rewrite(root, [Pattern(ExprOp.MUL, lambda e: e.args[1].op == ExprOp.CONST, drop_one)])
  assert out.args[0] is x and out.args[1] is x
  assert len(calls) == 1


def test_fixpoint_retries_per_node_and_single_application_does_not() -> None:
  x = al.sym("x", 2)
  root = -(-(-x))
  cancel = Pattern(ExprOp.NEG, lambda e: e.args[0].op == ExprOp.NEG, lambda e: e.args[0].args[0])
  assert rewrite(root, [cancel]) is -x
  assert rewrite(root, [cancel]) is rewrite(root, [cancel], fixpoint=False)
  peel = Pattern(ExprOp.NEG, lambda e: True, lambda e: e.args[0])
  assert rewrite(root, [peel]) is x
  # Children are rewritten first, so single application peels each level once: three levels gone.
  assert rewrite(root, [peel], fixpoint=False) is x
  assert rewrite(-x, [peel], fixpoint=False) is x


def test_revisit_walks_into_replacements_in_the_program_dialect() -> None:
  a = buffer("a", dtypes.float64, (2,))
  b = buffer("b", dtypes.float64, (2,))
  i = var("i")
  producers = {"a": add(load(view(b, [i])), const_float(1.0)), "b": const_float(2.0)}
  inline = Pattern(ProgramOp.LOAD, lambda n: n.args[0].attrs["buffer"] in producers, lambda n: producers[n.args[0].attrs["buffer"]])
  once = rewrite(load(view(a, [i])), [inline], rebuild=rebuild_program, fixpoint=False)
  assert once.args[0].op == ProgramOp.LOAD
  full = rewrite(load(view(a, [i])), [inline], rebuild=rebuild_program, fixpoint=False, revisit=True)
  assert full == add(const_float(2.0), const_float(1.0))


def test_program_rebuild_keeps_untouched_subtrees_shared() -> None:
  a = buffer("a", dtypes.float64, (2,))
  i, j = var("i"), var("j")
  root = add(load(view(a, [i])), load(view(a, [j])))
  out = rewrite(root, [Pattern(ProgramOp.VAR, lambda n: n.attrs["name"] == "i", lambda n: var("k"))], rebuild=rebuild_program, fixpoint=False)
  assert out.args[1] is root.args[1]
  assert out.args[0].args[0].args[0] == var("k")


def test_deep_program_chain_does_not_recurse() -> None:
  node: ProgramNode = const_float(0.0)
  for _ in range(5000):
    node = add(node, const_float(1.0))
  out = rewrite(
    node, [Pattern(ProgramOp.CONST_FLOAT, lambda n: n.attrs["value"] == 1.0, lambda n: const_float(2.0))], rebuild=rebuild_program, fixpoint=False
  )
  assert out.args[1] == const_float(2.0)


def test_max_steps_names_the_offending_pattern() -> None:
  x = al.sym("x", 2)

  def grow(e: al.Expr) -> al.Expr:
    return e + 1.0

  with pytest.raises(RuntimeError, match="grow"):
    rewrite(x + 1.0, [Pattern(ExprOp.ADD, lambda e: True, grow)], max_steps=10)


def test_revisit_rejects_a_replacement_that_contains_the_replaced_node() -> None:
  a = buffer("a", dtypes.float64, (2,))
  wrap = Pattern(ProgramOp.LOAD, lambda n: True, lambda n: add(n, const_float(1.0)))
  with pytest.raises(RuntimeError, match="contains the node it replaces"):
    rewrite(load(view(a, [var("i")])), [wrap], rebuild=rebuild_program, fixpoint=False, revisit=True)
