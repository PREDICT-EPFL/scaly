from __future__ import annotations

import pytest

from typing_playground.trees import _Group, arg, group


def test_leaf_shapes() -> None:
  assert arg("x", 3).shapes == ((3,),) and arg("P", (2, 2)).shapes == ((2, 2),) and arg("s", ()).shapes == ((),)
  with pytest.raises(TypeError, match="unresolved"):
    arg("f", ...).shapes


def test_group_widths_and_nesting() -> None:
  assert group(arg("a", 1), arg("b", 1)).names == ("a", "b")
  eight = group(*(arg(f"x{i}", 1) for i in range(8)))
  assert eight.size == 8
  assert group(arg("a", 1)).names == ("a",) and group(arg("a", 1)).symbols() == (arg("a", 1).symbols(),)  # a one-element tuple
  with pytest.raises(TypeError, match="1 to 8"):
    _Group(())
  with pytest.raises(TypeError, match="nest for more"):
    _Group(tuple(arg(f"x{i}", 1) for i in range(9)))
  nested = group(group(arg("a", 1), arg("b", 1)), arg("c", 1))
  assert nested.names == ("a", "b", "c") and nested.size == 3


def test_duplicate_names_rejected_within_a_tree() -> None:
  with pytest.raises(ValueError, match="duplicate"):
    group(arg("x", 3), arg("x", 3))
  with pytest.raises(ValueError, match="duplicate"):
    group(group(arg("x", 3), arg("y", 3)), arg("x", 1))


def test_symbols_follow_the_structure() -> None:
  syms = group(group(arg("a", 1), arg("b", 2)), arg("c", ())).symbols()
  assert isinstance(syms, tuple) and isinstance(syms[0], tuple)
  assert syms[0][1].name == "b" and syms[0][1].shape == (2,) and syms[1].shape == ()


def test_relabel_and_with_shapes_keep_structure_and_holes() -> None:
  t = group(arg("u", 2), arg("s", ...))
  assert t.has_holes and t.relabel("lam:").names == ("lam:u", "lam:s") and t.relabel("lam:").has_holes
  assert t.with_shapes(((2,), (1,))).shapes == ((2,), (1,))
  assert isinstance(t.with_shapes(((2,), (1,))), _Group)


def test_resolved_checks_count_and_declared_shapes() -> None:
  t = group(arg("u", 2), arg("s", ...))
  assert t.resolved(((2,), (5,))) == ((2,), (5,))
  with pytest.raises(TypeError, match="declared 2 leaves"):
    t.resolved(((2,),))
  with pytest.raises(TypeError, match=r"'u' declared with shape \(2,\)"):
    t.resolved(((3,), (5,)))
