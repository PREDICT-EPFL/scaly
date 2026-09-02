from __future__ import annotations

import pytest

from typing_playground.trees import G, L, _G


def test_leaf_shapes() -> None:
  assert L("x", 3).shapes == ((3,),) and L("P", (2, 2)).shapes == ((2, 2),) and L("s", ()).shapes == ((),)
  with pytest.raises(TypeError, match="unresolved"):
    L("f", ...).shapes


def test_group_widths_and_nesting() -> None:
  assert G(L("a", 1), L("b", 1)).names == ("a", "b")
  eight = G(*(L(f"x{i}", 1) for i in range(8)))
  assert eight.size == 8
  with pytest.raises(TypeError, match="2 to 8"):
    _G((L("a", 1),))
  with pytest.raises(TypeError, match="nest for more"):
    _G(tuple(L(f"x{i}", 1) for i in range(9)))
  nested = G(G(L("a", 1), L("b", 1)), L("c", 1))
  assert nested.names == ("a", "b", "c") and nested.size == 3


def test_duplicate_names_rejected_within_a_tree() -> None:
  with pytest.raises(ValueError, match="duplicate"):
    G(L("x", 3), L("x", 3))
  with pytest.raises(ValueError, match="duplicate"):
    G(G(L("x", 3), L("y", 3)), L("x", 1))


def test_symbols_follow_the_structure() -> None:
  syms = G(G(L("a", 1), L("b", 2)), L("c", ())).symbols()
  assert isinstance(syms, tuple) and isinstance(syms[0], tuple)
  assert syms[0][1].name == "b" and syms[0][1].shape == (2,) and syms[1].shape == ()


def test_relabel_and_with_shapes_keep_structure_and_holes() -> None:
  t = G(L("u", 2), L("s", ...))
  assert t.has_holes and t.relabel("lam:").names == ("lam:u", "lam:s") and t.relabel("lam:").has_holes
  assert t.with_shapes(((2,), (1,))).shapes == ((2,), (1,))
  assert isinstance(t.with_shapes(((2,), (1,))), _G)


def test_resolved_checks_count_and_declared_shapes() -> None:
  t = G(L("u", 2), L("s", ...))
  assert t.resolved(((2,), (5,))) == ((2,), (5,))
  with pytest.raises(TypeError, match="declared 2 leaves"):
    t.resolved(((2,),))
  with pytest.raises(TypeError, match=r"'u' declared with shape \(2,\)"):
    t.resolved(((3,), (5,)))
