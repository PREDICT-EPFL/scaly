"""Pass slots: an extension inserts a pass relative to a named one, and it runs there, under its own
name, on every lowering; anchors and names are checked."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

import scaly as sc
from scaly.ir.program import ProgramNode
from scaly.passes import program as pipeline_module
from scaly.passes.lowering import lower_function
from scaly.passes.program import PASS_PIPELINE, insert_after, insert_before, pipeline


@pytest.fixture
def scratch_slots() -> Iterator[None]:
  saved = list(pipeline_module._INSERTED)
  yield
  pipeline_module._INSERTED[:] = saved


def test_an_inserted_pass_runs_at_its_slot(scratch_slots: None) -> None:
  seen: list[str] = []

  def counting(prog: ProgramNode) -> ProgramNode:
    seen.append("ran")
    return prog

  insert_after("fuse_elementwise", "probe_after", counting)
  insert_before("probe_after", "probe_before", counting)
  names = [name for name, _ in pipeline()]
  assert names[names.index("fuse_elementwise") + 1 : names.index("fuse_elementwise") + 3] == ["probe_before", "probe_after"]
  assert [name for name, _ in PASS_PIPELINE] == [n for n in names if not n.startswith("probe")]
  x = sc.sym("x", 3)
  observed: list[str] = []
  lower_function(sc.Function.from_exprs("slot_probe", [x], [x.sin()], ["x"], ["y"]), observe=lambda name, _: observed.append(name))
  assert seen == ["ran", "ran"]
  assert observed[observed.index("pass:fuse_elementwise") + 1 : observed.index("pass:fuse_elementwise") + 3] == [
    "pass:probe_before",
    "pass:probe_after",
  ]


def test_anchors_and_names_are_checked(scratch_slots: None) -> None:
  with pytest.raises(ValueError, match="no pass 'nope'"):
    insert_after("nope", "probe", lambda prog: prog)
  with pytest.raises(ValueError, match="already in the pipeline"):
    insert_after("scalarize", "fold_arith", lambda prog: prog)
