"""Derived Functions built by AD inherit the primal callee's effective lowering hint."""

from __future__ import annotations

import re

import numpy as np
import pytest

from scaly.function.sugar import _mapped_call
import scaly as sc
from scaly.ad.forward import jvp, jvp_many
from scaly.ad.reverse import vjp
from scaly.ir.program import ProgramNode, ProgramOp, walk_program
from scaly.ir.types import Lowering
from scaly.passes.lowering import lower_function


def _callees(prog: ProgramNode) -> list[ProgramNode]:
  return list(prog.args[:-1])


@pytest.mark.parametrize("hint", ["scalar", "block"])
def test_derived_procs_inherit_stage_hint(hint: Lowering) -> None:
  @sc.function(sc.arg("x", 3), outputs=sc.arg("y"), name="hint_stage")
  def stage(x):
    return (x.sin() * (x @ sc.const(np.ones(3)))).with_lowering(hint)

  length = 4
  inputs = sc.group(sc.arg("z", 3 * length), sc.arg("lam", 3 * length), sc.arg("seed", 3 * length), sc.arg("seeds", (2, 3 * length)))
  outputs = sc.group(sc.arg("g"), sc.arg("h"), sc.arg("jm"), sc.arg("j"))

  @sc.function(inputs, outputs=outputs, name="hint_chain")
  def fn(inputs):
    z, lam, seed, seeds = inputs
    mapped = _mapped_call(stage, length, [(z, 0, 3)])
    grad = vjp((mapped,), (z,), (lam,))[0]
    return (
      grad,
      jvp_many(grad, z, sc.const(np.tile(np.eye(3), (2, length)))),
      jvp_many(mapped, z, seeds),
      jvp(mapped, z, seed),
    )

  callees = _callees(lower_function(fn))
  names = [str(proc.attrs["name"]) for proc in callees]
  helper = "[0-9a-f]{10}"
  assert sorted(re.sub(helper, "#", n) for n in names) == ["hint_stage_adj_#", "hint_stage_adj_#_fwd_#", "hint_stage_fwd_#", "hint_stage_fwd_#"]
  assert all(proc.attrs["lowering"] == hint for proc in callees)
  assert all(bool(proc.attrs.get("scalarized")) is (hint == "scalar") for proc in callees)
  assert not [
    (proc.attrs["name"], node)
    for proc in callees
    for node in walk_program(proc)
    if node.op == ProgramOp.RANGE and node.attrs["name"].startswith("c_")
  ]
