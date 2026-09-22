"""Derived Functions built by AD inherit the primal callee's effective lowering hint."""

from __future__ import annotations

import numpy as np
import pytest

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
  @sc.function(sc.L("x", 3), sc.L("y", ...), name="hint_stage")
  def stage(x):
    return (x.sin() * (x @ sc.const(np.ones(3)))).with_lowering(hint)

  length = 4
  inputs = sc.G(sc.L("z", 3 * length), sc.L("lam", 3 * length), sc.L("seed", 3 * length), sc.L("seeds", (2, 3 * length)))
  outputs = sc.G(sc.L("g", ...), sc.L("h", ...), sc.L("jm", ...), sc.L("j", ...))

  @sc.function(inputs, outputs, name="hint_chain")
  def fn(inputs):
    z, lam, seed, seeds = inputs
    mapped = sc.vmap(stage, length, [(z, 0, 3)])
    grad = vjp((mapped,), (z,), (lam,))[0]
    return (
      grad,
      jvp_many(grad, z, sc.const(np.tile(np.eye(3), (2, length)))),
      jvp_many(mapped, z, seeds),
      jvp(mapped, z, seed),
    )

  callees = _callees(lower_function(fn))
  names = [str(proc.attrs["name"]) for proc in callees]
  assert any("_adj0_0" in n and "fwd" not in n for n in names)
  assert any("_adj0_0_fwd6c" in n for n in names)
  assert any("_fwd2j" in n for n in names)
  assert any(n.endswith("_fwd0_0") for n in names)
  assert all(proc.attrs["lowering"] == hint for proc in callees)
  assert all(bool(proc.attrs.get("scalarized")) is (hint == "scalar") for proc in callees)
  assert not [
    (proc.attrs["name"], node)
    for proc in callees
    for node in walk_program(proc)
    if node.op == ProgramOp.RANGE and node.attrs["name"].startswith("c_")
  ]
