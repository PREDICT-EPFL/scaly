"""Derived Functions built by AD inherit the primal callee's effective lowering hint."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.ad.forward import jvp, jvp_many
from alloy.ad.reverse import vjp
from alloy.ir.program import ProgramNode
from alloy.ir.types import Lowering
from alloy.passes.lowering import lower_function


def _callees(prog: ProgramNode) -> list[ProgramNode]:
  return list(prog.args[:-1])


@pytest.mark.parametrize("hint", ["scalar", "block"])
def test_derived_procs_inherit_stage_hint(hint: Lowering) -> None:
  x = al.sym("x", 3)
  stage = al.Function._from_exprs("hint_stage", [x], [(x.sin() * (x @ al.const(np.ones(3)))).with_lowering(hint)], ["x"], ["y"])
  length = 4
  z = al.sym("z", 3 * length)
  lam = al.sym("lam", 3 * length)
  seed = al.sym("seed", 3 * length)
  seeds = al.sym("seeds", (2, 3 * length))
  mapped = al.vmap(stage, length, [(z, 0, 3)])
  grad = vjp((mapped,), (z,), (lam,))[0]
  outputs = [
    grad,
    jvp_many(grad, z, al.const(np.tile(np.eye(3), (2, length)))),
    jvp_many(mapped, z, seeds),
    jvp(mapped, z, seed),
  ]
  fn = al.Function._from_exprs("hint_chain", [z, lam, seed, seeds], outputs, ["z", "lam", "seed", "seeds"], ["g", "h", "jm", "j"])
  callees = _callees(lower_function(fn))
  names = [str(proc.attrs["name"]) for proc in callees]
  assert any("_adj0_0" in n and "fwd" not in n for n in names)
  assert any("_adj0_0_fwd6c" in n for n in names)
  assert any("_fwd2j" in n for n in names)
  assert any(n.endswith("_fwd0_0") for n in names)
  assert all(proc.attrs["lowering"] == hint for proc in callees)
  assert all(bool(proc.attrs.get("scalarized")) is (hint == "scalar") for proc in callees)
