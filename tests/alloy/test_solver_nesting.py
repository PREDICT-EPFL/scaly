"""Structural tests for embedding solver functions in Alloy graphs."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.solvers.registry import available_backends
from alloy.expr import topo
from alloy.ops import Ops

pytestmark = pytest.mark.skipif("piqp" not in available_backends(), reason="structural tests build al.qp and need the alloy-piqp plugin installed")


def test_solver_call_returns_expressions() -> None:
  mu = al.sym("mu", 2)
  qp = al.qp(P=al.const(np.eye(2)), c=-mu)
  out_exprs = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
  assert len(out_exprs) == len(qp.output_names)
  # call() inherits from Function and wraps each output in an Ops.CALL node
  # whose callee is the solver function — the inner SOLVER_CALL nodes live in
  # the callee's own semantic graph.
  for e in out_exprs:
    assert e.op == Ops.CALL
    assert e.attrs["callee"] is qp
  # Shapes line up with the descriptor's output signature.
  expected_shapes = tuple(s for _, s in qp.descriptor.output_signature)
  for e, expected in zip(out_exprs, expected_shapes, strict=True):
    assert e.shape == expected


def test_solver_descriptor_present_in_inner_graph() -> None:
  mu = al.sym("mu", 2)
  qp = al.qp(P=al.const(np.eye(2)), c=-mu)
  solver_calls = [node for node in topo(qp.outputs) if node.op == Ops.SOLVER_CALL]
  assert len(solver_calls) == len(qp.output_names)
  descriptors = {id(node.attrs["solver"]) for node in solver_calls}
  assert len(descriptors) == 1  # all outputs share one descriptor
  desc = solver_calls[0].attrs["solver"]
  assert desc.backend == "piqp"
  assert desc.n == 2


def test_solver_outputs_share_one_program_ir_call() -> None:
  """Distinct outputs of the same solver invocation lower to one CALL statement."""

  mu = al.sym("mu", 2)
  qp = al.qp(P=al.const(np.eye(2)), c=-mu)

  @al.function("multi_out", {"mu": (2,)})
  def multi_out(mu):
    out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
    return {"x": out[0], "cost": out[1], "lam_box": out[4]}

  from alloy.lowering import lower_function, main_proc
  from alloy.program import POps

  calls = [stmt for stmt in main_proc(lower_function(multi_out)).args if stmt.op == POps.CALL]
  assert len(calls) == 1
  assert calls[0].attrs["callee"] == qp.name
