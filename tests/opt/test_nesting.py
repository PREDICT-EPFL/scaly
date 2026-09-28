"""Structural tests for embedding solver functions in Scaly graphs."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from tests.opt.problem_helpers import build_qp
from scaly.opt.method import REGISTRY
from scaly.ir.expr import ExprOp, topo
from scaly.codegen import render_c_source
from scaly.opt.external.graph import solver_descriptor

pytestmark = pytest.mark.skipif(
  "piqp" not in REGISTRY.installed(), reason="structural tests build the private QP differential fixture and need the scaly-piqp plugin installed"
)


def test_solver_call_returns_expressions() -> None:
  mu = sc.sym("mu", 2)
  qp = build_qp(P=sc.const(np.eye(2)), c=-mu)
  out_exprs = qp.symbolic_call(sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), mu)
  *solution, info = out_exprs  # the solution's four groups, then the solver's Info
  assert len(solution) == 4 and isinstance(info, sc.opt.Info)
  # call() inherits from Function and wraps each output in an ExprOp.CALL node
  # whose callee is the solver function — the inner EXTERN_CALL nodes live in
  # the callee's own expression graph.
  for e in (*solution, info.status, info.iter, info.objective, info.primal_residual):
    assert e.op == ExprOp.CALL
    assert e.attrs["callee"] is qp
  # Shapes line up with the descriptor's output signature.
  expected_shapes = tuple(s for _, s in solver_descriptor(qp).output_signature)
  for e, expected in zip(solution, expected_shapes, strict=True):
    assert e.shape == expected


def test_solver_descriptor_present_in_inner_graph() -> None:
  mu = sc.sym("mu", 2)
  qp = build_qp(P=sc.const(np.eye(2)), c=-mu)
  solver_calls = [node for node in topo(qp.outputs) if node.op == ExprOp.EXTERN_CALL]
  assert len(solver_calls) == len(qp.output_names)
  descriptors = {id(node.attrs["extern"]) for node in solver_calls}
  assert len(descriptors) == 1  # all outputs share one descriptor
  desc = solver_calls[0].attrs["extern"]
  assert desc.backend == "piqp"
  assert desc.n == 2


def test_solver_outputs_share_one_program_ir_call() -> None:
  """Distinct outputs of the same solver invocation lower to one CALL statement."""

  mu = sc.sym("mu", 2)
  qp = build_qp(P=sc.const(np.eye(2)), c=-mu)

  @sc.function(sc.L("mu", (2,)), output=sc.G(sc.L("x", ...), sc.L("lam_box", ...), sc.L("lam_eq", ...)), name="multi_out")
  def multi_out(mu):
    out = qp.symbolic_call(sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), mu)
    return (out[0], out[1], out[2])

  from scaly.passes.lowering import lower_function, main_proc
  from scaly.ir.program import ProgramOp

  calls = [stmt for stmt in main_proc(lower_function(multi_out)).args if stmt.op == ProgramOp.CALL]
  assert len(calls) == 1
  assert calls[0].attrs["callee"] == qp.name


@pytest.mark.solver("piqp")
def test_nested_solver_stats_query_uses_compiled_host_handle() -> None:
  mu = sc.sym("mu", 2)
  qp = build_qp(P=sc.const(np.eye(2)), c=-mu, name="nested_stats_qp")
  out = qp.symbolic_call(sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), mu)
  host = sc.Function.from_exprs("nested_stats_host", [mu], [out[0]], ["mu"], ["x"])
  np.testing.assert_allclose(host(np.array([0.5, -0.25])), [0.5, -0.25], atol=1e-8)
  stats = sc.opt.solver_stats(host, "nested_stats_qp")
  assert stats.version == sc.opt.SCALY_SOLVER_STATS_VERSION
  assert stats.status == sc.Status.OK
  assert stats.n_eval_f == 1


def test_duplicate_nested_solver_names_fail_before_c_compilation() -> None:
  mu = sc.sym("mu", 2)
  qps = [build_qp(P=np.eye(2), c=-mu) for _ in range(2)]
  outs = [qp.symbolic_call(sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), mu) for qp in qps]
  host = sc.Function.from_exprs("duplicate_solver_host", [mu], [outs[0][0], outs[1][0]], ["mu"], ["x0", "x1"])
  with pytest.raises(ValueError, match="duplicate extern symbol 'problem_body_piqp'"):
    render_c_source(host)
