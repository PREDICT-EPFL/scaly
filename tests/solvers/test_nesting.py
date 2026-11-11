"""Structural tests for embedding solver functions in Scaly graphs."""

from __future__ import annotations

import numpy as np
import pytest

import scaly as sc
from tests.solvers.problem_helpers import build_qp
from scaly.solvers.registry import available_backends
from scaly.codegen import render_c_source

pytestmark = pytest.mark.skipif(
  "piqp" not in available_backends(), reason="structural tests build the private QP differential fixture and need the scaly-piqp plugin installed"
)


def test_solver_outputs_share_one_program_ir_call() -> None:
  """Distinct outputs of the same solver invocation lower to one CALL statement."""

  mu = sc.sym("mu", 2)
  qp = build_qp(P=sc.const(np.eye(2)), c=-mu)

  @sc.function(sc.arg("mu", (2,)), outputs=sc.group(sc.arg("x"), sc.arg("lam_box"), sc.arg("lam_eq")), name="multi_out")
  def multi_out(mu: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr]:
    out = qp(mu)
    return (out[0], out[1], out[2])

  from scaly.passes.lowering import lower_function, main_proc
  from scaly.ir.program import ProgramOp

  program = lower_function(multi_out)
  calls = [stmt for stmt in main_proc(program).args if stmt.op == ProgramOp.CALL]
  assert len(calls) == 1
  callee = next(proc for proc in program.args if proc.attrs["name"] == calls[0].attrs["callee"])
  external_calls = [stmt for stmt in callee.args if stmt.op == ProgramOp.CALL]
  assert len(external_calls) == 1 and external_calls[0].attrs["external"]


@pytest.mark.parametrize("backend", ["piqp", "ipopt"])
def test_solver_graph_uses_extern_and_requires_custom_derivative(backend: str) -> None:
  if backend not in available_backends():
    pytest.skip(f"scaly-{backend} is not installed")

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("target", 2), name="extern_tracking")
  def tracking(x: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - target))

  solve = sc.solver(tracking, backend)
  graph = solve.function.instantiate()
  assert all(output.op == sc.ExprOp.EXTERN for output in graph.outputs)
  assert all(isinstance(output.attrs["callee"].library, sc.CLibrary) for output in graph.outputs)
  target = sc.sym("active_target", 2)
  result = solve(target)[0]
  from scaly.ad.sparsity import jacobian_sparsity

  np.testing.assert_array_equal(jacobian_sparsity(result, target).to_mask(), np.ones((2, 2), dtype=bool))
  for derivative in (lambda: sc.jvp(result, target, sc.const(np.ones(2))), lambda: sc.vjp((result,), (target,), (sc.const(np.ones(2)),))):
    with pytest.raises(NotImplementedError, match="sc.custom_derivative"):
      derivative()


@pytest.mark.parametrize(
  "backend", [pytest.param("piqp", marks=pytest.mark.solver("piqp")), pytest.param("ipopt", marks=pytest.mark.solver("ipopt"))]
)
def test_solver_and_user_extern_share_a_function(backend: str) -> None:
  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("target", 2), name="mixed_tracking")
  def tracking(x: sc.Expr, target: sc.Expr) -> sc.ProblemSpec[sc.Expr]:
    return sc.ProblemSpec(minimize=sc.sumsqr(x - target))

  solve = sc.solver(tracking, backend, options={"max_iter": 100})
  twice = sc.extern(
    sc.CLibrary("twice", sources={"twice.c": "void twice(const double* x, double* y, double* w) { y[0]=2*x[0]; y[1]=2*x[1]; (void)w; }"}),
    "twice",
    inputs=(sc.arg("x", 2),),
    outputs=sc.arg("y", 2),
  )

  @sc.function(sc.arg("target", 2), outputs=sc.arg("result", 2), name=f"mixed_{backend}")
  def host(target: sc.Expr) -> sc.Expr:
    return twice(solve(target)[0]) + solve(twice(target))[0]

  target = np.array([0.3, -0.7])
  np.testing.assert_allclose(host(target), 4 * target, atol=1e-6)
  assert host.solver_stats().status == sc.ScalySolveStatus.OK

  @sc.function(sc.arg("target", 2), outputs=sc.arg("result", 2), name=f"alternate_{backend}")
  def alternate(target: sc.Expr) -> sc.Expr:
    return twice(target)

  @sc.function(sc.arg("which", ()), sc.arg("target", 2), outputs=sc.arg("result", 2), name=f"conditional_{backend}")
  def conditional(which: sc.Expr, target: sc.Expr) -> sc.Expr:
    return sc.cond(which > 0, (host, alternate), target)

  from scaly.codegen import render_c_module

  assert len(render_c_module(conditional)._libraries) == 2
  from scaly.codegen.jit import _LINKER_NAMESPACES, _linux_loader
  import sys

  if sys.platform == "linux":
    assert _LINKER_NAMESPACES and all(
      namespace.dependencies and namespace.libraries[0]._name == _linux_loader()[1] for namespace in _LINKER_NAMESPACES
    )


@pytest.mark.solver("piqp")
def test_nested_solver_stats_query_uses_compiled_host_handle() -> None:
  mu = sc.sym("mu", 2)
  qp = build_qp(P=sc.const(np.eye(2)), c=-mu, name="nested_stats_qp")

  @sc.function(sc.arg("mu", (2,)), outputs=sc.arg("x", 2), name="nested_stats_host")
  def host(mu: sc.Expr) -> sc.Expr:
    return qp(mu)[0]

  np.testing.assert_allclose(host(np.array([0.5, -0.25])), [0.5, -0.25], atol=1e-8)
  stats = host.solver_stats("nested_stats_qp")
  assert stats.version == sc.SCALY_SOLVER_STATS_VERSION
  assert stats.status == sc.ScalySolveStatus.OK
  assert stats.n_eval_f == 1


def test_duplicate_nested_solver_names_fail_before_c_compilation() -> None:
  mu = sc.sym("mu", 2)
  qps = [build_qp(P=np.eye(2), c=-mu) for _ in range(2)]

  @sc.function(sc.arg("mu", (2,)), outputs=sc.group(sc.arg("x0"), sc.arg("x1")), name="duplicate_solver_host")
  def host(mu: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return qps[0](mu)[0], qps[1](mu)[0]

  with pytest.raises(ValueError, match="duplicate solver symbol 'problem_body_piqp'"):
    render_c_source(host)
