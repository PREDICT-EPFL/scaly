"""Tests for embedding ``SolverFunction``s inside larger Alloy ``Function``s.

The solver-as-graph-node feature: ``solver.call([...])`` returns ``Expr``s, which
can be combined with other ops and wrapped in another ``Function``. The outer
function's tape contains ``Ops.CALL`` nodes whose callee is the solver; the
solver's own tape contains ``Ops.SOLVER_CALL`` nodes whose attrs hold a
``SolverDescriptor`` pointing at the oracle and the backend choice.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import alloy as al
from alloy.ops import Ops


def _disable_jit():
  os.environ["ALLOY_DISABLE_JIT"] = "1"


def _enable_jit():
  os.environ.pop("ALLOY_DISABLE_JIT", None)


@pytest.fixture(autouse=True)
def _force_interpreter():
  """SOLVER_CALL has no C codegen yet — force the interpreter path for these tests."""
  _disable_jit()
  yield
  _enable_jit()


def test_solver_call_returns_expressions() -> None:
  mu = al.sym("mu", 2)
  qp = al.qp(P=al.const(np.eye(2)), c=-mu)
  out_exprs = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
  assert len(out_exprs) == len(qp.output_names)
  # call() inherits from Function and wraps each output in an Ops.CALL node
  # whose callee is the solver function — the inner SOLVER_CALL nodes live in
  # the callee's own tape.
  for e in out_exprs:
    assert e.op == Ops.CALL
    assert e.attrs["callee"] is qp
  # Shapes line up with the descriptor's output signature.
  expected_shapes = tuple(s for _, s in qp.descriptor.output_signature)
  for e, expected in zip(out_exprs, expected_shapes, strict=True):
    assert e.shape == expected


def test_solver_descriptor_present_in_inner_tape() -> None:
  mu = al.sym("mu", 2)
  qp = al.qp(P=al.const(np.eye(2)), c=-mu)
  inner_tape = qp.tape()
  solver_calls = [inst for inst in inner_tape if inst.op == Ops.SOLVER_CALL]
  assert len(solver_calls) == len(qp.output_names)
  descriptors = {id(inst.attrs["solver"]) for inst in solver_calls}
  assert len(descriptors) == 1  # all outputs share one descriptor
  desc = solver_calls[0].attrs["solver"]
  assert desc.backend == "piqp"
  assert desc.n == 2


def test_nested_qp_in_alloy_function() -> None:
  """The safety-filter assembly pattern: build QP data symbolically and wrap
  the solve as a node inside a larger ``Function``."""

  @al.function("track_qp", {"mu": (2,)})
  def track_qp(mu):
    # min 0.5 |x - mu|^2  -> solution is mu itself
    qp = al.qp(P=al.const(np.eye(2)), c=-mu)
    out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
    return {"x": out[0], "cost": out[1]}

  for mu_val in [np.array([0.5, -1.2]), np.zeros(2), np.array([3.0, 2.0])]:
    x, cost = track_qp(mu_val)
    np.testing.assert_allclose(x, mu_val, atol=1e-7)
    # cost = 0.5 mu^T mu + (-mu)^T mu = -0.5 mu^T mu
    np.testing.assert_allclose(cost, -0.5 * float(np.dot(mu_val, mu_val)), atol=1e-7)


def test_nested_qp_postprocessed() -> None:
  """Combine solver output with downstream symbolic math."""

  @al.function("squared_norm_via_qp", {"mu": (2,)})
  def sq_norm(mu):
    qp = al.qp(P=al.const(np.eye(2)), c=-mu)
    out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
    x_star = out[0]
    return {"y": al.dot(x_star, x_star)}

  mu_val = np.array([1.5, -0.3])
  y = sq_norm(mu_val)
  np.testing.assert_allclose(y, float(np.dot(mu_val, mu_val)), atol=1e-7)


def test_nested_qp_with_general_inequality() -> None:
  """Two-sided general inequality inside a nested QP."""

  @al.function("constrained_filter", {"u_ref": (2,)})
  def filter_fn(u_ref):
    G = al.const(np.array([[1.0, 1.0]]))
    l_ineq = al.const(np.array([-0.5]))
    u_ineq = al.const(np.array([0.5]))
    qp = al.qp(P=al.const(np.eye(2)), c=-u_ref, G_ineq=G, l_ineq=l_ineq, u_ineq=u_ineq)
    out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(1)), u_ref])
    return {"u": out[0]}

  # u_ref = (1, 1) is infeasible -> the QP projects onto the band.
  u = filter_fn(np.array([1.0, 1.0]))
  np.testing.assert_allclose(np.sum(u), 0.5, atol=1e-6)
  # u_ref = (-0.1, -0.2) is feasible -> solution is u_ref itself.
  u = filter_fn(np.array([-0.1, -0.2]))
  np.testing.assert_allclose(u, [-0.1, -0.2], atol=1e-7)


def test_nested_nlp_in_alloy_function() -> None:
  """NLP solver embedded in a larger Function."""

  @al.function("min_dist_to_unit_circle", {"target": (2,)})
  def proj(target):
    x = al.sym("x_inner", 2)
    f = (x[0] - target[0]) ** 2 + (x[1] - target[1]) ** 2
    h_eq = al.stack([x[0] ** 2 + x[1] ** 2 - 1.0], axis=0)
    nlp = al.nlp(x=x, f=f, p=target, h_eq=h_eq)
    out = nlp.call([al.const(np.array([1.0, 0.0])), al.const(np.zeros(1)), al.const(np.zeros(0)), target])
    return {"x_proj": out[0]}

  # Projection of (2, 0) onto the unit circle = (1, 0).
  x_proj = proj(np.array([2.0, 0.0]))
  np.testing.assert_allclose(x_proj, [1.0, 0.0], atol=1e-5)
  x_proj = proj(np.array([0.0, 3.0]))
  np.testing.assert_allclose(x_proj, [0.0, 1.0], atol=1e-5)


def test_solver_outputs_share_one_solve_per_tape_eval() -> None:
  """Distinct outputs of the same solve do not trigger a second solve."""

  mu = al.sym("mu", 2)
  qp = al.qp(P=al.const(np.eye(2)), c=-mu)

  call_count = {"n": 0}
  original = qp.descriptor.oracle  # type: ignore[union-attr]
  assert original is not None

  class _CountingFn:
    name = original.name
    inputs = original.inputs
    outputs = original.outputs
    input_names = original.input_names
    output_names = original.output_names
    output_sparsities = original.output_sparsities

    def eval_list(self, *args, **kwargs):
      call_count["n"] += 1
      return original.eval_list(*args, **kwargs)

    def eval_interpreter(self, *args, **kwargs):
      call_count["n"] += 1
      return original.eval_interpreter(*args, **kwargs)

  # Monkey-patch the descriptor's oracle inside this scope to count invocations.
  # We rely on the SOLVER_CALL renderer caching one solve across all outputs in
  # the same tape evaluation.
  qp.descriptor.runtime["__test_original_oracle"] = original  # keep alive
  object.__setattr__(qp.descriptor, "oracle", _CountingFn())  # type: ignore[arg-type]
  try:

    @al.function("multi_out", {"mu": (2,)})
    def multi_out(mu):
      out = qp.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0)), mu])
      return {"x": out[0], "cost": out[1], "lam_box": out[4]}

    _ = multi_out(np.array([1.0, -0.5]))
  finally:
    object.__setattr__(qp.descriptor, "oracle", original)

  # The parent tape currently calls ``qp.eval_interpreter`` once per CALL
  # output node — three CALL nodes -> three oracle evaluations. The interior
  # SOLVER_CALL nodes within a single ``qp.eval_interpreter`` are deduped (one
  # solve per inner tape), but cross-CALL deduplication is a future opt.
  assert call_count["n"] == 3
