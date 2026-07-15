"""Sparse PIQP through the generated wrapper (L2: CSC pattern baked at codegen)."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.codegen import render_c_source
from alloy.toolchain import solver_diagnostic, solver_loadable

need_piqp = pytest.mark.skipif(not solver_loadable("piqp"), reason=solver_diagnostic("piqpc"))


def _sparse_problem(sparse: bool, name: str) -> al.SolverFunction:
  """Parameterized QP with structural zeros in P, A_eq, and G_ineq."""
  t = al.sym("t", 2)
  zero = al.const(0.0)
  P = al.stack(
    [
      al.stack([2.0 + t[0] * t[0], t[1], zero, zero]),
      al.stack([t[1], 3.0, zero, zero]),
      al.stack([zero, zero, 1.5 + t[1] * t[1], zero]),
      al.stack([zero, zero, zero, 1.0]),
    ],
    axis=0,
  )
  c = al.stack([t[0], -1.0, 0.5, t[1]])
  A_eq = al.stack([al.stack([1.0, 1.0, zero, zero])], axis=0)
  b_eq = al.stack([1.0 + t[0]])
  G_ineq = al.stack([al.stack([zero, 1.0, zero, -1.0]), al.stack([t[0], zero, 1.0, zero])], axis=0)
  return al.qp(
    P=P,
    c=c,
    A_eq=A_eq,
    b_eq=b_eq,
    G_ineq=G_ineq,
    l_ineq=np.array([-2.0, -3.0]),
    u_ineq=np.array([2.0, 3.0]),
    x_lb=np.full(4, -5.0),
    x_ub=np.full(4, 5.0),
    sparse=sparse,
    name=name,
  )


@need_piqp
def test_sparse_qp_patterns_exclude_structural_zeros() -> None:
  qp = _sparse_problem(sparse=True, name="sparse_pattern_qp")
  desc = qp.descriptor
  assert desc.P_sparsity is not None and desc.A_sparsity is not None and desc.G_sparsity is not None
  # P: upper triangle of {(0,0),(0,1),(1,1),(2,2),(3,3)}; dense triu would be 10.
  assert set(zip(desc.P_sparsity.rows, desc.P_sparsity.cols)) == {(0, 0), (0, 1), (1, 1), (2, 2), (3, 3)}
  assert set(zip(desc.A_sparsity.rows, desc.A_sparsity.cols)) == {(0, 0), (0, 1)}
  assert set(zip(desc.G_sparsity.rows, desc.G_sparsity.cols)) == {(0, 1), (0, 3), (1, 0), (1, 2)}
  # CSC order = identity value permutation (what the generated wrapper assumes).
  for sp in (desc.P_sparsity, desc.A_sparsity, desc.G_sparsity):
    assert sp.to_csc()[2] == tuple(range(sp.nnz))
  # The oracle emits compact value buffers.
  assert desc.oracle is not None
  out_sizes = dict(zip(desc.oracle.output_names, [int(e.size) for e in desc.oracle.outputs], strict=True))
  assert out_sizes["P"] == 5 and out_sizes["A_eq"] == 2 and out_sizes["G_ineq"] == 4


@need_piqp
def test_sparse_qp_renders_baked_csc_tables() -> None:
  qp = _sparse_problem(sparse=True, name="sparse_render_qp")
  source = render_c_source(qp)
  assert "piqp_setup_sparse" in source and "piqp_update_sparse" in source
  assert "static piqp_int P_p[5] = { 0, 1, 3, 4, 5 };" in source
  assert "static piqp_int P_i[5] = { 0, 0, 1, 2, 3 };" in source
  assert "piqp_setup_dense" not in source


@need_piqp
def test_sparse_qp_matches_dense_over_parameter_sweep() -> None:
  sparse_qp = _sparse_problem(sparse=True, name="sparse_parity_qp")
  dense_qp = _sparse_problem(sparse=False, name="dense_parity_qp")
  for tv in (np.array([0.3, -0.7]), np.array([1.1, 0.2]), np.array([-0.5, 0.9])):
    sparse_out = sparse_qp(np.zeros(4), np.zeros(1), np.zeros(2), tv)
    dense_out = dense_qp(np.zeros(4), np.zeros(1), np.zeros(2), tv)
    assert sparse_qp.last_stats is not None and sparse_qp.last_stats.status == al.AlloySolveStatus.OK
    for key in sparse_out:
      np.testing.assert_allclose(sparse_out[key], dense_out[key], rtol=1e-6, atol=1e-6, err_msg=f"output {key} diverges for t={tv}")


@need_piqp
def test_sparse_qp_constant_data_and_stats() -> None:
  qp = al.qp(
    P=np.diag([2.0, 1.0, 4.0]),
    c=np.array([-1.0, 0.5, 0.0]),
    x_lb=np.full(3, -1.0),
    x_ub=np.full(3, 1.0),
    sparse=True,
    name="sparse_const_qp",
  )
  assert qp.descriptor.P_sparsity is not None and qp.descriptor.P_sparsity.nnz == 3
  out = qp(np.zeros(3), np.zeros(0), np.zeros(0))
  np.testing.assert_allclose(out["x"], [0.5, -0.5, 0.0], atol=1e-7)
  stats = qp.last_stats
  assert stats is not None and stats.status == al.AlloySolveStatus.OK
  assert stats.obj == pytest.approx(float(out["cost"]), rel=1e-12, abs=1e-12)
  assert stats.t_total == pytest.approx(stats.t_fe + stats.t_solver + stats.t_glue, rel=0.1, abs=1e-12)


@need_piqp
def test_sparse_qp_bakes_exactly_the_upper_triangle() -> None:
  """The sparse path gathers triu(P) exactly (PIQP's symmetric-P contract):
  an out-of-contract asymmetric P behaves as if symmetrized from its upper
  triangle. Compare against a dense solve of that symmetrized matrix."""
  kwargs: dict = {"c": np.array([-1.0, -1.0]), "x_lb": np.full(2, -3.0), "x_ub": np.full(2, 3.0)}
  sparse_qp = al.qp(P=np.array([[4.0, 1.0], [7.0, 2.0]]), sparse=True, name="asym_sparse", **kwargs)
  assert sparse_qp.descriptor.P_sparsity is not None
  assert set(zip(sparse_qp.descriptor.P_sparsity.rows, sparse_qp.descriptor.P_sparsity.cols)) == {(0, 0), (0, 1), (1, 1)}
  dense_qp = al.qp(P=np.array([[4.0, 1.0], [1.0, 2.0]]), sparse=False, name="sym_dense", **kwargs)
  sparse_out = sparse_qp(np.zeros(2), np.zeros(0), np.zeros(0))
  dense_out = dense_qp(np.zeros(2), np.zeros(0), np.zeros(0))
  np.testing.assert_allclose(sparse_out["x"], [1.0 / 7.0, 3.0 / 7.0], atol=1e-7)
  for key in sparse_out:
    np.testing.assert_allclose(sparse_out[key], dense_out[key], rtol=1e-7, atol=1e-7, err_msg=key)


@need_piqp
def test_sparse_qp_structurally_zero_P_keeps_valid_csc_handle() -> None:
  """An all-zero P (an LP) keeps one padded (0,0) entry whose gathered value
  is the structural zero, so the baked CSC handle stays valid."""
  qp = al.qp(
    P=np.zeros((2, 2)),
    c=np.array([1.0, -1.0]),
    x_lb=np.array([-1.0, -1.0]),
    x_ub=np.array([1.0, 1.0]),
    sparse=True,
    name="sparse_zero_P_lp",
  )
  assert qp.descriptor.P_sparsity is not None
  assert list(zip(qp.descriptor.P_sparsity.rows, qp.descriptor.P_sparsity.cols)) == [(0, 0)]
  out = qp(np.zeros(2), np.zeros(0), np.zeros(0))
  assert qp.last_stats is not None and qp.last_stats.status == al.AlloySolveStatus.OK
  np.testing.assert_allclose(out["x"], [-1.0, 1.0], atol=1e-6)


@need_piqp
def test_sparse_qp_dependency_mask_keeps_entries_that_probe_to_zero() -> None:
  """A parameter-dependent entry whose value happens to be zero at the probe
  draw must stay in the pattern (the dependency mask, not the probe, keeps it)."""
  t = al.sym("t", 1)
  zero = al.const(0.0)
  P = al.stack([al.stack([al.const(2.0), t[0] - t[0]]), al.stack([zero, al.const(2.0)])], axis=0)
  qp = al.qp(P=P, c=al.stack([t[0], -1.0]), sparse=True, name="probe_zero_qp")
  assert qp.descriptor.P_sparsity is not None
  assert set(zip(qp.descriptor.P_sparsity.rows, qp.descriptor.P_sparsity.cols)) == {(0, 0), (0, 1), (1, 1)}


@need_piqp
def test_sparse_qp_rejects_nested_solver_data() -> None:
  """QP data computed from a nested solver output cannot be pattern-analyzed
  (SOLVER_CALL is an opaque zero to the dependency mask) and must fail loudly
  before the probe would execute the inner solve."""
  inner = al.qp(P=np.eye(2), c=np.array([-1.0, 0.0]), x_lb=np.zeros(2), x_ub=np.ones(2), name="inner_for_pattern")
  x_inner = inner.call([al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0))])[0]
  P = al.stack([al.stack([2.0 + x_inner[0], al.const(0.0)]), al.stack([al.const(0.0), al.const(2.0)])], axis=0)
  with pytest.raises(NotImplementedError, match="nested solver output"):
    al.qp(P=P, c=np.zeros(2), sparse=True, name="outer_sparse_over_solver")


@need_piqp
def test_two_solver_wrappers_in_one_translation_unit() -> None:
  """Two distinct solvers (one sparse, one dense) called from one host
  Function share a single generated TU; their static state must not collide."""

  @al.function("two_qp_host", {"t": (2,)})
  def host(t):
    qp_a = al.qp(P=np.diag([2.0, 4.0]), c=al.stack([t[0], t[1]]), sparse=True, name="tu_qp_a")
    qp_b = al.qp(P=np.diag([1.0, 1.0]), c=al.stack([t[1], -t[0]]), name="tu_qp_b")
    zeros = [al.const(np.zeros(2)), al.const(np.zeros(0)), al.const(np.zeros(0))]
    xa = qp_a.call([*zeros, t])[0]
    xb = qp_b.call([*zeros, t])[0]
    return {"x_sum": xa + xb}

  tv = np.array([1.0, -2.0])
  # qp_a: x = -c / diag(P) = [-0.5, 0.5]; qp_b: x = [-t1, t0] = [2, 1].
  np.testing.assert_allclose(host(tv), [1.5, 1.5], atol=1e-7)


@need_piqp
def test_nested_sparse_qp_in_alloy_function() -> None:
  @al.function("shifted_sparse_qp", {"t": (2,)})
  def solve_shifted(t):
    qp = al.qp(
      P=np.diag([2.0, 4.0]),
      c=al.stack([t[0], t[1]]),
      x_lb=np.full(2, -10.0),
      x_ub=np.full(2, 10.0),
      sparse=True,
      name="nested_sparse_qp",
    )
    out = qp.call(x0=al.const(np.zeros(2)), lam_eq0=al.const(np.zeros(0)), lam_ineq0=al.const(np.zeros(0)), t=t)
    return {"x": out[0]}

  tv = np.array([1.0, -2.0])
  np.testing.assert_allclose(solve_shifted(tv), [-0.5, 0.5], atol=1e-7)
