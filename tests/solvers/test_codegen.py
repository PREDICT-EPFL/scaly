"""Structural tests for solver function input validation and the shared solver ABI.

These cover Scaly's own machinery — argument checking, the universal C entry, the
`scaly_solver_stats` handshake, identifier mangling across one translation unit, and
JIT cache keying. PIQP is only the vehicle: a second QP backend would exercise the
same code. Backend-specific behaviour (PIQP's CSC baking, its status mapping) stays
in `plugins/scaly-piqp/tests/`.
"""

from __future__ import annotations

import ctypes

import numpy as np
import pytest

import scaly as sc
from tests.solvers.problem_helpers import build_qp
from scaly.codegen import render_c_api_header, render_c_source
from scaly.codegen.jit import CompiledFunction, JitError
from scaly.solvers.registry import available_backends
from scaly.solvers.stats import CSolverStats

pytestmark = pytest.mark.skipif(
  "piqp" not in available_backends(), reason="structural tests build the private QP differential fixture and need the scaly-piqp plugin installed"
)


def test_solver_function_signature_errors() -> None:
  """Mis-shaped or missing inputs raise ``TypeError`` / ``ValueError``."""
  P = np.eye(2)
  c = np.zeros(2)
  qp = build_qp(P=P, c=c)
  with pytest.raises(ValueError, match="declared structure"):
    qp.function.numerical_call(*(np.zeros(2), np.zeros(2), np.zeros(0), np.zeros(0)))  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match=r"expected shape \(2,\).*got \(3,\)"):
    qp((), x0=np.zeros(3))


def test_standalone_qp_renders_universal_entry_and_stats_query() -> None:
  qp = build_qp(P=np.eye(2), c=np.zeros(2), name="standalone_qp")
  source = render_c_source(qp.function)
  header = render_c_api_header(qp.function)
  assert "int standalone_qp(const double** arg, double** res, int* iw, double* w, int mem)" in source
  assert "int standalone_qp_stats(scaly_solver_stats* out);" in header
  assert "SCALY_SOLVER_STATS_VERSION 3" in header


@pytest.mark.solver("piqp")
def test_qp_settings_are_baked_into_jit_cache_key() -> None:
  def build(eps_abs: float) -> CompiledFunction:
    return CompiledFunction(build_qp(P=np.eye(2), c=np.zeros(2), name="settings_qp", options={"eps_abs": eps_abs}).function)

  first = build(1e-8)
  same = build(1e-8)
  changed = build(1e-5)
  assert first.cache_key == same.cache_key
  assert first.lib_path == same.lib_path
  assert first.cache_key != changed.cache_key


@pytest.mark.solver("piqp")
def test_solver_stats_reject_uninitialized_and_mismatched_versions() -> None:
  """The `scaly_solver_stats` handshake is Scaly's contract with every backend."""
  qp = build_qp(P=np.eye(2), c=np.zeros(2), name="stats_version_qp")
  compiled = CompiledFunction(qp.function)
  with pytest.raises(JitError, match="has not run yet"):
    compiled.solver_stats()

  def mismatched_stats(out: ctypes.c_void_p) -> int:
    ctypes.cast(out, ctypes.POINTER(CSolverStats)).contents.version = sc.SCALY_SOLVER_STATS_VERSION + 1
    return 0

  compiled._stats_entries["stats_version_qp"] = mismatched_stats
  with pytest.raises(JitError, match="ABI mismatch.*artifact version 4, expected 3"):
    compiled.solver_stats()


def test_sparse_qp_rejects_nested_solver_data() -> None:
  """QP data computed from a nested solver output cannot be pattern-analyzed
  so the structural QP proof must reject it before the sparse-pattern probe
  would execute the inner solve."""
  inner = build_qp(P=np.eye(2), c=np.array([-1.0, 0.0]), x_lb=np.zeros(2), x_ub=np.ones(2), name="inner_for_pattern")
  x_inner = inner.function.symbolic_call(*(sc.const(np.zeros(2)), sc.const(np.zeros(2)), sc.const(np.zeros(0)), sc.const(np.zeros(0)), ()))[0]
  P = sc.stack([sc.stack([2.0 + x_inner[0], sc.const(0.0)]), sc.stack([sc.const(0.0), sc.const(2.0)])], axis=0)
  with pytest.raises(sc.NotQuadratic, match="cannot prove QP structure through a nested solver"):
    build_qp(P=P, c=np.zeros(2), sparse=True, name="outer_sparse_over_solver")


@pytest.mark.solver("piqp")
def test_two_solver_wrappers_in_one_translation_unit() -> None:
  """Two distinct solvers (one sparse, one dense) called from one host
  Function share a single generated TU; their static state must not collide."""

  @sc.function(sc.arg("t", (2,)), outputs=sc.arg("x_sum", ...), name="two_qp_host")
  def host(t: sc.Expr) -> sc.Expr:
    qp_a = build_qp(P=np.diag([2.0, 4.0]), c=sc.stack([t[0], t[1]]), sparse=True, name="tu_qp_a")
    qp_b = build_qp(P=np.diag([1.0, 1.0]), c=sc.stack([t[1], -t[0]]), name="tu_qp_b")
    xa, *_ = qp_a(t)
    xb, *_ = qp_b(t)
    return xa + xb

  tv = np.array([1.0, -2.0])
  # qp_a: x = -c / diag(P) = [-0.5, 0.5]; qp_b: x = [-t1, t0] = [2, 1].
  np.testing.assert_allclose(host(tv), [1.5, 1.5], atol=1e-7)
