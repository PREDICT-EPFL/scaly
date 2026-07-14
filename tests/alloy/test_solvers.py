"""Structural tests for solver function input validation."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
from alloy.codegen import render_c_api_header, render_c_source
from alloy.jit import CompiledFunction
from alloy.solvers.registry import available_backends

pytestmark = pytest.mark.skipif("piqp" not in available_backends(), reason="structural tests build al.qp and need the alloy-piqp plugin installed")


def test_solver_function_signature_errors() -> None:
  """Mis-shaped or missing inputs raise ``TypeError`` / ``ValueError``."""
  P = np.eye(2)
  c = np.zeros(2)
  qp = al.qp(P=P, c=c)
  with pytest.raises(TypeError, match="missing keyword inputs"):
    qp(x0=np.zeros(2))
  with pytest.raises(ValueError, match="cannot reshape"):
    qp(x0=np.zeros(3), lam_eq0=np.zeros(0), lam_ineq0=np.zeros(0))


def test_standalone_qp_renders_universal_entry_and_stats_query() -> None:
  qp = al.qp(P=np.eye(2), c=np.zeros(2), name="standalone_qp")
  source = render_c_source(qp)
  header = render_c_api_header(qp)
  assert "int standalone_qp(const double** arg, double** res, int* iw, double* w, void* mem)" in source
  assert "int standalone_qp_stats(alloy_solver_stats* out);" in header
  assert "ALLOY_SOLVER_STATS_VERSION 1" in header


def test_qp_settings_are_baked_into_jit_cache_key() -> None:
  def build(eps_abs: float) -> CompiledFunction:
    return CompiledFunction(al.qp(P=np.eye(2), c=np.zeros(2), name="settings_qp", options={"eps_abs": eps_abs}))

  first = build(1e-8)
  same = build(1e-8)
  changed = build(1e-5)
  assert first.cache_key == same.cache_key
  assert first.lib_path == same.lib_path
  assert first.cache_key != changed.cache_key
