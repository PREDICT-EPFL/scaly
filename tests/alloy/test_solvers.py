"""Structural tests for solver function input validation."""

from __future__ import annotations

import numpy as np
import pytest

import alloy as al
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
