"""The public surface of ``scaly.ocp``: its names, and that ``sc.ocp`` is the package, loaded on first use."""

from __future__ import annotations

import importlib
import importlib.util

import pytest

import scaly as sc


def test_the_ocp_surface() -> None:
  ocp = importlib.import_module("scaly.ocp")
  assert sc.ocp is ocp
  assert ocp.__all__ == [
    "METHOD_API",
    "REGISTRY",
    "Collocation",
    "ContinuousOCP",
    "Direct",
    "DiscreteOCP",
    "Form",
    "ILQR",
    "Info",
    "Interval",
    "Layout",
    "MultipleShooting",
    "Param",
    "Path",
    "Pseudospectral",
    "Quadratic",
    "StageStructure",
    "TerminalEquality",
    "TinyADMM",
    "Transcription",
    "initial_guess",
    "largest_ellipsoid",
    "lqr",
    "max_invariant_set",
    "shift",
    "solver",
    "to_problem",
    "transcribe",
  ]
  assert "ocp" not in sc.__all__ and not hasattr(sc.integrators, "MultipleShooting")
  installed = set(ocp.REGISTRY.installed())
  assert {"direct", "ilqr", "tinyadmm"} <= installed <= {"altro", "direct", "ilqr", "scvx", "tinyadmm"}
  # The experimental methods are scaly-experimental's, resolved through the registry on first use.
  if {"altro", "scvx"} <= installed:
    assert ocp.ALTRO is ocp.REGISTRY.get("altro") and ocp.SCvx is ocp.REGISTRY.get("scvx")
  else:
    with pytest.raises(AttributeError, match="scaly-experimental"):
      ocp.ALTRO  # noqa: B018
  assert importlib.util.find_spec("scaly.mpc") is None  # the receding horizon is the user's loop over ocp.solver and ocp.shift
