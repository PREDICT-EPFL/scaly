"""The public surface of ``scaly.ocp``: its names, and that ``sc.ocp`` is the package, loaded on first use."""

from __future__ import annotations

import importlib
import importlib.util

import scaly as sc


def test_the_ocp_surface() -> None:
  ocp = importlib.import_module("scaly.ocp")
  assert sc.ocp is ocp
  assert ocp.__all__ == [
    "ALTRO",
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
    "SCvx",
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
  assert sorted(ocp.REGISTRY.installed()) == ["altro", "direct", "ilqr", "scvx", "tinyadmm"]
  assert importlib.util.find_spec("scaly.mpc") is None  # the receding horizon is the user's loop over ocp.solver and ocp.shift
