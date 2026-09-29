"""The public surface of ``scaly.ocp``: its names, and that ``sc.ocp`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_ocp_surface() -> None:
  ocp = importlib.import_module("scaly.ocp")
  assert sc.ocp is ocp
  assert ocp.__all__ == [
    "METHOD_API",
    "Collocation",
    "ContinuousOCP",
    "DiscreteOCP",
    "Form",
    "Interval",
    "Layout",
    "MultipleShooting",
    "Param",
    "Path",
    "Pseudospectral",
    "Quadratic",
    "StageStructure",
    "TerminalEquality",
    "Transcription",
    "to_problem",
    "transcribe",
  ]
  assert "ocp" not in sc.__all__ and not hasattr(sc.integrators, "MultipleShooting")
