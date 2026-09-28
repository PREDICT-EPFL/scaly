"""The public surface of ``scaly.integrators``: its names, and that ``sc.integrators`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_integrator_surface() -> None:
  integrators = importlib.import_module("scaly.integrators")
  assert sc.integrators is integrators
  assert integrators.__all__ == [
    "Collocation",
    "FAMILIES",
    "Interval",
    "MultipleShooting",
    "Pseudospectral",
    "TABLEAUS",
    "Tableau",
    "Transcription",
    "UNROLL_STEPS",
    "adaptive",
    "explicit",
    "foh",
    "gauss_legendre",
    "implicit",
    "linearize",
    "lobatto_iiia",
    "lobatto_iiic",
    "order_conditions",
    "radau_iia",
    "rk4",
    "symplectic",
    "tableau",
    "zoh",
  ]
  assert "integrators" not in sc.__all__ and not hasattr(sc, "rk4")
