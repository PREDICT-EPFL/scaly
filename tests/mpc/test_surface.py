"""The public surface of ``scaly.mpc``: its names, and that ``sc.mpc`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_mpc_surface() -> None:
  mpc = importlib.import_module("scaly.mpc")
  assert sc.mpc is mpc
  assert mpc.__all__ == [
    "MPC",
    "OCP",
    "ClosedLoop",
    "Ellipsoid",
    "Path",
    "Polytope",
    "Quadratic",
    "Solution",
    "TerminalEquality",
    "largest_ellipsoid",
    "linear",
    "lqr",
    "max_invariant_set",
    "simulate",
  ]
  assert "mpc" not in sc.__all__ and not hasattr(sc, "OCP")
