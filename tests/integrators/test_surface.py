"""The public surface of ``scaly.integrators``: its names, and that ``sc.integrators`` is the package, loaded on first use."""

from __future__ import annotations

import importlib

import scaly as sc


def test_the_integrator_surface() -> None:
  integrators = importlib.import_module("scaly.integrators")
  assert sc.integrators is integrators
  assert integrators.__all__ == [
    "BS32",
    "DOPRI5",
    "FAMILIES",
    "METHOD_API",
    "ODE",
    "REGISTRY",
    "RK3",
    "RK38",
    "RK4",
    "SDIRK2",
    "SDIRK3",
    "SSPRK3",
    "TABLEAUS",
    "Tableau",
    "UNROLL_STEPS",
    "Adaptive",
    "BackwardEuler",
    "Euler",
    "ExplicitRK",
    "GaussLegendre",
    "Heun",
    "ImplicitMidpoint",
    "ImplicitRK",
    "LobattoIIIA",
    "LobattoIIIC",
    "Midpoint",
    "RadauIIA",
    "Ralston",
    "StormerVerlet",
    "SymplecticEuler",
    "Trapezoidal",
    "Tsit5",
    "adaptive",
    "affine",
    "explicit",
    "foh",
    "gauss_legendre",
    "implicit",
    "lgl",
    "linearize",
    "lobatto_iiia",
    "lobatto_iiic",
    "order_conditions",
    "radau_iia",
    "rk4",
    "solver",
    "symplectic",
    "tableau",
    "zoh",
  ]
  assert "integrators" not in sc.__all__ and not hasattr(sc, "rk4")
