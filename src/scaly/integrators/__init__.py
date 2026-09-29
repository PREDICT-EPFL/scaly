"""Discretization of continuous-time models: Runge-Kutta maps built over the model's own signature, one method class per named method."""

from .explicit import adaptive, explicit, rk4, symplectic
from .implicit import implicit
from .linear import affine, foh, linearize, zoh
from .method import METHOD_API, ODE, REGISTRY, solver
from .methods import (
  BS32,
  DOPRI5,
  RK3,
  RK4,
  RK38,
  SDIRK2,
  SDIRK3,
  SSPRK3,
  Adaptive,
  BackwardEuler,
  Euler,
  ExplicitRK,
  GaussLegendre,
  Heun,
  ImplicitMidpoint,
  ImplicitRK,
  LobattoIIIA,
  LobattoIIIC,
  Midpoint,
  RadauIIA,
  Ralston,
  StormerVerlet,
  SymplecticEuler,
  Trapezoidal,
  Tsit5,
)
from .model import UNROLL_STEPS
from .polynomial import lgl
from .tableau import FAMILIES, TABLEAUS, Tableau, gauss_legendre, lobatto_iiia, lobatto_iiic, order_conditions, radau_iia, tableau
from .variational import variational

# The method classes of other distributions, loaded on first use from the registry.
__getattr__ = REGISTRY.attribute(__name__)

__all__ = [
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
  "variational",
  "zoh",
]
