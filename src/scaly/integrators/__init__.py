"""Discretization of continuous-time models: Runge-Kutta maps built over the model's own signature."""

from .explicit import adaptive, explicit, rk4, symplectic
from .implicit import implicit
from .linear import foh, linearize, zoh
from .model import UNROLL_STEPS
from .tableau import FAMILIES, TABLEAUS, Tableau, gauss_legendre, lobatto_iiia, lobatto_iiic, order_conditions, radau_iia, tableau

__all__ = [
  "FAMILIES",
  "TABLEAUS",
  "Tableau",
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
