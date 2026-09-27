"""Discretization of continuous-time models: Runge-Kutta maps built over the model's own signature."""

from .explicit import explicit, rk4
from .implicit import implicit
from .model import UNROLL_STEPS
from .tableau import FAMILIES, TABLEAUS, Tableau, gauss_legendre, lobatto_iiia, lobatto_iiic, order_conditions, radau_iia, tableau

__all__ = [
  "FAMILIES",
  "TABLEAUS",
  "Tableau",
  "UNROLL_STEPS",
  "explicit",
  "gauss_legendre",
  "implicit",
  "lobatto_iiia",
  "lobatto_iiic",
  "order_conditions",
  "radau_iia",
  "rk4",
  "tableau",
]
