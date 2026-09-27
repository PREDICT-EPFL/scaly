"""Discretization of continuous-time models: Runge-Kutta maps built over the model's own signature."""

from .explicit import explicit, rk4
from .model import UNROLL_STEPS
from .tableau import TABLEAUS, Tableau, order_conditions, tableau

__all__ = ["TABLEAUS", "Tableau", "UNROLL_STEPS", "explicit", "order_conditions", "rk4", "tableau"]
