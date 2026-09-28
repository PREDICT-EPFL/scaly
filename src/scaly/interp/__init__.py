"""Interpolation and lookup tables: tensor-product B-splines on rectilinear grids, fitted from data or given by their coefficients, evaluated as generated code."""

from .fit import BOUNDARIES, KINDS, interpolant
from .grid import Axis
from .spline import PP_BUDGET, BSpline, Index

__all__ = ["BOUNDARIES", "KINDS", "PP_BUDGET", "Axis", "BSpline", "Index", "interpolant"]
