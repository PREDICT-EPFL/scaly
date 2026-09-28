"""Interpolation and lookup tables: tensor-product B-splines on rectilinear grids, fitted from data or given by their coefficients, evaluated as generated code."""

from .constrained import constrained
from .fit import BOUNDARIES, KINDS, interpolant, smoothing
from .grid import Axis
from .spline import PP_BUDGET, BSpline, Index, Inverse

__all__ = ["BOUNDARIES", "KINDS", "PP_BUDGET", "Axis", "BSpline", "Index", "Inverse", "constrained", "interpolant", "smoothing"]
