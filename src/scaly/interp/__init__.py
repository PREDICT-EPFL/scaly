"""Interpolation and lookup tables: tensor-product B-splines on rectilinear grids, fitted from data or given by their coefficients, evaluated as generated code; one method class per kind of fit."""

from .constrained import constrained
from .fit import BOUNDARIES, KINDS, interpolant, smoothing
from .grid import Axis
from .method import METHOD_API, REGISTRY, Fit, solver
from .methods import PCHIP, ZOH, Akima, Constrained, Cubic, Interpolating, Linear, Makima, Nearest, PerAxis, Smoothing, SmoothLinear, Spline, Steffen
from .spline import PP_BUDGET, BSpline, Index, Inverse

__all__ = [
  "BOUNDARIES",
  "KINDS",
  "METHOD_API",
  "PCHIP",
  "PP_BUDGET",
  "REGISTRY",
  "ZOH",
  "Akima",
  "Axis",
  "BSpline",
  "Constrained",
  "Cubic",
  "Fit",
  "Index",
  "Interpolating",
  "Inverse",
  "Linear",
  "Makima",
  "Nearest",
  "PerAxis",
  "SmoothLinear",
  "Smoothing",
  "Spline",
  "Steffen",
  "constrained",
  "interpolant",
  "smoothing",
  "solver",
]
