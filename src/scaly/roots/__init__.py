"""Nonlinear equations and least squares: problems over typed unknowns and parameters, the Newton
family that solves them as generated loops, and ``custom_root``, the implicit derivative of a root
however it was found."""

from __future__ import annotations

from .implicit import Linear, Residual, custom_root
from .least_squares import GaussNewton, LevenbergMarquardt
from .method import METHOD_API, REGISTRY, Info
from .newton import LINEAR, Newton, NewtonBisection
from .problem import LeastSquares, Root, RootSpec, least_squares, root
from .solver import solver

__all__ = [
  "LINEAR",
  "METHOD_API",
  "REGISTRY",
  "GaussNewton",
  "Info",
  "LeastSquares",
  "LevenbergMarquardt",
  "Linear",
  "Newton",
  "NewtonBisection",
  "Residual",
  "Root",
  "RootSpec",
  "custom_root",
  "least_squares",
  "root",
  "solver",
]
