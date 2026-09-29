"""The methods of ``scaly.interp``: the ``Fit`` problem class, the method API its methods implement, their registry and ``solver``."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..function.method import registry
from ..ir.expr import Expr
from ..ir.types import DType

METHOD_API = 1
"""The version of the method API of ``Fit`` every interp method implements: bump it whenever what a
method's ``build`` receives, or the spline it must return, changes."""


@dataclass(frozen=True, eq=False)
class Fit:
  """Data a spline is fitted to, the problem every interp method solves, and how the spline it gives
  is evaluated.

  ``x`` and ``y`` are the data on a grid, a vector of sites with ``y`` of shape ``(n, *out_shape)``
  or a tuple of vectors with ``y`` of shape ``(n_1, ..., n_D, *out_shape)``, or scattered, points
  ``(m, D)`` with ``y`` of shape ``(m, *out_shape)``; ``y`` may be an ``Expr``, for the methods whose
  fit is a linear map of the data. ``extrap``, ``fill``, ``search``, ``strategy``, ``dtype`` and
  ``name`` are the spline's, as for ``BSpline``.

  A method builds a ``BSpline``, with its derivatives, integrals and inverse, not a Function: unlike
  an optimization problem's solver, it takes no warm start and returns no ``Info``."""

  method_api: ClassVar[int] = METHOD_API

  x: Any
  y: Any
  extrap: Any = None
  fill: float = math.nan
  search: Any = "auto"
  strategy: Any = "auto"
  dtype: DType | str = "float64"
  name: str = "interp"

  @property
  def gridded(self) -> bool:
    """Whether ``x`` is a grid: a vector of sites, or a tuple (or list) of them."""
    if isinstance(self.x, (tuple, list)) and self.x and all(np.ndim(g) == 1 for g in self.x):
      return True
    return np.ndim(self.x) == 1

  @property
  def ndim(self) -> int:
    """The number of axes the data span."""
    if isinstance(self.x, (tuple, list)) and self.gridded and np.ndim(self.x[0]) == 1:
      return len(self.x)
    return 1 if np.ndim(self.x) == 1 else int(np.shape(self.x)[1])

  @property
  def symbolic(self) -> bool:
    """Whether the values are an ``Expr``, a fit in the graph."""
    return isinstance(self.y, Expr)


REGISTRY = registry("interp", preference=("linear", "smoothing"))
"""Every interp method by short name (``"pchip"``, ``"smoothing"``); ``auto`` interpolates data on a
grid linearly and smooths scattered data."""


def solver(problem: Fit, method: Any = "auto", /, *, name: str | None = None) -> Any:
  """The spline ``method`` fits to ``problem``: a method with its options (``sc.interp.Cubic(bc=
  "natural")``), a method's name (``"pchip"``, with its default options), or ``"auto"``, linear
  interpolation for data on a grid and a smoothing spline for scattered data. Named ``name``, by
  default the problem's. The shorthands (``interpolant``, ``smoothing``, ``constrained``) build the
  same splines."""
  chosen = REGISTRY.resolve(method, problem)
  return chosen.build(problem, name=name or problem.name)


__all__ = ["METHOD_API", "REGISTRY", "Fit", "solver"]
