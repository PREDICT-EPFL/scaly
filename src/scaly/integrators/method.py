"""The methods of ``scaly.integrators``: the ``ODE`` problem class, the method API its methods implement, their registry and ``solver``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar

from ..function.method import registry
from ..function.model import Function

METHOD_API = 1
"""The version of the method API of ``ODE`` every integrator method implements: bump it whenever
what a method's ``build`` receives, or the map it must return, changes."""


@dataclass(frozen=True, eq=False)
class ODE:
  """The flow of ``x' = f(x, ...)`` over an interval ``dt``, the problem every integrator method
  solves. ``f`` is the model, a Function whose first parameter is the state and whose one output is
  its derivative; its other parameters (inputs, parameters) are held fixed over the interval. ``dt``
  is folded into the generated code, or ``None`` appends it to the map's parameters. ``name`` names
  the maps built from it, ``{name}_{method}``, by default the model's name.

  A method builds the discrete map ``F(x, ...) -> xnext`` over the model's own parameters (and
  ``dt``): unlike an optimization problem's solver, it takes no warm start and returns no ``Info``."""

  method_api: ClassVar[int] = METHOD_API

  f: Function[Any, Any, Any, Any]
  dt: float | None = None
  name: str = ""

  def __post_init__(self) -> None:
    if not isinstance(self.f, Function):
      raise TypeError(f"an ODE's model must be an sc.Function, got {type(self.f).__name__}")
    if self.dt is not None and not float(self.dt) > 0:
      raise ValueError(f"dt must be positive, or None to make it an input; got {self.dt}")
    if not self.name:
      object.__setattr__(self, "name", self.f.name)


REGISTRY = registry("integrators", preference=("rk4",))
"""Every integrator method by short name (``"rk4"``, ``"radau_iia"``, ``"adaptive"``); ``auto`` is
the classical Runge-Kutta method."""


def solver(problem: ODE, method: Any = "auto", /, *, name: str | None = None) -> Any:
  """The discrete map of ``problem``, ``F(x, ...) -> xnext``, built by ``method``: a method with its
  options (``sc.integrators.RadauIIA(3, newton=sc.roots.Newton(tol=1e-10))``), a method's name
  (``"tsit5"``, with its default options), or ``"auto"``, the classical Runge-Kutta method. The map
  is the one the shorthands (``rk4``, ``explicit``, ``implicit``, ...) build, named
  ``{problem.name}_{method.label}`` unless ``name`` says otherwise. A template model gives a
  template."""
  chosen = REGISTRY.resolve(method, problem)
  # Left to the map, the default name follows a template model's instances, as the shorthands' does.
  if name is None and problem.name != problem.f.name:
    name = f"{problem.name}_{chosen.label}"
  return chosen.build(problem, name=name)


__all__ = ["METHOD_API", "ODE", "REGISTRY", "solver"]
