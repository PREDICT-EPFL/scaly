"""``solver``: the Function that solves a roots problem, built by the method asked for or the first that supports it."""

from __future__ import annotations

from typing import Any, overload

from ..function import ConcreteFunction
from .method import REGISTRY, Info
from .problem import LeastSquares, Root


@overload
def solver[SV, NV](
  problem: Root[SV, NV, None, None] | LeastSquares[SV, NV, None, None], method: Any = "auto", /, *, name: str | None = None
) -> ConcreteFunction[[SV], [NV], tuple[SV, Info], tuple[NV, Info]]: ...


@overload
def solver[SV, NV, SP, NP](
  problem: Root[SV, NV, SP, NP] | LeastSquares[SV, NV, SP, NP], method: Any = "auto", /, *, name: str | None = None
) -> ConcreteFunction[[SV, SP], [NV, NP], tuple[SV, Info], tuple[NV, Info]]: ...


def solver(problem: Any, method: Any = "auto", /, *, name: str | None = None) -> Any:
  """The Function that solves ``problem``: the unknowns' warm start and the parameters (when the
  problem has any) in, the solution and an ``Info`` out, whichever method builds it. The solution
  is differentiable in the parameters by the implicit function theorem at the point found.

  ``method`` is a method with its options (``sc.roots.Newton(tol=1e-12)``), a method's name
  (``"newton"``, with its default options), or ``"auto"``, the first that supports the problem:
  Newton for a square system, the bracketing Newton for one bounded unknown, Levenberg-Marquardt for
  least squares. A method that cannot solve the problem raises, naming why."""
  chosen = REGISTRY.resolve(method, problem)
  return chosen.build(problem, name=name or f"{problem.name}_{chosen.name.removeprefix('roots.')}")


__all__ = ["solver"]
