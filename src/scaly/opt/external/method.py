"""``External``: the base of an optimization method that drives a solver library from generated C, as the solver plugins' methods do."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Literal

from ...function.method import Support
from ..method import METHOD_API
from ..nlp import build_nlp
from ..problem import NLP
from ..qp import NotQuadratic, build_qp, prove_qp

if TYPE_CHECKING:
  from ...function.model import ConcreteFunction
  from .wrapper import SolverWrapperCtx


@dataclass(frozen=True)
class External:
  """An opt method whose solve is a solver library's, called from a generated C wrapper around the
  problem's oracle Functions (``docs/dev/solver_plugins.md``).

  A plugin subclasses it with its library's facts as class attributes: ``name`` (``"opt.piqp"``,
  its entry-point name), ``kind`` (``"qp"``: it takes the quadratic normal form, and refuses a
  problem that is not quadratic; ``"nlp"``: it takes the oracles), ``lib_stem``, ``link_flags``,
  ``header`` (relative to ``include_dir()``), ``hess_triangle`` for an NLP solver, and the
  ``render_wrapper`` hook that writes the C. ``options`` go to the solver as they are; a plugin
  refuses the ones its solver does not take in ``check_options``, so a bad option fails when the
  method is made."""

  name: ClassVar[str]
  problem: ClassVar[type] = NLP
  api: ClassVar[int] = METHOD_API
  kind: ClassVar[Literal["qp", "nlp"]]
  lib_stem: ClassVar[str]
  link_flags: ClassVar[tuple[str, ...]]
  header: ClassVar[str]
  hess_triangle: ClassVar[Literal["lower", "upper"]] = "lower"

  options: dict[str, Any] = field(default_factory=dict)

  def __post_init__(self) -> None:
    object.__setattr__(self, "options", dict(self.options))
    if self.kind not in ("qp", "nlp"):
      raise TypeError(f"{self.name} must declare kind 'qp' or 'nlp', got {self.kind!r}")
    if self.kind == "nlp" and self.hess_triangle not in ("lower", "upper"):
      raise TypeError(f"{self.name} must declare hess_triangle as 'lower' or 'upper'")
    self.check_options()

  def check_options(self) -> None:
    """Refuse ``options`` the solver does not take; a plugin's hook, which takes all by default."""

  @staticmethod
  def include_dir() -> Path:
    """The directory ``header`` is relative to."""
    raise NotImplementedError

  @staticmethod
  def lib_dir() -> Path:
    """The directory holding ``lib<lib_stem>``."""
    raise NotImplementedError

  def render_wrapper(self, fun: ConcreteFunction, ctx: SolverWrapperCtx) -> list[str]:
    """The C of the wrapper that drives the solver for the solver Function ``fun``: it must define
    ``static void <ctx.raw_symbol>(...)`` with the descriptor's ``in*``/``out*`` signature and a
    trailing ``double* w``, call the oracles through ``ctx.raw_symbol_of``, and fill
    ``ctx.stats_symbol`` on every call."""
    raise NotImplementedError

  @property
  def backend(self) -> str:
    """The short name, ``"piqp"``: the solver descriptor's ``backend``."""
    return self.name.removeprefix("opt.")

  def supports(self, problem: Any) -> Support:
    if not isinstance(problem, NLP):
      return Support((f"{type(problem).__name__} is not an opt problem",))
    if self.kind == "qp":
      try:
        prove_qp(problem)
      except NotQuadratic as exc:
        return Support((str(exc),))
    return Support()

  def build(self, problem: NLP[Any, Any, Any, Any], *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    if self.kind == "qp":
      return build_qp(problem, self, name=name, options=dict(self.options), sparse=bool(getattr(self, "sparse", False)))
    return build_nlp(problem, self, name=name, options=dict(self.options))


__all__ = ["External"]
