"""The method interface: the methods of a problem class, their registry through entry points, and the ``Status`` and ``Info`` every built solver reports.

Every numerical domain follows one pattern: a problem class, any number of methods, and a
``solver(problem, method)`` that returns what solves the problem. A method is a frozen dataclass of
its options that says whether it can solve a problem (``supports``) and builds that (``build``):
for the domains that solve, ``opt`` and ``roots``, a ``ConcreteFunction`` with a warm start and the
parameters in and the solution and an ``Info`` out; each other domain says what it builds (an
integrator's discrete map, an interp fit's ``BSpline``). Methods are declared in the entry-point group ``scaly.methods`` under
``<domain>.<name>``, the built-in ones in their distribution's manifest and plugins in theirs, so
a domain's ``MethodRegistry`` finds them all the same way and loads one only when it is asked for.
The compiler names no domain and no method: a domain creates its registry with the install hints
for the methods it knows of.
"""

from __future__ import annotations

import enum
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field, fields
from functools import cached_property
from importlib.metadata import EntryPoint, entry_points
from typing import Any, ClassVar, Protocol, runtime_checkable

from .tree import L, Record

METHOD_ENTRY_POINTS = "scaly.methods"
"""The entry-point group every method is declared in, as ``<domain>.<name> = "module:Class"``."""


class Status(enum.IntEnum):
  """The outcome a method reports, whatever the domain and whichever method ran; each method maps its
  own outcomes onto these. ``OK`` and ``ACCEPTABLE`` are the successful ones. The codes are the
  ``status`` of the solver statistics ABI and of every ``Info``."""

  OK = 0
  ACCEPTABLE = 1
  MAX_ITER = 2
  PRIMAL_INFEASIBLE = 3
  DUAL_INFEASIBLE = 4
  NUMERICS = 5
  USER_STOP = 6
  ERROR = 7

  @property
  def ok(self) -> bool:
    """Whether the outcome is a solution: ``OK`` or ``ACCEPTABLE``."""
    return self in (Status.OK, Status.ACCEPTABLE)


@dataclass(frozen=True)
class Info:
  """What a built solver reports beside its solution, as outputs of the Function: ``status``, a
  ``Status`` code, and ``iter``, the iterations taken, both ``int64`` scalars in the graph. A domain
  subclasses it with its residuals, each a ``float64`` scalar unless the subclass's ``tree`` says
  otherwise. A numerical call returns every output as ``float64``, as the C entry does, so
  ``Status(int(info.status))`` names the outcome."""

  status: Any
  iter: Any

  @classmethod
  def tree(cls, prefix: str = "") -> Record:
    """The output tree of this ``Info``: ``status`` and ``iter`` as ``int64`` scalars, every other
    field a ``float64`` scalar, each leaf named after its field with ``prefix`` in front."""
    kinds = {"status": "int64", "iter": "int64"}
    names = [f.name for f in fields(cls)]
    return Record(cls, **{name: L(prefix + name, (), dtype=kinds.get(name, "float64")) for name in names})


@dataclass(frozen=True)
class Support:
  """Whether a method can solve a problem: true with no ``reasons``, otherwise false with the reasons
  it cannot (nonconvex, too large, needs bounds, ...)."""

  reasons: tuple[str, ...] = ()

  @property
  def ok(self) -> bool:
    return not self.reasons

  def __bool__(self) -> bool:
    return self.ok

  @staticmethod
  def unless(**conditions: bool) -> Support:
    """The reasons whose condition holds: ``Support.unless(nonconvex=not convex, unbounded=...)``."""
    return Support(tuple(reason.replace("_", " ") for reason, holds in conditions.items() if holds))


@runtime_checkable
class Method(Protocol):
  """A method of a problem class: a frozen dataclass of its options. ``name`` is its registry key
  (``"opt.ipm"``), ``problem`` the problem class it solves, and ``api`` the version of that class's
  method API it implements (the class's ``method_api``)."""

  name: ClassVar[str]
  problem: ClassVar[type]
  api: ClassVar[int]

  def supports(self, problem: Any) -> Support:
    """Whether this method, with these options, can solve ``problem``."""
    ...

  def build(self, problem: Any, *, name: str) -> Any:
    """What solves ``problem``: for a solving domain the Function with its parameters and a warm
    start in and the solution and an ``Info`` out; otherwise what the domain says."""
    ...


class MethodError(LookupError):
  """A method that is not installed, does not load, or does not fit its problem class."""


@dataclass(frozen=True)
class MethodHint:
  """A method a domain knows of that may not be installed: its short ``name``, the ``cls`` a
  namespace attribute names it by, and the ``distribution`` that provides it."""

  name: str
  cls: str
  distribution: str


@dataclass
class MethodRegistry:
  """The methods of one domain (``"opt"``, ``"roots"``, ...): the entry points under
  ``<domain>.<name>`` in ``scaly.methods``, loaded on first use.

  ``hints`` are the methods the domain knows of, so a missing one names the distribution to install;
  ``preference`` is the order ``auto`` tries methods in, the rest after it by name."""

  domain: str
  hints: tuple[MethodHint, ...] = ()
  preference: tuple[str, ...] = ()
  _loaded: dict[str, type[Any]] = field(default_factory=dict, repr=False)

  @cached_property
  def _hint(self) -> dict[str, MethodHint]:
    return {h.name: h for h in self.hints}

  def installed(self) -> dict[str, EntryPoint]:
    """The installed methods, by short name, not loaded."""
    prefix = f"{self.domain}."
    return {ep.name.removeprefix(prefix): ep for ep in entry_points(group=METHOD_ENTRY_POINTS) if ep.name.startswith(prefix)}

  def _short(self, name: str) -> str:
    return name.removeprefix(f"{self.domain}.")

  def _missing(self, name: str) -> MethodError:
    hint = self._hint.get(name)
    if hint is not None:
      return MethodError(f"method {self.domain}.{name} is not installed; it comes with {hint.distribution} (uv add {hint.distribution})")
    return MethodError(f"no method {self.domain}.{name}; installed: {sorted(self.installed())}")

  def get(self, name: str) -> type[Any]:
    """The method class ``name`` (``"piqp"`` or ``"opt.piqp"``), loaded and checked on first use."""
    short = self._short(name)
    if short in self._loaded:
      return self._loaded[short]
    ep = self.installed().get(short)
    if ep is None:
      raise self._missing(short)
    try:
      cls = ep.load()
    except Exception as exc:
      raise MethodError(f"method {self.domain}.{short} failed to load from {ep.value}: {exc}") from exc
    self._check(short, cls)
    self._loaded[short] = cls
    return cls

  def _check(self, short: str, cls: Any) -> None:
    key = f"{self.domain}.{short}"
    if getattr(cls, "name", None) != key:
      raise MethodError(f"method {key} declares name {getattr(cls, 'name', None)!r}; it must equal its entry-point name")
    problem: Any = getattr(cls, "problem", None)
    wanted = getattr(problem, "method_api", None)
    if wanted is None:
      raise MethodError(f"method {key} solves {problem!r}, which declares no method_api")
    if getattr(cls, "api", None) != wanted:
      raise MethodError(f"method {key} implements API {getattr(cls, 'api', None)} of {problem.__name__}; this scaly has API {wanted}")
    if not callable(getattr(cls, "supports", None)) or not callable(getattr(cls, "build", None)):
      raise MethodError(f"method {key} needs callable supports(problem) and build(problem, *, name)")

  def _order(self) -> list[str]:
    installed = self.installed()
    first = [name for name in self.preference if name in installed]
    return first + sorted(name for name in installed if name not in first)

  def auto(self, problem: Any) -> Any:
    """The first installed method, with its default options, that supports ``problem``: those in
    ``preference`` first, then the rest by name."""
    refused = []
    for name in self._order():
      try:
        cls = self.get(name)
      except MethodError as exc:  # one broken install must not hide the others
        warnings.warn(str(exc), RuntimeWarning, stacklevel=2)
        continue
      if not isinstance(problem, cls.problem):
        continue
      method = cls()
      support = method.supports(problem)
      if support:
        return method
      refused.append(f"{self.domain}.{name}: {', '.join(support.reasons)}")
    detail = "; ".join(refused) if refused else f"none is installed for {type(problem).__name__}"
    raise MethodError(f"no {self.domain} method supports this problem ({detail})")

  def resolve(self, method: Any, problem: Any) -> Any:
    """A method instance from what a ``solver`` call takes: an instance, a registry name (built
    with its default options), or ``"auto"``; checked against ``problem``."""
    if method == "auto":
      return self.auto(problem)
    if isinstance(method, str):
      method = self.get(method)()
    if not isinstance(problem, method.problem):
      raise TypeError(f"{method.name} solves {method.problem.__name__}, not {type(problem).__name__}")
    support = method.supports(problem)
    if not support:
      raise ValueError(f"{method.name} cannot solve this problem: {', '.join(support.reasons)}")
    return method

  def attribute(self, module: str) -> Callable[[str], type[Any]]:
    """A module ``__getattr__`` that resolves the domain's method classes by class name, lazily:
    ``sc.opt.PIQP`` loads the method whose entry point names ``PIQP``, and names the distribution to
    install when a known method is absent."""

    def __getattr__(attr: str) -> type[Any]:
      for name, ep in self.installed().items():
        if ep.attr == attr:
          return self.get(name)
      for hint in self.hints:
        if hint.cls == attr:
          raise AttributeError(f"{module}.{attr} needs {hint.distribution} (uv add {hint.distribution})")
      raise AttributeError(f"module {module!r} has no attribute {attr!r}")

    return __getattr__


_REGISTRIES: dict[str, MethodRegistry] = {}


def registry(domain: str, *, hints: Iterable[MethodHint] = (), preference: Iterable[str] = ()) -> MethodRegistry:
  """The registry of ``domain``, one per process: the first call, made by the domain's package,
  gives its hints and preference, and later calls return it unchanged."""
  if domain not in _REGISTRIES:
    _REGISTRIES[domain] = MethodRegistry(domain, tuple(hints), tuple(preference))
  return _REGISTRIES[domain]


def method_names() -> dict[str, list[str]]:
  """Every installed method, by domain: what ``scaly.methods`` declares, loaded or not."""
  out: dict[str, list[str]] = {}
  for ep in entry_points(group=METHOD_ENTRY_POINTS):
    domain, _, name = ep.name.partition(".")
    out.setdefault(domain, []).append(name)
  return {domain: sorted(names) for domain, names in sorted(out.items())}


__all__ = ["METHOD_ENTRY_POINTS", "Info", "Method", "MethodError", "MethodHint", "MethodRegistry", "Status", "Support", "method_names", "registry"]
