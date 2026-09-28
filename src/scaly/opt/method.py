"""The methods of ``scaly.opt``: their registry over the ``scaly.methods`` entry points, the method API they implement, the ``Info`` they report, and the external ones' view of the registry."""

from __future__ import annotations

import warnings
from dataclasses import dataclass, fields
from typing import Any

from ..function.method import Info as MethodInfo
from ..function.method import MethodError, MethodHint, registry
from ..function.tree import L, Record
from ..ir.types import TensorType

METHOD_API = 9
"""The version of the method API of ``NLP`` (and ``QP``) every opt method implements. It continues
the solver plugin protocol, whose last version was 8: bump it whenever what a method's ``build``
receives, what its Function must return, or the extern-callee and statistics ABI change."""


@dataclass(frozen=True)
class Info(MethodInfo):
  """What every opt solver returns beside the solution: ``status`` (a ``Status`` code), ``iter``,
  the ``objective`` at the solution and its ``primal_residual``, the largest constraint violation.
  All four are ``float64`` scalars, as an external solver's C wrapper writes them; a method that has
  no such measure reports 0."""

  objective: Any
  primal_residual: Any

  @classmethod
  def tree(cls, prefix: str = "info:") -> Record:
    return Record(cls, **{f.name: L(prefix + f.name, TensorType((), diff=False)) for f in fields(cls)})


REGISTRY = registry(
  "opt",
  hints=(MethodHint("piqp", "PIQP", "scaly-piqp"), MethodHint("ipopt", "IPOPT", "scaly-ipopt"), MethodHint("sqp", "SQP", "scaly-sqp")),
  preference=("piqp", "ipopt", "sqp"),
)
"""Every opt method, generated or external, by short name (``"piqp"``); ``auto`` tries a QP method
before the NLP ones."""


def short_name(name: str) -> str:
  """``"piqp"`` for ``"opt.piqp"`` or ``"piqp"``."""
  return name.removeprefix("opt.")


def is_external(cls: Any) -> bool:
  """Whether a method class drives a solver library from generated C (``opt.external.External``)."""
  return callable(getattr(cls, "render_wrapper", None)) and callable(getattr(cls, "lib_dir", None))


_EXTERNAL: dict[str, Any] = {}


def external_method(name: str) -> Any:
  """The installed external method ``name`` with its default options: what knows the solver's
  library, headers and C wrapper."""
  key = short_name(name)
  if key not in _EXTERNAL:
    cls = REGISTRY.get(key)
    if not is_external(cls):
      raise MethodError(f"opt.{key} is not an external solver")
    _EXTERNAL[key] = cls()
  return _EXTERNAL[key]


def external_methods() -> dict[str, Any]:
  """Every installed external method that loads, by short name. A broken install warns and is
  skipped, so it cannot hide the others; a generated method is not external and is left out."""
  out: dict[str, Any] = {}
  for key in sorted(REGISTRY.installed()):
    try:
      cls = REGISTRY.get(key)
    except MethodError as exc:
      warnings.warn(f"could not load opt.{key}: {exc}", RuntimeWarning, stacklevel=2)
      continue
    if is_external(cls):
      out[key] = external_method(key)
  return out


__all__ = ["METHOD_API", "REGISTRY", "Info", "external_method", "external_methods", "is_external", "short_name"]
