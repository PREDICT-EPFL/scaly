"""The key and the name of every derived callee: the forward and adjoint helpers AD builds for calls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any

from ..function.concrete import ConcreteFunction, Role
from ..ir.types import Lowering


@dataclass(frozen=True, slots=True)
class HelperKey:
  """Everything that tells two helpers of one callee apart.

  A helper is stored in its callee's ``_memo`` under this key, so the callee's identity is where the
  key lives rather than a field of it. ``nseed`` is ``None`` for one seed shaped like its input (or
  one cotangent shaped like its output) and the seed count of a seed-major ``(nseed, *shape)`` seed
  otherwise. ``constants`` holds, per active input, the dtype and bytes of a seed baked into the
  body, or ``None`` for a seed passed at run time. ``members`` are the helpers a pack concatenates.
  ``rules`` names the callee's custom rules, so its helpers and those of the function it was made
  from are named apart.
  """

  role: Role
  outputs: tuple[int, ...]
  active: tuple[int, ...]
  lowering: Lowering
  nseed: int | None = None
  constants: tuple[tuple[str, bytes] | None, ...] = ()
  members: tuple[ConcreteFunction, ...] = ()
  rules: tuple[str, ...] = ()

  @classmethod
  def of(cls, callee: ConcreteFunction, role: Role, outputs: tuple[int, ...], active: tuple[int, ...], **fields: Any) -> HelperKey:
    """The key of a helper of ``callee``, with its lowering hint and its rules."""
    return cls(role, outputs, active, callee._effective_lowering, rules=callee.rules.key if callee.rules else (), **fields)


def helper_name(callee: ConcreteFunction, key: HelperKey) -> str:
  """A readable stem and a digest of ``key``, such as ``car_fwd_0123456789``.

  Lowering reserves the emitted symbol through its ``NameScope``, which renames a clash between
  helpers of two callees sharing a name.
  """
  stem = {"forward": "fwd", "adjoint": "adj"}[key.role] + ("_pack" if key.members else "")
  fields = (key.role, key.outputs, key.active, key.lowering, key.nseed, key.constants, tuple(fn.name for fn in key.members))
  fields += (key.rules,) if key.rules else ()
  return f"{callee.name}_{stem}_{hashlib.sha1(repr(fields).encode()).hexdigest()[:10]}"
