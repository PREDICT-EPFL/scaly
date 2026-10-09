"""The key and the name of every derived callee: the forward and adjoint helpers AD builds for calls."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

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
  """

  role: Role
  outputs: tuple[int, ...]
  active: tuple[int, ...]
  lowering: Lowering
  nseed: int | None = None
  constants: tuple[tuple[str, bytes] | None, ...] = ()
  members: tuple[ConcreteFunction, ...] = ()


def helper_name(callee: ConcreteFunction, key: HelperKey) -> str:
  """A readable stem and a digest of ``key``, such as ``car_fwd_0123456789``.

  Lowering reserves the emitted symbol through its ``NameScope``, which renames a clash between
  helpers of two callees sharing a name.
  """
  stem = {"forward": "fwd", "adjoint": "adj"}[key.role] + ("_pack" if key.members else "")
  fields = (key.role, key.outputs, key.active, key.lowering, key.nseed, key.constants, tuple(fn.name for fn in key.members))
  return f"{callee.name}_{stem}_{hashlib.sha1(repr(fields).encode()).hexdigest()[:10]}"
