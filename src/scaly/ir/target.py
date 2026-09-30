"""The processor generated code is tuned for: ``Target``, its presets, the host's, and the target in force.

A target is read when a Function is rendered, never when its graph is built: one graph renders for
any number of targets, and the rendered C carries every choice a target made, so the JIT's cache
key, a hash of that C, tells two targets' builds apart without naming them. The JIT compiles for the
processor it runs on whatever the target, so rendering for another target in-process runs that
target's choices on this machine; ahead-of-time output takes ``Target.cflags`` for the machine it
is built for.
"""

from __future__ import annotations

import functools
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from typing import Any, Literal

from ..utils.env import CpuFacts, cpu_facts, env

type Rounding = Literal["target", "portable"]

_KIB = 1024


@dataclass(frozen=True, slots=True)
class Target:
  """A description of the processor generated code is tuned for.

  The fields are facts about the hardware; the properties below them are the code-generation
  choices derived from those facts. A choice was fixed by measurement on the reference machine, an
  Apple M3 (``PORTABLE``); on other targets it follows the same formula, unmeasured there.

  A choice can change the last bits of a result even where it keeps the order of every sum: the
  C compiler fuses a multiply and an add into one rounding (``-ffp-contract=on``) or not depending
  on how it vectorizes the code around them, and a block width changes that. Under
  ``rounding="portable"`` every choice is the reference machine's, whatever the target, so the
  generated C is the same for every target and results differ between machines only as far as
  their compilers do.

  Attributes:
    name: the preset this target is, or was derived from.
    cflags: the compiler flags that select this processor for ahead-of-time builds. The JIT
      ignores them and compiles for the processor it runs on.
    vector_bytes: the width of one SIMD register (8 for scalar code).
    l1i_bytes: the level-1 instruction cache of the core generated code runs on.
    l1d_bytes: the level-1 data cache of that core.
    rounding: ``"target"`` lets a choice that shapes floating-point code (a block width, a lane
      count) follow this target, so results may differ in the last bits between targets;
      ``"portable"`` gives every such choice the reference machine's value.
  """

  name: str = "generic"
  cflags: tuple[str, ...] = ()
  vector_bytes: int = 8
  l1i_bytes: int = 32 * _KIB
  l1d_bytes: int = 32 * _KIB
  rounding: Rounding = "target"

  def __post_init__(self) -> None:
    if self.rounding not in ("target", "portable"):
      raise ValueError(f"Target.rounding must be 'target' or 'portable', got {self.rounding!r}")
    if type(self.vector_bytes) is not int or self.vector_bytes not in (8, 16, 32, 64):
      raise ValueError(f"Target.vector_bytes must be 8, 16, 32 or 64, got {self.vector_bytes!r}")
    for name in ("l1i_bytes", "l1d_bytes"):
      value = getattr(self, name)
      if type(value) is not int or value <= 0:
        raise ValueError(f"Target.{name} must be a positive integer, got {value!r}")
    if not isinstance(self.cflags, tuple) or not all(isinstance(flag, str) for flag in self.cflags):
      raise TypeError(f"Target.cflags must be a tuple of strings, got {self.cflags!r}")

  @classmethod
  def preset(cls, name: str, **changes: Any) -> Target:
    """The preset ``name`` (one of ``Target.presets()``, or ``"host"``), with ``changes`` to its fields."""
    base = host_target() if name == "host" else PRESETS.get(name)
    if base is None:
      raise ValueError(f"unknown target preset {name!r}; known: {', '.join(('host', *PRESETS))}")
    return replace(base, **changes) if changes else base

  @staticmethod
  def presets() -> tuple[str, ...]:
    """The names of the presets."""
    return tuple(PRESETS)

  @property
  def choices(self) -> Target:
    """The target whose facts the code-generation choices follow: this one, or under
    ``rounding="portable"`` the reference machine (``PORTABLE``)."""
    return self if self.rounding == "target" else PORTABLE

  @property
  def vector_doubles(self) -> int:
    """The ``double`` lanes of one SIMD register."""
    return self.vector_bytes // 8

  @property
  def row_blocks(self) -> tuple[int, ...]:
    """The widths, widest first, of the blocks of a row that ``x @ b`` and ``a @ b`` keep in
    registers across the reduction: 8, 4 and 2 registers' worth of ``double``, 16, 8 and 4 columns
    on the reference machine. Each output still sums in order of ``k``, but the widths decide how
    the compiler vectorizes the sums, and so which of them it fuses into multiply-adds: they are a
    rounding choice."""
    v = self.choices.vector_doubles
    return (8 * v, 4 * v, 2 * v)

  @property
  def straight_line_ops(self) -> int:
    """The operations a dense factorization or triangular solve may take as straight-line code:
    under it the body is unrolled, over it the code loops. 4 096 on the reference machine, the
    budget scalar expansion keeps a body in registers under, past which straight-line code measured
    slower than loops there; other targets scale it by their instruction cache."""
    return self.choices.l1i_bytes // 48

  @property
  def panel_bytes(self) -> int:
    """The bytes of ``b`` a product ``a @ b`` keeps in the level-1 data cache while every row of
    ``a`` passes over it, when its column blocks run outermost (a ``b`` wider than
    ``row_blocked_max``, or larger than ``l1d_bytes``, and rows enough to share the copy): half of
    ``l1d_bytes``, the other half for the rows of ``a`` and the outputs. Each block's panel is
    copied contiguous in chunks of ``k`` that fit."""
    return self.choices.l1d_bytes // 2

  @property
  def row_blocked_max(self) -> int:
    """The widest row of a vector times a matrix that is blocked at all, four of the widest blocks
    (64 columns on the reference machine); a wider one streams with the reduction outermost."""
    return 4 * self.row_blocks[0]


PRESETS: dict[str, Target] = {
  t.name: t
  for t in (
    Target("generic"),
    Target("apple-m1", ("-mcpu=apple-m1",), 16, 192 * _KIB, 128 * _KIB),
    Target("apple-m2", ("-mcpu=apple-m2",), 16, 192 * _KIB, 128 * _KIB),
    Target("apple-m3", ("-mcpu=apple-m3",), 16, 192 * _KIB, 128 * _KIB),
    Target("apple-m4", ("-mcpu=apple-m4",), 16, 192 * _KIB, 128 * _KIB),
    Target("armv8-a", ("-march=armv8-a",), 16, 64 * _KIB, 64 * _KIB),
    Target("cortex-a53", ("-mcpu=cortex-a53",), 16, 32 * _KIB, 32 * _KIB),
    Target("cortex-a72", ("-mcpu=cortex-a72",), 16, 48 * _KIB, 32 * _KIB),
    Target("cortex-a76", ("-mcpu=cortex-a76",), 16, 64 * _KIB, 64 * _KIB),
    Target("neoverse-v2", ("-mcpu=neoverse-v2",), 16, 64 * _KIB, 64 * _KIB),
    Target("x86-64", ("-march=x86-64",), 16, 32 * _KIB, 32 * _KIB),
    Target("x86-64-v3", ("-march=x86-64-v3",), 32, 32 * _KIB, 32 * _KIB),
    Target("x86-64-v4", ("-march=x86-64-v4",), 64, 32 * _KIB, 48 * _KIB),
  )
}
"""The named targets. Apple's are their performance cores; ``armv8-a`` is any 64-bit Arm core without
a preset of its own; ``x86-64``, ``x86-64-v3`` and ``x86-64-v4`` are the x86-64 microarchitecture
levels (SSE2; AVX2 and FMA; AVX-512) with the caches of a typical core of each."""

PORTABLE = PRESETS["apple-m3"]
"""The reference machine: every choice was measured on it, and under ``rounding="portable"`` every
target makes its choices."""

# Linux's "CPU part" numbers: the Arm cores with a preset, and Apple's (implementer 0x61), which
# Linux runs on up to the M2.
_ARM_PARTS = {"0xd03": "cortex-a53", "0xd08": "cortex-a72", "0xd0b": "cortex-a76", "0xd4f": "neoverse-v2"}
_APPLE_PARTS = {
  **dict.fromkeys(("0x022", "0x023", "0x024", "0x025", "0x028", "0x029"), "apple-m1"),
  **dict.fromkeys(("0x032", "0x033", "0x034", "0x035", "0x038", "0x039"), "apple-m2"),
}
_AVX512 = frozenset({"avx512f", "avx512bw", "avx512dq", "avx512vl"})


def _apple_preset(facts: CpuFacts) -> Target:
  if facts.part.lower() in _APPLE_PARTS:
    return PRESETS[_APPLE_PARTS[facts.part.lower()]]
  for generation in ("m4", "m3", "m2", "m1"):
    if f"apple {generation}" in facts.brand.lower():
      return PRESETS[f"apple-{generation}"]
  return PRESETS["apple-m4"]  # a later Apple core, or one the system did not name: its newest known relative


def _from_facts(facts: CpuFacts) -> Target:
  """The preset that best matches ``facts``, with the cache sizes the system reports."""
  if facts.machine == "aarch64":
    if facts.system == "darwin" or facts.implementer.lower() == "0x61" or "apple" in facts.brand.lower():
      base = _apple_preset(facts)
    else:
      base = PRESETS.get(_ARM_PARTS.get(facts.part.lower(), ""), PRESETS["armv8-a"])
  elif facts.machine == "x86_64":
    base = PRESETS["x86-64-v4" if _AVX512 <= facts.features else "x86-64-v3" if {"avx2", "fma"} <= facts.features else "x86-64"]
  else:
    base = PRESETS["generic"]
  sizes = {"l1i_bytes": facts.l1i, "l1d_bytes": facts.l1d}
  changes = {name: value for name, value in sizes.items() if value and value != getattr(base, name)}
  return replace(base, **changes) if changes else base


@functools.cache
def host_target() -> Target:
  """The processor this process runs on: the preset that matches it, with its reported caches."""
  return _from_facts(cpu_facts())


_current: ContextVar[Target | None] = ContextVar("scaly_target", default=None)
_default: Target | None = None


def resolve_target(value: Target | str | None) -> Target:
  """``value`` as a ``Target``: itself, the preset it names, or, for None, the target in force."""
  if value is None:
    return get_target()
  if isinstance(value, Target):
    return value
  if isinstance(value, str):
    return Target.preset(value)
  raise TypeError(f"a target is a Target or a preset name, got {value!r}")


def get_target() -> Target:
  """The target in force: the innermost ``sc.target`` block, else the process default, which is the
  preset ``SCALY_TARGET`` names or, without it, the host."""
  current = _current.get()
  if current is not None:
    return current
  global _default
  if _default is None:
    _default = Target.preset(env("SCALY_TARGET") or "host")
  return _default


def set_target(value: Target | str | None) -> None:
  """Change the process default, which applies wherever no ``sc.target`` block is in force; None
  restores the one ``SCALY_TARGET`` or the host gives."""
  global _default
  _default = None if value is None else resolve_target(value)


@contextmanager
def target(value: Target | str) -> Iterator[Target]:
  """Render for ``value`` (a ``Target`` or a preset name) inside a ``with`` block, in this context
  only (the thread's or the task's, as ``sc.options``), and restore on exit."""
  token = _current.set(resolve_target(value))
  try:
    yield get_target()
  finally:
    _current.reset(token)


__all__ = ["PORTABLE", "PRESETS", "Target", "get_target", "host_target", "resolve_target", "set_target", "target"]
