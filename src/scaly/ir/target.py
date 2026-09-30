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

type Arch = Literal["aarch64", "x86_64", "generic"]
type Rounding = Literal["target", "portable"]

_KIB = 1024
_MIB = 1024 * 1024


@dataclass(frozen=True, slots=True)
class Target:
  """A description of the processor generated code is tuned for.

  The fields are facts about the hardware; the properties below them are the code-generation
  choices derived from those facts, each documented with the measurement that fixed it on the
  machine it was measured on. A choice whose derivation has not been measured on a target still
  follows the formula there.

  A choice can change the last bits of a result even where it keeps the order of every sum: the
  C compiler fuses a multiply and an add into one rounding (``-ffp-contract=on``) or not depending
  on how it vectorizes the code around them, and a block width changes that. Under
  ``rounding="portable"`` every choice that shapes floating-point code is the one measured on the
  reference machine, the Apple M3 (``PORTABLE``), whatever the target, so the generated C is the
  same for every target and results differ between machines only as far as their compilers do.

  Attributes:
    name: the preset this target is, or was derived from.
    arch: ``"aarch64"``, ``"x86_64"``, or ``"generic"`` for plain scalar C.
    cflags: the compiler flags that select this processor for ahead-of-time builds. The JIT
      ignores them and compiles for the processor it runs on.
    vector_bytes: the width of one SIMD register (8 for scalar code).
    vector_registers: the architectural SIMD (or floating-point) registers.
    fma_units: the fused multiply-adds issued per cycle.
    fma_latency: the cycles before a dependent fused multiply-add can start.
    fma_by_lane: whether a fused multiply-add can take one lane of a register as a scalar
      operand (NEON's ``fmla v, v, v.d[i]``).
    l1i_bytes, l1d_bytes, l2_bytes: the caches of the core generated code runs on.
    line_bytes: the cache line.
    rounding: ``"target"`` lets a choice that shapes floating-point code (a block width, a lane
      count) follow this target, so results may differ in the last bits between targets;
      ``"portable"`` gives every such choice the reference machine's value.
  """

  name: str = "generic"
  arch: Arch = "generic"
  cflags: tuple[str, ...] = ()
  vector_bytes: int = 8
  vector_registers: int = 16
  fma_units: int = 1
  fma_latency: int = 4
  fma_by_lane: bool = False
  l1i_bytes: int = 32 * _KIB
  l1d_bytes: int = 32 * _KIB
  l2_bytes: int = 256 * _KIB
  line_bytes: int = 64
  rounding: Rounding = "target"

  def __post_init__(self) -> None:
    if self.arch not in ("aarch64", "x86_64", "generic"):
      raise ValueError(f"Target.arch must be 'aarch64', 'x86_64' or 'generic', got {self.arch!r}")
    if self.rounding not in ("target", "portable"):
      raise ValueError(f"Target.rounding must be 'target' or 'portable', got {self.rounding!r}")
    if self.vector_bytes not in (8, 16, 32, 64):
      raise ValueError(f"Target.vector_bytes must be 8, 16, 32 or 64, got {self.vector_bytes!r}")
    counts = ("vector_registers", "fma_units", "fma_latency", "l1i_bytes", "l1d_bytes", "l2_bytes", "line_bytes")
    for name in counts:
      value = getattr(self, name)
      if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
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

  @classmethod
  def host(cls) -> Target:
    """The processor this process runs on, as its operating system describes it (``host_target``)."""
    return host_target()

  @property
  def vector_doubles(self) -> int:
    """The ``double`` lanes of one SIMD register."""
    return self.vector_bytes // 8

  @property
  def choices(self) -> Target:
    """The target whose facts the code-generation choices follow: this one, or under
    ``rounding="portable"`` the reference machine (``PORTABLE``)."""
    return self if self.rounding == "target" else PORTABLE

  @property
  def row_blocks(self) -> tuple[int, ...]:
    """The widths, widest first, of the blocks of a row that ``x @ b`` and ``a @ b`` keep in
    registers across the reduction: 8, 4 and 2 registers' worth of ``double``. On the M3 that is 16,
    8 and 4 columns (``notes/codegen_speed_o6_report.html``: ``matmul_48`` 2.13x faster than the
    reduction outermost). Each output still sums in order of ``k``, but the widths decide how the
    compiler vectorizes the sums, and so which of them it fuses into multiply-adds: they are a
    rounding choice."""
    v = self.choices.vector_doubles
    return (8 * v, 4 * v, 2 * v)

  @property
  def row_blocked_max(self) -> int:
    """The widest row that is blocked at all, four of the widest blocks: 64 columns on the M3,
    where a wider row streamed with the reduction outermost measured faster (unbumpercars' 128 x 256
    products, 1.13x)."""
    return 4 * self.row_blocks[0]


PRESETS: dict[str, Target] = {
  t.name: t
  for t in (
    Target("generic"),
    Target(
      "apple-m1",
      "aarch64",
      ("-mcpu=apple-m1",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=4,
      fma_latency=4,
      fma_by_lane=True,
      l1i_bytes=192 * _KIB,
      l1d_bytes=128 * _KIB,
      l2_bytes=12 * _MIB,
      line_bytes=128,
    ),
    Target(
      "apple-m2",
      "aarch64",
      ("-mcpu=apple-m2",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=4,
      fma_latency=4,
      fma_by_lane=True,
      l1i_bytes=192 * _KIB,
      l1d_bytes=128 * _KIB,
      l2_bytes=16 * _MIB,
      line_bytes=128,
    ),
    Target(
      "apple-m3",
      "aarch64",
      ("-mcpu=apple-m3",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=4,
      fma_latency=4,
      fma_by_lane=True,
      l1i_bytes=192 * _KIB,
      l1d_bytes=128 * _KIB,
      l2_bytes=16 * _MIB,
      line_bytes=128,
    ),
    Target(
      "apple-m4",
      "aarch64",
      ("-mcpu=apple-m4",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=4,
      fma_latency=4,
      fma_by_lane=True,
      l1i_bytes=192 * _KIB,
      l1d_bytes=128 * _KIB,
      l2_bytes=16 * _MIB,
      line_bytes=128,
    ),
    Target(
      "cortex-a53",
      "aarch64",
      ("-mcpu=cortex-a53",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=1,
      fma_latency=8,
      fma_by_lane=True,
      l1i_bytes=32 * _KIB,
      l1d_bytes=32 * _KIB,
      l2_bytes=512 * _KIB,
      line_bytes=64,
    ),
    Target(
      "cortex-a72",
      "aarch64",
      ("-mcpu=cortex-a72",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=2,
      fma_latency=7,
      fma_by_lane=True,
      l1i_bytes=48 * _KIB,
      l1d_bytes=32 * _KIB,
      l2_bytes=1 * _MIB,
      line_bytes=64,
    ),
    Target(
      "cortex-a76",
      "aarch64",
      ("-mcpu=cortex-a76",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=2,
      fma_latency=4,
      fma_by_lane=True,
      l1i_bytes=64 * _KIB,
      l1d_bytes=64 * _KIB,
      l2_bytes=512 * _KIB,
      line_bytes=64,
    ),
    Target(
      "neoverse-v2",
      "aarch64",
      ("-mcpu=neoverse-v2",),
      vector_bytes=16,
      vector_registers=32,
      fma_units=4,
      fma_latency=4,
      fma_by_lane=True,
      l1i_bytes=64 * _KIB,
      l1d_bytes=64 * _KIB,
      l2_bytes=1 * _MIB,
      line_bytes=64,
    ),
    Target(
      "x86-64",
      "x86_64",
      ("-march=x86-64",),
      vector_bytes=16,
      vector_registers=16,
      fma_units=2,
      fma_latency=4,
      l1i_bytes=32 * _KIB,
      l1d_bytes=32 * _KIB,
      l2_bytes=256 * _KIB,
      line_bytes=64,
    ),
    Target(
      "x86-64-v3",
      "x86_64",
      ("-march=x86-64-v3",),
      vector_bytes=32,
      vector_registers=16,
      fma_units=2,
      fma_latency=4,
      l1i_bytes=32 * _KIB,
      l1d_bytes=32 * _KIB,
      l2_bytes=1 * _MIB,
      line_bytes=64,
    ),
    Target(
      "x86-64-v4",
      "x86_64",
      ("-march=x86-64-v4",),
      vector_bytes=64,
      vector_registers=32,
      fma_units=2,
      fma_latency=4,
      l1i_bytes=32 * _KIB,
      l1d_bytes=48 * _KIB,
      l2_bytes=1 * _MIB,
      line_bytes=64,
    ),
  )
}
"""The named targets. Apple's are their performance cores; ``x86-64``, ``x86-64-v3`` and
``x86-64-v4`` are the x86-64 microarchitecture levels (SSE2; AVX2 and FMA; AVX-512) with the caches of
a typical core of each."""

PORTABLE = PRESETS["apple-m3"]
"""The reference machine: every choice was measured on it, and under ``rounding="portable"`` every
target makes its choices."""

# Linux's "CPU part" numbers of the Arm cores with a preset.
_ARM_PARTS = {"0xd03": "cortex-a53", "0xd08": "cortex-a72", "0xd0b": "cortex-a76", "0xd4f": "neoverse-v2"}


def _apple_preset(brand: str) -> Target:
  for generation in ("m4", "m3", "m2", "m1"):
    if f"apple {generation}" in brand.lower():
      return PRESETS[f"apple-{generation}"]
  return PRESETS["apple-m4"]  # a later Apple core: its newest known relative


def _from_facts(facts: CpuFacts) -> Target:
  """The preset that best matches ``facts``, with the cache sizes the system reports."""
  if facts.machine == "aarch64":
    if "apple" in facts.brand.lower():
      base = _apple_preset(facts.brand)
    else:
      base = PRESETS.get(_ARM_PARTS.get(facts.part.lower(), ""), PRESETS["cortex-a76"])
  elif facts.machine == "x86_64":
    base = PRESETS["x86-64-v4" if "avx512f" in facts.features else "x86-64-v3" if {"avx2", "fma"} <= facts.features else "x86-64"]
  else:
    base = PRESETS["generic"]
  sizes = {"l1i_bytes": facts.l1i, "l1d_bytes": facts.l1d, "l2_bytes": facts.l2, "line_bytes": facts.line}
  return replace(base, **{name: value for name, value in sizes.items() if value})


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
  """Change the process default, which every thread sees outside an ``sc.target`` block; None
  restores the one ``SCALY_TARGET`` or the host gives."""
  global _default
  _default = None if value is None else resolve_target(value)


@contextmanager
def target(value: Target | str) -> Iterator[Target]:
  """Render for ``value`` (a ``Target`` or a preset name) inside a ``with`` block, for this thread or
  task only, and restore on exit."""
  token = _current.set(resolve_target(value))
  try:
    yield get_target()
  finally:
    _current.reset(token)


__all__ = ["PORTABLE", "PRESETS", "Target", "get_target", "host_target", "resolve_target", "set_target", "target"]
