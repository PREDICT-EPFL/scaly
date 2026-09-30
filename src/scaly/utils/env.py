"""Process environment and platform facts scaly reads.

A leaf: it imports nothing from scaly, so the pieces that need an environment variable or a
shared-library suffix — AD, solver-library discovery, the JIT — can share one definition without
depending on each other. The compiler's variables are registered here; a package that reads its
own (the solvers) lists them through the ``scaly.env_vars`` entry points, and
``scaly_env_vars`` joins them, so the JIT, tests and diagnostics agree on one list.
"""

from __future__ import annotations

import functools
import os
import platform
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import entry_points
from pathlib import Path


class ToolchainError(RuntimeError):
  """Raised when native compiler/header/library discovery fails."""


@dataclass(frozen=True, slots=True)
class EnvVar:
  name: str
  default: str | None
  help: str


ENV_VARS: tuple[EnvVar, ...] = (
  EnvVar("SCALY_CACHE_DIR", None, "Override the JIT cache root."),
  EnvVar("SCALY_CC", None, "Override the C compiler used by the JIT."),
  EnvVar("SCALY_CC_OPT", "-O2", "Optimization flag the JIT passes to the C compiler."),
  EnvVar("SCALY_STRICT_JVP_MANY", "0", "Raise instead of using the unrolled multi-seed JVP fallback."),
  EnvVar("SCALY_TARGET", None, "The processor preset code is generated for when none is set in code (default: the host's)."),
  EnvVar("SCALY_VIZ_DIR", None, "Visualization recording directory."),
)


def env(name: str, default: str | None = None) -> str | None:
  return os.environ.get(name, default)


def env_bool(name: str, default: bool = False) -> bool:
  raw = os.environ.get(name)
  if raw is None or raw == "":
    return default
  val = raw.strip().lower()
  if val in {"1", "true", "yes", "on"}:
    return True
  if val in {"0", "false", "no", "off"}:
    return False
  raise ToolchainError(f"{name} must be a boolean (1/0, true/false, yes/no, on/off), got {raw!r}")


def env_path(name: str) -> Path | None:
  raw = os.environ.get(name)
  return Path(raw).expanduser() if raw else None


ENV_VAR_ENTRY_POINTS = "scaly.env_vars"
"""The entry-point group through which a package lists the environment variables it reads."""


def scaly_env_vars() -> tuple[EnvVar, ...]:
  """Every environment variable the installed packages read: the compiler's, then each package's."""
  extra = [var for ep in sorted(entry_points(group=ENV_VAR_ENTRY_POINTS), key=lambda ep: ep.name) for var in ep.load()]
  return (*ENV_VARS, *extra)


def shared_lib_ext() -> str:
  if sys.platform == "darwin":
    return ".dylib"
  if sys.platform == "win32":
    return ".dll"
  return ".so"


def shared_lib_flag() -> str:
  return "-dynamiclib" if sys.platform == "darwin" else "-shared"


@dataclass(frozen=True, slots=True)
class CpuFacts:
  """What the operating system says about the processor this process runs on. ``machine`` is
  ``aarch64``, ``x86_64`` or what ``platform.machine()`` says, ``system`` is ``sys.platform``.
  Caches are those of the core generated code runs on (Apple's performance cores); a size the system
  does not report is None. ``implementer`` and ``part`` are Arm's numbers on Linux, ``features`` the
  x86 extensions that decide a preset (``avx2``, ``fma``, ``avx512f``, ``avx512bw``, ``avx512dq``,
  ``avx512vl``)."""

  machine: str
  system: str = ""
  brand: str = ""
  implementer: str = ""
  part: str = ""
  features: frozenset[str] = frozenset()
  l1i: int | None = None
  l1d: int | None = None


_FEATURES = frozenset({"avx2", "fma", "avx512f", "avx512bw", "avx512dq", "avx512vl"})
_SYSCTL_SIZES = ("hw.perflevel0.l1icachesize", "hw.perflevel0.l1dcachesize", "hw.l1icachesize", "hw.l1dcachesize")
_SYSCTL_FEATURES = {"hw.optional.avx2_0": "avx2", "hw.optional.fma": "fma", **{f"hw.optional.{f}": f for f in _FEATURES if f.startswith("avx512")}}


def _sysctl() -> str:
  # Its own path first: a restricted PATH (cron's /usr/bin:/bin) does not reach /usr/sbin.
  return "/usr/sbin/sysctl" if Path("/usr/sbin/sysctl").exists() else "sysctl"


def _darwin_facts(machine: str, out: str | None = None) -> CpuFacts:
  """The facts ``sysctl`` gives, or ``out`` as it would print them."""
  if out is None:
    # One call for every key: a key this machine lacks is reported on stderr and the rest still print.
    keys = ("machdep.cpu.brand_string", *_SYSCTL_SIZES, *_SYSCTL_FEATURES)
    out = subprocess.run([_sysctl(), *keys], capture_output=True, text=True, check=False).stdout
  values = dict(line.split(": ", 1) for line in out.splitlines() if ": " in line)

  def size(*keys: str) -> int | None:
    return next((int(values[k]) for k in keys if values.get(k, "").strip().isdigit() and int(values[k]) > 0), None)

  return CpuFacts(
    machine,
    "darwin",
    brand=values.get("machdep.cpu.brand_string", "").strip(),
    features=frozenset(name for key, name in _SYSCTL_FEATURES.items() if values.get(key, "").strip() == "1"),
    l1i=size("hw.perflevel0.l1icachesize", "hw.l1icachesize"),
    l1d=size("hw.perflevel0.l1dcachesize", "hw.l1dcachesize"),
  )


def _sysfs_size(text: str) -> int | None:
  text = text.strip().upper()
  scale = {"K": 1024, "M": 1024 * 1024}.get(text[-1:], 1)
  digits = text.rstrip("KM")
  return int(digits) * scale if digits.isdigit() else None


def _linux_facts(machine: str, root: Path = Path("/")) -> CpuFacts:
  """The facts ``/proc/cpuinfo`` and ``/sys/devices/system/cpu`` give, under ``root``."""
  info: dict[str, str] = {}
  try:
    for line in (root / "proc/cpuinfo").read_text().splitlines():
      key, sep, value = line.partition(":")
      if sep and key.strip() not in info:
        info[key.strip()] = value.strip()
  except OSError:
    pass
  flags = set((info.get("flags") or info.get("Features") or "").split())
  sizes: dict[str, int | None] = {}
  for index in sorted((root / "sys/devices/system/cpu/cpu0/cache").glob("index*")):
    try:
      level, kind = (index / "level").read_text().strip(), (index / "type").read_text().strip()
      sizes.setdefault(f"{level}{kind}", _sysfs_size((index / "size").read_text()))
    except OSError:
      continue
  return CpuFacts(
    machine,
    "linux",
    brand=info.get("model name", ""),
    implementer=info.get("CPU implementer", ""),
    part=info.get("CPU part", ""),
    features=frozenset(flags & _FEATURES),
    l1i=sizes.get("1Instruction"),
    l1d=sizes.get("1Data"),
  )


@functools.cache
def cpu_facts() -> CpuFacts:
  """The facts about this process's processor that code generation reads, looked up once."""
  machine = platform.machine().lower()
  machine = {"arm64": "aarch64", "amd64": "x86_64"}.get(machine, machine)
  try:
    if sys.platform == "darwin":
      return _darwin_facts(machine)
    if sys.platform.startswith("linux"):
      return _linux_facts(machine)
  except (OSError, ValueError):
    pass
  return CpuFacts(machine, sys.platform)
