"""Alloy-owned solver statistics ABI."""

from __future__ import annotations

import ctypes
import enum
from dataclasses import dataclass, fields
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .solver_function import SolverStatus

ALLOY_SOLVER_STATS_VERSION = 1

# This order is the ABI: four int32, five doubles, then six int32 (80 bytes).
STATS_FIELDS = (
  ("version", "int32_t"),
  ("status", "int32_t"),
  ("native_status", "int32_t"),
  ("iter", "int32_t"),
  ("obj", "double"),
  ("t_total", "double"),
  ("t_fe", "double"),
  ("t_solver", "double"),
  ("t_glue", "double"),
  ("n_eval_f", "int32_t"),
  ("n_eval_grad_f", "int32_t"),
  ("n_eval_g", "int32_t"),
  ("n_eval_jac_g", "int32_t"),
  ("n_eval_h", "int32_t"),
  ("_pad0", "int32_t"),
)


class AlloySolveStatus(enum.IntEnum):
  OK = 0
  ACCEPTABLE = 1
  MAX_ITER = 2
  PRIMAL_INFEASIBLE = 3
  DUAL_INFEASIBLE = 4
  NUMERICS = 5
  USER_STOP = 6
  ERROR = 7


_CTYPE = {"int32_t": ctypes.c_int32, "double": ctypes.c_double}


class CSolverStats(ctypes.Structure):
  _fields_ = [(name, _CTYPE[c_type]) for name, c_type in STATS_FIELDS]


@dataclass(frozen=True, slots=True)
class SolverStats:
  version: int
  status: AlloySolveStatus
  native_status: int
  iter: int
  obj: float
  t_total: float
  t_fe: float
  t_solver: float
  t_glue: float
  n_eval_f: int
  n_eval_grad_f: int
  n_eval_g: int
  n_eval_jac_g: int
  n_eval_h: int
  _pad0: int = 0

  @classmethod
  def from_c(cls, value: CSolverStats) -> SolverStats:
    return cls(**{name: AlloySolveStatus(raw) if name == "status" else raw for name, _ in STATS_FIELDS if (raw := getattr(value, name)) is not None})

  def to_solver_status(self) -> SolverStatus:
    from .solver_function import SolverStatus

    counts = {name: getattr(self, name) for name, _ in STATS_FIELDS if name.startswith("n_eval_")}
    return SolverStatus(
      code=int(self.status),
      name=self.status.name,
      iter=self.iter,
      stats=counts,
      _ok=self.status in (AlloySolveStatus.OK, AlloySolveStatus.ACCEPTABLE),
    )


assert tuple(f.name for f in fields(SolverStats)) == tuple(name for name, _ in STATS_FIELDS)
assert ctypes.sizeof(CSolverStats) == 80


def stats_c_defs() -> list[str]:
  lines = [
    "#ifndef ALLOY_SOLVER_STATS_DEFINED",
    "#define ALLOY_SOLVER_STATS_DEFINED",
    f"#define ALLOY_SOLVER_STATS_VERSION {ALLOY_SOLVER_STATS_VERSION}",
  ]
  lines += [f"#define ALLOY_SOLVE_{status.name} {int(status)}" for status in AlloySolveStatus]
  lines += ["typedef struct {"]
  lines += [f"  {c_type} {name};" for name, c_type in STATS_FIELDS]
  lines += ["} alloy_solver_stats;", "#endif"]
  return lines


def stats_c_timing_defs() -> list[str]:
  return [
    "#ifndef ALLOY_SOLVER_TIMING_DEFINED",
    "#define ALLOY_SOLVER_TIMING_DEFINED",
    "static double alloy_clock_s(void) {",
    "#ifdef __APPLE__",
    "  return 1e-9 * (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);",
    "#else",
    "  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);",
    "  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;",
    "#endif",
    "}",
    "#endif",
  ]
