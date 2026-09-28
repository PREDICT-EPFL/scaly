"""Scaly-owned solver statistics ABI."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass, fields

from ...function.method import Status

SCALY_SOLVER_STATS_VERSION = 3

# This order is the ABI: four int32, seven doubles, six int32, then the v3
# diagnostics tail — four doubles (8-aligned at offset 96) and two int32
# (136 bytes total). New fields only ever append; never reorder existing ones.
STATS_FIELDS = (
  ("version", "int32_t"),
  ("status", "int32_t"),
  ("native_status", "int32_t"),
  ("iter", "int32_t"),
  ("obj", "double"),
  ("t_total", "double"),
  ("t_fe", "double"),
  ("t_solver", "double"),
  ("t_qp", "double"),
  ("t_globalization", "double"),
  ("t_glue", "double"),
  ("n_eval_f", "int32_t"),
  ("n_eval_grad_f", "int32_t"),
  ("n_eval_g", "int32_t"),
  ("n_eval_jac_g", "int32_t"),
  ("n_eval_h", "int32_t"),
  ("_pad0", "int32_t"),
  # v3 diagnostics: zero when the backend has no such concept.
  ("primal_viol", "double"),  # constraint violation (inf norm) at the returned x
  ("step_inf", "double"),  # inf norm of the last computed step
  ("alpha", "double"),  # last accepted line-search step length; 0.0 if no step was accepted
  ("merit_penalty", "double"),  # final merit penalty parameter
  ("backtracks", "int32_t"),  # total rejected line-search trial points across the solve
  ("qp_iter", "int32_t"),  # QP iteration count (accumulated across SQP iterations)
)


@dataclass(frozen=True, slots=True)
class SolverStatus:
  code: int
  name: str
  iter: int = 0
  stats: dict[str, int] | None = None
  _ok: bool | None = None

  @property
  def ok(self) -> bool:
    # code is the scaly status enum: OK == 0, ACCEPTABLE == 1.
    return self.code in (0, 1) if self._ok is None else self._ok


_CTYPE = {"int32_t": ctypes.c_int32, "double": ctypes.c_double}


class CSolverStats(ctypes.Structure):
  _fields_ = [(name, _CTYPE[c_type]) for name, c_type in STATS_FIELDS]


@dataclass(frozen=True, slots=True)
class SolverStats:
  version: int
  status: Status
  native_status: int
  iter: int
  obj: float
  t_total: float
  t_fe: float
  t_solver: float
  t_qp: float
  t_globalization: float
  t_glue: float
  n_eval_f: int
  n_eval_grad_f: int
  n_eval_g: int
  n_eval_jac_g: int
  n_eval_h: int
  _pad0: int = 0
  primal_viol: float = 0.0
  step_inf: float = 0.0
  alpha: float = 0.0
  merit_penalty: float = 0.0
  backtracks: int = 0
  qp_iter: int = 0

  @classmethod
  def from_c(cls, value: CSolverStats) -> SolverStats:
    return cls(**{name: Status(raw) if name == "status" else raw for name, _ in STATS_FIELDS if (raw := getattr(value, name)) is not None})

  def to_solver_status(self) -> SolverStatus:
    counts = {name: getattr(self, name) for name, _ in STATS_FIELDS if name.startswith("n_eval_")}
    return SolverStatus(
      code=int(self.status),
      name=self.status.name,
      iter=self.iter,
      stats=counts,
      _ok=self.status in (Status.OK, Status.ACCEPTABLE),
    )


assert tuple(f.name for f in fields(SolverStats)) == tuple(name for name, _ in STATS_FIELDS)
assert ctypes.sizeof(CSolverStats) == 136


def stats_c_defs() -> list[str]:
  lines = [
    "#ifndef SCALY_SOLVER_STATS_DEFINED",
    "#define SCALY_SOLVER_STATS_DEFINED",
    f"#define SCALY_SOLVER_STATS_VERSION {SCALY_SOLVER_STATS_VERSION}",
  ]
  lines += [f"#define SCALY_SOLVE_{status.name} {int(status)}" for status in Status]
  lines += ["typedef struct {"]
  lines += [f"  {c_type} {name};" for name, c_type in STATS_FIELDS]
  lines += ["} scaly_solver_stats;", "#endif"]
  return lines


def stats_c_timing_defs() -> list[str]:
  return [
    "#ifndef SCALY_SOLVER_TIMING_DEFINED",
    "#define SCALY_SOLVER_TIMING_DEFINED",
    "static double scaly_clock_s(void) {",
    "#ifdef __APPLE__",
    "  return 1e-9 * (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);",
    "#else",
    "  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);",
    "  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;",
    "#endif",
    "}",
    "#endif",
  ]
