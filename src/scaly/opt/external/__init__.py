"""External optimization solvers: the ``External`` method base the solver plugins build on, the extern callee a solver Function carries, its C wrapper frame, statistics and library paths."""

from .method import External
from .model import ExternalOracle, SolverDescriptor, descriptor_function
from .stats import SCALY_SOLVER_STATS_VERSION, CSolverStats, SolverStats, SolverStatus
from .wrapper import SolverWrapperCtx, solver_stats

__all__ = [
  "SCALY_SOLVER_STATS_VERSION",
  "CSolverStats",
  "External",
  "ExternalOracle",
  "SolverDescriptor",
  "SolverStats",
  "SolverStatus",
  "SolverWrapperCtx",
  "descriptor_function",
  "solver_stats",
]
