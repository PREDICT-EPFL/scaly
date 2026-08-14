"""QP and NLP solver builders.

This package exposes the user-facing builders :func:`qp` and :func:`nlp`. Both
return a callable :class:`SolverFunction` whose oracle is an ordinary Alloy
``Function`` and whose solve is a generated C wrapper driving the vendored
PIQP / IPOPT C APIs directly (``codegen/solver_c``) — JIT-compiled like any
other alloy function, with no Python in the solve loop.
"""

from __future__ import annotations

from .nlp import nlp
from .qp import qp
from .solver_function import ExternalOracle, SolverDescriptor, SolverFunction, SolverStatus
from .stats import ALLOY_SOLVER_STATS_VERSION, AlloySolveStatus, CSolverStats, SolverStats

__all__ = [
  "ALLOY_SOLVER_STATS_VERSION",
  "AlloySolveStatus",
  "CSolverStats",
  "ExternalOracle",
  "SolverDescriptor",
  "SolverFunction",
  "SolverStats",
  "SolverStatus",
  "nlp",
  "qp",
]
