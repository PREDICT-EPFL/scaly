"""Typed backend-free problems, solver selection, descriptors, and solver statistics."""

from __future__ import annotations

from .model import ExternalOracle, SolverDescriptor, descriptor_function
from .problem import Bounded, Problem, ProblemSpec, bounded, problem
from .solver import solver
from .stats import ALLOY_SOLVER_STATS_VERSION, AlloySolveStatus, CSolverStats, SolverStats, SolverStatus

__all__ = [
  "ALLOY_SOLVER_STATS_VERSION",
  "AlloySolveStatus",
  "Bounded",
  "CSolverStats",
  "ExternalOracle",
  "Problem",
  "ProblemSpec",
  "SolverDescriptor",
  "SolverStats",
  "SolverStatus",
  "bounded",
  "descriptor_function",
  "problem",
  "solver",
]
