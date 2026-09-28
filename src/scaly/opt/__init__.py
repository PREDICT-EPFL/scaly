"""Optimization problems and the solvers that solve them: problems declared over typed variables and
parameters, their quadratic and nonlinear normal forms, and ``solver``."""

from __future__ import annotations

from .external import SCALY_SOLVER_STATS_VERSION, SolverStats, SolverStatus, solver_stats
from .problem import NO_LB, NO_UB, Bounded, NLP, ProblemSpec, bounded, problem
from .nlp import NLPOracles, nlp_oracles
from .qp import QP, NotQuadratic, QPData, QPForm, extract_qp
from .method import REGISTRY, Info
from .solver import solver

# The methods' classes, loaded from their distributions on first use: ``sc.opt.PIQP``.
__getattr__ = REGISTRY.attribute(__name__)

__all__ = [
  "NLP",
  "NO_LB",
  "NLPOracles",
  "NO_UB",
  "QP",
  "SCALY_SOLVER_STATS_VERSION",
  "Bounded",
  "Info",
  "NotQuadratic",
  "ProblemSpec",
  "QPData",
  "QPForm",
  "REGISTRY",
  "SolverStats",
  "SolverStatus",
  "bounded",
  "extract_qp",
  "nlp_oracles",
  "problem",
  "solver",
  "solver_stats",
]
