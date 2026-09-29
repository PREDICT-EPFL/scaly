"""Optimal control: problems over a continuous-time model or a discrete-time map, their transcription from one to the other, their formulation as optimization problems, the methods that solve them, warm starts and terminal ingredients."""

from . import terminal as terminal
from .direct import Direct
from .formulate import Form, Layout, to_problem
from .ilqr import ILQR
from .method import REGISTRY, Info, solver
from .problem import METHOD_API, ContinuousOCP, DiscreteOCP, Param, Path, Quadratic, StageStructure, TerminalEquality, transcribe
from .terminal import largest_ellipsoid, lqr, max_invariant_set
from .tinyadmm import TinyADMM
from .transcription import Collocation, Interval, MultipleShooting, Pseudospectral, Transcription
from .warmstart import initial_guess, shift

# The method classes of other distributions, loaded on first use: ``sc.ocp.ALTRO`` and ``sc.ocp.SCvx``
# come with scaly-experimental, and name it when it is missing.
__getattr__ = REGISTRY.attribute(__name__)

__all__ = [
  "METHOD_API",
  "REGISTRY",
  "Collocation",
  "ContinuousOCP",
  "Direct",
  "DiscreteOCP",
  "Form",
  "ILQR",
  "Info",
  "Interval",
  "Layout",
  "MultipleShooting",
  "Param",
  "Path",
  "Pseudospectral",
  "Quadratic",
  "StageStructure",
  "TerminalEquality",
  "TinyADMM",
  "Transcription",
  "initial_guess",
  "largest_ellipsoid",
  "lqr",
  "max_invariant_set",
  "shift",
  "solver",
  "to_problem",
  "transcribe",
]
