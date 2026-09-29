"""Optimal control: problems over a continuous-time model or a discrete-time map, their transcription from one to the other, and their formulation as optimization problems."""

from .formulate import Form, Layout, to_problem
from .problem import METHOD_API, ContinuousOCP, DiscreteOCP, Param, Path, Quadratic, StageStructure, TerminalEquality, transcribe
from .transcription import Collocation, Interval, MultipleShooting, Pseudospectral, Transcription

__all__ = [
  "METHOD_API",
  "Collocation",
  "ContinuousOCP",
  "DiscreteOCP",
  "Form",
  "Interval",
  "Layout",
  "MultipleShooting",
  "Param",
  "Path",
  "Pseudospectral",
  "Quadratic",
  "StageStructure",
  "TerminalEquality",
  "Transcription",
  "to_problem",
  "transcribe",
]
