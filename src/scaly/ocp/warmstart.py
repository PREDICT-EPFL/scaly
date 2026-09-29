"""The warm start of a receding horizon: a primal-dual point moved up one stage (``shift``), and a first one (``initial_guess``)."""

from __future__ import annotations

from typing import Any

import numpy as np

from ..function.model import ConcreteFunction
from ..ir.expr import Expr, concat
from .formulate import Layout
from .method import REGISTRY, discrete
from .problem import DiscreteOCP


def shift(problem: DiscreteOCP, method: Any) -> ConcreteFunction[Any, Any, Any, Any]:
  """``point -> warm``: a point ``method`` reached moved up one stage, each stage's block of
  variables and multipliers by one, the last repeated: the warm start of the next solve of a
  receding horizon."""
  return REGISTRY.resolve(method, discrete(problem)).shift(problem)


def initial_guess(problem: DiscreteOCP, method: Any, x0: Any, u: Any = None) -> np.ndarray:
  """A first warm start for ``method``: every state ``x0``, every control ``u`` (zeros by default),
  a transcription's own variables from its ``guess``, slacks and multipliers zero."""
  return REGISTRY.resolve(method, discrete(problem)).initial_guess(problem, x0, u)


def split(layout: Layout, n_eq: int, flat: Any) -> tuple[list[Any], list[Any], Any, Any]:
  """A flat primal-dual point in its parts: the variables and their bounds' multipliers per leaf,
  the equality and the inequality multipliers."""
  cut, n = 0, layout.n_vars
  primal, box = [], []
  for size in layout.var_sizes:
    primal.append(flat[cut : cut + size])
    box.append(flat[n + cut : n + cut + size])
    cut += size
  return primal, box, flat[2 * n : 2 * n + n_eq], flat[2 * n + n_eq :]


def shifted(problem: DiscreteOCP, layout: Layout, n_eq: int, n_ineq: int, flat: Expr) -> Expr:
  """The point moved up one stage: each per-stage block of a variable, of a bound's multiplier, of
  the dynamics' and the path constraints' multipliers, by one stage, the last repeated."""
  primal, box, lam_eq, lam_ineq = split(layout, n_eq, flat)

  def moved(seg: Expr, block: int) -> Expr:
    return seg if block == 0 or seg.size <= block else concat([seg[block:], seg[seg.size - block :]])

  def runs(seg: Expr, blocks: list[tuple[int, int, int]]) -> Expr:
    pieces, cut = [], 0
    for offset, size, block in blocks:
      pieces += [seg[cut:offset], moved(seg[offset : offset + size], block)]
      cut = offset + size
    pieces.append(seg[cut:])
    return concat([p for p in pieces if p.size]) if any(p.size for p in pieces) else seg

  per_leaf = [_leaf_runs(problem, layout, name, size) for name, size in zip(layout.var_names, layout.var_sizes, strict=True)]
  eq_runs, ineq_runs = layout.multiplier_blocks()
  parts = [runs(p, r) for p, r in zip(primal, per_leaf, strict=True)] + [runs(b, r) for b, r in zip(box, per_leaf, strict=True)]
  return concat([*parts, *([runs(lam_eq, eq_runs)] if n_eq else []), *([runs(lam_ineq, ineq_runs)] if n_ineq else [])])


def _leaf_runs(problem: DiscreteOCP, layout: Layout, name: str, size: int) -> list[tuple[int, int, int]]:
  if name != "slack":
    return [(0, size, layout.var_blocks[layout.var_names.index(name)])]
  return [
    (offset, fn.outputs[0].size * problem.N, fn.outputs[0].size)
    for offset, fn in zip(layout.slack_offsets, problem.paths, strict=True)
    if offset >= 0
  ]


__all__ = ["initial_guess", "shift", "shifted", "split"]
