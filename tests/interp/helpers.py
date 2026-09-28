"""Evaluating an interpolant through the JIT at a batch of points, with its derivatives in both modes."""

from __future__ import annotations

import itertools

import numpy as np

import scaly as sc
from scaly.interp import BSpline

_COUNTER = itertools.count()


def evaluate(f: BSpline, points: np.ndarray, *, derivatives: bool = False) -> dict[str, np.ndarray]:
  """``f`` at ``points`` (``(N,)`` in 1-D, ``(N, D)``) as one batch Function. With ``derivatives``:
  ``grad`` (reverse) and ``jvp`` (forward) hold the first derivatives, ``(N,)`` in 1-D and ``(N, D)``,
  each summed over the outputs; ``hess`` (reverse over reverse) and ``hess_fwd`` (forward over
  forward) the second, ``(N,)`` or ``(N, D, D)``."""
  points = np.asarray(points, dtype=np.float64)
  x = sc.sym("x", points.shape)
  y = f(x)
  outs, names = [y], ["y"]
  if derivatives:
    total = y.sum() if y.shape else y
    grad = sc.gradient(total, x)
    n, width = points.shape[0], int(np.prod(f.out_shape))

    def over_outputs(e: sc.Expr) -> sc.Expr:
      return e.reshape((n, width)) @ sc.const(np.ones(width)) if f.out_shape else e

    if f.ndim == 1:
      ones = sc.const(np.ones(points.shape))
      jvp = over_outputs(sc.jvp(y, x, ones))
      hess = sc.gradient(grad.sum(), x)
      hess_fwd = sc.jvp(jvp, x, ones)
    else:
      d = f.ndim
      seeds = [sc.const(np.repeat(np.eye(d)[i][None, :], n, axis=0)) for i in range(d)]
      jvp = sc.stack([over_outputs(sc.jvp(y, x, s)) for s in seeds], axis=1)
      hess = sc.stack([sc.gradient(grad[:, i].sum(), x) for i in range(d)], axis=1)
      hess_fwd = sc.stack([sc.stack([sc.jvp(jvp[:, i], x, s) for s in seeds], axis=1) for i in range(d)], axis=1)
    outs += [grad, jvp, hess, hess_fwd]
    names += ["grad", "jvp", "hess", "hess_fwd"]
  fn = sc.Function._from_exprs(f"interp_eval_{next(_COUNTER)}", [x], outs, ["x"], names)
  values = fn(points)
  values = values if isinstance(values, tuple) else (values,)
  return dict(zip(names, (np.asarray(v) for v in values), strict=True))


def inside_points(edges: tuple[np.ndarray, ...], rng: np.random.Generator, n_random: int) -> np.ndarray:
  """Every edge, the floats just inside the partition on both sides of each edge, and random points,
  per axis; points of an n-D grid combine the axes' lists at random."""
  per_axis = []
  for e in edges:
    near = np.concatenate([e, np.nextafter(e[1:], -np.inf), np.nextafter(e[:-1], np.inf), rng.uniform(e[0], e[-1], n_random)])
    per_axis.append(near)
  if len(edges) == 1:
    return per_axis[0]
  n = max(p.size for p in per_axis)
  return np.column_stack([rng.choice(p, n) if p.size < n else rng.permutation(p) for p in per_axis])


def scale(reference: np.ndarray) -> float:
  finite = np.abs(reference[np.isfinite(reference)])
  return float(max(1.0, finite.max())) if finite.size else 1.0


def numbers(values: object) -> np.ndarray:
  """A spline's coefficients, or an ``Expr``'s constant value, known here to be numbers."""
  value = values.value if isinstance(values, sc.Expr) else values
  assert isinstance(value, np.ndarray)
  return value
