# /// script
# requires-python = ">=3.12"
# dependencies = ["scaly"]
# ///
"""Entropic optimal transport: log-domain Sinkhorn with segment reductions, and its gradient.

Moving the histogram ``a`` on points ``x`` to the histogram ``b`` on points ``y`` at cost
``C_ij = |x_i - y_j|^2`` is regularized with entropy ``eps``; its dual is solved by the Sinkhorn
iteration on the potentials ``(f, g)``,

    f_i = eps log a_i - eps LSE_j((g_j - C_ij) / eps),   g_j = eps log b_j - eps LSE_i((f_i - C_ij) / eps),

written in the log domain so that small ``eps`` does not underflow. The two log-sum-exps are per
row and per column of an ``n x m`` array, which Scaly expresses with ``sc.segment_max`` (the
stabilizing shift) and ``sc.segment_sum`` over static segment ids: the flat entry ``(i, j)`` belongs
to row segment ``i`` and column segment ``j``. The iteration runs in a ``sc.while_loop`` until the
row marginal of the plan matches ``a``.

The regularized cost ``W_eps(a, b) = <f, a> + <g, b>`` is differentiable in ``a``; the envelope
theorem says its gradient is the potential ``f`` (up to a constant, since ``a`` lives on the
simplex). ``sc.gradient`` through the loop, which differentiates the iterations actually taken,
gives the same answer, and so do central differences. For a small ``eps`` the transport cost of the
plan, ``<P, C>``, approaches the exact one-dimensional ``W_2^2``, computed here from the quantile
functions.

The generated C lands in ``examples/generated/sinkhorn_transport/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import scaly as sc
from scaly.codegen import write_module

GENERATED = Path(__file__).resolve().parents[1] / "generated" / "sinkhorn_transport"
N, M = 100, 80
EPS, TOL, MAX_ITER = 2e-3, 1e-11, 3000
ROW = np.repeat(np.arange(N), M)  # the segment of flat entry (i, j) in the row reductions
COL = np.tile(np.arange(M), N)  # and in the column reductions


def lse(values: sc.Expr, segments: np.ndarray, count: int) -> sc.Expr:
  """``log sum exp`` of ``values`` in each segment, shifted by the segment maximum."""
  shift = sc.segment_max(values, segments, count)
  return shift + sc.segment_sum((values - sc.gather(shift, segments)).exp(), segments, count).log()


def cost_matrix(x: sc.Expr, y: sc.Expr) -> sc.Expr:
  d = sc.gather(x, ROW) - sc.gather(y, COL)
  return d * d  # flat, (N * M,)


@sc.function
def sinkhorn_step(fg: sc.Expr, a: sc.Expr, b: sc.Expr, c: sc.Expr) -> sc.Expr:
  g = fg[N : N + M]
  f = EPS * a.log() - EPS * lse((sc.gather(g, COL) - c) / EPS, ROW, N)
  g = EPS * b.log() - EPS * lse((sc.gather(f, ROW) - c) / EPS, COL, M)
  plan = ((sc.gather(f, ROW) + sc.gather(g, COL) - c) / EPS).exp()
  row_error = (sc.segment_sum(plan, ROW, N) - a).abs().sum()
  return sc.concat([f, g, row_error.reshape((1,))])


@sc.function
def not_converged(fg: sc.Expr, a: sc.Expr, b: sc.Expr, c: sc.Expr) -> sc.Expr:
  return sc.greater(fg[N + M], TOL)


def entropic_cost(a: sc.Expr, b: sc.Expr, x: sc.Expr, y: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  c = cost_matrix(x, y)
  start = sc.concat([sc.const(np.zeros(N + M)), sc.const(np.ones(1))])
  fg, n_iter = sc.while_loop(not_converged, sinkhorn_step, start, max_iter=MAX_ITER, params=(a, b, c))
  f, g = fg[:N], fg[N : N + M]
  plan = ((sc.gather(f, ROW) + sc.gather(g, COL) - c) / EPS).exp()
  return (f * a).sum() + (g * b).sum(), (plan * c).sum(), f, n_iter


@sc.function(N, M, N, M, output=sc.G("W", "plan_cost", "f", "dW_da", "iterations"))
def transport(a: sc.Expr, b: sc.Expr, x: sc.Expr, y: sc.Expr) -> tuple[sc.Expr, sc.Expr, sc.Expr, sc.Expr, sc.Expr]:
  w, plan_cost, f, n_iter = entropic_cost(a, b, x, y)
  return w, plan_cost, f, sc.gradient(w, a), n_iter


def mixture(points: np.ndarray, centers: list[float], widths: list[float], weights: list[float]) -> np.ndarray:
  h = sum(w * np.exp(-0.5 * ((points - c) / s) ** 2) for c, s, w in zip(centers, widths, weights, strict=True)) + 1e-3
  return h / h.sum()


def exact_w2(a: np.ndarray, x: np.ndarray, b: np.ndarray, y: np.ndarray) -> float:
  """One-dimensional ``W_2^2`` by integrating the squared difference of the quantile functions."""
  levels = np.unique(np.r_[0.0, np.cumsum(a), np.cumsum(b)].clip(0, 1))
  mid = 0.5 * (levels[1:] + levels[:-1])
  qa = x[np.minimum(np.searchsorted(np.cumsum(a), mid), len(x) - 1)]
  qb = y[np.minimum(np.searchsorted(np.cumsum(b), mid), len(y) - 1)]
  return float(np.sum(np.diff(levels) * (qa - qb) ** 2))


def main() -> dict:
  x, y = np.linspace(0, 1, N), np.linspace(0, 1, M)
  a = mixture(x, [0.2, 0.6], [0.05, 0.1], [1.0, 0.5])
  b = mixture(y, [0.45, 0.8], [0.08, 0.04], [0.7, 1.0])
  w, plan_cost, f, grad, iterations = transport(a, b, x, y)
  center = lambda v: v - v.mean()  # noqa: E731 - gradients on the simplex are defined up to a constant
  rng = np.random.default_rng(0)
  direction = center(rng.standard_normal(N))
  h = 1e-6
  fd = (transport(a + h * direction, b, x, y)[0] - transport(a - h * direction, b, x, y)[0]) / (2 * h)
  return {
    "W": float(w),
    "plan_cost": float(plan_cost),
    "W2_exact": exact_w2(a, x, b, y),
    "iterations": int(iterations),
    "grad_vs_potential": np.abs(center(grad) - center(f)).max() / np.abs(center(f)).max(),
    "directional_fd": (float(fd), float(grad @ direction)),
  }


if __name__ == "__main__":
  out = main()
  print(f"Sinkhorn with eps = {EPS}: {out['iterations']} iterations, W_eps = {out['W']:.6f}")
  print(f"the plan's transport cost <P, C> = {out['plan_cost']:.6f}; the exact W_2^2 = {out['W2_exact']:.6f}")
  print(f"reverse mode through the loop vs the potential f (envelope theorem): relative difference {out['grad_vs_potential']:.1e}")
  fd, ad = out["directional_fd"]
  print(f"directional derivative: {ad:.8f} by AD, {fd:.8f} by central differences")
  write_module(transport, GENERATED)
  print(f"generated C in {GENERATED}")
