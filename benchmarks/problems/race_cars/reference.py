"""Minimum-curvature spline reference generator.

A closed cubic spline is fitted once to the track center line, sampled uniformly
in arc length, and turned into a constant-speed time parameterisation. At every
control step the car's position is projected onto the spline and the reference
horizon is read off in time, so the reference re-anchors to the car instead of
drifting away from it.

Ported from ``minimal_tracking_nmpc/motion_planning.py``. The one change is the
spline fit: the original solves the equality-constrained QP with OSQP through
``qpsolvers``/``scipy.sparse``, which alloy does not depend on. The same problem
is solved here as a dense KKT system, which is both dependency-free and more
accurate (continuity residuals ~1e-14 instead of ~1e-9).
"""

from __future__ import annotations

import numpy as np

SPLINE_SAMPLES = 500
CURVATURE_WEIGHT = 2.0


def teds_projection(x: np.ndarray, a: float) -> np.ndarray:
  """Project ``x`` onto ``[a, a + 2*pi)``."""
  return np.mod(x - a, 2 * np.pi) + a


def unwrap_to_pi(x: np.ndarray) -> np.ndarray:
  """Remove the 2*pi jumps a per-sample ``atan2`` heading leaves behind."""
  diffs = np.diff(x)
  diffs[diffs > 1.5 * np.pi] -= 2 * np.pi
  diffs[diffs < -1.5 * np.pi] += 2 * np.pi
  return np.insert(x[0] + np.cumsum(diffs), 0, x[0])


def fit_spline(path: np.ndarray, curv_weight: float = CURVATURE_WEIGHT) -> tuple[np.ndarray, np.ndarray]:
  """Fit a closed cubic spline through ``path``, one cubic per interval.

  ``path`` is ``(N, 2)`` and closed, with the first and last points distinct. Each
  segment ``i`` is ``sum_k coeffs[i, k] t**k`` for ``t`` in ``[0, 1]``. The
  objective trades waypoint fit against curvature; value, tangent, and curvature
  continuity at every knot are equality constraints, so the fit is a QP whose KKT
  system is solved directly.
  """
  if path.ndim != 2 or path.shape[1] != 2:
    raise ValueError(f"path must have shape (N, 2), got {path.shape}")
  n = path.shape[0]
  delta_s = np.concatenate([np.linalg.norm(np.diff(path, axis=0), axis=1), [np.linalg.norm(path[0] - path[-1])]])
  # rho[i] rescales segment i+1's parameter derivative to segment i's arc-length scale
  rho = delta_s / np.roll(delta_s, -1)

  index = np.arange(n)
  nxt = 4 * ((index + 1) % n)
  constraints = np.zeros((3 * n, 4 * n))
  constraints[3 * index + 0, 4 * index + 0] = 1.0
  constraints[3 * index + 0, 4 * index + 1] = 1.0
  constraints[3 * index + 0, 4 * index + 2] = 1.0
  constraints[3 * index + 0, 4 * index + 3] = 1.0
  constraints[3 * index + 1, 4 * index + 1] = 1.0
  constraints[3 * index + 1, 4 * index + 2] = 2.0
  constraints[3 * index + 1, 4 * index + 3] = 3.0
  constraints[3 * index + 2, 4 * index + 2] = 2.0
  constraints[3 * index + 2, 4 * index + 3] = 6.0
  constraints[3 * index + 0, nxt + 0] -= 1.0
  constraints[3 * index + 1, nxt + 1] -= rho
  constraints[3 * index + 2, nxt + 2] -= 2.0 * rho**2

  fit = np.zeros((n, 4 * n))
  fit[index, 4 * index] = 1.0
  curvature = np.zeros((n, 4 * n))
  curvature[index, 4 * index + 2] = 2.0 / delta_s**2
  curvature[index, 4 * index + 3] = 6.0 / delta_s**2
  hessian = fit.T @ fit + curv_weight * (curvature.T @ curvature) + 1e-10 * np.eye(4 * n)

  kkt = np.block([[hessian, constraints.T], [constraints, np.zeros((3 * n, 3 * n))]])
  rhs = np.vstack([fit.T @ path, np.zeros((3 * n, 2))])
  solution = np.linalg.solve(kkt, rhs)[: 4 * n]
  return solution[:, 0].reshape(n, 4), solution[:, 1].reshape(n, 4)


def _polyval(coeffs: np.ndarray, index: np.ndarray, t: np.ndarray) -> np.ndarray:
  return coeffs[index, 0] + coeffs[index, 1] * t + coeffs[index, 2] * t**2 + coeffs[index, 3] * t**3


def _polyval_d(coeffs: np.ndarray, index: np.ndarray, t: np.ndarray) -> np.ndarray:
  return coeffs[index, 1] + 2 * coeffs[index, 2] * t + 3 * coeffs[index, 3] * t**2


def spline_interval_lengths(coeffs_x: np.ndarray, coeffs_y: np.ndarray, samples: int = 100) -> np.ndarray:
  """Arc length of each spline segment, by chord summation over ``samples`` points."""
  t = np.linspace(0.0, 1.0, samples)[np.newaxis, :]
  index = np.arange(coeffs_x.shape[0])[:, np.newaxis]
  points = np.stack([_polyval(coeffs_x, index, t), _polyval(coeffs_y, index, t)], axis=-1)
  return np.sum(np.linalg.norm(np.diff(points, axis=1), axis=2), axis=1)


def sample_spline(coeffs_x: np.ndarray, coeffs_y: np.ndarray, delta_s: np.ndarray, n_samples: int):
  """Sample ``n_samples`` points spaced uniformly in arc length over one lap."""
  ends = np.cumsum(delta_s)
  s = np.linspace(0.0, ends[-1], n_samples, endpoint=False)
  index = np.argmax(s[:, np.newaxis] < ends, axis=1)
  starts = np.concatenate([[0.0], ends[:-1]])
  t = (s - starts[index]) / delta_s[index]
  x, y = _polyval(coeffs_x, index, t), _polyval(coeffs_y, index, t)
  heading = np.arctan2(_polyval_d(coeffs_y, index, t), _polyval_d(coeffs_x, index, t))
  return x, y, heading, s


class MotionPlanner:
  """Constant-speed reference horizons along a track's center line."""

  def __init__(self, center_line: np.ndarray, *, horizon: int, dt: float, v_ref: float):
    self.horizon, self.dt, self.v_ref = horizon, dt, v_ref
    coeffs_x, coeffs_y = fit_spline(center_line)
    delta_s = spline_interval_lengths(coeffs_x, coeffs_y)
    x, y, heading, s = sample_spline(coeffs_x, coeffs_y, delta_s, SPLINE_SAMPLES)

    closing = float(np.hypot(x[-1] - x[0], y[-1] - y[0]))
    self.lap_length = float(s[-1] + closing)
    self.lap_time = self.lap_length / v_ref
    self.center_path = np.stack([x, y], axis=1)

    # three laps laid end to end so a horizon may cross the start line without special cases
    self.s_ref = np.concatenate([s - self.lap_length, s, s + self.lap_length])
    self.x_ref = np.tile(x, 3)
    self.y_ref = np.tile(y, 3)
    self.phi_ref = unwrap_to_pi(np.tile(heading, 3))

  def project(self, x: float, y: float, s_guess: float, tolerance: float = 10.0) -> float:
    """Arc length of the point on the spline closest to ``(x, y)``, searched near ``s_guess``."""
    low = int(np.searchsorted(self.s_ref, s_guess - tolerance))
    high = int(np.searchsorted(self.s_ref, s_guess + tolerance))
    point = np.array([x, y])
    window = np.stack([self.x_ref[low:high], self.y_ref[low:high]], axis=1)
    nearest = low + int(np.argmin(np.linalg.norm(window - point, axis=1)))

    best_s, best_distance = self.s_ref[nearest], np.inf
    for first in (max(nearest - 1, 0), min(nearest, self.s_ref.size - 2)):
      a = np.array([self.x_ref[first], self.y_ref[first]])
      b = np.array([self.x_ref[first + 1], self.y_ref[first + 1]])
      segment = b - a
      ratio = float(np.clip(np.dot(point - a, segment) / np.dot(segment, segment), 0.0, 1.0))
      distance = float(np.linalg.norm(point - (a + ratio * segment)))
      if distance < best_distance:
        best_s = self.s_ref[first] + ratio * (self.s_ref[first + 1] - self.s_ref[first])
        best_distance = distance
    return float(best_s)

  def plan(self, x: float, y: float, phi: float, s_guess: float) -> tuple[float, np.ndarray]:
    """Return the projected arc length and a ``(horizon + 1, 4)`` ``[X, Y, phi, v]`` reference."""
    s0 = self.project(x, y, s_guess)
    # constant reference speed, so sampling uniformly in time is sampling uniformly in arc length
    s = s0 + self.v_ref * self.dt * np.arange(self.horizon + 1)
    reference = np.stack(
      [
        np.interp(s, self.s_ref, self.x_ref),
        np.interp(s, self.s_ref, self.y_ref),
        teds_projection(np.interp(s, self.s_ref, self.phi_ref), phi - np.pi),
        np.full(self.horizon + 1, self.v_ref),
      ],
      axis=1,
    )
    return s0, reference
