"""The data both halves of each pair use, generated from fixed seeds, so the two sides read the
same numbers and neither side's line count includes them."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


def lut_tables() -> dict[str, tuple[tuple[np.ndarray, ...], np.ndarray]]:
  """Three gridded tables: a smooth curve on 1 000 non-uniform sites (each a random fifth of a cell
  off a uniform grid), a smooth surface on a uniform 64 x 64 grid, and random data on a uniform
  20 x 20 x 20 grid. The uniform grids step by powers of two, so their spacings are exactly equal,
  which CasADi's ``lookup_mode="exact"`` checks."""
  rng = np.random.default_rng(0)
  g1 = np.linspace(0.0, 10.0, 1000)
  g1[1:-1] += rng.uniform(-0.2, 0.2, 998) * (g1[1] - g1[0])
  y1 = np.sin(g1) + 0.05 * g1**2 + 0.3 * np.sin(7.0 * g1)
  g2 = np.arange(64) / 64.0
  x2, v2 = np.meshgrid(g2, g2, indexing="ij")
  y2 = np.sin(3.0 * x2) * np.cos(2.0 * v2) + 0.3 * x2 * v2
  g3 = (np.arange(20) - 10.0) / 8.0
  y3 = rng.normal(size=(20, 20, 20))
  return {"1d": ((g1,), y1), "2d": ((g2, g2), y2), "3d": ((g3, g3, g3), y3)}


def lut_points(grid: tuple[np.ndarray, ...], n: int, seed: int, margin: float = 0.0) -> np.ndarray:
  """``(n, D)`` uniform random points in the grid's box, widened by ``margin`` of its width per side."""
  rng = np.random.default_rng(seed)
  lo = np.array([g[0] for g in grid])
  hi = np.array([g[-1] for g in grid])
  width = hi - lo
  return rng.uniform(lo - margin * width, hi + margin * width, size=(n, len(grid)))


def lut_nodes(grid: tuple[np.ndarray, ...]) -> np.ndarray:
  """Every grid node, ``(prod n_d, D)``, the first axis slowest."""
  return np.stack(np.meshgrid(*grid, indexing="ij"), axis=-1).reshape(-1, len(grid))


def calibration_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """A 12 x 12 grid on [0, 1]^2 and 3 000 noisy measurements of an efficiency-like map at scattered
  points: ``(grid, points (3000, 2), measured (3000,), truth at the grid (12, 12))``."""
  rng = np.random.default_rng(5)
  grid = np.linspace(0.0, 1.0, 12)

  def truth(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 0.36 - 0.12 * (x - 0.45) ** 2 - 0.18 * (y - 0.7) ** 2 + 0.02 * np.sin(6.0 * x) * y

  points = rng.uniform(0.0, 1.0, size=(3000, 2))
  measured = truth(points[:, 0], points[:, 1]) + 0.005 * rng.normal(size=3000)
  gx, gy = np.meshgrid(grid, grid, indexing="ij")
  return grid, points, measured, truth(gx, gy)


HAMMERSTEIN_KNOTS = np.concatenate([[-1.0] * 3, np.linspace(-1.0, 1.0, 18), [1.0] * 3])  # a cubic's 20 coefficients


def hammerstein_data() -> dict[str, np.ndarray]:
  """2 000 samples of a Hammerstein system, a static nonlinearity and then second-order dynamics:
  ``v = N(u)``, ``y[k+1] = a1 y[k] + a2 y[k-1] + v[k]``, measured with noise. The input holds random
  levels for 1 to 20 samples; ``N`` is a saturating curve, and ``c_true`` is its least-squares fit
  by the cubic spline on ``HAMMERSTEIN_KNOTS`` that generates the data."""
  from scipy.interpolate import make_lsq_spline

  rng = np.random.default_rng(7)
  n = 2000
  u = np.empty(n)
  k = 0
  while k < n:
    hold = int(rng.integers(1, 21))
    u[k : k + hold] = rng.uniform(-1.0, 1.0)
    k += hold
  sites = np.linspace(-1.0, 1.0, 400)
  c_true = make_lsq_spline(sites, np.tanh(2.0 * sites) + 0.3 * sites**3, HAMMERSTEIN_KNOTS, k=3).c
  from scipy.interpolate import BSpline

  v = BSpline(HAMMERSTEIN_KNOTS, c_true, 3)(u)
  a_true = np.array([1.5, -0.7])
  y = np.zeros(n + 1)
  prev = 0.0
  for i in range(n):
    y[i + 1], prev = a_true[0] * y[i] + a_true[1] * prev + v[i], y[i]
  measured = y[1:] + 0.01 * rng.normal(size=n)
  greville = np.array([HAMMERSTEIN_KNOTS[i + 1 : i + 4].mean() for i in range(20)])
  return {"u": u, "y": measured, "a_true": a_true, "c_true": c_true, "a_guess": np.array([1.0, -0.3]), "c_guess": greville}


class Track(NamedTuple):
  s: np.ndarray
  xy: np.ndarray
  width: float
  length: float


def track() -> Track:
  """The FSDS "competition 1" track's centre line (vendored with the race-car benchmark), closed and
  parametrized by chord length: ``s`` (n + 1,) from 0 to the lap length, ``xy`` (n + 1, 2) with the
  first point repeated at the end, the narrowest half ``width`` and the ``length``."""
  from pathlib import Path

  table = np.genfromtxt(Path(__file__).resolve().parents[1] / "data" / "fsds_competition_1.csv", delimiter=",", skip_header=1)
  xy = np.vstack([table[:, :2], table[:1, :2]])
  s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
  return Track(s, xy, float(table[:, 2:4].min()), float(s[-1]))


@dataclass(frozen=True)
class Mpcc:
  """The contouring controller both halves of the contouring_mpc pair build."""

  dt: float = 0.05
  horizon: int = 40
  steps: int = 200
  wheelbase: float = 1.5
  q_contour: float = 2.0
  q_lag: float = 200.0
  q_progress: float = 2.0
  r: tuple[float, float, float] = (0.02, 2.0, 0.01)  # acceleration, steering, progress speed
  u_lo: tuple[float, float, float] = (-4.0, -0.4, 0.0)
  u_hi: tuple[float, float, float] = (3.0, 0.4, 10.0)
  v_max: float = 8.0
  margin: float = 0.5  # kept from the track's edge [m]


MPCC = Mpcc()


def mpcc_start() -> np.ndarray:
  """``(X, Y, psi, v, theta)``: on the centre line at the start, along it, at 2 m/s."""
  t = track()
  xy = t.xy
  heading = np.arctan2(xy[1, 1] - xy[-2, 1], xy[1, 0] - xy[-2, 0])  # the chord across the start
  return np.array([xy[0, 0], xy[0, 1], heading, 2.0, 0.0])


@dataclass(frozen=True)
class HeatPump:
  """A house heated by an air-to-water heat pump, the heat_pump_mpc pair's model. Temperatures in C,
  heat in kW, capacities in kWh/K, times in hours."""

  dt: float = 0.25
  horizon: int = 96
  c_air: float = 1.5  # indoor air and furniture
  c_mass: float = 15.0  # the building's mass
  h_air_mass: float = 1.0
  h_air_out: float = 0.1
  h_mass_out: float = 0.08
  k_emitter: float = 0.4  # heat into the room per kelvin of supply temperature above the room's
  p_max: float = 4.0  # electrical power [kW]
  t_supply: tuple[float, float] = (25.0, 55.0)
  t_room: tuple[float, float] = (20.0, 23.0)
  x0: tuple[float, float] = (20.5, 20.0)  # room, mass


HEAT_PUMP = HeatPump()


def cop_table() -> tuple[tuple[np.ndarray, np.ndarray], np.ndarray]:
  """A heat pump's COP on a 9 x 9 grid of outdoor and supply temperatures: 45 % of the Carnot COP
  between the supply temperature plus 5 K and the outdoor temperature minus 5 K. The grid reaches
  5 K past the supply temperature's bounds, since IPOPT relaxes a bound by a hair and CasADi's
  ``bspline`` is zero outside its grid."""
  t_out = np.linspace(-20.0, 20.0, 9)
  t_supply = np.linspace(20.0, 60.0, 9)
  hot = t_supply[None, :] + 273.15 + 5.0
  cold = t_out[:, None] + 273.15 - 5.0
  return (t_out, t_supply), 0.45 * hot / (hot - cold)


def day_ahead() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  """Hourly forecasts for one day: ``(hours, price [EUR/kWh], outdoor temperature [C])``."""
  hours = np.arange(24.0)
  price = 0.15 + 0.2 * np.exp(-(((hours - 8.0) / 1.5) ** 2)) + 0.45 * np.exp(-(((hours - 18.5) / 2.0) ** 2)) - 0.08 * np.exp(-(((hours - 3.0) / 2.0) ** 2))
  t_out = -5.0 + 4.0 * np.sin(2.0 * np.pi * (hours - 9.0) / 24.0)
  return hours, np.round(price, 4), np.round(t_out, 2)
