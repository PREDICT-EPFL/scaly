"""FSDS track data: center lines, per-side widths, and cone positions.

The CSV files under ``data/tracks/`` are vendored from the Formula Student
Driverless Simulator via ``minimal_tracking_nmpc``. Each track is closed and its
first and last center-line waypoints are distinct, which is what the spline fit
in :mod:`.reference` assumes.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

TRACKS_DIR = Path(__file__).resolve().parent / "data" / "tracks"
CONE_COLORS = ("blue", "yellow", "big_orange", "small_orange")


def track_names() -> tuple[str, ...]:
  return tuple(sorted(path.name for path in TRACKS_DIR.iterdir() if (path / "center_line.csv").is_file()))


def load_center_line(path: Path) -> tuple[np.ndarray, np.ndarray]:
  """Return the ``(N, 2)`` center line and the ``(N, 2)`` ``[right, left]`` widths."""
  arr = np.genfromtxt(path, delimiter=",", dtype=float, skip_header=1)
  return arr[:, :2], arr[:, 2:4]


def load_cones(path: Path) -> dict[str, np.ndarray]:
  """Return ``(N, 2)`` cone positions keyed by cone colour."""
  arr = np.genfromtxt(path, delimiter=",", dtype=str, skip_header=1)
  return {color: arr[arr[:, 0] == color][:, 1:3].astype(float) for color in CONE_COLORS}


@dataclass(frozen=True)
class Track:
  name: str
  center_line: np.ndarray
  widths: np.ndarray
  cones: dict[str, np.ndarray]

  @property
  def half_width(self) -> float:
    """Narrowest of the per-side widths — the corridor the car actually has."""
    return float(np.min(self.widths))


def load_track(name: str) -> Track:
  directory = TRACKS_DIR / name
  if not directory.is_dir():
    raise ValueError(f"unknown track {name!r}; available: {', '.join(track_names())}")
  center_line, widths = load_center_line(directory / "center_line.csv")
  return Track(name, center_line, widths, load_cones(directory / "cones.csv"))
