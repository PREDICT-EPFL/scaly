"""Shared Matplotlib style for the example notebooks: one categorical order, thin marks, a recessive frame."""

from __future__ import annotations

import matplotlib as mpl
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
# Categorical slots, always assigned in this order (validated for colour-vision deficiency as adjacent pairs).
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"
SERIES = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
NEUTRAL = "#8c8b86"  # reference lines, bounds, obstacles
# Sequential magnitude: one hue, light to dark.
BLUES = LinearSegmentedColormap.from_list("blues", ["#f3f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"])


def use() -> None:
  mpl.rcParams.update(
    {
      "figure.facecolor": SURFACE,
      "axes.facecolor": SURFACE,
      "savefig.facecolor": SURFACE,
      "figure.dpi": 100,
      "figure.figsize": (7.5, 3.6),
      "figure.constrained_layout.use": True,
      "font.size": 10,
      "axes.titlesize": 11,
      "axes.titleweight": "bold",
      "axes.titlelocation": "left",
      "axes.labelcolor": TEXT_SECONDARY,
      "axes.edgecolor": GRID,
      "axes.linewidth": 0.8,
      "axes.spines.top": False,
      "axes.spines.right": False,
      "axes.grid": True,
      "axes.axisbelow": True,
      "axes.prop_cycle": cycler(color=SERIES),
      "grid.color": GRID,
      "grid.linewidth": 0.6,
      "xtick.color": TEXT_SECONDARY,
      "ytick.color": TEXT_SECONDARY,
      "text.color": TEXT,
      "lines.linewidth": 2.0,
      "lines.markersize": 5,
      "legend.frameon": False,
      "image.cmap": "Blues",
    }
  )
