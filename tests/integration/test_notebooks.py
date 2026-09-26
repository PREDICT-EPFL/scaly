"""The example notebooks (``examples/notebooks``) run top to bottom without error.

No Jupyter is needed: each notebook's code cells are executed in order in one namespace, from the
notebook's directory, with Matplotlib's non-interactive backend. The notebooks carry their own
checks in their printed output; this test keeps them from silently breaking as the API moves.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

NOTEBOOKS = Path(__file__).resolve().parents[2] / "examples" / "notebooks"
SOLVER = {
  "cbf_safety_filter": ("piqp",),
  "kinetics_estimation": ("ipopt",),
  "nmpc_cartpole": ("ipopt",),
  "opf_day_ahead": ("ipopt", "piqp"),
  "spike_deconvolution": ("piqp",),
  "surrogate_optimization": ("ipopt",),
}
NAMES = [
  "bratu_newton",
  "cbf_safety_filter",
  "circuit_transient",
  "conductivity_inversion",
  "ekf_identification",
  "gaussian_process",
  "ilqr",
  "kinetics_estimation",
  "lagrangian_mechanics",
  "nmpc_cartpole",
  "opf_day_ahead",
  "pose_graph_slam",
  "sparse_fem_topology",
  "sparse_kkt_mpc",
  "spike_deconvolution",
  "surrogate_optimization",
]


def _params() -> list:
  return [pytest.param(name, marks=[pytest.mark.solver(s) for s in SOLVER[name]]) if name in SOLVER else name for name in NAMES]


@pytest.mark.parametrize("name", _params())
def test_notebook_runs(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
  mpl = pytest.importorskip("matplotlib")
  mpl.use("Agg")
  import matplotlib.pyplot as plt

  monkeypatch.chdir(NOTEBOOKS)
  monkeypatch.setattr(sys, "path", [str(NOTEBOOKS), *sys.path])
  monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
  cells = json.loads((NOTEBOOKS / f"{name}.ipynb").read_text())["cells"]
  namespace: dict = {"__name__": f"notebook_{name}"}
  for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
      exec(compile("".join(cell["source"]), f"{name}.ipynb[{index}]", "exec"), namespace)  # noqa: S102
  plt.close("all")
