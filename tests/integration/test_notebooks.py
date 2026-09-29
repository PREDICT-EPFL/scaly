"""The example notebooks (``examples/notebooks``, ``examples/integrators``, ``examples/interp``,
``examples/ocp``) run top to bottom without error.

No Jupyter is needed: each notebook's code cells are executed in order in one namespace, from the
notebook's directory, with Matplotlib's non-interactive backend. The notebooks in
``examples/notebooks`` carry their own checks in their printed output; this test keeps them from
silently breaking as the API moves. Those in ``examples/integrators``, ``examples/interp`` and
``examples/ocp`` end with a cell of assertions against their references, so running them checks them.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
NOTEBOOKS = EXAMPLES / "notebooks"  # also holds plotstyle.py, which every notebook imports
SOLVER = {
  "cbf_safety_filter": ("piqp",),
  "kinetics_estimation": ("ipopt",),
  "nmpc_cartpole": ("ipopt",),
  "opf_day_ahead": ("ipopt", "piqp"),
  "spike_deconvolution": ("piqp",),
  "surrogate_optimization": ("ipopt",),
  "interp/contouring_control": ("ipopt",),
  "interp/learning_tables": ("ipopt",),
  "interp/shape_constrained": ("piqp",),
  "interp/spline_trajectories": ("ipopt", "piqp"),
  "ocp/linear_mpc": ("piqp",),
  "ocp/nmpc_closed_loop": ("ipopt", "sqp"),
  "ocp/nmpc_transcriptions": ("ipopt",),
  "ocp/reference_tracking": ("piqp",),
  "ocp/terminal_sets": ("piqp", "ipopt"),
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
  "integrators/adaptive_plant",
  "integrators/discrete_maps",
  "integrators/explicit_methods",
  "integrators/frequency_response",
  "integrators/linearize_and_discretize",
  "integrators/polynomials",
  "integrators/stiff_implicit",
  "integrators/symplectic_orbits",
  "integrators/transcriptions",
  "interp/contouring_control",
  "interp/interpolation_kinds",
  "interp/learning_tables",
  "interp/lookup_tables_nd",
  "interp/shape_constrained",
  "interp/spline_trajectories",
  "ocp/linear_mpc",
  "ocp/nmpc_closed_loop",
  "ocp/nmpc_transcriptions",
  "ocp/reference_tracking",
  "ocp/terminal_sets",
]


def _params() -> list:
  return [pytest.param(name, marks=[pytest.mark.solver(s) for s in SOLVER[name]]) if name in SOLVER else name for name in NAMES]


@pytest.mark.parametrize("name", _params())
def test_notebook_runs(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
  mpl = pytest.importorskip("matplotlib")
  mpl.use("Agg")
  import matplotlib.pyplot as plt

  path = EXAMPLES / f"{name}.ipynb" if "/" in name else NOTEBOOKS / f"{name}.ipynb"
  monkeypatch.chdir(path.parent)
  monkeypatch.setattr(sys, "path", [str(NOTEBOOKS), *sys.path])
  monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
  cells = json.loads(path.read_text())["cells"]
  namespace: dict = {"__name__": f"notebook_{name.replace('/', '_')}"}
  for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
      exec(compile("".join(cell["source"]), f"{name}.ipynb[{index}]", "exec"), namespace)  # noqa: S102
  plt.close("all")
