"""The example runner: every example script and notebook outside ``examples/case_studies`` runs,
skipped where a requirement it declares (a PEP 723 header, a notebook's metadata) is not installed.

A notebook's code cells run in order in one namespace, from the notebook's directory, with
Matplotlib's non-interactive backend and no Jupyter. A script runs as ``uv run`` runs it, in its own
process from its directory. Examples that name a solver plugin carry that plugin's ``method`` marks,
so CI's method job runs them. Scripts that another test already runs, against a reference and at a
smaller size, are listed in ``ELSEWHERE`` (a test checks the name is really there); the few that
only measure are in ``NOT_RUN`` with the reason.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from scaly.testing import examples

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
TESTS = ROOT / "tests"

ELSEWHERE = {
  "core/deploy_in_c.py": "integration/test_examples_gallery.py",
  "core/derivatives_tour.py": "integration/test_examples_gallery.py",
  "core/mdp_value_iteration.py": "integration/test_examples.py",
  "core/sinkhorn_transport.py": "integration/test_examples_gallery.py",
  "integrators/ekf_identification.py": "integration/test_examples_gallery.py",
  "integrators/lagrangian_mechanics.py": "integration/test_examples_gallery.py",
  "linalg/gaussian_process.py": "integration/test_examples_gallery.py",
  "linalg/heat_control.py": "integration/test_examples.py",
  "linalg/kalman_update.py": "linalg/test_examples.py",
  "linalg/lasso_admm.py": "integration/test_examples.py",
  "linalg/lqr_tuning.py": "integration/test_examples.py",
  "linalg/robot_arm_ik.py": "integration/test_examples_gallery.py",
  "linalg/sqp_newton_sparse.py": "linalg/test_examples.py",
  "linalg/truss_sizing.py": "integration/test_examples.py",
  "ocp/ilqr.py": "integration/test_examples_gallery.py",
  "ocp/tinympc/random_mpc.py": "integration/test_tinympc.py",
  "ocp/tinympc/rocket_landing.py": "integration/test_tinympc.py",
  "ocp/tinympc/safety_filter.py": "integration/test_tinympc.py",
  "opt/cbf_safety_filter.py": "integration/test_examples_gallery.py",
  "opt/mhe.py": "integration/test_examples_gallery.py",
  "opt/nmpc_cartpole.py": "integration/test_examples_gallery.py",
  "opt/optimal_power_flow.py": "integration/test_examples_gallery.py",
  "opt/portfolio_qp.py": "integration/test_examples_gallery.py",
  "opt/tiny_qp.py": "integration/test_examples_gallery.py",
  "roots/bratu_newton.py": "integration/test_examples_gallery.py",
  "roots/hanging_chain.py": "integration/test_examples.py",
  "roots/option_greeks.py": "integration/test_examples_gallery.py",
  "roots/periodic_orbit.py": "integration/test_examples_gallery.py",
}
NOT_RUN = {
  "casadi/compare.py": "times every pair in fresh processes, minutes per pair",
  "opt/qp_solvers/compare.py": "measures five solvers on four families, about 15 minutes",
  "opt/qp_solvers/compare.ipynb": "plots compare.py's recorded results, or measures again for 15 minutes",
  "ocp/tinympc/run_benchmark.py": "clones and builds the TinyMPC library before measuring",
  "ocp/tinympc/benchmark/report.py": "rewrites the recorded benchmark report under internal/notes",
}


def _examples(suffix: str) -> list[str]:
  found = []
  for path in sorted(EXAMPLES.rglob(f"*{suffix}")):
    rel = path.relative_to(EXAMPLES).as_posix()
    if rel.startswith(("case_studies/", "generated/", "gen/")) or ".ipynb_checkpoints" in rel:
      continue
    if suffix == ".py" and examples.requirements(path) is None:
      continue  # a module the examples beside it import, not an example
    found.append(rel)
  return found


SCRIPTS = _examples(".py")
NOTEBOOKS = _examples(".ipynb")


def _params(names: list[str]) -> list:
  out = []
  for name in names:
    if name in ELSEWHERE or name in NOT_RUN:
      continue
    reqs = examples.requirements(EXAMPLES / name)
    assert reqs is not None, name
    marks = [pytest.mark.method(m) for m in examples.methods(reqs)]
    if missing := examples.unmet(reqs):
      marks.append(pytest.mark.skip(reason="; ".join(missing)))
    out.append(pytest.param(name, marks=marks, id=name))
  return out


def test_every_example_runs_somewhere() -> None:
  """An example this runner leaves to another test is run by that test, and nothing is listed twice
  or listed and gone."""
  for name, test in ELSEWHERE.items():
    assert name in SCRIPTS, f"{name} is listed but is not an example"
    assert name.removesuffix(".py").split("/")[-1] in (TESTS / test).read_text(), f"{test} does not run {name}"
  for name in NOT_RUN:
    assert name in SCRIPTS or name in NOTEBOOKS, f"{name} is listed but is not an example"
  assert not set(ELSEWHERE) & set(NOT_RUN)


@pytest.mark.parametrize("name", _params(SCRIPTS))
def test_script_runs(name: str) -> None:
  path = EXAMPLES / name
  env = {**os.environ, "MPLBACKEND": "Agg"}
  proc = subprocess.run([sys.executable, path.name], cwd=path.parent, env=env, capture_output=True, text=True, timeout=900, check=False)
  assert proc.returncode == 0, proc.stdout[-2000:] + proc.stderr[-4000:]


@pytest.mark.parametrize("name", _params(NOTEBOOKS))
def test_notebook_runs(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
  mpl = pytest.importorskip("matplotlib")
  mpl.use("Agg")
  import matplotlib.pyplot as plt

  path = EXAMPLES / name
  monkeypatch.chdir(path.parent)
  monkeypatch.setattr(sys, "path", [str(path.parent), *sys.path])  # as a kernel started there has it
  for module, loaded in list(sys.modules.items()):  # another folder's plotstyle, problems, ... must not stand in
    origin = getattr(loaded, "__file__", None)
    if origin and Path(origin).resolve().is_relative_to(EXAMPLES):
      monkeypatch.delitem(sys.modules, module)
  monkeypatch.setattr(plt, "show", lambda *args, **kwargs: plt.close("all"))
  cells = json.loads(path.read_text())["cells"]
  namespace: dict = {"__name__": f"notebook_{name.replace('/', '_').removesuffix('.ipynb')}"}
  for index, cell in enumerate(cells):
    if cell["cell_type"] == "code":
      exec(compile("".join(cell["source"]), f"{name}[{index}]", "exec"), namespace)  # noqa: S102
  plt.close("all")
