"""``examples/qp_solvers``: the generated PIQP, reached from a ``sc.problem``, against the PIQP library.

The example's front end (``generated_piqp.solver``) extracts a problem's QP the way ``sc.solver`` does
and generates the solver from it. On small instances of the four families it must take the vendored
library's iterations, backend for backend, and return its solution; ``qp_data`` must reproduce the
problem's objective.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

import numpy as np
import pytest

import scaly as sc

HERE = Path(__file__).resolve().parents[2] / "examples" / "qp_solvers"


@pytest.fixture(scope="module")
def example() -> dict:
  sys.path.insert(0, str(HERE))
  try:
    yield {"problems": runpy.run_path(str(HERE / "problems.py")), "generated_piqp": runpy.run_path(str(HERE / "generated_piqp.py"))}
  finally:
    sys.path.remove(str(HERE))


SMALL = {
  "mpc": ("mpc", {"horizon": 4}),
  "portfolio": ("portfolio", {"n_assets": 12, "n_factors": 2}),
  "svm": ("svm", {"n_samples": 16, "n_features": 3}),
  "dense": ("dense", {"n": 6, "n_eq": 2, "n_ineq": 8}),
}


@pytest.mark.solver("piqp")
@pytest.mark.parametrize("backend", ["sparse", "dense"])
@pytest.mark.parametrize("family", sorted(SMALL))
def test_generated_piqp_matches_the_library(example: dict, family: str, backend: str) -> None:
  make, kwargs = SMALL[family]
  case = example["problems"][make](**kwargs)
  gp = example["generated_piqp"]
  params = case.params()
  x, _, _, _, status, iters, obj = gp["solver"](case.problem, backend, name=f"test_{case.name}_{family}_{backend}")(params)

  library = sc.solver(case.problem, "piqp", name=f"test_{case.name}_{family}_lib_{backend}", options={"sparse": backend == "sparse"})
  p = case.problem
  zeros = p.vars.unflatten(tuple(np.zeros(s) for s in p.vars.shapes))
  out = library((zeros, zeros, np.zeros(p.n_eq), np.zeros(p.n_ineq), params))
  x_lib = np.concatenate([np.ravel(v) for v in p.vars.flatten_numerical(out[0], "x")])
  stats = library.solver_stats()

  assert int(status) == 1 and stats.status.name == "OK"
  assert int(iters) == stats.iter
  np.testing.assert_allclose(x, x_lib, atol=1e-8 * (1 + np.abs(x_lib).max()))

  (P, c), _, _, _, f0 = gp["qp_data"](case.problem)(params)
  assert abs(float(obj) - (0.5 * x @ P @ x + c @ x + f0)) < 1e-10 * (1 + abs(float(obj)))
