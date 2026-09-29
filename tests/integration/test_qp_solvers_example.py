"""``examples/qp_solvers``: the generated PIQP (``opt.ipm``) against the PIQP library (``opt.piqp``).

On small instances of the four families the generated solver must take the vendored library's
iterations, backend for backend, and return its solution and multipliers; its objective must be the
problem's own, as the example's ``qp_data`` reproduces it.
"""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest

import scaly as sc

HERE = Path(__file__).resolve().parents[2] / "examples" / "qp_solvers"


@pytest.fixture(scope="module")
def example() -> dict:
  return {"problems": runpy.run_path(str(HERE / "problems.py")), "compare": runpy.run_path(str(HERE / "compare.py"))}


SMALL = {
  "mpc": ("mpc", {"horizon": 4}),
  "portfolio": ("portfolio", {"n_assets": 12, "n_factors": 2}),
  "svm": ("svm", {"n_samples": 16, "n_features": 3}),
  "dense": ("dense", {"n": 6, "n_eq": 2, "n_ineq": 8}),
}


@pytest.mark.method("opt.piqp")
@pytest.mark.parametrize("backend", ["sparse", "dense"])
@pytest.mark.parametrize("family", sorted(SMALL))
def test_generated_piqp_matches_the_library(example: dict, family: str, backend: str) -> None:
  make, kwargs = SMALL[family]
  case = example["problems"][make](**kwargs)
  params = case.params()
  p = case.problem
  zeros = p.vars.unflatten(tuple(np.zeros(s) for s in p.vars.shapes))
  args = (zeros, zeros, np.zeros(p.n_eq), np.zeros(p.n_ineq), params)
  sparse = backend == "sparse"
  generated = sc.opt.solver(p, sc.opt.IPM(sparse=sparse), name=f"test_{case.name}_{family}_{backend}")(*args)
  library = sc.opt.solver(p, sc.opt.PIQP(sparse=sparse), name=f"test_{case.name}_{family}_lib_{backend}")(*args)

  def flat(tree: object) -> np.ndarray:
    return np.concatenate([np.ravel(v) for v in p.vars.flatten_numerical(tree, "x")])

  info, lib_info = generated[-1], library[-1]
  assert int(info.status) == int(lib_info.status) == sc.Status.OK
  assert int(info.iter) == int(lib_info.iter)
  x = flat(generated[0])
  scale = 1 + np.abs(flat(library[0])).max()
  np.testing.assert_allclose(x, flat(library[0]), atol=1e-8 * scale)
  np.testing.assert_allclose(flat(generated[1]), flat(library[1]), atol=1e-6 * (1 + np.abs(flat(library[1])).max()))
  for got, want in zip(generated[2:4], library[2:4], strict=True):
    np.testing.assert_allclose(got, want, atol=1e-6 * (1 + np.abs(want).max(initial=0.0)))

  (P, c), _, _, _, f0 = example["compare"]["qp_data"](p)(params)
  assert abs(float(info.objective) - (0.5 * x @ P @ x + c @ x + f0)) < 1e-10 * (1 + abs(float(info.objective)))
