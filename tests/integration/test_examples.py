"""The control, optimization and numerical-solve examples, each checked against a NumPy/SciPy reference."""

from __future__ import annotations

import runpy
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str) -> dict:
  return runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name=f"example_{name}")


def _central_difference(fun, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
  return np.array([(fun(x + eps * e) - fun(x - eps * e)) / (2 * eps) for e in np.eye(x.size)])


@pytest.mark.parametrize("slip, puddle", [(0.0, 6.0), (0.1, 6.0), (0.2, 2.0)])
def test_value_iteration_matches_numpy_and_its_policy_is_greedy(slip: float, puddle: float) -> None:
  ns = _load("mdp_value_iteration")
  out = ns["main"](slip, puddle)
  v_ref, q_ref = ns["reference"](slip, puddle)
  np.testing.assert_allclose(out["value"], v_ref, rtol=1e-12, atol=1e-12)
  policy = out["policy"].astype(int)
  assert np.all((0 <= policy) & (policy < 4))
  np.testing.assert_allclose(q_ref[np.arange(ns["S"]), policy], q_ref.min(axis=1), rtol=1e-9)
  assert out["value"][ns["GOAL"]] == 0.0 and 1 < out["sweeps"] < ns["MAX_SWEEPS"]


def test_lasso_meets_its_optimality_conditions() -> None:
  ns = _load("lasso_admm")
  out = ns["main"]()
  x, lam, a, b = out["x"], float(out["lam"]), ns["A_CONST"], out["b"]
  grad = a.T @ (a @ x - b)  # optimality: grad + lam * sign(x) = 0 on the support, |grad| <= lam off it
  on = x != 0.0
  np.testing.assert_array_equal(out["support"], on)
  np.testing.assert_allclose(grad[on], -lam * np.sign(x[on]), atol=1e-6)
  assert np.all(np.abs(grad[~on]) <= lam * (1 + 1e-6))
  assert out["iterations"] < ns["MAX_ITER"]
  largest = np.sort(np.argsort(-np.abs(x))[: ns["K_TRUE"]])
  np.testing.assert_array_equal(largest, np.flatnonzero(out["truth"]))


def test_lqr_cost_and_gradient_through_both_scans() -> None:
  ns = _load("lqr_tuning")
  w = np.random.default_rng(3).normal(size=ns["NX"] + 1) * 0.5
  cost, grad = ns["tuning_objective"](w)
  np.testing.assert_allclose(cost, ns["reference_cost"](w), rtol=1e-12)
  np.testing.assert_allclose(grad, _central_difference(ns["reference_cost"], w), rtol=1e-6, atol=1e-8)
  out = ns["main"](max_iter=30)
  assert out["tuned_cost"] < 0.6 * out["initial_cost"]


def test_truss_analysis_and_design() -> None:
  ns = _load("truss_sizing")
  structure = (ns["COORDS"].reshape(-1), ns["BARS"], ns["FREE"], ns["LOAD"])
  areas = np.random.default_rng(0).uniform(0.1, 1.0, ns["NB"])
  compliance, grad, lengths = ns["analyse"](areas, *structure)
  ref_c, ref_g = ns["reference"](areas)
  np.testing.assert_allclose(compliance, ref_c, rtol=1e-12)
  np.testing.assert_allclose(grad, ref_g, rtol=1e-10, atol=1e-12)
  out = ns["optimality_criteria"](iterations=40)
  np.testing.assert_allclose(out["areas"] @ out["lengths"], out["volume"], rtol=1e-8)
  assert out["history"][-1] < 0.5 * out["history"][0]
  # The same compiled analysis takes another truss of the same size: move a node and reverse the bars.
  moved = ns["COORDS"].copy()
  moved[-1] += [0.1, -0.2]
  c_moved, _, _ = ns["analyse"](areas, moved.reshape(-1), ns["BARS"][:, ::-1].copy(), ns["FREE"], ns["LOAD"])
  ns["COORDS"][:] = moved
  np.testing.assert_allclose(c_moved, ns["reference"](areas)[0], rtol=1e-12)


def test_heat_control_simulation_gradient_and_sparse_signature() -> None:
  ns = _load("heat_control")
  u = np.random.default_rng(1).uniform(0.0, 50.0, (ns["STEPS"], ns["N_HEATERS"]))
  kappa = ns["KAPPA"]
  k = ns["system"](kappa)
  assert isinstance(k, sparse.csc_array) and k.nnz == ns["NODES"] + len(ns["FACE_P"])
  np.testing.assert_allclose(ns["simulate"](k, u), ns["reference_final"](kappa, u), rtol=1e-11, atol=1e-13)

  def cost(flat: np.ndarray) -> float:
    t = ns["reference_final"](kappa, flat.reshape(u.shape))
    return 0.5 * np.sum((ns["WEIGHT"] * (t - ns["TARGET"])) ** 2) + 0.5 * ns["ALPHA"] * np.sum(flat**2)

  _, grad = ns["objective"](kappa, u)
  picks = [0, 17, 50, u.size - 1]
  fd = [(cost(u.reshape(-1) + 1e-5 * e) - cost(u.reshape(-1) - 1e-5 * e)) / 2e-5 for e in np.eye(u.size)[picks]]
  np.testing.assert_allclose(grad.reshape(-1)[picks], fd, rtol=1e-6, atol=1e-9)
  with pytest.raises(ValueError, match="different sparsity pattern"):
    ns["simulate"](k + k.T, u)  # the full symmetric matrix is not the declared lower triangle
  out = ns["main"](max_iter=40)
  assert out["cost"] < 0.5 * out["initial_cost"]


def test_chain_equilibrium_gradient_and_fit() -> None:
  ns = _load("hanging_chain")
  import scaly as sc

  p, params = sc.sym("p", ns["NP"]), sc.sym("params", 3)
  grad_fn = sc.Function._from_exprs("chain_grad_check", [p, params], [sc.gradient(ns["energy"](p, params), p)], ["p", "params"], ["g"])
  for theta, weight in ((np.log([60.0, 0.03]), 0.0), (np.log([10.0, 0.2]), 0.05)):
    params_v = np.r_[theta, weight]
    assert np.abs(grad_fn((ns["equilibrium"](params_v), params_v))).max() < 1e-9
  photos = np.stack([ns["shape"](np.log([60.0, 0.03]), w)[ns["OBSERVED"]] for w in (0.0, ns["WEIGHT"])]) + 0.01
  theta = np.log([20.0, 0.1])
  _, grad = ns["misfit"](theta, photos)

  def loss(t: np.ndarray) -> float:
    return sum(0.5 * np.sum((ns["shape"](t, w)[ns["OBSERVED"]] - photos[i]) ** 2) for i, w in enumerate((0.0, ns["WEIGHT"])))

  np.testing.assert_allclose(grad, _central_difference(loss, theta), rtol=1e-6)
  out = ns["main"]()
  np.testing.assert_allclose(np.exp(out["theta"]), np.exp(out["truth"]), rtol=0.15)
