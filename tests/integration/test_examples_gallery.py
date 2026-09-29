"""The feature-gallery examples (``examples/README.md``), each checked against its own reference.

Every example's ``main()`` returns what it compared: NumPy/SciPy references, central differences,
published values. These tests hold those comparisons to tolerances, so an example that silently
drifts fails here. The solver examples carry the ``solver`` marker.
"""

from __future__ import annotations

import runpy
import shutil
from pathlib import Path

import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


def _load(name: str) -> dict:
  return runpy.run_path(str(EXAMPLES / f"{name}.py"), run_name=f"example_{name.replace('/', '_')}")


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler on PATH")
def test_deploy_in_c_program_matches_the_jit(tmp_path: Path) -> None:
  ns = _load("core/deploy_in_c")
  out = ns["main"]()
  np.testing.assert_allclose(out["q"], out["q_ref"], atol=1e-12)
  np.testing.assert_allclose(out["jac_last"], out["jac_bias_fd"], atol=1e-8)
  ns["run_in_c"].__globals__["GENERATED"] = tmp_path
  c_out = ns["run_in_c"](out["gyro"], out["bias"])
  np.testing.assert_allclose(c_out[:4], out["q"], atol=1e-12)
  np.testing.assert_allclose(c_out[4:], out["jac_bias"].reshape(-1), atol=1e-12)


def test_derivatives_tour_against_differences_and_lj7() -> None:
  ns = _load("core/derivatives_tour")
  out = ns["main"]()
  for name, err in out["checks"].items():
    assert err < 1e-6, name
  assert abs(float(out["energy"]) - ns["E_MIN"]) < 1e-6
  assert np.abs(out["modes"][:6]).max() < 1e-6 and out["modes"][6] > 1.0
  assert int(out["rigidity_rank"]) == 3 * ns["N"] - 6


def test_option_greeks_and_implied_vols() -> None:
  out = _load("roots/option_greeks")["main"]()
  for key in ("price_error", "delta_error", "vega_error", "gamma_error"):
    assert out[key] < 1e-10, key
  assert out["gradient_vs_fd"] < 1e-6
  assert out["iv_error"] < 1e-8 and out["iv_brentq_error"] < 1e-9
  assert out["sensitivities_error"] < 1e-12


def test_robot_arm_kinematics_and_batched_ik() -> None:
  ns = _load("linalg/robot_arm_ik")
  out = ns["main"]()
  assert out["fk_error"] < 1e-12 and out["jacobian_error"] < 1e-8
  assert np.sum(out["ik_error_numpy"] < 1e-8) >= 0.98 * ns["BATCH"]
  np.testing.assert_allclose(out["ik_error"], out["ik_error_numpy"], atol=1e-12)
  assert out["within_limits"]


def test_lagrangian_mechanics_textbook_energy_and_lyapunov() -> None:
  out = _load("integrators/lagrangian_mechanics")["main"]()
  assert out["textbook_error"] < 1e-12
  assert out["energy_drift"] < 1e-6
  assert abs(out["ftle"] - out["ftle_twin"]) < 1e-2


def test_bratu_newton_matches_scipy_and_stays_definite() -> None:
  ns = _load("roots/bratu_newton")
  out = ns["main"]()
  assert out["reference_error"] < 1e-10
  n = ns["N_GRID"] ** 2
  for _, _, residual, iterations, inertia in out["continuation"]:
    assert residual < ns["TOL"] and iterations < ns["MAX_NEWTON"]
    assert inertia == (n, 0, 0)
  assert len({colors for _, _, colors in out["coloring"]}) == 1


def test_periodic_orbit_period_and_floquet_multipliers() -> None:
  ns = _load("roots/periodic_orbit")
  out = ns["main"]()
  assert abs(out["period"] - ns["PERIOD_MU_1"]) < 1e-8
  assert abs(out["multipliers"][1] - 1.0) < 1e-8
  np.testing.assert_allclose(out["det_monodromy"], out["liouville"], rtol=1e-6)
  periods = [t for _, _, t, _, _ in out["sweep"]]
  assert all(np.diff(periods) > 0)


def test_sinkhorn_gradient_is_the_potential() -> None:
  out = _load("core/sinkhorn_transport")["main"]()
  assert out["grad_vs_potential"] < 1e-8
  fd, ad = out["directional_fd"]
  assert abs(fd - ad) < 1e-4 * abs(ad)
  assert abs(out["plan_cost"] - out["W2_exact"]) < 0.05 * out["W2_exact"]


def test_gaussian_process_likelihood_derivatives_and_fit() -> None:
  out = _load("linalg/gaussian_process")["main"]()
  assert out["value_error"] < 1e-10 and out["grad_error"] < 1e-10 and out["hess_error"] < 1e-5
  assert out["grad_at_optimum"] < 1e-3 and out["hess_eigenvalues"].min() > 0
  assert out["coverage"] > 0.9


def test_ekf_likelihood_gradient_and_identification() -> None:
  ns = _load("integrators/ekf_identification")
  out = ns["main"]()
  assert out["grad_error"] < 1e-6
  assert out["nll_fit"] <= out["nll_true"]
  fitted, true = np.exp(out["theta"]), np.exp(ns["THETA_TRUE"])
  np.testing.assert_allclose(fitted[[0, 1, 3]], true[[0, 1, 3]], rtol=0.15)


def test_ilqr_reaches_a_stationary_point_clear_of_obstacles() -> None:
  out = _load("ocp/ilqr")["main"]()
  assert out["gradient_norm"] < 1e-4
  assert abs(out["cost"] - out["lbfgs_cost"]) < 1e-4 * out["cost"]
  assert out["clearance"] > 0 and np.abs(out["final_error"]).max() < 0.05


@pytest.mark.method("opt.piqp")
def test_tiny_qp_matches_the_closed_form() -> None:
  ns = _load("opt/tiny_qp")
  for _, _, _, _, err, kkt in ns["main"]()["rows"]:
    assert err < 1e-6 and kkt < 1e-6


@pytest.mark.method("opt.piqp")
def test_cbf_filter_is_safe_and_matches_slsqp() -> None:
  ns = _load("opt/cbf_safety_filter")
  out = ns["main"]()
  assert out["barriers"].min() > -1e-8
  assert out["scipy_error"] < 1e-6
  assert len(out["active"]) > 0


@pytest.mark.method("opt.piqp")
def test_portfolio_frontier_and_references() -> None:
  ns = _load("opt/portfolio_qp")
  out = ns["main"]()
  assert out["dense_error"] < 1e-6 and out["slsqp_error"] < 1e-6
  returns = [r for _, r, _, _, _, _, _ in out["frontier"]]
  risks = [s for _, _, s, _, _, _, _ in out["frontier"]]
  assert all(np.diff(returns) <= 1e-9) and all(np.diff(risks) <= 1e-9)
  for _, _, _, _, total, lo, hi in out["frontier"]:
    assert abs(total - 1) < 1e-6 and lo > -1e-8 and hi < ns["X_MAX"] + 1e-8


@pytest.mark.method("opt.ipopt")
def test_nmpc_cartpole_swings_up() -> None:
  ns = _load("opt/nmpc_cartpole")
  out = ns["main"]()
  assert set(out["statuses"]) == {"OK"}
  assert np.abs(out["final"]).max() < 1e-2
  assert np.abs(out["inputs"]).max() <= ns["U_MAX"] + 1e-6


@pytest.mark.method("opt.sqp")
@pytest.mark.method("opt.ipopt")
def test_mhe_tracks_the_pendulum_and_agrees_with_ipopt() -> None:
  ns = _load("opt/mhe")
  out = ns["main"]()
  assert out["sqp_vs_ipopt"] < 1e-8
  assert out["rate_rmse"] < out["naive_rmse"] / 3
  assert abs(out["b"][-1] - ns["B_TRUE"]) < 0.15


@pytest.mark.method("opt.ipopt")
def test_opf_case9_reaches_the_published_optimum() -> None:
  ns = _load("opt/optimal_power_flow")
  out = ns["main"]()
  assert out["status"].ok
  assert abs(out["cost"] - ns["OPTIMUM"]) < 0.01
  assert out["mismatch"] < 1e-9
