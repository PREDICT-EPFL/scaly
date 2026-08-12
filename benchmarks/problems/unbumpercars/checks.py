"""Formulation correctness gates for the unbumpercars HCBF filter problem.

These are properties of *this benchmark problem* — the agreement between the Alloy
and CasADi oracles, the ordering of the parameter tail, the pair barrier's relative
degree, and the exact-Hessian path over closed-loop samples. They run before any timing is recorded, via
``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite
covers Alloy's core and must not depend on a benchmark problem. The sparse Lagrangian
Hessian and CasADi-differential behaviours these lean on have self-contained
reproductions in ``tests/alloy/test_alloy_sparsity.py`` and
``tests/alloy/test_factory_casadi.py``.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator

import numpy as np

from benchmarks.problems.unbumpercars.common import (
  DT_VF_DEADZONE,
  CarPhysics,
  ClosedLoopConfig,
  DTMLPWeights,
  FilterConfig,
  HCBFConfig,
  NSTATE,
  dt_mlp_step_np,
  dt_mlp_step_smooth_np,
  load_ct_full_weights,
  load_dt_mlp_weights,
  pair_b_np,
  pose_rk4_np,
  rk4_step_np,
  sample_initial_states,
  wall_b_np,
  wall_h_np,
)

EXACT_HESS_ENV = "ALLOY_RUN_UNBUMPERCARS_EXACT_HESS"


def check_default_output_dir() -> None:
  """The direct module and unified runner share the canonical closed-loop tree."""
  from benchmarks.harness import CLOSED_LOOP_RESULTS
  from benchmarks.problems.unbumpercars.run_closed_loop import DEFAULT_OUT_DIR, result_dir

  assert DEFAULT_OUT_DIR == CLOSED_LOOP_RESULTS
  assert result_dir(DEFAULT_OUT_DIR) == CLOSED_LOOP_RESULTS / "unbumpercars"


def check_oracle_matches_casadi() -> None:
  """Alloy and CasADi build the same cost and constraint rows for the same filter, either model."""
  from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter, build_alloy_oracle

  for model in ("ct", "dt"):
    loop_cfg = ClosedLoopConfig(ncars=2)
    filt_cfg = FilterConfig(model=model)
    weights = load_dt_mlp_weights() if model == "dt" else load_ct_full_weights()
    oracle = build_alloy_oracle(loop_cfg, filt_cfg)
    ca_filt = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)

    rng = np.random.default_rng(3)
    bar_x = sample_initial_states(loop_cfg).reshape(-1)
    u_des = rng.uniform(-0.5, 0.5, 2 * loop_cfg.ncars)
    slack = rng.uniform(0.0, 0.05, loop_cfg.n_slack)
    z = np.concatenate([np.clip(u_des + rng.normal(scale=0.1, size=u_des.size), -1.0, 1.0), slack])
    physics, dt = loop_cfg.physics.array(), np.array([loop_cfg.dt])
    p = np.concatenate([bar_x, u_des, weights.packed, physics, dt])

    cost_al, g_al = oracle(z, bar_x, u_des, weights.packed, physics, dt)
    np.testing.assert_allclose(np.asarray(cost_al).reshape(-1), np.asarray(ca_filt.cost_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(g_al).reshape(-1), np.asarray(ca_filt.g_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
    assert np.asarray(g_al).size == loop_cfg.n_slack == loop_cfg.n_pairs + 4 * loop_cfg.ncars


def check_parameter_tail_order() -> None:
  """An independent NumPy reference with a distinct value per physics entry pins the p-tail order."""
  from benchmarks.problems.unbumpercars.filters import build_alloy_oracle

  physics = CarPhysics(lf=0.9, lr=1.3, max_delta=0.7, steering_time_constant=0.45, x_min=-3.0, x_max=11.0, y_min=1.0, y_max=17.0)
  loop_cfg = ClosedLoopConfig(ncars=2, physics=physics, dt=0.17)
  weights = load_ct_full_weights()
  oracle = build_alloy_oracle(loop_cfg, FilterConfig(model="ct"))

  rng = np.random.default_rng(9)
  states = sample_initial_states(loop_cfg)
  u = rng.uniform(-0.8, 0.8, (2, 2))
  slack = rng.uniform(0.01, 0.05, loop_cfg.n_slack)
  z = np.concatenate([u.reshape(-1), slack])
  _, g_al = oracle(z, states.reshape(-1), np.zeros(4), weights.packed, physics.array(), np.array([loop_cfg.dt]))

  # rk4_step_np's theta wrap is a no-op here: the barriers read theta only through
  # sin/cos, and the wall rows through pose and world velocity.
  nxt = np.stack([rk4_step_np(states[i], u[i], loop_cfg.dt, weights, physics) for i in range(2)])
  rows = [pair_b_np(nxt[0], nxt[1], loop_cfg) - (1.0 - loop_cfg.pair_gamma) * pair_b_np(states[0], states[1], loop_cfg)]
  for i in range(2):
    for b_next, b_cur in zip(wall_b_np(nxt[i], loop_cfg), wall_b_np(states[i], loop_cfg), strict=True):
      rows.append(b_next - (1.0 - loop_cfg.wall_gamma) * b_cur)
  np.testing.assert_allclose(np.asarray(g_al).reshape(-1), np.array(rows) + slack, rtol=1e-9, atol=1e-9)


def check_pair_barrier_is_order_one() -> None:
  """One step of the discrete model moves the pair barrier, and braking is what moves it up.

  This is the property the whole formulation rests on: at dt = 0.1 the position
  barrier it replaced barely responds to the input within one step, so the filter had
  no authority over it; the hyperbolic barrier constrains closing speed and does.
  The second assertion pins the envelope's meaning — closing head-on at 2 m/s with
  0.72 m of clearance is not stoppable, so the barrier must be negative there.
  """
  loop_cfg = ClosedLoopConfig(ncars=2)
  weights = load_ct_full_weights()
  head_on = (
    np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]),
    np.array([3.0, 0.0, np.pi, 1.0, 0.0, 0.0, 0.0]),
  )

  def after(u: list[float]) -> float:
    nxt = [rk4_step_np(x, np.array(u), loop_cfg.dt, weights, loop_cfg.physics) for x in head_on]
    return pair_b_np(nxt[0], nxt[1], loop_cfg)

  assert after([-1.0, 0.0]) - after([1.0, 0.0]) > 0.1, "braking must buy at least 0.1 m/s of closing speed in one step"
  assert pair_b_np(*head_on, loop_cfg) < 0.0
  # a larger braking envelope claims more stopping authority, so the same state looks safer
  bolder = ClosedLoopConfig(ncars=2, hcbf=HCBFConfig(envelope_c=2.0 * loop_cfg.hcbf.envelope_c))
  assert pair_b_np(*head_on, bolder) > pair_b_np(*head_on, loop_cfg)


def check_velocity_wall_barrier_has_control_authority() -> None:
  """The wall row responds to control and contains a car requesting straight-ahead motion.

  The discrete model advances pose from the current velocity, held over the step, so its
  one-step position is exactly independent of control. A position DTCBF can only consume slack.
  The order-1 barrier instead reads the predicted velocity and must make the filter intervene
  before the car crosses the wall.
  """
  from benchmarks.problems.unbumpercars.filters import AlloyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=1, steps=150, target_center=False)
  filt_cfg = FilterConfig(model="dt", limited_memory_hessian=True)
  weights = load_dt_mlp_weights()
  state = np.array([13.0, 7.5, 0.0, 0.8, 0.0, 0.0, 0.0])
  brake = dt_mlp_step_smooth_np(state, np.array([-1.0, 0.0]), loop_cfg.dt, weights, loop_cfg.physics)
  throttle = dt_mlp_step_smooth_np(state, np.array([1.0, 0.0]), loop_cfg.dt, weights, loop_cfg.physics)
  assert wall_b_np(brake, loop_cfg)[1] - wall_b_np(throttle, loop_cfg)[1] > 0.1

  alloy_filter = AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  casadi_filter = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  sim = Simulator(sample_initial_states(loop_cfg), loop_cfg)
  min_clearance = float("inf")
  max_override = 0.0
  for step in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    assert desired[0, 1] == 0.0
    safe = alloy_filter.compute_safe_input(sim.states, desired, step)
    safe_casadi = casadi_filter.compute_safe_input(sim.states, desired, step)
    np.testing.assert_allclose(safe, safe_casadi, rtol=0.0, atol=1e-7)
    max_override = max(max_override, float(np.max(np.abs(safe - desired))))
    sim.step(safe)
    min_clearance = min(min_clearance, min(wall_h_np(sim.states[0], loop_cfg)))

  assert all(item.success for safety_filter in (alloy_filter, casadi_filter) for item in safety_filter.stats_history)
  assert max_override > 0.1, "the wall barrier never materially changed the requested control"
  assert min_clearance > -0.01, f"the car escaped {abs(min_clearance):.3f} m past the wall margin"


def check_dt_plant_pieces() -> None:
  """The plant's three pieces stay separate: analytic pose, learned velocity, first-order steering.

  Only the middle piece is learned, so perturbing the network must leave the pose rows
  bit-identical — the pose is integrated with the velocity held over the step, not with the
  network's prediction. The steering row is the checkpoint's own actuator at ``tau``; the
  continuous-time model's ``3 tau`` would give a third of the motion.
  """
  physics = CarPhysics()
  weights = load_dt_mlp_weights()
  dt = 0.1
  rng = np.random.default_rng(11)
  state = np.array([1.0, 2.0, 0.4, 0.9, 0.12, -0.05, 0.3])
  u = np.array([0.3, -0.6])

  nxt = dt_mlp_step_np(state, u, dt, weights, physics)
  bumped = DTMLPWeights(
    weights.path,
    weights.x_scale,
    *(np.asarray(w) + rng.normal(scale=0.1, size=np.shape(w)) for w in (weights.w0, weights.b0, weights.w1, weights.b1, weights.w2, weights.b2)),
  )
  perturbed = dt_mlp_step_np(state, u, dt, bumped, physics)
  np.testing.assert_array_equal(perturbed[:3], nxt[:3])
  assert np.abs(perturbed[3:6] - nxt[3:6]).max() > 1e-3, "the perturbation must actually move the learned rows"
  np.testing.assert_allclose(nxt[:3], pose_rk4_np(state, dt, physics), rtol=0.0, atol=0.0)

  delta_ref = u[1] * physics.max_delta
  expected_delta = state[6] + dt * (delta_ref - state[6]) / physics.steering_time_constant
  np.testing.assert_allclose(nxt[6], expected_delta, rtol=1e-12)
  assert abs(nxt[6] - state[6]) > 2.0 * abs(dt * (delta_ref - state[6]) / (3.0 * physics.steering_time_constant)), "actuator must be the fast one"


def check_dt_plant_control_order() -> None:
  """The network's two controls are fed in the opposite order to ours, and the deadzone bites.

  Under the correct order full throttle beats full brake by >0.05 m/s of next-step speed at
  every speed in range. Swapping the pair leaves under 0.035 m/s and flips sign at the top of
  the range, which is how the ordering was recovered in the first place.
  """
  physics = CarPhysics()
  weights = load_dt_mlp_weights()
  dt = 0.1
  for vf in (0.05, 0.25, 0.5, 1.0, 1.5, 2.0):
    state = np.array([0.0, 0.0, 0.0, vf, 0.0, 0.0, 0.0])
    throttle = dt_mlp_step_np(state, np.array([1.0, 0.0]), dt, weights, physics)[3]
    brake = dt_mlp_step_np(state, np.array([-1.0, 0.0]), dt, weights, physics)[3]
    assert throttle - brake > 0.05, f"at vf={vf} the throttle/brake authority is {throttle - brake:.4f}, too small to be the right control order"

  creeping = np.array([0.0, 0.0, 0.0, 0.5 * DT_VF_DEADZONE, 0.0, 0.0, 0.0])
  assert dt_mlp_step_np(creeping, np.array([-1.0, 0.0]), dt, weights, physics)[3] == 0.0


def check_dt_filter_model_matches_numpy() -> None:
  """Both symbolic DT prediction paths reproduce `dt_mlp_step_smooth_np`, row for row.

  This is the gate on the filter's *model*, independent of the barrier wrapped around it: the
  constraint rows only ever see the step's output, so if the step is right and the barrier is
  gated separately, the oracle is right. It also pins the two departures from the plant — the
  smoothed ReLU and the dropped deadzone — by checking the filter against a reference that has
  them and the plant against one that does not (`dt_plant_pieces`).
  """
  from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter, alloy_dt_mlp_step_fn

  physics = CarPhysics()
  weights = load_dt_mlp_weights()
  filt_cfg = FilterConfig(model="dt")
  loop_cfg = ClosedLoopConfig(ncars=2, dt=0.1)
  ca_filt = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, _build_solver=False)
  ca = ca_filt.ca
  x_sym, u_sym = ca.MX.sym("x", NSTATE), ca.MX.sym("u", 2)
  pw_sym, ph_sym, dt_sym = ca.MX.sym("pw", weights.packed.size), ca.MX.sym("ph", 8), ca.MX.sym("dt")
  ca_step = ca.Function("dt_step", [x_sym, u_sym, pw_sym, ph_sym, dt_sym], [ca_filt._dt_step(x_sym, u_sym, pw_sym, ph_sym, dt_sym)])

  rng = np.random.default_rng(5)
  pw, ph = weights.packed, physics.array()
  for _ in range(12):
    state = np.concatenate(
      [rng.uniform(2.0, 13.0, 2), rng.uniform(-np.pi, np.pi, 1), rng.uniform(0.0, 2.0, 1), rng.normal(0.0, 0.3, 2), rng.uniform(-2.0, 2.0, 1)]
    )
    u = rng.uniform(-1.0, 1.0, 2)
    want = dt_mlp_step_smooth_np(state, u, loop_cfg.dt, weights, physics)
    got_ca = np.asarray(ca_step(state, u, pw, ph, loop_cfg.dt), dtype=np.float64).reshape(-1)
    got_al = np.asarray(alloy_dt_mlp_step_fn(state, u, pw, ph, np.array([loop_cfg.dt])), dtype=np.float64).reshape(-1)
    np.testing.assert_allclose(got_ca, want, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(got_al, want, rtol=1e-10, atol=1e-10)
    # The smoothing is an approximation of the plant, not a rewrite of it. Its pose rows are the
    # plant's exactly, and its velocity rows are within 3 mm/s — except where the plant's dropped
    # deadzone fires, which is the one place the two models genuinely disagree: with vf near zero
    # under hard braking the network extrapolates negative and the plant clamps to a standstill.
    plant = dt_mlp_step_np(state, u, loop_cfg.dt, weights, physics)
    np.testing.assert_allclose(want[:3], plant[:3], rtol=0.0, atol=1e-15)
    if plant[3] > 0.0:
      assert np.abs(want[3:] - plant[3:]).max() < 3e-3, f"smoothing moved the velocity block by {np.abs(want[3:] - plant[3:]).max():.4f}"
    else:
      assert want[3] < DT_VF_DEADZONE, "the deadzone fired in the plant, so the filter's vf must be the small value it clamped"


def check_backends_solve_alike_per_step() -> None:
  """Both backends return the same safe input when handed the same state.

  This is the agreement that survives the DT plant's chaotic closed loop, where the two
  backends' *trajectories* separate from rounding alone (see the README, "Why the two
  backends' trajectories differ"). Comparing rollouts cannot distinguish that amplification
  from a real solver-plumbing divergence; comparing per-step solutions on a shared state can.
  Both filters see the same state at every step, so warm starts stay in lockstep too.

  The cars start on a collision course on purpose. From `sample_initial_states` they are far
  enough apart that every row is slack, the filter returns the desired input untouched, and
  the comparison holds no matter what either backend computes.
  """
  from benchmarks.problems.unbumpercars.filters import AlloyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=3, steps=5)
  filt_cfg = FilterConfig(model="ct")
  weights = load_ct_full_weights()
  alloy_filt = AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  casadi_filt = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  # two closing head-on 3 m apart, inside the 2.28 m envelope's reach, plus a third crossing
  converging = np.array(
    [
      [5.0, 7.5, 0.0, 1.2, 0.0, 0.0, 0.0],
      [8.0, 7.5, np.pi, 1.2, 0.0, 0.0, 0.0],
      [6.5, 9.6, -0.5 * np.pi, 1.0, 0.0, 0.0, 0.0],
    ]
  )
  sim = Simulator(converging, loop_cfg)

  acting = slacked = 0
  for step in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    u_alloy = alloy_filt.compute_safe_input(sim.states, desired, step)
    u_casadi = casadi_filt.compute_safe_input(sim.states, desired, step)
    alloy_stats, casadi_stats = alloy_filt.stats_history[-1], casadi_filt.stats_history[-1]
    assert alloy_stats.success and casadi_stats.success
    acting += np.abs(u_alloy - desired).max() > 1e-3
    slacked += alloy_stats.slack_l1 > 1e-3
    np.testing.assert_allclose(u_alloy, u_casadi, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(alloy_stats.objective, casadi_stats.objective, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(alloy_stats.slack_l1, casadi_stats.slack_l1, rtol=0.0, atol=1e-7)
    sim.step(u_alloy)
  # Non-vacuity is a property of the run, not of each step: the filter must override the
  # desired input on most steps, so the solution actually depends on the formulation, and the
  # L1 slack kink must be reached at least once. How many steps need slack depends on how well
  # the braking envelope describes the plant, so that count is deliberately only bounded below.
  assert acting >= 3 and slacked >= 1, f"only {acting}/{loop_cfg.steps} acting and {slacked} slacked steps; the comparison is near-vacuous"


def check_exact_hess_matches_casadi_on_closed_loop_samples() -> None:
  """Opt-in: the exact Lagrangian Hessian tracks CasADi along a real closed-loop rollout."""
  from benchmarks.problems.unbumpercars.filters import AlloyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  os.environ["ALLOY_STRICT_JVP_MANY"] = "1"
  try:
    loop_cfg = ClosedLoopConfig(ncars=2, steps=5)
    filt_cfg = FilterConfig(model="ct", limited_memory_hessian=False)
    weights = load_ct_full_weights()
    alloy_filt = AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
    casadi_filt = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
    assert alloy_filt.hess_fn is not None and casadi_filt.hess_fn is not None
    sim = Simulator(sample_initial_states(loop_cfg), loop_cfg)

    for step in range(loop_cfg.steps):
      desired = sim.desired_inputs()
      safe = alloy_filt.compute_safe_input(sim.states, desired, step)
      assert alloy_filt.last_z is not None and alloy_filt.last_mult_g is not None
      z, lam = alloy_filt.last_z, alloy_filt.last_mult_g
      bar_x, u_des = sim.states.reshape(-1), desired.reshape(-1)
      physics, dt = loop_cfg.physics.array(), np.array([loop_cfg.dt])
      p = np.concatenate([bar_x, u_des, weights.packed, physics, dt])
      alloy_values = np.asarray(alloy_filt.hess_fn.eval_list(z, 1.0, lam, bar_x, u_des, weights.packed, physics, dt)[0], dtype=np.float64).reshape(
        -1
      )[alloy_filt.hess_lower_mask]
      casadi_dense = np.asarray(casadi_filt.hess_fn(z, p, 1.0, lam), dtype=np.float64)
      np.testing.assert_allclose(alloy_values, casadi_dense[alloy_filt.hess_rows, alloy_filt.hess_cols], rtol=1e-8)
      sim.step(safe)
  finally:
    os.environ.pop("ALLOY_STRICT_JVP_MANY", None)


# name -> (check, requires an IPOPT-backed solve, requires CasADi)
CHECKS: dict[str, tuple[Callable[[], None], bool, bool]] = {
  "default_output_dir": (check_default_output_dir, False, False),
  "oracle_matches_casadi": (check_oracle_matches_casadi, False, True),
  "parameter_tail_order": (check_parameter_tail_order, False, False),
  "pair_barrier_is_order_one": (check_pair_barrier_is_order_one, False, False),
  "velocity_wall_barrier_has_control_authority": (check_velocity_wall_barrier_has_control_authority, True, True),
  "dt_plant_pieces": (check_dt_plant_pieces, False, False),
  "dt_plant_control_order": (check_dt_plant_control_order, False, False),
  "dt_filter_model_matches_numpy": (check_dt_filter_model_matches_numpy, False, True),
  "backends_solve_alike": (check_backends_solve_alike_per_step, True, True),
  "exact_hess": (check_exact_hess_matches_casadi_on_closed_loop_samples, True, True),
}


def run_checks() -> Iterator[tuple[str, str]]:
  """Yield ``(name, outcome)`` for each gate; ``outcome`` is "ok", "skipped: ..." or raises."""
  from alloy.toolchain import solver_loadable

  have_ipopt = solver_loadable("ipopt")
  try:
    import casadi  # noqa: F401

    have_casadi = True
  except ImportError:
    have_casadi = False
  for name, (check, needs_ipopt, needs_casadi) in CHECKS.items():
    if name == "exact_hess" and os.environ.get(EXACT_HESS_ENV) != "1":
      yield name, f"skipped: opt-in, set {EXACT_HESS_ENV}=1"
    elif needs_ipopt and not have_ipopt:
      yield name, "skipped: IPOPT plugin not loadable"
    elif needs_casadi and not have_casadi:
      yield name, "skipped: casadi not installed"
    else:
      check()
      yield name, "ok"
