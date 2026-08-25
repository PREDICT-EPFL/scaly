"""Formulation correctness gates for the unbumpercars HCBF filter problem.

These are properties of *this benchmark problem* — the agreement between the Alloy
and CasADi oracles, the ordering of the parameter tail, the pair barrier's relative
degree, and the exact-Hessian path over closed-loop samples. They run before any timing is recorded, via
``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite
covers Alloy's core and must not depend on a benchmark problem. The sparse Lagrangian
Hessian and CasADi-differential behaviours these lean on have self-contained
reproductions in ``tests/ad/test_sparsity.py`` and
``tests/function/test_factory.py``.
"""

from __future__ import annotations

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


def check_oracles_solve_alike_per_step() -> None:
  """Both IPOPT oracle providers return the same safe input when handed the same state.

  This is the agreement that survives the DT plant's chaotic closed loop, where the two
  providers' *trajectories* separate from rounding alone (see the README, "Why the two
  providers' trajectories differ"). Comparing rollouts cannot distinguish that amplification
  from a real solver-plumbing divergence; comparing per-step solutions on a shared state can.
  Both filters see the same state at every step, so warm starts stay in lockstep too.

  The cars start on a collision course on purpose. From `sample_initial_states` they are far
  enough apart that every row is slack, the filter returns the desired input untouched, and
  the comparison holds no matter what either provider computes.
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
  """The default exact Lagrangian Hessian tracks CasADi along a real closed-loop rollout."""
  from benchmarks.problems.unbumpercars.filters import AlloyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=2, steps=5)
  filt_cfg = FilterConfig(model="ct")
  assert not filt_cfg.limited_memory_hessian
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
    alloy_values = np.asarray(alloy_filt.hess_fn.eval_list(z, 1.0, lam, bar_x, u_des, weights.packed, physics, dt)[0], dtype=np.float64).reshape(-1)[
      alloy_filt.hess_lower_mask
    ]
    casadi_dense = np.asarray(casadi_filt.hess_fn(z, p, 1.0, lam), dtype=np.float64)
    np.testing.assert_allclose(alloy_values, casadi_dense[alloy_filt.hess_rows, alloy_filt.hess_cols], rtol=1e-8)
    sim.step(safe)


def check_canonical_hessian_handoff() -> None:
  """A canonical C=8 artifact drives both exact-Hessian codegen providers."""
  import tempfile
  from pathlib import Path

  from benchmarks.harness.sweep import _samples, build_kernel

  cfg = ClosedLoopConfig()
  weights = load_dt_mlp_weights()
  state = sample_initial_states(cfg).reshape(-1)
  desired = np.tile([cfg.nominal_speed, 0.0], cfg.ncars)
  arrays = {
    "z": np.concatenate([desired, np.zeros(cfg.n_slack)]),
    "lam_f": np.array(1.0),
    "lam_g": np.linspace(0.1, 1.0, cfg.n_slack),
    "bar_x": state,
    "u_des": desired,
    "pw": weights.packed,
    "physics": cfg.physics.array(),
    "dt": np.array([cfg.dt]),
  }
  with tempfile.TemporaryDirectory() as directory:
    root = Path(directory)
    artifact = root / "representative_fe_inputs.npz"
    np.savez_compressed(artifact, **arrays)
    with np.load(artifact) as loaded:
      harvested = {name: np.asarray(loaded[name], dtype=np.float64) for name in loaded.files}
    for backend in ("alloy", "casadi_mx"):
      output = root / backend
      output.mkdir()
      info = build_kernel("unbumpercars", cfg.ncars, backend, output)
      _samples("unbumpercars", cfg.ncars, info, output, harvested)


def check_casadi_ipopt_is_compiled() -> None:
  """The timed CasADi column is generated C linked to Alloy's IPOPT."""
  from pathlib import Path

  from alloy.solvers.paths import solver_paths
  from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter

  loop_cfg = ClosedLoopConfig(ncars=2, steps=1)
  controller = CasadiDTCBFSafetyFilter(loop_cfg, FilterConfig(ipopt_max_iter=40), load_dt_mlp_weights())
  expected = solver_paths(required=True).loads["ipopt"]
  assert controller.solver.compiled and not controller.solver.expand and expected is not None
  assert controller.solver.resolved_ipopt_library.read_bytes() == Path(expected).read_bytes()
  states = sample_initial_states(loop_cfg)
  controller.compute_safe_input(states, np.zeros((loop_cfg.ncars, 2)))
  assert controller.stats_history[-1].eval_counts["hess_lag"] > 0


def check_sqp_matches_ipopt_per_step() -> None:
  """SQP and IPOPT return the same safe input when handed the same state.

  Per-step on shared states for the same reason as ``oracles_solve_alike``: the DT plant's
  chaotic closed loop amplifies rounding, so rollout comparison cannot separate that from a
  solver divergence. The converging pair is deliberately *asymmetric* (lateral offsets and
  different speeds): the head-on symmetric start has two mirror-image optima and the two
  solvers legitimately pick different ones, which says nothing about solver health.
  Measured healthy-step differences at the shipped settings (exact Hessian, filter first,
  default dual_tol 1e-4) are control/z <= 1.69e-5 and objective <= 9.59e-5 on obj ~1; the
  1e-4/5e-4 envelopes keep the gate close to those declared workload settings. On
  divergence the message carries the first diverging step with both solvers' status and
  constraint violation — the signal the Phase 9 robustness work consumes.
  """
  from benchmarks.problems.unbumpercars.filters import AlloyDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=2, steps=4, target_center=False)
  filt_cfg = FilterConfig(ipopt_max_iter=40)
  weights = load_dt_mlp_weights()
  ipopt_filter = AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  sqp_filter = AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, solver="sqp")
  initial = np.zeros((2, NSTATE), dtype=np.float64)
  initial[:, :2] = [[6.3, 7.35], [8.7, 7.7]]
  initial[:, 2] = [0.0, np.pi]
  initial[:, 3] = [0.75, 0.85]
  sim = Simulator(initial, loop_cfg)
  acted = False
  for step in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    u_ipopt = ipopt_filter.compute_safe_input(sim.states, desired, step)
    u_sqp = sqp_filter.compute_safe_input(sim.states, desired, step)
    stats_i, stats_s = ipopt_filter.stats_history[-1], sqp_filter.stats_history[-1]
    du = float(np.max(np.abs(u_ipopt - u_sqp)))
    dz = float(np.max(np.abs(np.asarray(ipopt_filter.last_z) - np.asarray(sqp_filter.last_z))))
    dobj = abs(stats_i.objective - stats_s.objective)
    viol_i, viol_s = max(0.0, -stats_i.min_g), max(0.0, -stats_s.min_g)
    assert stats_i.success and stats_s.success and du <= 1e-4 and dz <= 1e-4 and dobj <= 5e-4 and max(viol_i, viol_s) <= 1e-6, (
      f"SQP first diverges from IPOPT at step {step}: |du|={du:.3e} |dz|={dz:.3e} |dobj|={dobj:.3e}; "
      f"ipopt: status={stats_i.status} success={stats_i.success} obj={stats_i.objective:.6e} violation={viol_i:.3e}; "
      f"sqp: status={stats_s.status} success={stats_s.success} obj={stats_s.objective:.6e} violation={viol_s:.3e}"
    )
    acted |= bool(np.max(np.abs(u_ipopt - desired)) > 0.1)
    sim.step(u_ipopt)
  assert acted, "the SQP-versus-IPOPT gate never exercised a binding constraint"


def check_sqp_oracles_agree() -> None:
  """The SQP's Alloy and CasADi C oracles return the same controls and timing split."""
  from benchmarks.problems.unbumpercars.filters import AlloyDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=2, steps=2, target_center=False)
  filt_cfg = FilterConfig(ipopt_max_iter=40)
  assert not filt_cfg.limited_memory_hessian
  weights = load_dt_mlp_weights()
  alloy_filter = AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, solver="sqp")
  casadi_filter = AlloyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, solver="sqp", oracle_provider="casadi")
  initial = np.zeros((2, NSTATE), dtype=np.float64)
  initial[:, :2] = [[6.3, 7.5], [8.7, 7.5]]
  initial[:, 2] = [0.0, np.pi]
  initial[:, 3] = 0.8
  sim = Simulator(initial, loop_cfg)
  acted = False
  for step in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    alloy_u = alloy_filter.compute_safe_input(sim.states, desired, step)
    casadi_u = casadi_filter.compute_safe_input(sim.states, desired, step)
    np.testing.assert_allclose(alloy_u, casadi_u, rtol=1e-9, atol=1e-9)
    acted |= bool(np.max(np.abs(alloy_u - desired)) > 1e-3)
    for controller in (alloy_filter, casadi_filter):
      stats = controller.nlp.last_stats
      report = controller.stats_history[-1]
      assert stats is not None and stats.status.name == "OK" and report.success and report.min_g >= -1e-6
      assert stats.t_qp > 0.0 and stats.n_eval_h > 0
      np.testing.assert_allclose(stats.t_total, stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rtol=1e-10)
    sim.step(alloy_u)
  assert acted, "the SQP oracle-provider gate never exercised a binding constraint"


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
  "oracles_solve_alike": (check_oracles_solve_alike_per_step, True, True),
  "exact_hess": (check_exact_hess_matches_casadi_on_closed_loop_samples, True, True),
  "canonical_hessian_handoff": (check_canonical_hessian_handoff, False, True),
  "casadi_ipopt_compiled": (check_casadi_ipopt_is_compiled, True, True),
  "sqp_matches_ipopt": (check_sqp_matches_ipopt_per_step, True, False),
  "sqp_oracles_agree": (check_sqp_oracles_agree, True, True),
}


def run_checks() -> Iterator[tuple[str, str]]:
  """Yield ``(name, outcome)`` for each gate; ``outcome`` is "ok", "skipped: ..." or raises."""
  from alloy.solvers.paths import solver_loadable

  have_ipopt = solver_loadable("ipopt")
  try:
    import casadi  # noqa: F401

    have_casadi = True
  except ImportError:
    have_casadi = False
  for name, (check, needs_ipopt, needs_casadi) in CHECKS.items():
    if needs_ipopt and not have_ipopt:
      yield name, "skipped: IPOPT plugin not loadable"
    elif needs_casadi and not have_casadi:
      yield name, "skipped: casadi not installed"
    else:
      check()
      yield name, "ok"
