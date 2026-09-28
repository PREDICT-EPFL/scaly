"""Formulation correctness gates for the unbumpercars HCBF filter problem.

These are properties of *this benchmark problem* — the agreement between the Scaly
and CasADi oracles, the ordering of the parameter tail, the pair barrier's relative
degree, and the exact-Hessian path over closed-loop samples. They run before any timing is recorded, via
``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite
covers Scaly's core and must not depend on a benchmark problem. The sparse Lagrangian
Hessian and CasADi-differential behaviours these lean on have self-contained
reproductions in ``tests/ad/test_sparsity.py`` and
``tests/function/test_factory.py``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from itertools import product
from typing import cast

import scaly as sc
from scaly.opt.external.graph import solver_descriptor
import numpy as np

from benchmarks.problems.unbumpercars.common import (
  DT_VF_DEADZONE,
  CarPhysics,
  ClosedLoopConfig,
  DTMLPWeights,
  FilterConfig,
  HCBFConfig,
  NCTRL,
  NSTATE,
  N_PHYSICS,
  N_PW_DT,
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
  """Scaly and CasADi build the same cost and constraint rows for the same filter, either model."""
  from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter, build_scaly_oracle

  for model, ncars, arena in product(("ct", "dt"), (1, 4), (False, True)):
    loop_cfg = ClosedLoopConfig(ncars=ncars, arena_avoidance=arena)
    filt_cfg = FilterConfig(model=model)
    weights = load_dt_mlp_weights() if model == "dt" else load_ct_full_weights()
    oracle = build_scaly_oracle(loop_cfg, filt_cfg)
    ca_filt = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, _build_solver=False)
    rng = np.random.default_rng(3)
    bar_x = sample_initial_states(loop_cfg).reshape(-1)
    u_des = rng.uniform(-0.5, 0.5, 2 * loop_cfg.ncars)
    slack = rng.uniform(0.0, 0.05, loop_cfg.n_slack)
    z = np.concatenate([np.clip(u_des + rng.normal(scale=0.1, size=u_des.size), -1.0, 1.0), slack])
    physics, dt = loop_cfg.physics.array(), np.array([loop_cfg.dt])
    p = np.concatenate([bar_x, u_des, weights.packed, physics, dt])

    cost_al, g_al = oracle((z, bar_x, u_des, weights.packed, physics, dt))
    np.testing.assert_allclose(np.asarray(cost_al).reshape(-1), np.asarray(ca_filt.cost_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(np.asarray(g_al).reshape(-1), np.asarray(ca_filt.g_fn(z, p)).reshape(-1), rtol=1e-9, atol=1e-9)
    assert np.asarray(g_al).size == loop_cfg.n_slack
    if loop_cfg.n_pairs:
      args = (z, bar_x, u_des, weights.packed, physics, dt)
      for spec, reference in (
        (sc.factory.SpJac("g", "z"), np.asarray(ca_filt.jac_fn(z, p))),
        (sc.factory.SpHess("gamma", "z"), np.asarray(ca_filt.hess_fn(z, p, 0.0, np.arange(1.0, loop_cfg.n_slack + 1)))),
      ):
        is_hess = isinstance(spec, sc.factory.SpHess)
        derivative = oracle.factory("pair_derivative", [*oracle.input_names, *(["lam:g"] if is_hess else [])], [spec], aux={"gamma": ["g"]})
        sparsity = derivative.output_sparsities[0]
        assert sparsity is not None
        values = derivative((*args, np.arange(1.0, loop_cfg.n_slack + 1)) if is_hess else args)
        np.testing.assert_allclose(values, reference[sparsity.rows, sparsity.cols], rtol=1e-8, atol=1e-9)
        np.testing.assert_allclose(reference[~sparsity.to_mask()], 0.0, atol=1e-12)


def check_pair_jac_codegen_growth() -> None:
  """Mapped pair and wall rows stay retained in the Jacobian and exact Hessian."""
  from scaly.codegen.aot import render_c_module, render_c_source
  from scaly.ir.program import ProgramOp
  from benchmarks.problems.unbumpercars.filters import build_scaly_oracle

  lines = []
  hess_families = []
  for ncars in (2, 4, 8):
    oracle = build_scaly_oracle(ClosedLoopConfig(ncars=ncars), FilterConfig())
    jac = oracle.factory("pair_jac", list(oracle.input_names), [sc.factory.SpJac("g", "z")])
    lines.append(len(render_c_source(jac).splitlines()))
    if ncars < 4:
      continue
    hess = oracle.factory(
      "row_hess",
      [*oracle.input_names, "lam:cost", "lam:g"],
      [sc.factory.SpHess("gamma", "z")],
      aux={"gamma": ["cost", "g"]},
    )
    program = render_c_module(hess, typed_buffers=False).program
    procs = {proc.attrs["name"]: proc for proc in program.args[: int(program.attrs["proc_count"])]}

    def walk(nodes):
      stack = list(nodes)
      while stack:
        node = stack.pop()
        yield node
        stack.extend(node.args)

    root = procs[hess.name]
    families: dict[str, list[tuple[tuple[tuple[int, ...], ...], tuple[tuple[int, ...], ...], int, int]]] = {"pair": [], "wall": []}
    for node in walk(root.args[int(root.attrs["param_count"]) :]):
      if node.op != ProgramOp.CALL or node.attrs["callee"] not in procs:
        continue
      helper = procs[node.attrs["callee"]]
      origin = str(helper.attrs.get("hoisted_from", helper.attrs["name"]))
      family = next((name for name in families if origin.startswith(f"{name}_hcbf")), None)
      if family is None:
        continue
      params = helper.args[: int(helper.attrs["param_count"])]
      body = helper.args[len(params) :]
      body_nodes = list(walk(body))
      families[family].append(
        (
          tuple(tuple(param.attrs["shape"]) for param in params),
          tuple(tuple(item.attrs["shape"]) for item in body_nodes if item.op == ProgramOp.BUFFER),
          len(body),
          sum(item.op == ProgramOp.FOR for item in body_nodes),
        )
      )
    assert all(families.values()), f"exact Hessian lost a mapped row family at C={ncars}: {families}"
    hess_families.append({name: sorted(signatures) for name, signatures in families.items()})
  assert lines[-1] - lines[-2] < 100, f"Jacobian source lines grew with a row family for C=2,4,8: {lines}"
  assert hess_families[0] == hess_families[1], "exact Hessian pair/wall call families changed between C=4 and C=8"


def check_parameter_tail_order() -> None:
  """An independent NumPy reference with a distinct value per physics entry pins the p-tail order."""
  from benchmarks.problems.unbumpercars.filters import build_scaly_oracle

  physics = CarPhysics(lf=0.9, lr=1.3, max_delta=0.7, steering_time_constant=0.45, x_min=-3.0, x_max=11.0, y_min=1.0, y_max=17.0)
  loop_cfg = ClosedLoopConfig(ncars=2, physics=physics, dt=0.17)
  weights = load_ct_full_weights()
  oracle = build_scaly_oracle(loop_cfg, FilterConfig(model="ct"))

  rng = np.random.default_rng(9)
  states = sample_initial_states(loop_cfg)
  u = rng.uniform(-0.8, 0.8, (2, 2))
  slack = rng.uniform(0.01, 0.05, loop_cfg.n_slack)
  z = np.concatenate([u.reshape(-1), slack])
  _, g_al = oracle((z, states.reshape(-1), np.zeros(4), weights.packed, physics.array(), np.array([loop_cfg.dt])))

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
  from benchmarks.problems.unbumpercars.filters import ScalyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=1, steps=150, target_center=False)
  filt_cfg = FilterConfig(model="dt", limited_memory_hessian=True)
  weights = load_dt_mlp_weights()
  state = np.array([13.0, 7.5, 0.0, 0.8, 0.0, 0.0, 0.0])
  brake = dt_mlp_step_smooth_np(state, np.array([-1.0, 0.0]), loop_cfg.dt, weights, loop_cfg.physics)
  throttle = dt_mlp_step_smooth_np(state, np.array([1.0, 0.0]), loop_cfg.dt, weights, loop_cfg.physics)
  assert wall_b_np(brake, loop_cfg)[1] - wall_b_np(throttle, loop_cfg)[1] > 0.1

  scaly_filter = ScalyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  casadi_filter = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  sim = Simulator(sample_initial_states(loop_cfg), loop_cfg)
  min_clearance = float("inf")
  max_override = 0.0
  for step in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    assert desired[0, 1] == 0.0
    safe = scaly_filter.compute_safe_input(sim.states, desired, step)
    safe_casadi = casadi_filter.compute_safe_input(sim.states, desired, step)
    np.testing.assert_allclose(safe, safe_casadi, rtol=0.0, atol=1e-7)
    max_override = max(max_override, float(np.max(np.abs(safe - desired))))
    sim.step(safe)
    min_clearance = min(min_clearance, min(wall_h_np(sim.states[0], loop_cfg)))

  assert all(item.success for safety_filter in (scaly_filter, casadi_filter) for item in safety_filter.stats_history)
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
  from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter, scaly_dt_mlp_step_fn

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
    got_al = np.asarray(scaly_dt_mlp_step_fn((state, u, pw, ph, np.array([loop_cfg.dt]))), dtype=np.float64).reshape(-1)
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
  from benchmarks.problems.unbumpercars.filters import ScalyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=3, steps=5)
  filt_cfg = FilterConfig(model="ct")
  weights = load_ct_full_weights()
  scaly_filt = ScalyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
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
    u_scaly = scaly_filt.compute_safe_input(sim.states, desired, step)
    u_casadi = casadi_filt.compute_safe_input(sim.states, desired, step)
    scaly_stats, casadi_stats = scaly_filt.stats_history[-1], casadi_filt.stats_history[-1]
    assert scaly_stats.success and casadi_stats.success
    acting += np.abs(u_scaly - desired).max() > 1e-3
    slacked += scaly_stats.slack_l1 > 1e-3
    np.testing.assert_allclose(u_scaly, u_casadi, rtol=0.0, atol=1e-7)
    np.testing.assert_allclose(scaly_stats.objective, casadi_stats.objective, rtol=0.0, atol=1e-9)
    np.testing.assert_allclose(scaly_stats.slack_l1, casadi_stats.slack_l1, rtol=0.0, atol=1e-7)
    sim.step(u_scaly)
  # Non-vacuity is a property of the run, not of each step: the filter must override the
  # desired input on most steps, so the solution actually depends on the formulation, and the
  # L1 slack kink must be reached at least once. How many steps need slack depends on how well
  # the braking envelope describes the plant, so that count is deliberately only bounded below.
  assert acting >= 3 and slacked >= 1, f"only {acting}/{loop_cfg.steps} acting and {slacked} slacked steps; the comparison is near-vacuous"


def check_exact_hess_matches_casadi_on_closed_loop_samples() -> None:
  """The default exact Lagrangian Hessian tracks CasADi along a real closed-loop rollout."""
  from benchmarks.problems.unbumpercars.filters import ScalyDTCBFSafetyFilter, CasadiDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=2, steps=5)
  filt_cfg = FilterConfig(model="ct")
  assert not filt_cfg.limited_memory_hessian
  weights = load_ct_full_weights()
  scaly_filt = ScalyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  casadi_filt = CasadiDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  assert scaly_filt.hess_fn is not None and casadi_filt.hess_fn is not None
  sim = Simulator(sample_initial_states(loop_cfg), loop_cfg)

  for step in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    safe = scaly_filt.compute_safe_input(sim.states, desired, step)
    assert scaly_filt.last_z is not None and scaly_filt.last_mult_g is not None
    z, lam = scaly_filt.last_z, scaly_filt.last_mult_g
    bar_x, u_des = sim.states.reshape(-1), desired.reshape(-1)
    physics, dt = loop_cfg.physics.array(), np.array([loop_cfg.dt])
    p = np.concatenate([bar_x, u_des, weights.packed, physics, dt])
    hess_fn = cast(sc.Function, scaly_filt.hess_fn)
    hess_inputs = ((z, (bar_x, u_des, weights.packed, physics, dt)), (np.array(1.0), lam))
    scaly_values = np.asarray(hess_fn(hess_inputs), dtype=np.float64).reshape(-1)
    casadi_dense = np.asarray(casadi_filt.hess_fn(z, p, 1.0, lam), dtype=np.float64)
    np.testing.assert_allclose(scaly_values, casadi_dense[scaly_filt.hess_rows, scaly_filt.hess_cols], rtol=1e-8)
    sim.step(safe)


def check_synthetic_hessian_inputs_at_range() -> None:
  """The C=16 and C=32 kernels have finite inputs without collision-free placement."""
  from benchmarks.harness.sweep import _unbumpercars_hessian_inputs

  for size in (16, 32):
    pieces, expected = _unbumpercars_hessian_inputs(size, None)
    assert pieces["bar_x"].shape == (NSTATE * size,)
    assert np.all(np.isfinite(expected)) and np.any(expected != 0.0)


def check_canonical_hessian_handoff() -> None:
  """A canonical C=8 artifact drives both exact-Hessian codegen providers."""
  import tempfile
  from pathlib import Path

  from benchmarks.harness.sweep import _samples, build_kernel

  cfg = ClosedLoopConfig()
  weights = load_dt_mlp_weights()
  state = sample_initial_states(cfg).reshape(-1)
  desired = np.tile([cfg.nominal_speed, 0.0], cfg.ncars)
  arrays: dict[str, np.ndarray] = {
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
    np.savez_compressed(
      artifact,
      z=arrays["z"],
      lam_f=arrays["lam_f"],
      lam_g=arrays["lam_g"],
      bar_x=arrays["bar_x"],
      u_des=arrays["u_des"],
      pw=arrays["pw"],
      physics=arrays["physics"],
      dt=arrays["dt"],
    )
    with np.load(artifact) as loaded:
      harvested = {name: np.asarray(loaded[name], dtype=np.float64) for name in loaded.files}
    for backend in ("scaly", "casadi_sx", "casadi_mx"):
      output = root / backend
      output.mkdir()
      info = build_kernel("unbumpercars", cfg.ncars, backend, output)
      if backend == "scaly":
        assert info["layout"] == "lower"
      else:
        assert info["symbol"] == "nlp_hess_l"
        assert info["inputs"] == [
          ("x", NCTRL * cfg.ncars + cfg.n_slack),
          ("p", NSTATE * cfg.ncars + NCTRL * cfg.ncars + N_PW_DT + N_PHYSICS + 1),
          ("lam_f", 1),
          ("lam_g", cfg.n_slack),
        ]
        assert info["output_index"] == 0 and info["requested_output_indices"] == (0,)
        assert info["layout"] == "upper"
      _samples("unbumpercars", cfg.ncars, info, output, harvested)


def check_casadi_ipopt_is_compiled() -> None:
  """The timed CasADi column is generated C linked to Scaly's IPOPT."""
  from pathlib import Path
  from types import SimpleNamespace
  from unittest.mock import patch

  from benchmarks.problems.unbumpercars import filters

  from scaly.opt.external.paths import solver_paths
  from benchmarks.problems.unbumpercars.filters import CasadiDTCBFSafetyFilter

  loop_cfg = ClosedLoopConfig(ncars=2, steps=1)
  controller = CasadiDTCBFSafetyFilter(loop_cfg, FilterConfig(ipopt_max_iter=40), load_dt_mlp_weights())
  expected = solver_paths(required=True).loads["ipopt"]
  assert controller.solver.compiled and not controller.solver.expand and expected is not None
  assert controller.solver.resolved_ipopt_library.read_bytes() == Path(expected).read_bytes()
  states = sample_initial_states(loop_cfg)
  clock = [0.0]
  native_call = type(controller.solver).__call__

  def timed_solve(instance, *args):
    clock[0] += 0.002
    return native_call(instance, *args)

  def diagnostic(*args):
    clock[0] += 1.0
    return 1000.0

  with (
    patch.object(filters, "time", SimpleNamespace(perf_counter=lambda: clock[0])),
    patch.object(type(controller.solver), "__call__", timed_solve),
    patch.object(controller, "_time_eval", diagnostic),
  ):
    controller.compute_safe_input(states, np.zeros((loop_cfg.ncars, 2)))
  assert abs(controller.last_solve_wall_ms - 2.0) < 1e-12
  assert clock[0] > 5.0
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
  from benchmarks.problems.unbumpercars.filters import ScalyDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=2, steps=4, target_center=False)
  filt_cfg = FilterConfig(ipopt_max_iter=40)
  weights = load_dt_mlp_weights()
  ipopt_filter = ScalyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights)
  sqp_filter = ScalyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, solver="sqp")
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
  """The SQP's Scaly and CasADi C oracles return the same controls and timing split."""
  from benchmarks.problems.unbumpercars.filters import ScalyDTCBFSafetyFilter
  from benchmarks.problems.unbumpercars.run_closed_loop import Simulator

  loop_cfg = ClosedLoopConfig(ncars=2, steps=2, target_center=False)
  filt_cfg = FilterConfig(ipopt_max_iter=40)
  assert not filt_cfg.limited_memory_hessian
  weights = load_dt_mlp_weights()
  scaly_filter = ScalyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, solver="sqp")
  casadi_filter = ScalyDTCBFSafetyFilter(loop_cfg, filt_cfg, weights, solver="sqp", oracle_provider="casadi")
  initial = np.zeros((2, NSTATE), dtype=np.float64)
  initial[:, :2] = [[6.3, 7.5], [8.7, 7.5]]
  initial[:, 2] = [0.0, np.pi]
  initial[:, 3] = 0.8
  sim = Simulator(initial, loop_cfg)
  acted = False
  for step in range(loop_cfg.steps):
    desired = sim.desired_inputs()
    scaly_u = scaly_filter.compute_safe_input(sim.states, desired, step)
    casadi_u = casadi_filter.compute_safe_input(sim.states, desired, step)
    np.testing.assert_allclose(scaly_u, casadi_u, rtol=1e-9, atol=1e-9)
    acted |= bool(np.max(np.abs(scaly_u - desired)) > 1e-3)
    for controller in (scaly_filter, casadi_filter):
      stats = sc.opt.solver_stats(controller.nlp)
      report = controller.stats_history[-1]
      assert stats is not None and stats.status.name == "OK" and report.success and report.min_g >= -1e-6
      assert stats.t_qp > 0.0 and stats.n_eval_h > 0
      np.testing.assert_allclose(stats.t_total, stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rtol=1e-10)
    sim.step(scaly_u)
  assert acted, "the SQP oracle-provider gate never exercised a binding constraint"


def check_typed_problem_keeps_hessian_in_place() -> None:
  """The typed problem preserves bounded Hessian source and workspace sizes."""
  from scaly.codegen import render_c_module

  from benchmarks.problems.unbumpercars.common import ClosedLoopConfig, FilterConfig
  from benchmarks.problems.unbumpercars.filters import build_scaly_nlp

  hessian = solver_descriptor(build_scaly_nlp(ClosedLoopConfig(ncars=2), FilterConfig(model="dt"))).hess
  module = render_c_module(hessian, typed_buffers=False)

  # Baselines are about 86 KB and 34k doubles. Headroom catches a CALL boundary materializing
  # batched-JVP seed tables without pinning harmless local code-generation changes.
  assert len(module.body.encode()) < 200_000
  assert module.workspace_size < 100_000


# name -> (check, requires an IPOPT-backed solve, requires CasADi)
CHECKS: dict[str, tuple[Callable[[], None], bool, bool]] = {
  "typed_hessian_in_place": (check_typed_problem_keeps_hessian_in_place, False, False),
  "default_output_dir": (check_default_output_dir, False, False),
  "oracle_matches_casadi": (check_oracle_matches_casadi, False, True),
  "pair_jac_codegen_growth": (check_pair_jac_codegen_growth, False, False),
  "parameter_tail_order": (check_parameter_tail_order, False, False),
  "pair_barrier_is_order_one": (check_pair_barrier_is_order_one, False, False),
  "velocity_wall_barrier_has_control_authority": (check_velocity_wall_barrier_has_control_authority, True, True),
  "dt_plant_pieces": (check_dt_plant_pieces, False, False),
  "dt_plant_control_order": (check_dt_plant_control_order, False, False),
  "dt_filter_model_matches_numpy": (check_dt_filter_model_matches_numpy, False, True),
  "oracles_solve_alike": (check_oracles_solve_alike_per_step, True, True),
  "exact_hess": (check_exact_hess_matches_casadi_on_closed_loop_samples, True, True),
  "synthetic_hessian_inputs_at_range": (check_synthetic_hessian_inputs_at_range, False, True),
  "canonical_hessian_handoff": (check_canonical_hessian_handoff, False, True),
  "casadi_ipopt_compiled": (check_casadi_ipopt_is_compiled, True, True),
  "sqp_matches_ipopt": (check_sqp_matches_ipopt_per_step, True, False),
  "sqp_oracles_agree": (check_sqp_oracles_agree, True, True),
}


def run_checks() -> Iterator[tuple[str, str]]:
  """Yield ``(name, outcome)`` for each gate; ``outcome`` is "ok", "skipped: ..." or raises."""
  from scaly.opt.external.paths import solver_loadable

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
