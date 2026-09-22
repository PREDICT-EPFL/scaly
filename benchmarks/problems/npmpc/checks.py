"""Formulation correctness gates for the neural-process MPC problem.

These are properties of *this benchmark problem* — the vendored checkpoint and reference episode,
the parameter tail's order, the pinned latent code and physical constants, the terminal Riccati
weight, the analytic plant, the transcribed NLP's constraint rows, and the agreement of this
formulation with the reference implementation that produced the episode. They run before any timing
is recorded, via ``benchmarks/run.py smoke``.

They deliberately do **not** live in ``tests/``: per `AGENTS.md`, the pytest suite covers Scaly's
core and must not depend on a benchmark problem. The IR behaviour this problem leans on — a dense
matmul body used through VMAP over a horizon, differentiated to second order — has a self-contained
reproduction in ``tests/integration/test_vmap_mlp.py``, so retiring this problem cannot drop the
compiler coverage.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import replace

import numpy as np

import scaly as sc
from scaly.solvers.paths import solver_loadable, solver_paths
from benchmarks.harness import problem_stats, solve_problem
from benchmarks.problems.npmpc import (
  DT,
  HORIZON,
  LATENT_SYS3,
  NLATENT,
  NU,
  NX,
  NY,
  PHI_LIMIT,
  SLACK_LIMIT,
  TORQUE_LIMIT,
  X0_BAND,
  CostWeights,
  Decoder,
  FurutaParams,
  decoder_mu_np,
  furuta_midpoint_np,
  furuta_ode_np,
  linearize,
  load_decoder_weights,
  load_reference_config,
  load_reference_episode,
  constraint_counts,
  n_dec,
  n_param,
  npmpc_bounds,
  npmpc_ineq_bounds,
  npmpc_constraint_exprs,
  npmpc_eq_function,
  npmpc_lag_function,
  npmpc_nlp,
  pack_nlp_params,
  pack_params,
  plant_step,
  riccati_residual,
  sample_inputs,
  stage_function,
  step_np,
  terminal_P,
)
from benchmarks.problems.npmpc.closed_loop import (
  EpisodeConfig,
  build_solver,
  horizon_eq_violation,
  initial_guess,
  run_episode,
  settling_step,
  upright_error,
)

# The parameter tail, spelled out independently of `Decoder.blocks` so a silent reordering there
# moves the residual this file checks (see `check_parameter_tail_order`).
TAIL_ORDER = ("x_scale_w", "x_scale_b", "y_inv_w", "y_inv_b", "w0", "w1", "w2", "bias", "latent")
TAIL_SIZES = (5, 5, 2, 2, 288, 1024, 64, 2, 4)

# Their released episode was produced by a float32 checkpoint through CasADi, and their open-loop
# rollouts are recorded from a torch float32 model, so agreement is expected at float32 precision
# rather than at ours. Measured on the vendored episode: one-step decoder residual 1.7e-5 against
# velocity changes of magnitude up to 11, and the analytic-plant rollout 1.3e-4 over twelve steps.
DECODER_STEP_TOL = 5e-5
PLANT_ROLLOUT_TOL = 5e-4
# Measured: their recorded solutions violate our dynamics equalities by at most 5.4e-6, which is
# their solver tolerance plus the same float32 noise.
REFERENCE_FEASIBILITY_TOL = 5e-5
# Applied-control agreement against their episode, after their own reported settling step (14 at a
# 10-degree tolerance). Measured over steps 14..99: max 9.4e-4 on a torque bounded by 0.05. Before
# settling the swing-up admits more than one local minimum and the two runs pick different ones --
# measured max 1.6e-2 at step 8 -- so that window is gated on the objective instead. The headroom
# here is deliberately thin (2.7x rather than the ~100x the other problems use) because the gate
# stops discriminating anything at 5e-3: zeroing the terminal weight moves the applied torque by
# only 3.2e-3.
REFERENCE_CONTROL_TOL = 2.5e-3
REFERENCE_SETTLING_STEP = 14


def check_dims_and_checkpoint() -> None:
  """The declared dimensions, the vendored checkpoint's shape, and the pinned latent code."""
  assert (NX, NU, NY, NLATENT) == (4, 1, 2, 4)
  assert n_dec(HORIZON) == 65, n_dec(HORIZON)
  decoder = Decoder()
  assert decoder.hidden == (32, 32) and decoder.n_pw == 1396, (decoder.hidden, decoder.n_pw)
  assert n_param(decoder) == 1427, n_param(decoder)
  assert decoder.weight_shapes == ((32, 9), (32, 32), (2, 32)), decoder.weight_shapes
  # `load_decoder_weights` raises if the packed size is wrong, so only finiteness is left to check
  assert np.all(np.isfinite(load_decoder_weights(decoder)))

  # The recovered latent code for system 3. Moving it moves every trajectory downstream of it, so it
  # is pinned rather than recomputed; `check_decoder_matches_reference_rollout` is what validates it.
  np.testing.assert_allclose(LATENT_SYS3, [-2.34141442, 0.16122490, -10.00758909, -1.65464341], rtol=0.0, atol=0.0)


def check_matches_reference_config() -> None:
  """Every number this formulation shares with the reference controller comes from their own config.

  The cost weights, the bounds, the slack limit, the sample time, the horizon length and system 3's
  geometry are read out of the vendored `reference_config.json` -- their `model/furuta_mpc.json`,
  copied verbatim -- rather than compared against a transcription of them. That matters because the
  cross-implementation gate below turns out *not* to be sensitive to the cost weights, so this is
  the check that actually holds them to the reference.
  """
  config = load_reference_config()
  assert config["method"] == "neural" and config["system"] == "furuta" and config["integrator"] == "midpoint"
  assert config["horizon_steps"] == HORIZON and config["dt"] == DT

  weights = CostWeights()
  assert list(weights.x) == config["cost"]["x"], (weights.x, config["cost"]["x"])
  assert list(weights.x_diff) == config["cost"]["x_diff"], (weights.x_diff, config["cost"]["x_diff"])
  assert list(weights.x_end) == config["cost"]["x_end"], (weights.x_end, config["cost"]["x_end"])
  assert [weights.u] == config["cost"]["u"], (weights.u, config["cost"]["u"])
  # their `compute_terminal_P` builds the Riccati `R` from the same `cost['u']` as the stage weight
  assert weights.u_end == weights.u, (weights.u_end, weights.u)
  # their `MPCBase.slack_weight`, the one cost number that lives in code rather than in the config
  assert weights.slack == 1000.0

  # the arm angle is the only bounded state, and the only one the slack softens
  bounds = config["hard_bound"]["x"]
  assert bounds == [[-np.inf, np.inf], [-PHI_LIMIT, PHI_LIMIT], [-np.inf, np.inf], [-np.inf, np.inf]], bounds
  assert config["hard_bound"]["u"] == [[-TORQUE_LIMIT, TORQUE_LIMIT]], config["hard_bound"]["u"]
  assert config["slack_bound"]["x"] == [[np.inf], [SLACK_LIMIT], [np.inf], [np.inf]], config["slack_bound"]["x"]

  plant = FurutaParams()
  assert [plant.l_p, plant.m_p, plant.l_r, plant.m_r] == config["p"][0], (plant, config["p"])
  assert plant.gravity == 9.81, plant.gravity
  np.testing.assert_allclose(EpisodeConfig().x_start, config["experiment_options"]["x0"], rtol=0.0, atol=1e-6)
  assert X0_BAND == 1e-3


def check_parameter_tail_order() -> None:
  """Pins the order of ``pw`` by rebuilding the decoder from a tail assembled without `Decoder.slice`.

  Every block gets its own distinguishable values, and the reference evaluation indexes the tail by
  the order written out in ``TAIL_ORDER`` rather than by asking the `Decoder` where things are. A
  permutation of `Decoder.blocks` therefore moves the residual instead of moving both sides
  together, which is the failure a self-consistent check would miss.
  """
  decoder = Decoder()
  assert tuple(name for name, _ in decoder.blocks) == TAIL_ORDER, decoder.blocks
  assert tuple(size for _, size in decoder.blocks) == TAIL_SIZES, decoder.blocks

  rng = np.random.default_rng(5)
  parts = {name: rng.normal(scale=0.3, size=size) + index for index, (name, size) in enumerate(zip(TAIL_ORDER, TAIL_SIZES, strict=True))}
  pw = np.concatenate([parts[name] for name in TAIL_ORDER])
  assert pw.size == decoder.n_pw

  x, u = np.array([0.4, -0.3, 1.7, -2.1]), np.array([0.02])
  h = np.concatenate([np.array([np.sin(x[0]), np.cos(x[0]), x[2], x[3], u[0]]) * parts["x_scale_w"] + parts["x_scale_b"], parts["latent"]])
  h = 1.0 / (1.0 + np.exp(-(parts["w0"].reshape(32, 9) @ h)))
  h = 1.0 / (1.0 + np.exp(-(parts["w1"].reshape(32, 32) @ h)))
  expected = (parts["w2"].reshape(2, 32) @ h + parts["bias"]) * parts["y_inv_w"] + parts["y_inv_b"]
  np.testing.assert_allclose(decoder_mu_np(decoder, pw, x, u), expected, rtol=0.0, atol=1e-13)

  xnext = np.array([0.1, 0.2, 0.3, 0.4])
  residual = np.asarray(stage_function(decoder)((x, xnext, u, pw, np.array(DT)))).reshape(-1)
  np.testing.assert_allclose(residual, x + np.concatenate([DT * (x[2:4] + expected / 2.0), expected]) - xnext, rtol=0.0, atol=1e-12)


def check_decoder_matches_reference_rollout() -> None:
  """The learned model reproduces the reference implementation's own open-loop rollout.

  This is what validates the recovered latent code and the extracted weights together: their
  `x_np` is a full neural-process rollout from their torch model, and our one-step map has to land
  on it from every recorded state. The Scaly stage function then has to land on our NumPy one.
  """
  decoder = Decoder()
  pw = pack_params(decoder, load_decoder_weights(decoder))
  episode = load_reference_episode()
  x_np, u = episode["x_np"], episode["u"]
  worst, where = 0.0, (0, 0)
  for step in range(x_np.shape[0]):
    for stage in range(u.shape[1]):
      error = float(np.max(np.abs(step_np(decoder, pw, x_np[step, stage], u[step, stage]) - x_np[step, stage + 1])))
      if error > worst:
        worst, where = error, (step, stage)
  assert worst <= DECODER_STEP_TOL, f"decoder one-step residual {worst:.3e} at step {where[0]}, stage {where[1]}"

  stage_fn = stage_function(decoder)
  rng = np.random.default_rng(2)
  for _ in range(20):
    x = np.array([rng.uniform(-np.pi, np.pi), rng.uniform(-2.0, 2.0), rng.normal(scale=4.0), rng.normal(scale=4.0)])
    u_sample = rng.uniform(-TORQUE_LIMIT, TORQUE_LIMIT, NU)
    xnext = step_np(decoder, pw, x, u_sample)
    np.testing.assert_allclose(np.asarray(stage_fn((x, xnext, u_sample, pw, np.array(DT)))).reshape(-1), np.zeros(NX), rtol=0.0, atol=1e-12)


def check_plant_matches_reference_oracle() -> None:
  """The analytic plant reproduces their analytic open-loop rollout, and its substepping has converged.

  Their `x_oracle` is the same recorded control sequence rolled through the true pendulum with one
  midpoint step per 20 ms interval, so it pins the rigid-body model and the state order. The plant
  the closed loop drives takes 80 substeps per interval instead, which is a different quantity: one
  midpoint step over a whole interval is off by up to 0.25 rad/s on this pendulum, while quadrupling
  the substep count moves the substepped answer by 3e-6 relative.
  """
  episode = load_reference_episode()
  x, u, x_oracle = episode["x"], episode["u"], episode["x_oracle"]
  worst, where = 0.0, 0
  for step in range(x.shape[0]):
    state = x[step, 0]
    for stage in range(u.shape[1]):
      state = furuta_midpoint_np(state, u[step, stage], DT)
      error = float(np.max(np.abs(state - x_oracle[step, stage + 1])))
      if error > worst:
        worst, where = error, step
  assert worst <= PLANT_ROLLOUT_TOL, f"analytic rollout differs by {worst:.3e} at step {where}"

  start = np.array([np.pi, 0.0, 0.0, 0.0])
  torque = np.array([TORQUE_LIMIT])
  coarse, fine = plant_step(start, torque), plant_step(start, torque, substeps=4 * 80)
  np.testing.assert_allclose(coarse, fine, rtol=1e-5, atol=1e-9)
  assert float(np.max(np.abs(coarse - furuta_midpoint_np(start, torque, DT)))) > 0.1, "substepping must matter at this sample time"

  # Both poses of the pendulum are equilibria of the unforced plant, and only one is stable. The
  # inertia matrix is of order 1e-4 here, so inverting it multiplies the 1.2e-16 that `sin(pi)`
  # actually evaluates to by about 1e8 -- hence the tolerance rather than exact zeros.
  for theta in (0.0, np.pi):
    at_rest = np.array([theta, 0.3, 0.0, 0.0])
    np.testing.assert_allclose(furuta_ode_np(at_rest, np.zeros(NU)), np.zeros(NX), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(plant_step(at_rest, np.zeros(NU)), at_rest, rtol=0.0, atol=1e-12)
  # Gravity accelerates the pendulum towards hanging from anywhere in between, which is what makes
  # `theta = 0` the unstable equilibrium the controller has to reach and `theta = pi` the one it
  # starts from. A sign error in the gravity term would reverse both.
  for theta in (0.01, 0.5 * np.pi, np.pi - 0.01):
    assert furuta_ode_np(np.array([theta, 0.0, 0.0, 0.0]), np.zeros(NU))[2] > 0.0, f"gravity pushes the wrong way at theta={theta:.3f}"


def check_terminal_riccati_weight() -> None:
  """The terminal weight solves the Riccati equation for the linearization it claims to come from.

  `linearize` reads `A` and `B` off `sc.factory.Jac` on the stage residual rather than from an autograd
  pass, so the finite-difference comparison is what keeps that shortcut honest, and the residual is
  what stops a pinned `P` from drifting away from the model it was solved for.
  """
  decoder = Decoder()
  pw = pack_params(decoder, load_decoder_weights(decoder))
  A, B = linearize(decoder, pw)
  step = 1e-6
  for column in range(NX):
    delta = np.zeros(NX)
    delta[column] = step
    expected = (step_np(decoder, pw, delta, np.zeros(NU)) - step_np(decoder, pw, -delta, np.zeros(NU))) / (2.0 * step)
    np.testing.assert_allclose(A[:, column], expected, rtol=1e-6, atol=1e-9)
  expected_b = (step_np(decoder, pw, np.zeros(NX), np.array([step])) - step_np(decoder, pw, np.zeros(NX), np.array([-step]))) / (2.0 * step)
  np.testing.assert_allclose(B[:, 0], expected_b, rtol=1e-6, atol=1e-9)

  P = terminal_P(decoder, pw)
  assert P.shape == (NX, NX)
  np.testing.assert_allclose(P, P.T, rtol=0.0, atol=1e-12)
  assert np.all(np.linalg.eigvalsh(P) > 0.0), np.linalg.eigvalsh(P)
  residual = riccati_residual(P, A, B)
  assert residual < 1e-8, f"Riccati residual {residual:.3e}"


def check_constraint_rows_and_bounds() -> None:
  """The transcribed inequalities and box bounds are the ones section 3.2 specifies.

  The rows are compared against a hand-written NumPy evaluation, so a mis-strided arm-angle gather
  or a slack on the wrong side of a row shows up as a value difference rather than as a shape one.
  """
  horizon = 3
  lower, upper = npmpc_ineq_bounds(horizon)

  @sc.function(
    sc.G(sc.L("z", n_dec(horizon)), sc.L("xstart", sc.TensorType((NX,), diff=False))),
    sc.L("g", ...),
    name="npmpc_ineq_check",
  )
  def constraints(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    z, xstart = inputs
    rows, _, _ = npmpc_constraint_exprs(z, xstart, horizon)
    return rows

  rng = np.random.default_rng(9)
  z = rng.normal(size=n_dec(horizon))
  xstart = rng.normal(size=NX)
  states = z[: NX * (horizon + 1)].reshape(horizon + 1, NX)
  slack = z[-1]
  expected = np.concatenate([states[0] - xstart, states[:, 1] + slack, states[:, 1] - slack])
  np.testing.assert_allclose(np.asarray(constraints((z, xstart))).reshape(-1), expected, rtol=0.0, atol=1e-14)

  assert constraints.outputs[0].size == NX + 2 * (horizon + 1)
  np.testing.assert_array_equal(lower[:NX], -X0_BAND)
  np.testing.assert_array_equal(upper[:NX], X0_BAND)
  np.testing.assert_array_equal(lower[NX : NX + horizon + 1], -PHI_LIMIT)
  # the sign of the open side matters: `+inf` where `-inf` belongs makes the row infeasible instead
  # of unbounded, so `isinf` alone would not catch a flip
  np.testing.assert_array_equal(upper[NX : NX + horizon + 1], np.inf)
  np.testing.assert_array_equal(lower[NX + horizon + 1 :], -np.inf)
  np.testing.assert_array_equal(upper[NX + horizon + 1 :], PHI_LIMIT)

  box_lower, box_upper = npmpc_bounds(horizon)
  offset = NX * (horizon + 1)
  assert np.all(np.isinf(box_lower[:offset])) and np.all(np.isinf(box_upper[:offset]))
  np.testing.assert_array_equal(box_lower[offset : offset + NU * horizon], -TORQUE_LIMIT)
  np.testing.assert_array_equal(box_upper[offset : offset + NU * horizon], TORQUE_LIMIT)
  assert (box_lower[-1], box_upper[-1]) == (0.0, SLACK_LIMIT)


def _runtime_parameter_cases() -> tuple[Decoder, np.ndarray, tuple[np.ndarray, ...]]:
  decoder = Decoder()
  z, pw = sample_inputs(2, decoder)
  P = np.diag(CostWeights().x_end)
  base = pack_nlp_params(decoder, z[:NX], pw, P)
  changed_dt = pack_nlp_params(decoder, z[:NX], pw, P, dt=1.5 * DT)
  changed_cost = pack_nlp_params(decoder, z[:NX], pw, P, weights=replace(CostWeights(), u=10.0 * CostWeights().u))
  changed_P = pack_nlp_params(decoder, z[:NX], pw, 2.0 * P)
  return decoder, z, (base, changed_dt, changed_cost, changed_P)


def check_runtime_tuning_parameters() -> None:
  """One compiled Function accepts new numerical tuning data without reconstruction."""
  decoder, z, parameters = _runtime_parameter_cases()
  base = parameters[0]
  np.testing.assert_array_equal(base[:NX], z[:NX])
  assert base[NX + decoder.n_pw] == DT
  np.testing.assert_array_equal(base[NX + decoder.n_pw + 1 : NX + decoder.n_pw + 11], [*CostWeights().x, *CostWeights().x_diff, 1.0, 1000.0])
  np.testing.assert_array_equal(base[-NX * NX :].reshape(NX, NX), np.diag(CostWeights().x_end))
  lag = npmpc_lag_function(2, decoder)
  (base_cost, base_eq), (dt_cost, dt_eq), (weight_cost, weight_eq), (P_cost, P_eq) = (lag((z, p)) for p in parameters)
  assert not np.allclose(dt_eq, base_eq)
  np.testing.assert_allclose(dt_cost, base_cost, rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(weight_eq, base_eq, rtol=0.0, atol=1e-12)
  np.testing.assert_allclose(P_eq, base_eq, rtol=0.0, atol=1e-12)
  assert not np.allclose(weight_cost, base_cost, rtol=0.0, atol=1e-8)
  assert not np.allclose(P_cost, base_cost, rtol=0.0, atol=1e-8)


def check_casadi_runtime_parameters_match() -> None:
  """CasADi reads the same runtime tuning fields as Scaly for every changed field."""
  import casadi as ca

  from benchmarks.problems.npmpc import _ca_npmpc_joint_parameter_pieces

  decoder, z, parameters = _runtime_parameter_cases()
  scaly = npmpc_lag_function(2, decoder)
  pieces = _ca_npmpc_joint_parameter_pieces(2, decoder, ca.MX)
  casadi = ca.Function("npmpc_runtime_parameters", [pieces["z"], pieces["p"]], [pieces["f"], pieces["h_eq"]])
  for p in parameters:
    scaly_cost, scaly_eq = scaly((z, p))
    casadi_cost, casadi_eq = casadi(z, p)
    np.testing.assert_allclose(np.asarray(casadi_cost), np.asarray(scaly_cost), rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(casadi_eq).reshape(-1), np.asarray(scaly_eq).reshape(-1), rtol=0.0, atol=1e-12)


def check_nlp_uses_an_exact_hessian() -> None:
  """The IPOPT column really *evaluates* the generated exact Lagrangian Hessian.

  The typed solver builder attaches a Hessian oracle unconditionally, so `descriptor.hess is not None` is true of
  every NLP and on its own proves nothing. What separates an exact-Hessian column from a
  quasi-Newton one is whether IPOPT calls the oracle: `n_eval_h` counts once per iteration here and
  drops to exactly zero the moment `hessian_approximation` is set to `limited-memory`. The
  controller is the one `build_solver` returns, so this is the column that gets timed rather than a
  locally rebuilt lookalike.
  """
  config = replace(EpisodeConfig(), horizon=4, steps=1)
  pw = pack_params(config.decoder, load_decoder_weights(config.decoder))
  P = terminal_P(config.decoder, pw, config.weights, config.dt)
  controller = build_solver(config, "ipopt", "scaly")
  assert controller.function.descriptor.hess is not None
  requested = dict(controller.function.descriptor.options).get("hessian_approximation")
  assert requested is None, f"the IPOPT column asks for hessian_approximation={requested!r}"

  n_eq, n_ineq = constraint_counts(config.horizon)
  start = np.array(config.x_start)
  solve_problem(
    controller,
    initial_guess(start, config),
    np.zeros(n_eq),
    np.zeros(n_ineq),
    np.zeros(n_dec(config.horizon)),
    pack_nlp_params(config.decoder, start, pw, P, weights=config.weights, dt=config.dt),
  )
  stats = problem_stats(controller)
  assert stats is not None and stats.iter > 0, "the probe solve did not iterate"
  assert stats.n_eval_h > 0, "IPOPT never evaluated the Lagrangian Hessian, so this column is quasi-Newton"


def check_casadi_ipopt_is_compiled() -> None:
  """The timed CasADi column is generated C linked to Scaly's IPOPT."""
  from pathlib import Path

  config = replace(EpisodeConfig.smoke(), steps=1)
  pw = pack_params(config.decoder, load_decoder_weights(config.decoder))
  P = terminal_P(config.decoder, pw, config.weights, config.dt)
  controller = build_solver(config, "ipopt", "casadi")
  expected = solver_paths(required=True).loads["ipopt"]
  assert controller.compiled and not controller.expand and expected is not None
  assert controller.resolved_ipopt_library.read_bytes() == Path(expected).read_bytes()
  n_eq, n_ineq = constraint_counts(config.horizon)
  start = np.array(config.x_start)
  solve_problem(
    controller,
    initial_guess(start, config),
    np.zeros(n_eq),
    np.zeros(n_ineq),
    np.zeros(n_dec(config.horizon)),
    pack_nlp_params(config.decoder, start, pw, P, weights=config.weights, dt=config.dt),
  )
  stats = problem_stats(controller)
  assert stats is not None and stats.n_eval_h > 0


def check_episode_artifacts() -> None:
  """A short episode produces the declared shapes, stays finite, and respects the transcription's bounds."""
  config = EpisodeConfig.smoke()
  result = run_episode(config)

  assert result.states.shape == (config.steps + 1, NX), result.states.shape
  assert result.controls.shape == (config.steps, NU), result.controls.shape
  assert result.predictions.shape == (config.steps, config.horizon + 1, NX), result.predictions.shape
  assert result.plans.shape == (config.steps, config.horizon, NU), result.plans.shape
  assert result.slacks.shape == (config.steps,) and len(result.telemetry) == config.steps
  assert result.terminal_weight.shape == (NX, NX)
  for array in (result.states, result.controls, result.predictions, result.plans, result.slacks):
    assert np.all(np.isfinite(array))
  for step, stats in enumerate(result.telemetry):
    assert stats.status.value <= 1, f"step {step}: {stats.status.name}"
    assert stats.t_total >= 0.0 and stats.t_fe >= 0.0 and stats.t_glue >= 0.0
  # the first horizon node is pinned to the measurement, inside the band the reference uses
  for step in range(config.steps):
    assert np.max(np.abs(result.predictions[step, 0] - result.states[step])) <= X0_BAND + 1e-9
    np.testing.assert_allclose(result.plans[step, 0], result.controls[step], rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(result.oracle_inputs[step]["p"][:NX], result.states[step], rtol=0.0, atol=0.0)


def check_episode_swings_up() -> None:
  """The canonical episode swings the pendulum up and holds it, with the slack and the bounds inactive.

  This is the closed-loop claim the problem exists to make, so it is gated rather than reported:
  from hanging, the learned model has to bring the real pendulum upright and keep it there. Measured
  on the canonical episode: settles at step 10 (their released episode settles at 14, because their
  plant keeps moving during each solve), final offset from upright 0.01 degrees, arm angle never
  past 1.4 rad of its 2.0 bound, so the slack stays at zero throughout.
  """
  result = run_episode(EpisodeConfig())
  settled = settling_step(result.states)
  assert settled is not None and settled <= 20, f"pendulum settles at {settled}"
  # `settling_step` is defined as the index after the last excursion, so "stays within tolerance
  # afterwards" is implied by it and is not asserted again; what is worth asserting is that the
  # pendulum ends up far tighter than the 10-degree settling tolerance rather than just inside it
  final = float(np.rad2deg(upright_error(result.states[-1])))
  assert final < 1.0, f"final offset from upright {final:.3f} degrees"
  assert float(np.max(np.abs(result.states[:, 1]))) < PHI_LIMIT, "the arm angle bound became active"
  assert float(np.max(result.slacks)) < 1e-6, f"slack rose to {result.slacks.max():.3e} with the bound inactive"
  # `plans` carries the solution's own controls, before the runner clips them, so a widened or
  # missing torque bound shows up here rather than being hidden by the clip
  assert float(np.max(np.abs(result.plans))) <= TORQUE_LIMIT + 1e-6, f"torque reached {np.max(np.abs(result.plans)):.6f}"
  # a settled pendulum needs almost no torque, which is what distinguishes holding from thrashing
  assert float(np.max(np.abs(result.controls[settled + 10 :]))) < 0.1 * TORQUE_LIMIT


def check_matches_reference_episode() -> None:
  """Solve our NLP from every step of their released episode and compare against what they applied.

  This is the cross-implementation gate: it validates the model, the cost, the weights, the latent
  code and the terminal matrix against an independent implementation, without depending on plant
  reproducibility, because every solve starts from a state and warm start their run recorded. Three
  things are checked, in increasing strength of claim:

  * their recorded solution satisfies *our* dynamics equalities (measured max 5.4e-6);
  * our solve never lands on a worse objective than theirs, evaluated with our own cost -- measured
    strictly better at every step, by 0.29 to 10.7 on objectives of 1.5 to 170, which says their
    real-time-capped solves stop short rather than that the two objectives disagree;
  * after their own reported settling step, the applied torque agrees to `REFERENCE_CONTROL_TOL`.

  The swing-up window is excluded from the last one on purpose. There the problem admits more than
  one local minimum and the two runs pick different ones: at step 8 their horizon and ours differ by
  2.7 rad/s in pendulum velocity while both remain feasible and ours costs 10.7 less. Loosening the
  tolerance until that passed would have made the gate unable to fail, so the window is reported
  instead, and the first diverging step is named on failure.

  What this gate does **not** establish is the cost weights, and that is worth stating rather than
  implying. Their recorded solutions are feasible but well short of stationary for the problem
  either implementation writes down -- restarting our IPOPT from their step-50 solution moves the
  horizon by 0.46 and drops the objective from 2.11 to 1.52, almost all of it in the tail. The
  applied torque is insensitive to that tail, which is why it agrees so well; it is equally
  insensitive to the arm-angle stage weight, which can be moved from 1.5 to 6.0 without this gate
  noticing. `check_matches_reference_config` is what pins the weights, by reading them out of their
  own configuration file.
  """
  decoder = Decoder()
  pw = pack_params(decoder, load_decoder_weights(decoder))
  P = terminal_P(decoder, pw)
  episode = load_reference_episode()
  horizon = int(episode["horizon_steps"])
  assert horizon == HORIZON and float(episode["dt"]) == DT, (horizon, float(episode["dt"]))
  offset = NX * (horizon + 1)

  eq = npmpc_eq_function(horizon, decoder)
  lag = npmpc_lag_function(horizon, decoder)
  controller = npmpc_nlp(
    horizon,
    decoder,
    options={"print_level": 0, "sb": "yes", "tol": 1e-6, "max_iter": 50, "warm_start_init_point": "yes"},
  )
  n_eq, n_ineq = constraint_counts(horizon)
  zeros = (np.zeros(n_eq), np.zeros(n_ineq), np.zeros(n_dec(horizon)))
  for step in range(1, int(episode["steps"])):
    theirs = np.concatenate([episode["x"][step].reshape(-1), episode["u"][step].reshape(-1), np.zeros(1)])
    xstart = step_np(decoder, pw, episode["x0"][step], episode["u0"][step])
    p = pack_nlp_params(decoder, xstart, pw, P)
    feasibility = float(np.max(np.abs(np.asarray(eq((theirs, p))))))
    assert feasibility <= REFERENCE_FEASIBILITY_TOL, f"their step {step} violates our dynamics by {feasibility:.3e}"

    # their warm start, rebuilt: the previous solution's controls shifted, the measurement advanced
    # one step through the learned model, and the shifted controls rolled out from there
    previous = episode["u"][step - 1]
    shifted = np.concatenate([previous[1:], previous[-1:]])
    states = [xstart]
    for stage in range(horizon):
      states.append(step_np(decoder, pw, states[-1], shifted[stage]))
    guess = np.concatenate([np.asarray(states).reshape(-1), shifted.reshape(-1), np.zeros(1)])

    out = solve_problem(controller, guess, *zeros, p)
    stats = problem_stats(controller)
    status = None if stats is None else stats.to_solver_status()
    assert status is not None and stats is not None and status.ok, f"step {step}: {None if stats is None else stats.status.name}"
    ours = np.asarray(out["x"], dtype=np.float64).reshape(-1)
    gap = float(out["f"]) - float(np.asarray(lag((theirs, p))[0]))
    assert gap <= 1e-6 * (1.0 + abs(float(out["f"]))), f"step {step}: our objective is {gap:.3e} worse than theirs"
    if step < REFERENCE_SETTLING_STEP:
      continue
    difference = float(np.max(np.abs(ours[offset : offset + NU] - episode["u"][step, 0])))
    assert difference <= REFERENCE_CONTROL_TOL, (
      f"applied torque first diverges from the reference episode at step {step}: |du|={difference:.3e} "
      f"(tolerance {REFERENCE_CONTROL_TOL:.1e}); ours={ours[offset]:+.6f}, theirs={episode['u'][step, 0, 0]:+.6f}, "
      f"objective gap {gap:.3e}, their feasibility {feasibility:.3e}"
    )


# The solver-comparison gates run the canonical horizon for eight steps: the cold start plus the
# fast part of the swing-up, which is where the two solvers are most likely to part company. Every
# worst-case difference below is the same at eight steps as over the full hundred.
COMPARISON_STEPS = 8
# Measured over those steps. The two oracle providers solve a deliberately identical problem, so
# under IPOPT they agree to round-off and under the SQP to the tolerance of its own linear algebra.
# Over the full canonical episode: 4.5e-15 in applied torque and 8.8e-13 in state for the IPOPT
# pair, 3.5e-11 and 5.1e-9 for the SQP pair.
ORACLE_IPOPT_TOL = 1e-12
ORACLE_SQP_TOL = 1e-10
# SQP against IPOPT: different algorithms, so small differences are allowed and large ones are not.
# Measured healthy-step worst cases: applied torque 3.6e-7, plan 1.4e-4, objective 6.5e-4 on
# objectives up to 170, equality violation 4e-10.
SQP_CONTROL_TOL = 3e-5
SQP_PLAN_TOL = 1e-2
SQP_OBJECTIVE_RTOL = 1e-4
SQP_VIOLATION_TOL = 1e-6


def _comparison_config() -> EpisodeConfig:
  return replace(EpisodeConfig(), steps=COMPARISON_STEPS)


def check_oracles_agree() -> None:
  """Scaly and CasADi oracles drive the same IPOPT to the same episode.

  The two columns solve a deliberately identical problem -- same decision-variable and parameter
  layout, same cost, same rows in the same order, same IPOPT with the same options -- so the only
  difference is who differentiates and evaluates. Measured: identical iteration counts at every step
  and trajectories agreeing to 8.8e-13 in state over the full episode.

  This gate pays the CasADi column's generated-C build. The column is only a baseline if it is
  compiled, and a gate that skipped the build would check a configuration that no timing uses.
  """
  config = _comparison_config()
  scaly = run_episode(config, solver="ipopt", oracle="scaly")
  casadi = run_episode(config, solver="ipopt", oracle="casadi")
  assert [stats.iter for stats in scaly.telemetry] == [stats.iter for stats in casadi.telemetry], "iteration counts diverge"
  np.testing.assert_allclose(scaly.controls, casadi.controls, rtol=0.0, atol=ORACLE_IPOPT_TOL)
  np.testing.assert_allclose(scaly.states, casadi.states, rtol=0.0, atol=1e2 * ORACLE_IPOPT_TOL)
  np.testing.assert_allclose(scaly.predictions, casadi.predictions, rtol=0.0, atol=1e2 * ORACLE_IPOPT_TOL)


def check_sqp_oracles_agree() -> None:
  """The same SQP produces the same episode from Scaly and CasADi generated-C oracles.

  Both columns run `scaly-sqp` with its PIQP subsolver at identical settings, and both feed it
  code-generated, compiled C. Both IPOPT columns are compiled too, so all four columns are
  compiled-code comparisons; what differs between the two *pairs* is the mix of
  oracle calls each optimizer makes, which is why their function-evaluation ratios differ.
  """
  config = _comparison_config()
  scaly = run_episode(config, solver="sqp", oracle="scaly")
  casadi = run_episode(config, solver="sqp", oracle="casadi")
  assert [stats.iter for stats in scaly.telemetry] == [stats.iter for stats in casadi.telemetry], "iteration counts diverge"
  np.testing.assert_allclose(scaly.controls, casadi.controls, rtol=0.0, atol=ORACLE_SQP_TOL)
  np.testing.assert_allclose(scaly.states, casadi.states, rtol=0.0, atol=1e2 * ORACLE_SQP_TOL)
  for run in (scaly, casadi):
    for step, stats in enumerate(run.telemetry):
      assert stats.status.value <= 1 and stats.t_qp > 0.0, f"step {step}: {stats.status.name}, qp time {stats.t_qp}"
      np.testing.assert_allclose(stats.t_total, stats.t_fe + stats.t_solver + stats.t_qp + stats.t_globalization + stats.t_glue, rtol=1e-10)


def check_sqp_matches_ipopt() -> None:
  """SQP and IPOPT drive the same episode to the same per-step solutions.

  Different algorithms are allowed small differences, not major ones, and the tolerances come from
  measured healthy-step differences with roughly a hundredfold margin. On divergence the message
  carries the first diverging step with both solvers' status, objective and equality violation.
  """
  config = _comparison_config()
  ipopt = run_episode(config, solver="ipopt", oracle="scaly")
  sqp = run_episode(config, solver="sqp", oracle="scaly")
  assert len(ipopt.controls) == len(sqp.controls) == config.steps
  for step in range(config.steps):
    stats_i, stats_s = ipopt.telemetry[step], sqp.telemetry[step]
    assert stats_i.status.value <= 1, f"step {step}: ipopt reported {stats_i.status.name}"
    assert stats_s.status.value <= 1, f"step {step}: sqp reported {stats_s.status.name}"
    assert stats_s.iter == 0 or stats_s.alpha > 0.0, f"step {step}: sqp accepted no step (iter={stats_s.iter})"
    du = float(np.max(np.abs(ipopt.controls[step] - sqp.controls[step])))
    dplan = float(np.max(np.abs(ipopt.predictions[step] - sqp.predictions[step])))
    violations = tuple(horizon_eq_violation(run, step) for run in (ipopt, sqp))
    assert (
      du <= SQP_CONTROL_TOL
      and dplan <= SQP_PLAN_TOL
      and abs(stats_i.obj - stats_s.obj) <= SQP_OBJECTIVE_RTOL * (1.0 + abs(stats_i.obj))
      and max(violations) <= SQP_VIOLATION_TOL
    ), (
      f"SQP first diverges from IPOPT at step {step}: |du|={du:.3e} |dplan|={dplan:.3e} "
      f"|dobj|={abs(stats_i.obj - stats_s.obj):.3e}; "
      f"ipopt: status={stats_i.status.name} obj={stats_i.obj:.6e} eq_violation={violations[0]:.3e}; "
      f"sqp: status={stats_s.status.name} obj={stats_s.obj:.6e} eq_violation={violations[1]:.3e}"
    )


def check_recorded_scene() -> None:
  """The runner feeds the 3D scene builders: the static rig once, and every step's pose, plan and torque.

  The builders are generic geometry covered by ``tests/viz/test_recording.py``; what belongs here is
  that the npmpc runner calls them with this problem's own data -- the arm angle in the slot the
  transcription keeps it in, one plan per applied control covering every horizon node, and the
  physical lengths rather than the drawing defaults.
  """
  import tempfile
  from pathlib import Path

  from benchmarks.harness import recording
  from benchmarks.harness.closed_loop import run_npmpc

  record_state, record_plan = recording.NpmpcRecorder.record_furuta, recording.NpmpcRecorder.record_furuta_plan
  record_static = recording.NpmpcRecorder.record_furuta_static
  poses: list[tuple[float, float, float]] = []
  plans: list[int] = []
  statics: list[tuple[float, float, float]] = []

  def spy_state(self, state, _seen=poses):  # type: ignore[no-untyped-def]
    logged = record_state(self, state)
    _seen.append((logged.theta, logged.phi, logged.torque))
    return logged

  def spy_plan(self, plan, _seen=plans):  # type: ignore[no-untyped-def]
    logged = record_plan(self, plan)
    _seen.append(len(logged.theta))
    return logged

  def spy_static(self, _seen=statics):  # type: ignore[no-untyped-def]
    _seen.append((self.shape.arm_length, self.shape.pendulum_length, self.shape.arm_limit))
    return record_static(self)

  recording.NpmpcRecorder.record_furuta, recording.NpmpcRecorder.record_furuta_plan = spy_state, spy_plan
  recording.NpmpcRecorder.record_furuta_static = spy_static
  try:
    with tempfile.TemporaryDirectory() as directory:
      output = run_npmpc(smoke=True, out_dir=Path(directory), cli_args=[])
      assert (output / "episode.mcap").stat().st_size > 0, "empty MCAP"
  finally:
    recording.NpmpcRecorder.record_furuta, recording.NpmpcRecorder.record_furuta_plan = record_state, record_plan
    recording.NpmpcRecorder.record_furuta_static = record_static

  config = EpisodeConfig.smoke()
  plant = config.plant
  assert statics == [(plant.l_r, plant.l_p, PHI_LIMIT)], statics
  assert len(poses) == config.steps + 1, len(poses)
  assert poses[-1][2] == 0.0, "the final pose has no torque to draw"
  # one plan per applied control, each covering every node of the horizon it solved
  assert plans == [config.horizon + 1] * config.steps, plans
  # the two angles reach the scene from the slots the transcription keeps them in; the episode is
  # deterministic, so re-running it is enough to say which number should have gone where
  episode = run_episode(config)
  np.testing.assert_allclose([theta for theta, _, _ in poses], episode.states[:, 0], rtol=0.0, atol=0.0)
  np.testing.assert_allclose([phi for _, phi, _ in poses], episode.states[:, 1], rtol=0.0, atol=0.0)
  np.testing.assert_allclose([torque for _, _, torque in poses[:-1]], episode.controls[:, 0], rtol=0.0, atol=0.0)


def check_initial_guess_reaches_upright() -> None:
  """The cold start interpolates towards the *forward* upright pose, which picks the swing direction.

  Interpolating towards `0` instead of `2 pi` is a one-character change that reverses the swing-up,
  so the direction is worth pinning: the guess has to end at the upright pose the reference
  implementation aims at, with no torque and no slack.
  """
  config = EpisodeConfig()
  guess = initial_guess(np.array([np.pi, 0.0, 0.0, 0.0]), config)
  assert guess.shape == (n_dec(config.horizon),)
  states = guess[: NX * (config.horizon + 1)].reshape(config.horizon + 1, NX)
  np.testing.assert_allclose(states[0], [np.pi, 0.0, 0.0, 0.0], rtol=0.0, atol=0.0)
  np.testing.assert_allclose(states[-1], [2.0 * np.pi, 0.0, 0.0, 0.0], rtol=0.0, atol=1e-12)
  assert np.all(np.diff(states[:, 0]) > 0.0), "the cold start must rotate forward, not back"
  np.testing.assert_allclose(guess[NX * (config.horizon + 1) :], 0.0, rtol=0.0, atol=0.0)


# name -> (check, requires an IPOPT-backed solve, requires CasADi)
CHECKS: dict[str, tuple[Callable[[], None], bool, bool]] = {
  "dims_and_checkpoint": (check_dims_and_checkpoint, False, False),
  "reference_config": (check_matches_reference_config, False, False),
  "parameter_tail_order": (check_parameter_tail_order, False, False),
  "decoder_rollout": (check_decoder_matches_reference_rollout, False, False),
  "plant_rollout": (check_plant_matches_reference_oracle, False, False),
  "terminal_riccati": (check_terminal_riccati_weight, False, False),
  "constraint_rows": (check_constraint_rows_and_bounds, False, False),
  "runtime_tuning_parameters": (check_runtime_tuning_parameters, False, False),
  "casadi_runtime_parameters": (check_casadi_runtime_parameters_match, False, True),
  "initial_guess": (check_initial_guess_reaches_upright, False, False),
  "exact_hessian": (check_nlp_uses_an_exact_hessian, True, False),
  "casadi_ipopt_compiled": (check_casadi_ipopt_is_compiled, True, True),
  "episode_artifacts": (check_episode_artifacts, True, False),
  "episode_swings_up": (check_episode_swings_up, True, False),
  "reference_episode": (check_matches_reference_episode, True, False),
  "sqp_matches_ipopt": (check_sqp_matches_ipopt, True, False),
  "oracles_agree": (check_oracles_agree, True, True),
  "sqp_oracles_agree": (check_sqp_oracles_agree, True, True),
  "recorded_scene": (check_recorded_scene, True, False),
}


def run_checks() -> Iterator[tuple[str, str]]:
  """Yield ``(name, outcome)`` for each gate; ``outcome`` is "ok", "skipped: ..." or raises."""
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
