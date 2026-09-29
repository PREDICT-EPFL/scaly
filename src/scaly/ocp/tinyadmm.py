"""``TinyADMM``: the ADMM of TinyMPC for a linear OCP with box constraints, a Riccati-cached LQR as its primal step, the whole solve generated as loops."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, ClassVar

import numpy as np

from ..ad.derivatives import jacobian
from ..ad.sparsity import _jac_mask
from ..function.method import Status, Support
from ..function.model import ConcreteFunction, Function
from ..function.sugar import scan, vmap, while_loop
from ..function.tree import G, L, param_list
from ..ir.expr import Expr, cast, concat, greater, less, less_equal, logical_and, maximum, minimum, norm_inf, stack, where
from ..ir.types import Lowering
from ..passes.expr import simplify_cse_fixpoint
from .formulate import step_map
from .method import Info
from .problem import METHOD_API, DiscreteOCP, Quadratic


@dataclass(frozen=True)
class Cone:
  """A second-order cone on ``dim`` consecutive entries starting at ``start``: the last one is the
  axis, ``|s[:-1]| <= mu * s[-1]``."""

  start: int
  dim: int
  mu: float


@dataclass(frozen=True)
class Settings:
  """The ADMM's stopping rule and the box constraints it enforces: it stops when the primal residuals
  (``x - v``, ``u - z``) and ``rho`` times the slacks' change are below the tolerances, or after
  ``max_iter`` iterations. A disabled bound keeps its slack and dual but clips nothing."""

  abs_pri_tol: float = 1e-3
  abs_dua_tol: float = 1e-3
  max_iter: int = 1000
  en_state_bound: bool = True
  en_input_bound: bool = True


@dataclass(frozen=True, eq=False)
class LQRCache:
  """The primal step's LQR with the penalty folded in, ``u_i = -K_i x_i - d_i``: the gain, the cost to
  go at the last knot ``P``, ``(R + rho I + B'P_{i+1}B)^-1``, ``(A - B K_i)'`` and the affine-term
  corrections ``(A - B K_i)' P_{i+1} f`` and ``B' P_{i+1} f``. Each is one matrix (the time-invariant
  cache of TinyMPC) or one per stage ``i = 0 .. N - 2`` along a leading axis."""

  K: np.ndarray
  P: np.ndarray
  Quu_inv: np.ndarray
  AmBKt: np.ndarray
  APf: np.ndarray
  BPf: np.ndarray

  @property
  def varying(self) -> bool:
    return self.K.ndim == 3


def _weights(q: Any, rho: float) -> np.ndarray:
  q = np.asarray(q, dtype=np.float64)
  return np.diag(q + rho) if q.ndim == 1 else q + rho * np.eye(q.shape[0])


def tinympc_cache(A: Any, B: Any, Q: Any, R: Any, rho: float, f: Any = None) -> LQRCache:
  """TinyMPC's ``tiny_precompute_and_set_cache``: the infinite-horizon LQR of ``Q + rho I`` and
  ``R + rho I`` (``Q``, ``R`` diagonals or matrices), by the Riccati recursion from ``P = rho I``,
  stopped when ``K`` moves by less than 1e-5."""
  a, b = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
  nx, nu = b.shape
  q1, r1 = _weights(Q, rho), _weights(R, rho)
  k_prev, p_next = np.zeros((nu, nx)), rho * np.eye(nx)
  kinf, pinf = k_prev, p_next
  for _ in range(1000):
    kinf = np.linalg.inv(r1 + b.T @ p_next @ b) @ b.T @ p_next @ a
    pinf = q1 + a.T @ p_next @ (a - b @ kinf)
    if np.abs(kinf - k_prev).max() < 1e-5:
      break
    k_prev, p_next = kinf, pinf
  quu_inv = np.linalg.inv(r1 + b.T @ pinf @ b)
  ambkt = (a - b @ kinf).T
  fdyn = np.zeros(nx) if f is None else np.asarray(f, dtype=np.float64)
  return LQRCache(kinf, pinf, quu_inv, ambkt, ambkt @ pinf @ fdyn, b.T @ pinf @ fdyn)


def finite_cache(A: Any, B: Any, Q: Any, R: Any, QN: Any, rho: float, knots: int, f: Any = None) -> LQRCache:
  """The LQR of ``Q + rho I``, ``R + rho I`` over ``knots`` points ending in ``QN + rho I``: the
  backward Riccati recursion, one gain per stage, so the primal step is the exact minimizer for a
  horizon whose terminal cost is ``QN``."""
  a, b = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
  nx, nu = b.shape
  q1, r1 = _weights(Q, rho), _weights(R, rho)
  fdyn = np.zeros(nx) if f is None else np.asarray(f, dtype=np.float64)
  p_next = _weights(QN, rho)
  terminal = p_next
  stages = []
  for _ in range(knots - 1):
    quu_inv = np.linalg.inv(r1 + b.T @ p_next @ b)
    k = quu_inv @ b.T @ p_next @ a
    ambkt = (a - b @ k).T
    stages.append((k, quu_inv, ambkt, ambkt @ p_next @ fdyn, b.T @ p_next @ fdyn))
    p_next = q1 + a.T @ p_next @ (a - b @ k)
  k, quu_inv, ambkt, apf, bpf = (np.stack(m) for m in zip(*reversed(stages), strict=True))
  return LQRCache(k, terminal, quu_inv, ambkt, apf, bpf)


STATE_FIELDS = ("x", "u", "v", "vnew", "z", "znew", "g", "y", "gc", "yc")
"""The solver's warm-start state, in the order it is packed: trajectories, box slacks and their
previous values, box duals, and cone duals. Arrays are stage-major, ``(N, nx)`` and ``(N - 1, nu)``
for ``N`` knots."""


@dataclass(frozen=True)
class Layout:
  """Offsets of each field in the solver state and in the while-loop carry."""

  offsets: dict[str, tuple[int, int]]
  size: int

  def __getitem__(self, name: str) -> tuple[int, int]:
    return self.offsets[name]


@dataclass(frozen=True)
class Shape:
  """The sizes the ADMM's layout depends on: ``knots`` points of ``nx`` states and ``nu`` controls,
  and whether it carries the duals of state and of input cones."""

  knots: int
  nx: int
  nu: int
  state_cones: bool = False
  input_cones: bool = False

  def field_size(self, name: str) -> int:
    per_state = name in ("x", "v", "vnew", "g", "gc", "vcnew", "qref", "x_min", "x_max")
    return self.knots * self.nx if per_state else (self.knots - 1) * self.nu

  def fields(self) -> tuple[str, ...]:
    """The state fields: the cone duals only with cones."""
    return tuple(f for f in STATE_FIELDS if (f != "gc" or self.state_cones) and (f != "yc" or self.input_cones))

  def state_layout(self) -> Layout:
    offsets, at = {}, 0
    for name in self.fields():
      offsets[name] = (at, at + self.field_size(name))
      at += self.field_size(name)
    return Layout(offsets, at)


def _carry_layout(shape: Shape, settings: Settings, runtime_bounds: bool) -> Layout:
  """The carry is the state, the per-solve slacks of the cones, the loop invariants (linear cost of
  the references, bounds) and the convergence flag. A loop body reads only its carry."""
  names = list(shape.fields())
  names += ["vcnew"] if shape.state_cones else []
  names += ["zcnew"] if shape.input_cones else []
  names += ["qref", "rref", "pnref"]
  names += ["x_min", "x_max"] if settings.en_state_bound and runtime_bounds else []
  names += ["u_min", "u_max"] if settings.en_input_bound and runtime_bounds else []
  offsets, at = {}, 0
  for name in names:
    size = shape.nx if name == "pnref" else shape.field_size(name)
    offsets[name] = (at, at + size)
    at += size
  offsets["done"] = (at, at + 1)
  return Layout(offsets, at + 1)


def project_cones(w: Expr, rows: int, width: int, cones: tuple[Cone, ...]) -> Expr:
  """TinyMPC's ``project_soc`` on every stage (row of ``w`` reshaped to ``(rows, width)``) for every
  cone: zero below the polar cone, unchanged inside, else onto the boundary in the metric that scales
  the axis by ``mu``."""
  m = w.reshape((rows, width))
  cols = [m[:, j] for j in range(width)]
  for c in cones:
    axis = cols[c.start + c.dim - 1]
    rest = cols[c.start : c.start + c.dim - 1]
    u0 = axis * c.mu
    a = sum((r * r for r in rest[1:]), rest[0] * rest[0]).sqrt()
    below, inside = less_equal(a, -u0), less_equal(a, u0)
    scale = 0.5 * (1.0 + u0 / a)

    def pick(value: Expr, keep: Expr) -> Expr:
      return where(below, 0.0, where(inside, keep, value))

    for j, r in enumerate(rest):
      cols[c.start + j] = pick(scale * r, r)
    cols[c.start + c.dim - 1] = pick(scale * (a / c.mu), axis)
  return stack(cols).T.reshape((rows * width,)) if len(cols) > 1 else cols[0]


def _flat_table(matrices: np.ndarray) -> Expr:
  return Expr.const(np.concatenate([m.reshape(-1) for m in matrices]))


def _backward_step(name: str, B: np.ndarray, c: LQRCache, lowering: Lowering) -> Function[Any, Any, Any, Any]:
  """``d_i = Quu_i^-1 (B' p_{i+1} + r_i + BPf_i)``, ``p_i = q_i + (A - B K_i)' p_{i+1} - K_i' r_i + APf_i``."""
  nx, nu = B.shape
  pn, r, q = Expr.sym("p", nx), Expr.sym("r", nu), Expr.sym("q", nx)
  inputs, names = [pn, r, q], ["p", "r", "q"]
  if c.varying:
    stage = [Expr.sym(n, size) for n, size in (("quu", nu * nu), ("ambkt", nx * nx), ("k", nu * nx), ("apf", nx), ("bpf", nu))]
    quu, ambkt, k = stage[0].reshape((nu, nu)), stage[1].reshape((nx, nx)), stage[2].reshape((nu, nx))
    apf, bpf = stage[3], stage[4]
    inputs += stage
    names += ["quu", "ambkt", "k", "apf", "bpf"]
  else:
    quu, ambkt, k, apf, bpf = (Expr.const(m) for m in (c.Quu_inv, c.AmBKt, c.K, c.APf, c.BPf))
  d = quu @ (Expr.const(B.T) @ pn + r + bpf)
  p_prev = q + ambkt @ pn - k.T @ r + apf
  outs = [e.with_lowering(lowering) for e in (p_prev, d)]
  return Function.from_exprs(f"{name}_backward", inputs, outs, names, ["p_prev", "d"])


def _forward_step(name: str, A: np.ndarray, B: np.ndarray, f: np.ndarray | None, c: LQRCache, lowering: Lowering) -> Function[Any, Any, Any, Any]:
  """``u_i = -K_i x_i - d_i``, ``x_{i+1} = A x_i + B u_i + f``."""
  nx, nu = B.shape
  x, d = Expr.sym("x", nx), Expr.sym("d", nu)
  inputs, names = [x, d], ["x", "d"]
  if c.varying:
    k_flat = Expr.sym("k", nu * nx)
    k = k_flat.reshape((nu, nx))
    inputs.append(k_flat)
    names.append("k")
  else:
    k = Expr.const(c.K)
  u = -(k @ x) - d
  x_next = Expr.const(A) @ x + Expr.const(B) @ u
  if f is not None:
    x_next = x_next + Expr.const(f)
  outs = [e.with_lowering(lowering) for e in (x_next, u, x_next)]
  return Function.from_exprs(f"{name}_forward", inputs, outs, names, ["x_next", "u", "x_out"])


def admm_solver(
  A: Any,
  B: Any,
  cache: LQRCache,
  *,
  knots: int,
  rho: float,
  linear_cost: Callable[[Expr, Expr], tuple[Expr, Expr, Expr]],
  f: Any = None,
  settings: Settings = Settings(),
  state_cones: tuple[Cone, ...] = (),
  input_cones: tuple[Cone, ...] = (),
  fixed_bounds: dict[str, np.ndarray] | None = None,
  lowering: Lowering = "auto",
  name: str = "tinyadmm",
) -> Function[Any, Any, Any, Any]:
  """One call of TinyMPC's ``tiny_solve`` as a Function,

      (state, x0, xref, uref, [x_min, x_max], [u_min, u_max]) -> (state_next, iterations, solved, u0),

  for ``x_{i+1} = A x_i + B u_i + f`` over ``knots`` points: the ADMM loop is a ``while_loop``, and
  inside it the backward pass for the affine terms and the forward rollout are two ``scan``s over the
  horizon, the ``cache``'s gains and matrices constants of the generated code. ``linear_cost(xref,
  uref)`` gives the linear terms of the references in the penalized cost, ``(q, r, p_N)``, which
  fixes the problem's convention. ``state`` packs ``STATE_FIELDS`` stage-major and comes back as
  the next warm start. Bounds are inputs of the Function unless ``fixed_bounds`` makes them
  constants (``x_min``, ``x_max``, ``u_min``, ``u_max``, flat); a disabled one is neither.

  ``lowering`` is the lowering of the two stage steps: ``"auto"`` expands small ones into
  straight-line code with the matrices as literals, ``"block"`` keeps them as loops over constant
  tables, ``"scalar"`` forces the expansion."""
  a, b = np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64)
  fdyn = None if f is None else np.asarray(f, dtype=np.float64)
  n, (nx, nu) = knots, b.shape
  if n < 2:
    raise ValueError("the ADMM needs a horizon of at least two knot points")
  for cones, width in ((state_cones, nx), (input_cones, nu)):
    for cone in cones:
      if cone.dim < 2 or cone.start < 0 or cone.start + cone.dim > width:
        raise ValueError(f"cone {cone} does not fit a vector of {width}")
  shape = Shape(n, nx, nu, bool(state_cones), bool(input_cones))
  slay, clay = shape.state_layout(), _carry_layout(shape, settings, fixed_bounds is None)
  backward, forward = _backward_step(name, b, cache, lowering), _forward_step(name, a, b, fdyn, cache, lowering)
  st = settings

  # Per-stage LQR tables, read backwards (i = N-2 .. 0) by the backward pass and forwards by the rollout.
  stage_tables: list[tuple[Expr, int]] = []
  if cache.varying:
    stage_tables = [(_flat_table(m), m[0].size) for m in (cache.Quu_inv, cache.AmBKt, cache.K, cache.APf, cache.BPf)]
  k_table = [(_flat_table(cache.K), cache.K[0].size)] if cache.varying else []

  carry = Expr.sym("carry", clay.size)

  def get(field: str) -> Expr:
    if fixed_bounds is not None and field in ("x_min", "x_max", "u_min", "u_max"):
      return Expr.const(np.asarray(fixed_bounds[field], dtype=np.float64).reshape(-1))
    lo, hi = clay[field]
    return carry[lo:hi]

  g, y = get("g"), get("y")
  # update_linear_cost
  q = get("qref") - rho * (get("vnew") - g)
  r = get("rref") - rho * (get("znew") - y)
  pn = get("pnref") - rho * (get("vnew")[(n - 1) * nx :] - g[(n - 1) * nx :])
  if state_cones:
    q = q - rho * (get("vcnew") - get("gc"))
    pn = pn - rho * (get("vcnew")[(n - 1) * nx :] - get("gc")[(n - 1) * nx :])
  if input_cones:
    r = r - rho * (get("zcnew") - get("yc"))
  # backward_pass_grad: i = N-2, ..., 0 reads r_i and q_i backwards; d comes out in that order
  back_specs = [(r, (n - 2) * nu, -nu), (q, (n - 2) * nx, -nx), *((t, (n - 2) * size, -size) for t, size in stage_tables)]
  _, d_rev = scan(backward, pn, back_specs, length=n - 1)
  # forward_pass: i = 0, ..., N-2 reads d_i from the reversed stack
  x0 = get("x")[:nx]
  _, u, x_tail = scan(forward, x0, [(d_rev, (n - 2) * nu, -nu), *((t, 0, size) for t, size in k_table)], length=n - 1)
  x = concat([x0, x_tail])
  # update_slack
  vnew, znew = x + g, u + y
  if st.en_state_bound:
    vnew = minimum(get("x_max"), maximum(get("x_min"), vnew))
  if st.en_input_bound:
    znew = minimum(get("u_max"), maximum(get("u_min"), znew))
  # update_dual
  new = {"x": x, "u": u, "vnew": vnew, "znew": znew, "g": g + x - vnew, "y": y + u - znew}
  if state_cones:
    vcnew = project_cones(x + get("gc"), n, nx, state_cones)
    new["vcnew"], new["gc"] = vcnew, get("gc") + x - vcnew
  if input_cones:
    zcnew = project_cones(u + get("yc"), n - 1, nu, input_cones)
    new["zcnew"], new["yc"] = zcnew, get("yc") + u - zcnew
  # termination_condition; v and z keep their values in the iteration that stops
  done = logical_and(
    logical_and(less(norm_inf(x - vnew), st.abs_pri_tol), less(norm_inf(u - znew), st.abs_pri_tol)),
    logical_and(less(norm_inf(get("v") - vnew) * rho, st.abs_dua_tol), less(norm_inf(get("z") - znew) * rho, st.abs_dua_tol)),
  )
  new["v"] = where(done, get("v"), vnew)
  new["z"] = where(done, get("z"), znew)
  new["done"] = cast(done, "float64").reshape((1,))
  parts = [new.get(field, get(field)) for field in clay.offsets]
  body = Function.from_exprs(f"{name}_iteration", [carry], [concat(parts)], ["carry"], ["next"])
  cond = Function.from_exprs(f"{name}_not_converged", [carry], [less(get("done")[0], 0.5)], ["carry"], ["go_on"])

  state, x0_in = Expr.sym("state", slay.size), Expr.sym("x0", nx)
  xref, uref = Expr.sym("xref", n * nx), Expr.sym("uref", (n - 1) * nu)
  inputs, names = [state, x0_in, xref, uref], ["state", "x0", "xref", "uref"]
  bounds = {}
  if st.en_state_bound and fixed_bounds is None:
    bounds["x_min"], bounds["x_max"] = Expr.sym("x_min", n * nx), Expr.sym("x_max", n * nx)
  if st.en_input_bound and fixed_bounds is None:
    bounds["u_min"], bounds["u_max"] = Expr.sym("u_min", (n - 1) * nu), Expr.sym("u_max", (n - 1) * nu)
  inputs += list(bounds.values())
  names += list(bounds)

  def field(name_: str) -> Expr:
    lo, hi = slay[name_]
    return state[lo:hi]

  x_start = concat([x0_in, field("x")[nx:]])  # tiny_set_x0
  init = {f_: field(f_) for f_ in slay.offsets}
  init["x"] = x_start
  if state_cones:
    init["vcnew"] = x_start
  if input_cones:
    init["zcnew"] = field("u")
  init["qref"], init["rref"], init["pnref"] = linear_cost(xref, uref)
  init.update(bounds)
  init["done"] = Expr.const(np.zeros(1))
  final, iterations = while_loop(cond, body, concat([init[k] for k in clay.offsets]), max_iter=st.max_iter)
  out_state = concat([final[clay[f_][0] : clay[f_][1]] for f_ in slay.offsets])
  solved = greater(final[clay["done"][0]], 0.5)
  u0 = final[clay["u"][0] : clay["u"][0] + nu]
  # The output is not called ``state``: an output named like an input is read back from the output
  # buffer by the generated code (todo C-124), so the names stay distinct.
  return Function.from_exprs(name, inputs, [out_state, iterations, solved, u0], names, ["state_next", "iterations", "solved", "u0"])


@dataclass(frozen=True)
class TinyADMM:
  """The ADMM of TinyMPC (Nguyen et al., ICRA 2024) as an OCP method, for a ``DiscreteOCP`` with an
  affine map without parameters, ``Quadratic`` costs, box bounds and no other constraints.

  Every bound gets a slack copy of the trajectory, so the primal step is an LQR problem with the
  penalty ``rho`` folded into ``Q`` and ``R``: its gains are computed once, offline, by the backward
  Riccati recursion from the terminal cost (``finite_cache``), and each iteration is a backward pass
  for the affine terms, a forward rollout, the clip onto the bounds and the dual update. The linear
  terms of the references are those of the problem, ``-Q x_ref`` and ``-R u_ref`` (TinyMPC's library
  uses ``Q + rho I``, which ``admm_solver`` also takes), so a converged solve is the problem's
  solution to the tolerances. It stops when the primal residuals and ``rho`` times the slacks' change
  are below ``abs_pri_tol`` and ``abs_dua_tol``, with ``OK``, or after ``max_iter`` iterations, with
  ``MAX_ITER``.

  The warm start, and the point a solve returns, is the ADMM's whole state (``STATE_FIELDS``
  without the cone duals, over the ``N + 1`` knots), flat; ``shift`` moves each field up one stage."""

  name: ClassVar[str] = "ocp.tinyadmm"
  problem: ClassVar[type] = DiscreteOCP
  api: ClassVar[int] = METHOD_API
  label: ClassVar[str] = "tinyadmm"

  rho: float = 1.0
  abs_pri_tol: float = 1e-3
  abs_dua_tol: float = 1e-3
  max_iter: int = 1000

  def __post_init__(self) -> None:
    if self.rho <= 0 or self.max_iter < 1:
      raise ValueError("TinyADMM needs rho > 0 and max_iter >= 1")

  def supports(self, problem: Any) -> Support:
    """Whether this method can solve ``problem``: the reasons it cannot, if any."""
    if not isinstance(problem, DiscreteOCP):
      return Support((f"{type(problem).__name__} is not a DiscreteOCP",))
    reasons = []
    if not problem.shooting:
      reasons.append("TinyADMM takes a discrete map or multiple shooting")
    if problem.stage_quadratic is None or (problem.terminal_cost is not None and problem.terminal_quadratic is None):
      reasons.append("TinyADMM takes Quadratic costs")
    if problem.stage_quadratic is not None and problem.stage_quadratic.R is None:
      reasons.append("TinyADMM needs a control weight R")
    if problem.cost_rule == "integral":
      reasons.append("TinyADMM takes the running cost at the points")
    if problem.constraints:
      reasons.append("TinyADMM takes box bounds only, no path constraints")
    if problem.terminal is not None:
      reasons.append("TinyADMM takes no terminal set or equality")
    specs = [(q, attr) for q, attrs in ((problem.stage_quadratic, ("x_ref", "u_ref")), (problem.terminal_quadratic, ("x_ref",))) for attr in attrs]
    references = {getattr(q, attr) for q, attr in specs if q is not None and isinstance(getattr(q, attr), str)}
    others = [p.name for p in problem.params if p.name not in references]
    if others:
      reasons.append(f"TinyADMM's model and weights take no parameters, and {others} are not references")
    if not reasons and not _affine(problem):
      reasons.append("TinyADMM takes a map affine in the state and the control, with no parameters")
    return Support(tuple(reasons))

  def warm_size(self, problem: DiscreteOCP) -> int:
    """The size of the ADMM's state over ``N + 1`` knots, flat."""
    return Shape(problem.N + 1, problem.nx, problem.nu).state_layout().size

  def build(self, problem: DiscreteOCP, *, name: str) -> ConcreteFunction[Any, Any, Any, Any]:
    """The solver Function: ``(x0, *params, warm) -> (xs, us, point, info)``."""
    nx, nu, n = problem.nx, problem.nu, problem.N
    knots = n + 1
    a, b, f = _affine_parts(problem)
    stage, terminal = problem.stage_quadratic, problem.terminal_quadratic
    assert stage is not None and stage.R is not None
    scale = problem.dt if problem.continuous and problem.dt is not None else 1.0
    # TinyMPC's costs are 1/2 x'Qx; a Quadratic is x'Qx, and the running cost is scaled at the points.
    q = 2.0 * scale * np.atleast_2d(np.asarray(stage.Q, dtype=np.float64))
    r = 2.0 * scale * np.atleast_2d(np.asarray(stage.R, dtype=np.float64))
    qn = 2.0 * np.atleast_2d(np.asarray(terminal.Q, dtype=np.float64)) if terminal is not None else np.zeros((nx, nx))
    cache = finite_cache(a, b, q, r, qn, self.rho, knots, f)
    x_lo, x_hi = _box(problem.x_bounds, nx)
    u_lo, u_hi = _box(problem.u_bounds, nu)
    settings = Settings(
      self.abs_pri_tol, self.abs_dua_tol, self.max_iter, bool(np.isfinite(np.r_[x_lo, x_hi]).any()), bool(np.isfinite(np.r_[u_lo, u_hi]).any())
    )
    # The initial state is data: its knot is unbounded.
    fixed = {
      "x_min": np.r_[np.full(nx, -np.inf), np.tile(x_lo, n)],
      "x_max": np.r_[np.full(nx, np.inf), np.tile(x_hi, n)],
      "u_min": np.tile(u_lo, n),
      "u_max": np.tile(u_hi, n),
    }

    def linear_cost(xref: Expr, uref: Expr) -> tuple[Expr, Expr, Expr]:
      xs_ref, us_ref = xref.reshape((knots, nx)), uref.reshape((n, nu))
      qref = -(xs_ref @ Expr.const(q.T)).reshape((knots * nx,))
      rref = -(us_ref @ Expr.const(r.T)).reshape((n * nu,))
      return qref, rref, -(Expr.const(qn) @ xref[n * nx :])

    admm = admm_solver(
      a, b, cache, knots=knots, rho=self.rho, linear_cost=linear_cost, f=f, settings=settings, fixed_bounds=fixed, name=f"{name}_admm"
    )
    slay = Shape(knots, nx, nu).state_layout()
    size = slay.size

    def reference(spec: Quadratic | None, attr: str, width: int, count: int, values: dict[str, Expr]) -> Expr:
      ref = None if spec is None else getattr(spec, attr)
      if isinstance(ref, str):
        value = values[ref]
        return value[: count * width] if ref in problem.varying else concat([value.reshape((width,))] * count)
      return Expr.const(np.tile(np.zeros(width) if ref is None else np.asarray(ref, dtype=np.float64).reshape(-1), count))

    def terminal_reference(values: dict[str, Expr]) -> Expr:
      spec = terminal if terminal is not None else stage
      ref = spec.x_ref
      if isinstance(ref, str):
        value = values[ref]
        return value[n * nx :] if ref in problem.varying else value.reshape((nx,))
      return Expr.const(np.zeros(nx) if ref is None else np.asarray(ref, dtype=np.float64).reshape(-1))

    def body(x0: Expr, *rest: Expr) -> Any:
      *raw, warm = rest
      values = {p.name: v.reshape((v.size,)) for p, v in zip(problem.params, raw, strict=True)}
      xref = concat([reference(stage, "x_ref", nx, n, values), terminal_reference(values)])
      uref = reference(stage, "u_ref", nu, n, values)
      state_next, iterations, solved, _ = admm((warm, x0, xref, uref))
      xs = state_next[slay["x"][0] : slay["x"][1]]
      us = state_next[slay["u"][0] : slay["u"][1]]
      residual = maximum(norm_inf(xs - state_next[slay["vnew"][0] : slay["vnew"][1]]), norm_inf(us - state_next[slay["znew"][0] : slay["znew"][1]]))
      specs = [(xs, 0, nx), (us, 0, nu), *((raw[i], 0, p.type.size if p.name in problem.varying else 0) for i, p in enumerate(problem.params))]
      assert problem.stage_cost is not None
      cost = vmap(problem.stage_cost, n, specs).sum() * scale
      if problem.terminal_cost is not None:
        ends = [problem.at_end({p.name: raw[i]}, p) for i, p in enumerate(problem.params)]
        cost = cost + problem.terminal_cost(xs[n * nx :], *ends)
      status = where(solved, float(Status.OK), float(Status.MAX_ITER))
      info = Info(status=status, iter=cast(iterations, "float64"), objective=cost, primal_residual=residual)
      return xs.reshape((knots, nx)), us.reshape((n, nu)), state_next, info

    inputs = param_list(L("x0", nx), *(L(p.name, (problem.param_size(p),)) for p in problem.params), L("warm", size))
    outputs = G(L("xs", (knots, nx)), L("us", (n, nu)), L("point", size), Info.tree())
    return ConcreteFunction(name, body, inputs, outputs)

  def shift(self, problem: DiscreteOCP) -> ConcreteFunction[Any, Any, Any, Any]:
    """``point -> warm``: every field of the ADMM's state moved up one stage, the last repeated
    (``sc.ocp.shift``)."""
    nx, nu, knots = problem.nx, problem.nu, problem.N + 1
    shape = Shape(knots, nx, nu)
    slay = shape.state_layout()

    def body(point: Expr) -> Expr:
      parts = []
      for field in slay.offsets:
        lo, hi = slay[field]
        block = nx if shape.field_size(field) == knots * nx else nu
        seg = point[lo:hi]
        parts.append(concat([seg[block:], seg[seg.size - block :]]) if seg.size > block else seg)
      return concat(parts)

    return ConcreteFunction(f"{problem.name}_shift_tinyadmm", body, param_list(L("point", slay.size)), L("warm", slay.size))

  def initial_guess(self, problem: DiscreteOCP, x0: Any, u: Any = None) -> np.ndarray:
    """A first warm start (``sc.ocp.initial_guess``): the states ``x0`` and the controls ``u`` (zeros
    by default), their slacks equal to them, every dual zero."""
    nx, nu, knots = problem.nx, problem.nu, problem.N + 1
    x = np.tile(np.ravel(np.asarray(x0, dtype=np.float64)), knots)
    uu = np.tile(np.zeros(nu) if u is None else np.ravel(np.asarray(u, dtype=np.float64)), knots - 1)
    fields = {"x": x, "v": x, "vnew": x, "u": uu, "z": uu, "znew": uu}
    shape = Shape(knots, nx, nu)
    return np.concatenate([fields.get(name, np.zeros(shape.field_size(name))) for name in shape.fields()])


def _box(bounds: tuple[Any, Any] | None, size: int) -> tuple[np.ndarray, np.ndarray]:
  lo, hi = (None, None) if bounds is None else bounds
  side = lambda v, fill: np.full(size, fill) if v is None else np.broadcast_to(np.asarray(v, dtype=np.float64), (size,)).copy()  # noqa: E731
  return side(lo, -np.inf), side(hi, np.inf)


def _symbolic_map(problem: DiscreteOCP) -> tuple[Expr, Expr, list[Expr], Expr]:
  x, u = Expr.sym("x", problem.nx), Expr.sym("u", problem.nu)
  params = [Expr.sym(p.name, p.type.shape) for p in problem.params]
  return x, u, params, step_map(problem)(x, u, *params)


def _affine(problem: DiscreteOCP) -> bool:
  """Whether the map is affine in the state and the control and reads no parameter."""
  x, u, params, out = _symbolic_map(problem)
  for wrt in (x, u):
    if _jac_mask(simplify_cse_fixpoint(jacobian(out, wrt)), wrt, {}).nnz:
      return False
  return not any(_jac_mask(out, p, {}).nnz for p in params)


def _affine_parts(problem: DiscreteOCP) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
  """``A``, ``B`` and ``f`` of the affine map ``x_next = A x + B u + f``, evaluated once."""
  x, u, params, out = _symbolic_map(problem)
  parts = ConcreteFunction.from_exprs(
    f"{problem.name}_affine_parts",
    [x, u, *params],
    [jacobian(out, x), jacobian(out, u), out],
    ["x", "u", *(p.name for p in problem.params)],
    ["a", "b", "f"],
  )
  a, b, f = parts((np.zeros(problem.nx), np.zeros(problem.nu), *(np.zeros(p.type.shape) for p in problem.params)))
  f = np.asarray(f, dtype=np.float64)
  return np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64), None if not f.any() else f


__all__ = ["STATE_FIELDS", "Cone", "LQRCache", "Settings", "Shape", "TinyADMM", "admm_solver", "finite_cache", "project_cones", "tinympc_cache"]
