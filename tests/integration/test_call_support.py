"""Argument support through the chain M=9 Runge-Kutta forward helpers."""

import numpy as np

import scaly as sc
from scaly.ad.forward import _pushforward
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.ir.expr import Expr, OP_INFO, topo
from scaly.ir.support import specialize_support


def _chain_ode() -> ConcreteFunction:
  @sc.function(sc.arg("dist", 3), outputs=sc.arg("force"), name="support_chain_link")
  def link(dist):
    return (1 / 0.033) * (1 - 0.033 / sc.norm_2(dist)) * dist

  @sc.function(sc.arg("left", 3), sc.arg("pos", 3), sc.arg("right", 3), outputs=sc.arg("accel"), name="support_chain_accel")
  def accel(left, pos, right):
    return link(right - pos) - link(pos - left) + sc.const([0, 0, -9.81])

  @sc.function(sc.arg("x", 45), sc.arg("u", 3), outputs=sc.arg("xdot"), name="support_chain_ode_M9")
  def ode(x, u):
    positions = sc.concat([sc.const([0, 0, 0]), x[:24]])
    acceleration = sc.vmap(accel, 7)(sc.window(positions, 0, 3), sc.window(positions, 3, 3), sc.window(positions, 6, 3)).vec()
    return sc.concat([x[24:], u, acceleration])

  return as_concrete(ode)


def _seed_image(expr: Expr, wrt: Expr, seeds: np.ndarray) -> np.ndarray:
  pattern = sc.jacobian_sparsity(expr, wrt)
  mask = np.zeros(pattern.shape, dtype=bool)
  mask[pattern.rows, pattern.cols] = True
  return seeds @ mask.T


def test_chain_M9_rk4_seed_axis_support_matches_independent_jacobian_images() -> None:
  ode = _chain_ode()
  x, u = ode.inputs
  dx, du = sc.sym("chain_dx", (42, 45)), sc.sym("chain_du", (42, 3))
  (tangent,) = _pushforward(ode.outputs, {x: dx, u: du}, 42)
  assert tangent is not None
  helper = ConcreteFunction._from_exprs("support_chain_fwd", [x, u, dx, du], [tangent], ["x", "u", "dx", "du"], ["tangent"])
  seed_nodes = [node for node in topo(helper.results) if node.shape and node.shape[0] == 42]
  assert len(seed_nodes) > 10
  z = sc.sym("chain_stage_z", 48)
  seeds = np.eye(42, dtype=bool)[:, np.arange(48) % 42]
  state = z[:45]
  control = z[45:]
  support_counts = []
  for stage in range(4):
    state_support = _seed_image(state, z, seeds)
    control_support = _seed_image(control, z, seeds)
    actual_dx = sc.sym(f"chain_actual_dx_{stage}", dx.shape) * sc.const(state_support, dtype=sc.dtypes.float64)
    actual_du = sc.sym(f"chain_actual_du_{stage}", du.shape) * sc.const(control_support, dtype=sc.dtypes.float64)
    call = helper(state, control, actual_dx, actual_du)
    call_support = call.support
    context = specialize_support(helper, (state.support, control.support, actual_dx.support, actual_du.support), OP_INFO)
    for node in seed_nodes:
      expected = _seed_image(node, dx, state_support.reshape(1, -1)) | _seed_image(node, du, control_support.reshape(1, -1))
      assert context(node) == frozenset(map(int, np.flatnonzero(expected)))
    primal = ode(state, control)
    expected_output = _seed_image(primal, z, seeds)
    assert call_support == frozenset(map(int, np.flatnonzero(expected_output)))
    support_counts.append(int(np.count_nonzero(state_support)))
    if stage < 3:
      state = z[:45] + (0.1 if stage < 2 else 0.2) * primal
  assert support_counts == [45, 249, 567, 825]
