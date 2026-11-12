"""Argument support through the chain M=9 Hessian's Runge-Kutta forward helpers."""

import numpy as np

import scaly as sc
from scaly.ad.sparse import _sparse_hessian
from scaly.ad.sparsity import star_coloring
from scaly.ir.expr import Expr, ExprOp, OP_INFO, substitute, topo
from scaly.ir.support import specialize_support


def _chain_stage():
  @sc.function(sc.arg("dist", 3), *(sc.arg(name, 1) for name in ("mass", "spring_d", "rest_len")), outputs=sc.arg("accel"), name="support_chain_link")
  def link(dist, mass, spring_d, rest_len):
    return (spring_d[0] / mass[0]) * (1 - rest_len[0] / sc.norm_2(dist)) * dist

  constant = sc.TensorType((1,), diff=False)
  parameters = ("mass", "spring_d", "rest_len", "gravity")

  @sc.function(
    *(sc.arg(name, 3) for name in ("left", "pos", "right")),
    *(sc.arg(name, 1) for name in parameters),
    outputs=sc.arg("accel"),
    name="support_chain_accel",
  )
  def accel(left, pos, right, mass, spring_d, rest_len, gravity):
    return link(right - pos, mass, spring_d, rest_len) - link(pos - left, mass, spring_d, rest_len) + sc.stack([0, 0, gravity[0]])

  @sc.function(
    sc.arg("x", 45),
    sc.arg("u", 3),
    *(sc.arg(name, constant) for name in parameters),
    outputs=sc.arg("xdot"),
    name="support_chain_ode_M9",
  )
  def ode(x, u, mass, spring_d, rest_len, gravity):
    positions = sc.concat([sc.const([0, 0, 0]), x[:24]])
    acceleration = sc.vmap(accel, 7)(
      sc.window(positions, 0, 3),
      sc.window(positions, 3, 3),
      sc.window(positions, 6, 3),
      *(sc.broadcast(value) for value in (mass, spring_d, rest_len, gravity)),
    ).vec()
    return sc.concat([x[24:], u, acceleration])

  @sc.function(
    sc.arg("x", 45),
    sc.arg("u", 3),
    *(sc.arg(name, constant) for name in (*parameters, "dt")),
    outputs=sc.arg("next"),
    name="support_chain_step_M9",
  )
  def step(x, u, mass, spring_d, rest_len, gravity, dt):
    h = dt[0]
    k1 = ode(x, u, mass, spring_d, rest_len, gravity)
    k2 = ode(x + h / 2 * k1, u, mass, spring_d, rest_len, gravity)
    k3 = ode(x + h / 2 * k2, u, mass, spring_d, rest_len, gravity)
    k4 = ode(x + h * k3, u, mass, spring_d, rest_len, gravity)
    return x + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

  @sc.function(
    sc.arg("z", 48),
    sc.arg("xnext", 45),
    sc.arg("params", sc.TensorType((5,), diff=False)),
    outputs=sc.arg("eq"),
    name="support_chain_eq_stage_M9",
  )
  def stage(z, xnext, params):
    return (step(z[:45], z[45:], *(params[i : i + 1] for i in range(5))) - xnext).scalar()

  return stage


def _seed_image(expr: Expr, wrt: Expr, seeds: np.ndarray) -> np.ndarray:
  pattern = sc.jacobian_sparsity(expr, wrt)
  mask = np.zeros(pattern.shape, dtype=bool)
  mask[pattern.rows, pattern.cols] = True
  return seeds @ mask.T


def test_chain_M9_hessian_seed_axis_support_matches_independent_jacobian_images() -> None:
  z, p, lam = sc.sym("chain_z", 141), sc.sym("chain_p", 50, diff=False), sc.sym("chain_lam", 135, diff=False)
  mapped = sc.vmap(_chain_stage(), 2)(sc.window(z, 0, 48), sc.window(z, 48, 48), sc.window(p, 45, 0)).vec()
  hessian = _sparse_hessian(lam @ sc.concat([z[:45] - p[:45], mapped]), z)
  colors = star_coloring(hessian.sparsity)
  assert hessian.coloring_width == 42
  seeds = np.eye(hessian.coloring_width, dtype=bool)[:, colors]
  assert seeds.shape == (42, 141)
  np.testing.assert_array_equal(seeds[:, :48], seeds[:, 48:96])
  stage_seeds = seeds[:, :48]
  pending, seen, sites = [(hessian.values,)], set(), []
  while pending:
    for node in topo(pending.pop()):
      if node.op in {ExprOp.CALL, ExprOp.VMAP}:
        callee = node.attrs["callee"]
        if callee not in seen:
          seen.add(callee)
          pending.append(callee.results)
        if callee.name.startswith("support_chain_ode_M9") and any(arg.shape == (42, 45) for arg in node.args):
          sites.append(node)
  assert len(sites) == 2
  counts = []
  for call in sites:
    helper = call.attrs["callee"]
    wrt = helper.inputs[-1]
    assert wrt.shape == (42, 45)
    state = call.args[0]
    primal_inputs = [node for node in topo((state,)) if node.op == ExprOp.INPUT and node.type.diff]
    (stage_input,) = primal_inputs
    assert stage_input.shape == (48,)
    expected_input = _seed_image(state, stage_input, stage_seeds)
    assert call.args[-1].support == frozenset(map(int, np.flatnonzero(expected_input)))
    context = specialize_support(helper, tuple(arg.support for arg in call.args), OP_INFO)
    seed_nodes = [node for node in topo(helper.results) if node.shape and node.shape[0] == 42]
    seed_constants = [node for node in seed_nodes if node.op == ExprOp.CONST and np.any(node.value)]
    assert len(seed_constants) == 1
    np.testing.assert_array_equal(seed_constants[0].value, stage_seeds[:, 45:])
    constant = seed_constants[0]
    constant_seed = sc.sym("chain_constant_seed", constant.shape)
    for node in seed_nodes:
      expected = _seed_image(node, wrt, expected_input.reshape(1, -1))
      expected |= _seed_image(substitute(node, {constant: constant_seed}), constant_seed, stage_seeds[:, 45:].reshape(1, -1))
      assert context(node) == frozenset(map(int, np.flatnonzero(expected)))
    counts.append((len(call.args[-1].support), len(call.support)))
  assert sorted(counts) == [(249, 564), (567, 813)]
