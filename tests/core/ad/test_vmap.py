from __future__ import annotations

import numpy as np
import pytest

import scaly as sc


@sc.function(sc.G(sc.L("x", 3), sc.L("p", 3)), output=sc.L("y", ...), name="scale_add")
def scale_add(inputs):
  x, p = inputs
  return 2.0 * x + p


def test_jvp_many_of_vmap_matches_unrolled_jvp() -> None:
  N = 4
  z = sc.sym("z", 3 * N)
  p = sc.sym("p", 3 * N)

  mapped = sc.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  unrolled = sc.concat([scale_add((z[i * 3 : (i + 1) * 3], p[i * 3 : (i + 1) * 3])) for i in range(N)])

  seeds = sc.const(np.eye(3 * N))
  jvp_vmap = sc.jvp_many(mapped, z, seeds)
  jvp_ref = sc.jvp_many(unrolled, z, seeds)

  fn_vmap = sc.Function.from_exprs("jvp_vmap", [z, p], [jvp_vmap], ["z", "p"], ["dy"])
  fn_ref = sc.Function.from_exprs("jvp_ref", [z, p], [jvp_ref], ["z", "p"], ["dy"])

  rng = np.random.default_rng(0)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)

  np.testing.assert_allclose(fn_vmap((zv, pv)), fn_ref((zv, pv)), rtol=1e-10, atol=1e-10)


def test_jacobian_of_vmap_matches_finite_differences() -> None:
  N = 5
  z = sc.sym("z", 3 * N)
  p = sc.sym("p", 3 * N)
  mapped = sc.vmap(scale_add, N, [(z, 0, 3), (p, 0, 3)])
  fn = sc.Function.from_exprs("mapped", [z, p], [mapped], ["z", "p"], ["y"])

  seeds = sc.const(np.eye(3 * N))
  jacobian = sc.jvp_many(mapped, z, seeds).T  # (3N, 3N)
  jac_fn = sc.Function.from_exprs("jac", [z, p], [jacobian], ["z", "p"], ["jac"])

  rng = np.random.default_rng(1)
  zv = rng.normal(size=3 * N)
  pv = rng.normal(size=3 * N)
  jac = jac_fn((zv, pv))

  eps = 1e-7
  y0 = fn((zv, pv))
  assert isinstance(y0, np.ndarray)
  fd = np.zeros((3 * N, 3 * N))
  for j in range(3 * N):
    zp = zv.copy()
    zp[j] += eps
    yj = fn((zp, pv))
    assert isinstance(yj, np.ndarray)
    fd[:, j] = (yj - y0) / eps
  np.testing.assert_allclose(jac, fd, atol=1e-5)


def test_grad_factory_over_vmap_matches_unrolled_and_finite_difference() -> None:
  from scaly.ad import finite_difference
  from scaly.ir.expr import topo

  @sc.function(sc.L("x", 2), output=sc.L("y", ...), name="vmap_grad_piece")
  def piece(x):
    return x * x + x.sin()

  N = 4
  z = sc.sym("z", 2 * N)
  mapped = sc.vmap(piece, N, [(z, 0, 2)])
  unrolled = sc.concat([piece(z[2 * it : 2 * (it + 1)]) for it in range(N)])
  mapped_fn = sc.Function.from_exprs("vmap_grad_factory", [z], [mapped], ["z"], ["y"])
  unrolled_fn = sc.Function.from_exprs("vmap_grad_unrolled", [z], [unrolled], ["z"], ["y"])
  mapped_grad = mapped_fn.factory("vmap_grad_factory_grad", ["z", "lam:y"], [sc.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  unrolled_grad = unrolled_fn.factory("vmap_grad_unrolled_grad", ["z", "lam:y"], [sc.factory.Grad("gamma", "z")], aux={"gamma": ["y"]})
  vmap_nodes = [node for node in topo(mapped_grad.outputs) if node.op == sc.ExprOp.VMAP]
  assert len(vmap_nodes) == 1
  assert "_adj0_0" in vmap_nodes[0].attrs["callee"].name

  zv = np.random.default_rng(7).normal(size=2 * N)
  lamv = np.random.default_rng(8).normal(size=2 * N)
  np.testing.assert_allclose(mapped_grad((zv, lamv)), unrolled_grad((zv, lamv)), rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(
    mapped_grad((zv, lamv)), finite_difference(lambda value: np.dot(lamv, np.asarray(mapped_fn(value))), zv).reshape(-1), rtol=1e-6, atol=1e-7
  )


@sc.function(sc.G(sc.L("x", 3), sc.L("q", 2)), output=sc.L("y", ...), name="vmap_duality_piece")
def duality_piece(inputs):
  x, q = inputs
  return sc.stack([x[0] * x[1] * q[0].sin() + x[2].exp(), (1.0 + sc.dot(x, x)).sqrt() * q[1] + x[0] * x[2]])


def test_forward_and_adjoint_of_vmap_match_unrolled_and_are_dual() -> None:
  N = 5
  z, q = sc.sym("z", 3 * N), sc.sym("q", 2 * N)
  mapped = sc.vmap(duality_piece, N, [(z, 0, 3), (q, 0, 2)])
  unrolled = sc.concat([duality_piece((z[3 * k : 3 * (k + 1)], q[2 * k : 2 * (k + 1)])) for k in range(N)])
  fn_vmap = sc.Function.from_exprs("duality_vmap", [z, q], [mapped], ["z", "q"], ["y"])
  fn_unroll = sc.Function.from_exprs("duality_unroll", [z, q], [unrolled], ["z", "q"], ["y"])

  rng = np.random.default_rng(5)
  zv, qv = rng.normal(size=3 * N), rng.normal(size=2 * N)
  v, w = rng.normal(size=3 * N), rng.normal(size=2 * N)
  # Inputs are unit normal and the callee holds exp and products of three, so |y|, |J| stay around 1.
  fwd = {name: np.asarray(sc.forward(fn, "y", "z")((zv, qv), v)) for name, fn in (("vmap", fn_vmap), ("unroll", fn_unroll))}
  adj = {name: np.asarray(sc.adjoint(fn, "y", "z")((zv, qv), w)) for name, fn in (("vmap", fn_vmap), ("unroll", fn_unroll))}
  jac = np.asarray(sc.jacobian(fn_unroll, "y", "z")((zv, qv)))
  np.testing.assert_allclose(fwd["vmap"], fwd["unroll"], rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(adj["vmap"], adj["unroll"], rtol=1e-12, atol=1e-12)
  np.testing.assert_allclose(fwd["vmap"], jac @ v, rtol=1e-10, atol=1e-10)
  np.testing.assert_allclose(adj["vmap"], jac.T @ w, rtol=1e-10, atol=1e-10)
  # Forward/reverse duality: <J v, w> == <v, J^T w>.
  np.testing.assert_allclose(w @ fwd["vmap"], v @ adj["vmap"], rtol=1e-12)


# --- a mapped callee with a reverse rule of its own -------------------------------------------------

LANES = 5


def _layers(name: str, *, reads: str) -> tuple[sc.Function, sc.Function]:
  """``z = W x``, ``y = tanh(z)`` as a Function of outputs ``(y, [z, 2 z])``, plain and with a
  reverse rule. The rule needs ``1 - y^2``: it takes it from the output ``y`` (``reads="y"``), from
  the second output, which is twice as long (``"z"``), or computes it again from the inputs
  (``"inputs"``)."""
  x, w = sc.sym("x", 4), sc.sym("w", 12)
  W = w.reshape((3, 4))
  z = W @ x
  plain = sc.Function.from_exprs(name, [x, w], [z.tanh(), sc.concat([z, 2.0 * z])], ["x", "w"], ["y", "z"])
  y_in, z_in, ybar, zbar = sc.sym("y", 3), sc.sym("z", 6), sc.sym("ybar", 3), sc.sym("zbar", 6)
  slope = {"y": 1.0 - y_in * y_in, "z": 1.0 - z_in[:3].tanh() ** 2, "inputs": 1.0 - (W @ x).tanh() ** 2}[reads]
  g = ybar * slope + zbar[:3] + 2.0 * zbar[3:]
  rule = sc.Function.from_exprs(
    f"{name}_rule",
    [x, w, y_in, z_in, ybar, zbar],
    [W.T @ g, (g.reshape((3, 1)) * x.reshape((1, 4))).reshape((12,))],
    ["x", "w", "y", "z", "ybar", "zbar"],
    ["xbar", "wbar"],
  )
  return plain, sc.custom_derivative(plain, vjp=rule)


def _calls(prog, *, looped: bool | None = None) -> dict[str, list[str]]:
  """Per procedure of a lowered program, the procedures it calls (from inside a loop, from outside
  every loop, or anywhere)."""
  from scaly.ir.program import ProgramOp

  out: dict[str, list[str]] = {}
  for proc in prog.args[: int(prog.attrs["proc_count"])]:
    found: list[str] = []

    def walk(node, inside: bool, found: list[str] = found) -> None:
      if node.op == ProgramOp.CALL and looped in (None, inside):
        found.append(node.attrs["callee"])
      for arg in node.args:
        walk(arg, inside or node.op == ProgramOp.FOR)

    for stmt in proc.args[int(proc.attrs["param_count"]) :]:
      walk(stmt, False)
    out[proc.attrs["name"]] = found
  return out


def _lowered(fn: sc.Function):
  """``fn``'s program as lowering leaves it, before any pass."""
  from scaly.passes.lowering import lower_function

  stages: dict = {}
  lower_function(fn, observe=lambda name, prog: stages.__setitem__(name, prog))
  return stages["lowered"]


def _loss_gradients(name: str, callee: sc.Function, *, linear: bool = False) -> sc.Function:
  xs, ws = sc.sym("xs", LANES * 4), sc.sym("ws", LANES * 12)
  y = sc.vmap(callee, LANES, [(xs, 0, 4), (ws, 0, 12)])
  loss = (y * sc.const(np.linspace(0.5, 2.0, LANES * 3))).sum() if linear else (y * y * y).sum()
  return sc.Function.from_exprs(name, [xs, ws], list(sc.vjp((loss,), (xs, ws), (sc.const(1.0),))), ["xs", "ws"], ["gx", "gw"])


@pytest.mark.parametrize("reads", ["y", "z"])
def test_a_mapped_rule_reads_the_maps_own_outputs(reads: str) -> None:
  """The gradient of a loss that is not linear in a mapped callee's output needs that output twice:
  for the loss's own cotangent, and for the callee's reverse rule, which takes the outputs. Both
  read the one map: it is one loop (whichever outputs the rule reads, they are outputs of one
  call), the lane of the adjoint map calls the rule and nothing else, and the gradient is the one
  AD gives through the body. Before, every lane called the callee again for the rule."""
  plain, custom = _layers(f"ruled_{reads}", reads=reads)
  ruled, reference = _loss_gradients(f"ruled_{reads}_grad", custom), _loss_gradients(f"plain_{reads}_grad", plain)
  rng = np.random.default_rng(31)
  xv, wv = rng.normal(size=LANES * 4), rng.normal(size=LANES * 12)
  for got, want in zip(ruled((xv, wv)), reference((xv, wv)), strict=True):
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
  lowered = _lowered(ruled)  # before the passes, which expand small procedures into their callers
  calls = _calls(lowered)
  (lane,) = [n for n in calls if "_adj0_" in n]
  assert calls[lane] == [f"ruled_{reads}_rule"]
  assert sorted(_calls(lowered, looped=True)[f"ruled_{reads}_grad"]) == sorted([custom.name, lane])


def test_a_rule_that_does_not_read_the_outputs_leaves_the_map_alone() -> None:
  """A rule that works from the inputs takes no output, and a loss linear in the map never needs
  its value: the gradient does not call the callee at all."""
  plain, custom = _layers("unread", reads="inputs")
  ruled, reference = _loss_gradients("unread_grad", custom, linear=True), _loss_gradients("unread_plain_grad", plain, linear=True)
  rng = np.random.default_rng(32)
  xv, wv = rng.normal(size=LANES * 4), rng.normal(size=LANES * 12)
  for got, want in zip(ruled((xv, wv)), reference((xv, wv)), strict=True):
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-14)
  assert custom.name not in {name for called in _calls(_lowered(ruled)).values() for name in called}


@pytest.mark.parametrize("dtype", ["int64", "bool"])
@pytest.mark.parametrize("reads", [False, True], ids=["unread", "read"])
def test_a_rule_is_given_a_count_or_a_flag_it_does_not_read_as_a_zero_of_that_type(dtype: str, reads: bool) -> None:
  """A Function with a reverse rule returns a count or a flag beside its value. The rule is passed
  a zero for an output it does not read, and that zero has the output's own type: a float zero for
  an integer or a boolean formal is no call at all. In a call, under a map and in a scan's body the
  gradient is the one the rule gives, as it is when the rule reads the output."""
  tag = f"{dtype}_{'read' if reads else 'unread'}"
  x = sc.sym("x", 3)
  above = x > 0.5
  extra = above if dtype == "bool" else sc.cast(above.cast("float64").sum(), dtype).reshape((1,))
  plain = sc.Function.from_exprs(f"flag_{tag}", [x], [x * x * x, extra], ["x"], ["y", "n"])
  y, n, ybar, nbar = sc.sym("y", 3), sc.sym("n", extra.shape, dtype=dtype), sc.sym("ybar", 3), sc.sym("nbar", extra.shape)
  slope = 3.0 * x * x
  if reads:  # times one, computed from the output that is not a float
    slope = slope * (sc.where(n, 1.0, 1.0) if dtype == "bool" else sc.cast(n, "float64").sum() * 0.0 + 1.0)
  rule = sc.Function.from_exprs(f"flag_{tag}_rule", [x, y, n, ybar, nbar], [slope * ybar], ["x", "y", "n", "ybar", "nbar"], ["xbar"])
  ruled = sc.custom_derivative(plain, vjp=rule)
  point, weights = np.array([0.3, 0.9, 1.4]), np.array([1.0, -2.0, 0.5])
  q = sc.sym("q", 3)
  called = sc.Function.from_exprs(f"flag_{tag}_call", [q], [sc.gradient((weights * ruled(q)[0]).sum(), q)], ["q"], ["g"])
  np.testing.assert_allclose(called(point), 3 * point**2 * weights, rtol=1e-13)
  qs, points = sc.sym("qs", 6), np.concatenate([point, point + 0.1])
  mapped = sc.Function.from_exprs(f"flag_{tag}_map", [qs], [sc.gradient((sc.vmap(ruled, 2, [(qs, 0, 3)]) ** 2).sum(), qs)], ["qs"], ["g"])
  np.testing.assert_allclose(mapped(points), 6 * points**5, rtol=1e-13)
  carry = sc.sym("carry", 3)
  body = sc.Function.from_exprs(f"flag_{tag}_body", [carry], [ruled(carry)[0] * 0.5], ["carry"], ["next"])
  scanned = sc.Function.from_exprs(f"flag_{tag}_scan", [q], [sc.gradient(sc.scan(body, q, length=2)[0].sum(), q)], ["q"], ["g"])
  np.testing.assert_allclose(scanned(point), 9 * point**8 / 16, rtol=1e-13)  # ((x^3 / 2)^3) / 2 = x^9 / 16


def test_second_derivatives_go_through_a_mapped_rule_and_the_outputs_it_reads() -> None:
  """The lane is a function of the inputs, the outputs and the cotangent, and the outputs are the
  map's: forward mode over the gradient differentiates all three paths. The Hessian of the loss
  matches the one AD gives through the body."""
  plain, custom = _layers("second", reads="y")
  xs, ws = sc.sym("xs", LANES * 4), sc.sym("ws", LANES * 12)
  hessians = []
  for label, callee in (("ruled", custom), ("plain", plain)):
    y = sc.vmap(callee, LANES, [(xs, 0, 4), (ws, 0, 12)])
    (gx,) = sc.vjp(((y * y * y).sum(),), (xs,), (sc.const(1.0),))
    hessians.append(sc.Function.from_exprs(f"second_{label}", [xs, ws], [sc.jacobian(gx, xs)], ["xs", "ws"], ["h"]))
  rng = np.random.default_rng(33)
  xv, wv = rng.normal(size=LANES * 4), rng.normal(size=LANES * 12)
  got, want = hessians[0]((xv, wv)), hessians[1]((xv, wv))
  assert np.abs(want).max() > 0.1
  np.testing.assert_allclose(got, want, rtol=1e-11, atol=1e-13)
