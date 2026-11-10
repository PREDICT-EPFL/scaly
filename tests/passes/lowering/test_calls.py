"""Call invocations keep sibling outputs and custom-rule residuals together."""

import numpy as np
import pytest

import scaly as sc
from scaly.function.concrete import ConcreteFunction
from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
from scaly.ir.program import ProgramOp, walk_program
from scaly.passes.lowering import lower_function, main_proc


def _lowered_main(fun):
  stages = {}
  lower_function(fun, observe=lambda phase, program: stages.setdefault(phase, program))
  return main_proc(stages["lowered"])


@pytest.mark.parametrize("length", [0, 1, 5])
def test_multi_output_map_uses_one_loop_and_matches_numpy(length) -> None:
  @sc.function(sc.arg("x", 2), sc.arg("p", ()))
  def pair(x, p):
    return x.sin() + p, x * x, x.sum() * p

  @sc.function(sc.arg("xs", 2 * length + 1), sc.arg("p", ()))
  def outer(xs, p):
    a, _, b = sc.vmap(pair, length)(sc.window(xs, 1, 2), sc.broadcast(p))
    return b, a, b

  xs, p = np.linspace(-0.8, 0.9, 2 * length + 1), np.array(1.3)
  b, a, again = outer(xs, p)
  rows = xs[1:].reshape(length, 2)
  np.testing.assert_allclose(a, np.sin(rows) + p)
  np.testing.assert_allclose(b, rows.sum(axis=1) * p)
  np.testing.assert_array_equal(again, b)
  nodes = walk_program(_lowered_main(outer))
  calls = [node for node in nodes if node.op == ProgramOp.CALL]
  assert len(calls) == int(length > 0)
  if calls:
    assert calls[0].attrs["n_out"] == 3
    assert [arg.op for arg in calls[0].args[2:]] == [ProgramOp.VIEW, ProgramOp.BUFFER, ProgramOp.VIEW]


@pytest.mark.parametrize("mapped", [False, True])
def test_custom_rule_residual_shares_its_primal_invocation(mapped) -> None:
  @sc.function(sc.arg("x", ()))
  def square(x):
    return x * x

  @sc.function()
  def fwd(x):
    return x * x, sc.stack([x, 2.0 * x])

  @sc.function()
  def bwd(residual, cotangent):
    return residual[1] * cotangent

  callee = as_concrete(sc.custom_derivative(square, fwd=fwd, bwd=bwd))
  length = 5 if mapped else 1
  x = sc.sym("x", length if mapped else ())
  if mapped:
    primal, residual = (_mapped_call(callee, length, [(x, 0, 1)], i) for i in range(2))
  else:
    primal, residual = callee._call((x,))
  fn = ConcreteFunction._from_exprs("residual_siblings", [x], [primal, residual], ["x"], ["value", "residual"])
  values = np.linspace(-0.7, 1.1, length) if mapped else np.array(0.3)
  y, res = fn(values)
  np.testing.assert_allclose(y, values * values)
  expected = np.stack([values, 2.0 * values], axis=-1)
  np.testing.assert_allclose(res, expected.reshape(res.shape))
  calls = [node for node in walk_program(_lowered_main(fn)) if node.op == ProgramOp.CALL]
  assert len(calls) == 1
  assert calls[0].attrs["n_out"] == 2


def test_different_map_layouts_are_distinct_invocations() -> None:
  @sc.function(sc.arg("x", 2))
  def pair(x):
    return x.sin(), x.sum()

  @sc.function(sc.arg("xs", 9))
  def outer(xs):
    a, b = sc.vmap(pair, 3)(sc.window(xs, 0, 2))
    c, d = sc.vmap(pair, 3)(sc.window(xs, 1, 2))
    e, f = sc.vmap(pair, 3)(sc.window(xs, 0, 3))
    g, h = sc.vmap(pair, 2)(sc.window(xs, 0, 2))
    return a, b, c, d, e, f, g, h

  values = np.linspace(-0.8, 1.3, 9)
  got = outer(values)
  for i, (start, stride, length) in enumerate(((0, 2, 3), (1, 2, 3), (0, 3, 3), (0, 2, 2))):
    rows = np.stack([values[start + k * stride : start + k * stride + 2] for k in range(length)])
    np.testing.assert_allclose(got[2 * i], np.sin(rows))
    np.testing.assert_allclose(got[2 * i + 1], rows.sum(axis=1))
  assert sum(node.op == ProgramOp.CALL for node in walk_program(_lowered_main(outer))) == 4
