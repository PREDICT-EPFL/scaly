from __future__ import annotations

import numpy as np
import pytest

from scaly.ad.reverse import _gather_vjp, _unbroadcast
from scaly.ir.expr import ExprOp, topo

from scaly.function.model import as_concrete
import scaly as sc


def test_structural_transpose_concat_vec_eval_and_ad() -> None:
  @sc.function(sc.arg("x", (2, 2)), outputs=sc.arg("y"))
  def f(x):
    return sc.concat([x.T, x + 1.0], axis=1).vec()

  jf = sc.jacobian(f, "y", "x")
  xv = np.array([[1.0, 2.0], [3.0, 4.0]])

  np.testing.assert_allclose(f(xv), np.concatenate([xv.T, xv + 1.0], axis=1).reshape(8))
  np.testing.assert_allclose(
    jf(xv),
    np.array(
      [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ]
    ),
  )


def test_slice_split_eval_and_ad() -> None:
  @sc.function(sc.arg("x", 4), outputs=sc.arg("y"))
  def f(x):
    left, right = sc.split(x, [2, 2])
    return sc.stack([x[0], x[2:4].sum(), sc.concat([left, right])[3]])

  (x,) = as_concrete(f).inputs
  jf = sc.jacobian(f, "y", "x")
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  np.testing.assert_allclose(f(xv), np.array([1.0, 7.0, 4.0]))
  np.testing.assert_allclose(jf(xv), np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))

  try:
    _ = sc.split(x, [1, 2])
  except ValueError as e:
    assert "do not sum to axis length 4" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid split should fail")

  try:
    _ = sc.split(x, [-1, 5])
  except ValueError as e:
    assert "cannot contain negative entries" in str(e)
  else:  # pragma: no cover
    raise AssertionError("negative split size should fail")


def test_gather_scatter_eval_and_ad() -> None:
  @sc.function(sc.arg("x", 5), outputs=sc.arg("y"))
  def f(x):
    return sc.scatter(x.gather([3, 1, 4]), [0, 2, 3], 5)

  (x,) = as_concrete(f).inputs
  jf = sc.jacobian(f, "y", "x")
  xv = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

  np.testing.assert_allclose(f(xv), np.array([13.0, 0.0, 11.0, 14.0, 0.0]))
  np.testing.assert_allclose(
    jf(xv),
    np.array(
      [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
      ]
    ),
  )

  try:
    _ = sc.gather(x, [5])
  except IndexError as e:
    assert "indices must be in [0, 5)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("out-of-range gather should fail")


def _solver_call(parameter: sc.Expr) -> sc.Expr:
  from scaly.solvers.model import SolverDescriptor

  descriptor = SolverDescriptor("ad_test_solver", "test", parameter.size, 0, 0, (("p", parameter.shape),), (("y", parameter.shape),), (), 1)
  return sc.Expr(
    sc.ExprOp.SOLVER_CALL, (parameter,), sc.TensorType(parameter.shape, diff=False), attrs={"solver": descriptor, "output": 0, "output_name": "y"}
  )


@pytest.mark.parametrize("mode", ["jvp", "jvp_many", "vjp", "sparsity"])
@pytest.mark.parametrize("wrapped", ["direct", "call", "mapped"])
def test_active_solver_derivative_refused(mode, wrapped) -> None:
  from scaly.ad.forward import jvp, jvp_many
  from scaly.ad.reverse import vjp
  from scaly.ad.sparsity import jacobian_sparsity
  from scaly.function.concrete import ConcreteFunction

  x = sc.sym("solver_parameter", 2)
  result = _solver_call(x)
  if wrapped != "direct":
    fn = ConcreteFunction._from_exprs("opaque_test", [x], [result], ["x"], ["y"])
    if wrapped == "mapped":
      from scaly.function.sugar import _mapped_call

      result = _mapped_call(fn, 1, [(x, 0, 0)])
    else:
      result = fn(x * 2.0)
  with pytest.raises(NotImplementedError, match="SOLVER_CALL"):
    if mode == "jvp":
      jvp(result, x, sc.const(np.ones(2)))
    elif mode == "jvp_many":
      jvp_many(result, x, sc.const(np.eye(2)))
    elif mode == "vjp":
      vjp((result,), (x,), (sc.const(np.ones(2)),))
    else:
      jacobian_sparsity(result, x)


def test_inactive_solver_derivatives_remain_zero() -> None:
  from scaly.ad.forward import jvp
  from scaly.ad.reverse import vjp
  from scaly.ad.sparsity import jacobian_sparsity

  x, unrelated = sc.sym("solver_inactive", 2), sc.sym("unrelated", 2)
  result = _solver_call(x)
  np.testing.assert_array_equal(jvp(result, unrelated, sc.const(np.ones(2))).value, np.zeros(2))
  np.testing.assert_array_equal(jvp(result, x, sc.const(np.zeros(2))).value, np.zeros(2))
  np.testing.assert_array_equal(vjp((result,), (unrelated,), (sc.const(np.ones(2)),))[0].value, np.zeros(2))
  np.testing.assert_array_equal(vjp((result,), (x,), (sc.const(np.zeros(2)),))[0].value, np.zeros(2))
  assert jacobian_sparsity(result, unrelated).nnz == 0


@pytest.mark.parametrize("mapped", [False, True])
def test_solver_call_with_inactive_parameter_allows_other_derivatives(mapped) -> None:
  from scaly.ad.forward import jvp, jvp_many
  from scaly.ad.reverse import vjp
  from scaly.ad.sparsity import jacobian_sparsity
  from scaly.function.concrete import ConcreteFunction
  from scaly.function.sugar import _mapped_call

  p, q, x = sc.sym("inactive_p", 2), sc.sym("active_q", 2), sc.sym("outer_x", 2)
  solver = _solver_call(p)
  fn = ConcreteFunction._from_exprs("mixed_solver_test", [p, q], [solver + q], ["p", "q"], ["y"])
  constant = sc.const(np.ones(2))
  result = _mapped_call(fn, 1, [(constant, 0, 0), (x, 0, 0)]) if mapped else fn(constant, x)

  def evaluate(expr):
    derivative = ConcreteFunction._from_exprs(f"inactive_solver_{mapped}_{expr.id}", [x], [expr], ["x"], ["y"])
    return derivative(np.ones(2))

  np.testing.assert_array_equal(evaluate(jvp(result, x, constant)), np.ones(2))
  np.testing.assert_array_equal(evaluate(jvp_many(result, x, sc.const(np.eye(2)))), np.eye(2))
  np.testing.assert_array_equal(evaluate(vjp((result,), (x,), (constant,))[0]), np.ones(2))
  np.testing.assert_array_equal(jacobian_sparsity(result, x).to_mask(), np.eye(2, dtype=bool))


@pytest.mark.parametrize("scalar", [False, True])
@pytest.mark.parametrize("ids", [[4, 0, 2, 5, 1], [1, 3, 1, 1, 0], [2, 2, 2, 2, 2], [0, 1, 2, 0, 1]])
def test_scatter_accumulates_in_index_order(ids: list[int], scalar: bool) -> None:
  @sc.function(sc.arg("v", 5), outputs=sc.arg("y"))
  def f(v):
    out = sc.scatter(v, ids, 6)
    return out.scalar() if scalar else out

  for values in (np.array([1.5, -2.0, 3.0, 0.5, -1.0]), np.array([1e16, 1.0, -1e16, 2.0, 3.0]), np.array([-0.0, 0.0, -0.0, 0.0, -0.0])):
    expected = np.zeros(6)
    if len(set(ids)) == len(ids):
      expected[ids] = values
    else:
      np.add.at(expected, ids, values)
    result = np.asarray(f(values))
    np.testing.assert_array_equal(result, expected)
    if len(set(ids)) == len(ids):
      np.testing.assert_array_equal(result.view(np.uint64), expected.view(np.uint64))


def test_segment_sum_is_scatter_builder() -> None:
  v = sc.sym("v", 5)
  ids = [1, 3, 1, 1, 0]
  assert sc.segment_sum(v, ids, 6) is sc.scatter(v, ids, 6)


@pytest.mark.parametrize("n", [20, 2000])
def test_gather_adjoint_has_one_scatter(n: int) -> None:
  ids = np.arange(2 * n) % n
  cot = sc.sym("cot", (2, n))
  adj = _gather_vjp(cot, ids.reshape(2, n), (n,))
  assert adj.op == ExprOp.SCATTER
  assert len(topo((adj,))) == 2
  assert adj.attrs["indices"].size == 2 * n


def test_repeated_gather_reverse_adjoint_matches_numpy() -> None:
  ids = np.array([[3, 1, 3], [0, 3, 1]])

  @sc.function(sc.group(sc.arg("x", 5), sc.arg("cot", (2, 3))), outputs=sc.arg("y"))
  def f(inputs):
    x, cot = inputs
    return sc.vjp((sc.gather(x, ids),), (x,), (cot,))[0]

  values = np.array([[1e16, 2.0, -1e16], [4.0, 1.0, 3.0]])
  expected = np.zeros(5)
  np.add.at(expected, ids.reshape(-1), values.reshape(-1))
  np.testing.assert_array_equal(f((np.ones(5), values)), expected)


@pytest.mark.parametrize("shape,out_shape", [((1, 3), (4, 3)), ((2, 1), (2, 4)), ((3,), (2, 4, 3))])
def test_unbroadcast_uses_one_scatter(shape: tuple[int, ...], out_shape: tuple[int, ...]) -> None:
  cot = sc.sym("cot", out_shape)
  adj = _unbroadcast(cot, shape, out_shape)
  assert adj.op == ExprOp.SCATTER
  assert len(topo((adj,))) == 2

  @sc.function(sc.arg("cot", out_shape), outputs=sc.arg("y"))
  def f(cot):
    return _unbroadcast(cot, shape, out_shape)

  values = np.arange(np.prod(out_shape), dtype=float).reshape(out_shape)
  ids = np.broadcast_to(np.arange(np.prod(shape)).reshape(shape), out_shape).reshape(-1)
  expected = np.zeros(np.prod(shape))
  np.add.at(expected, ids, values.reshape(-1))
  np.testing.assert_array_equal(f(values), expected.reshape(shape))


def test_constant_scatter_accumulates() -> None:
  @sc.function(sc.arg("x", 1), outputs=sc.arg("y"))
  def f(x):
    return sc.scatter(sc.const([2.0, 5.0, -1.0]), [1, 1, 0], 3) + x

  np.testing.assert_array_equal(f(np.array([0.0])), [-1.0, 7.0, 0.0])
