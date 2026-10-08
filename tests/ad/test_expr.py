from __future__ import annotations

import numpy as np
import pytest

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

  try:
    _ = sc.scatter(sc.sym("v", 2), [1, 1], 3)
  except ValueError as e:
    assert "scatter indices must be unique" in str(e)
  else:  # pragma: no cover
    raise AssertionError("duplicate scatter should fail")


@pytest.mark.parametrize("mode", ["jvp", "jvp_many", "vjp", "sparsity"])
@pytest.mark.parametrize("wrapped", ["direct", "call", "mapped"])
def test_active_solver_derivative_refused(mode, wrapped, monkeypatch) -> None:
  from scaly.ad.forward import jvp, jvp_many
  from scaly.ad.reverse import vjp
  from scaly.ad.sparsity import jacobian_sparsity
  from scaly.function.concrete import ConcreteFunction
  from scaly.ir.types import TensorType

  x = sc.sym("solver_parameter", 2)
  result = sc.Expr(sc.ExprOp.SOLVER_CALL, (x,), TensorType((2,), diff=False), attrs={"output": 0})
  if wrapped != "direct":
    fn = ConcreteFunction._from_exprs("opaque_test", [x], [result], ["x"], ["y"])
    if wrapped == "mapped":
      from scaly.function.sugar import _mapped_call

      result = _mapped_call(fn, 1, [(x, 0, 0)])
    else:
      result = fn(x * 2.0)
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
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
  from scaly.ir.types import TensorType

  x, unrelated = sc.sym("solver_inactive", 2), sc.sym("unrelated", 2)
  result = sc.Expr(sc.ExprOp.SOLVER_CALL, (x,), TensorType((2,), diff=False), attrs={"output": 0})
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
  from scaly.ir.types import TensorType

  p, q, x = sc.sym("inactive_p", 2), sc.sym("active_q", 2), sc.sym("outer_x", 2)
  solver = sc.Expr(sc.ExprOp.SOLVER_CALL, (p,), TensorType((2,), diff=False), attrs={"output": 0})
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
