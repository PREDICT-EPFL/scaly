"""Conditional calls select one branch, including in nested loops and derivatives."""

import numpy as np
import pytest

import scaly as sc
from scaly.function.concrete import ConcreteFunction
from scaly.ir.expr import callees_of, invocation_key

I64 = sc.TensorType((), sc.dtypes.int64)
BOOL = sc.TensorType((), sc.dtypes.bool_)


@sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y", 3), sc.arg("cost", ())))
def square(x):
  return x * x, sc.sumsqr(x)


@sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y", 3), sc.arg("cost", ())))
def sine(x):
  return x.sin(), x.sum()


@sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y", 3), sc.arg("cost", ())))
def affine(x):
  return 2 * x + 1, 3 * x.sum()


def choose(dtype=I64, branches=(square, sine, affine)):
  @sc.function(sc.arg("which", ()), sc.arg("x", 3))
  def selected(which, x):
    return sc.cond(which.cast(dtype.dtype), branches, x)

  return selected


@pytest.mark.parametrize("which", [-9, 0, 1, 2, 20])
def test_integer_selection_clamps_and_groups_results(which):
  f = choose()
  x = np.array([0.4, -0.7, 1.2])
  expected = (square, sine, affine)[max(0, min(which, 2))](x)
  got = f(np.int64(which), x)
  for actual, ref in zip(got, expected, strict=True):
    np.testing.assert_allclose(actual, ref, rtol=1e-14)
  nodes = [n for n in f.instantiate().nodes if n.op == sc.ExprOp.COND]
  assert len(nodes) == 2 and invocation_key(nodes[0]) == invocation_key(nodes[1])
  assert len(callees_of(nodes[0])) == 3


@pytest.mark.parametrize("which", [False, True])
def test_boolean_false_selects_first_branch(which):
  x = np.array([0.4, -0.7, 1.2])
  got = choose(BOOL, (square, sine))(which, x)
  expected = (x * x, x @ x) if not which else (np.sin(x), x.sum())
  for actual, ref in zip(got, expected, strict=True):
    np.testing.assert_allclose(actual, ref, rtol=1e-14)


@pytest.mark.parametrize("which", [0, 1, 2])
def test_both_ad_modes_match_finite_differences(which):
  selector, x = sc.sym("selector"), sc.sym("x", 3)
  y, cost = sc.cond(selector.cast(sc.dtypes.int64), (square, sine, affine), x)
  seed = sc.const(np.array([0.3, -0.5, 0.7]))
  weight = sc.const(np.array([-0.2, 0.4, 0.8]))
  fwd = sc.jvp(y, x, seed)
  rev = sc.vjp((y, cost), (x,), (weight, sc.const(0.6)))[0]
  f = ConcreteFunction._from_exprs("cond_ad", (selector, x), (y, cost, fwd, rev), ("selector", "x"), ("y", "cost", "fwd", "rev"))
  xv, h = np.array([0.4, -0.7, 1.2]), 1e-6
  branch = (lambda z: (z * z, z @ z), lambda z: (np.sin(z), z.sum()), lambda z: (2 * z + 1, 3 * z.sum()))[which]
  assert seed.value is not None and weight.value is not None
  yp, _ = branch(xv + h * seed.value)
  ym, _ = branch(xv - h * seed.value)

  def objective(z):
    a, b = branch(z)
    return weight.value @ a + 0.6 * b

  grad = [(objective(xv + h * d) - objective(xv - h * d)) / (2 * h) for d in np.eye(3)]
  got = f(np.int64(which), xv)
  np.testing.assert_allclose(got[2], (yp - ym) / (2 * h), atol=1e-9)
  np.testing.assert_allclose(got[3], grad, atol=1e-9)


def test_cond_inside_scan_and_map():
  @sc.function(sc.arg("c", ()), sc.arg("x", ()))
  def add(c, x):
    return c + x

  @sc.function(sc.arg("c", ()), sc.arg("x", ()))
  def subtract(c, x):
    return c - x

  @sc.function(sc.arg("c", ()), sc.arg("x", ()))
  def step(c, x):
    cnext = sc.cond(x < 0, (add, subtract), c, x)
    return cnext, cnext

  @sc.function(sc.arg("xs", 5))
  def scan(xs):
    c, (ys,) = sc.scan(step, sc.const(0.0), (xs,), length=5)
    return c, ys

  xs = np.array([0.3, -0.7, 0.4, -0.2, 0.9])
  c, ys = scan(xs)
  np.testing.assert_allclose(ys, np.cumsum(np.abs(xs)))
  np.testing.assert_allclose(c, np.abs(xs).sum())
  got = sc.vmap(step, 5)(np.zeros(5), xs)
  np.testing.assert_allclose(got[0], np.abs(xs))


def test_support_and_sparsity_union_specialize_all_branches():
  @sc.function(sc.arg("x", 3))
  def left(x):
    return x * sc.const([1.0, 0.0, 0.0])

  @sc.function(sc.arg("x", 3))
  def right(x):
    return sc.stack([sc.const(0.0), x[0] + x[1], sc.const(0.0)])

  selector, x = sc.sym("s", dtype=sc.dtypes.int64), sc.sym("x", 3)
  y = sc.cond(selector, (left, right), x)
  assert y.support == frozenset({0, 1})
  np.testing.assert_array_equal(sc.jacobian_sparsity(y, x).to_mask(), [[1, 0, 0], [1, 1, 0], [0, 0, 0]])
  z = sc.cond(selector, (left, right), x * sc.const([0.0, 1.0, 0.0]))
  assert z.support == frozenset({1})
  np.testing.assert_array_equal(sc.jacobian_sparsity(z, x).to_mask(), [[0, 0, 0], [0, 1, 0], [0, 0, 0]])
  assert sc.jacobian_sparsity(y, selector).nnz == 0


@pytest.mark.parametrize("which", [0.0, 1.0])
def test_derivatives_of_a_mapped_conditional_with_multiple_operands(which):
  @sc.function(sc.arg("x", ()), sc.arg("a", ()))
  def product(x, a):
    return x * a

  @sc.function(sc.arg("x", ()), sc.arg("a", ()))
  def wave(x, a):
    return x.sin()

  @sc.function(sc.arg("s", ()), sc.arg("x", ()), sc.arg("a", ()))
  def branch(s, x, a):
    return sc.cond(s.cast(sc.dtypes.bool_), (product, wave), x, a)

  s, x, a = sc.sym("s"), sc.sym("x", 5), sc.sym("a", 5)
  y = sc.vmap(branch, 5)(sc.broadcast(s), x, a)
  seeds = sc.const(np.eye(5))
  forward = sc.jvp_many(y, x, seeds)
  gx, ga = sc.vjp((y,), (x, a), (sc.const(np.ones(5)),))
  fun = ConcreteFunction._from_exprs("mapped_cond_ad", [s, x, a], [y, forward, gx, ga], ["s", "x", "a"], ["y", "fwd", "gx", "ga"])
  xv, av = np.linspace(-0.5, 0.5, 5), np.linspace(1.0, 2.0, 5)
  result = fun(which, xv, av)
  slope = np.cos(xv) if which else av
  np.testing.assert_allclose(result[0], np.sin(xv) if which else xv * av)
  np.testing.assert_allclose(result[1], np.diag(slope))
  np.testing.assert_allclose(result[2], slope)
  np.testing.assert_allclose(result[3], np.zeros(5) if which else xv)


def test_zero_operand_and_single_branch_calls():
  @sc.function()
  def first():
    return sc.const(2.0)

  @sc.function()
  def second():
    return sc.const(3.0)

  @sc.function(sc.arg("s", ()))
  def f(s):
    return sc.cond(s.cast(sc.dtypes.int32), (first, second)), sc.cond(s.cast(sc.dtypes.int64), (first,))

  np.testing.assert_array_equal(f(np.array(-10.0)), [2.0, 2.0])
  np.testing.assert_array_equal(f(np.array(10.0)), [3.0, 2.0])
  assert "cond" in sc.__all__ and sc.cond.__doc__


@pytest.mark.parametrize(
  "selector,branches,match",
  [(0.0, (square, sine), "selector"), (True, (square, sine, affine), "two branches"), (0, (), "tuple"), (0, [square, sine], "tuple")],
)
def test_builder_rejects_invalid_selectors_and_branches(selector, branches, match):
  with pytest.raises((ValueError, TypeError), match=match):
    sc.cond(selector, branches, sc.sym("x", 3))


def test_verifier_rejects_mismatched_branch_results():
  from scaly.ir.spec import VerifyError

  @sc.function(sc.arg("x", 3))
  def bad(x):
    return x.sum(), x

  selector, x = sc.sym("selector", dtype=sc.dtypes.int64), sc.sym("x", 3)
  with pytest.raises(ValueError, match="matching output shapes"):
    sc.cond(selector, (square, bad), x)
  y = sc.Expr(sc.ExprOp.COND, (selector, x), x.type, attrs={"branches": (square.instantiate(), bad.instantiate()), "output": 0})
  with pytest.raises(VerifyError, match="outputs must match"):
    sc.verify_expr(y)


def test_sparse_branch_layouts_and_both_derivatives():
  from scaly.ir.types import CanonicalPattern

  pattern = CanonicalPattern((3, 3), (0, 1, 2, 2), (0, 1, 0, 2))
  type_ = sc.TensorType((3, 3), pattern=pattern)

  @sc.function(sc.arg("x", type_, layout="csr"), outputs=sc.arg("y", type_, layout="csc"))
  def squared(x):
    return x * x

  @sc.function(sc.arg("x", type_, layout="csc"), outputs=sc.arg("y", type_, layout="csr"))
  def doubled(x):
    return 2 * x

  s, x = sc.sym("s"), sc.sym("x", (3, 3))
  sx = sc.from_dense(x, pattern)
  y = sc.cond(s.cast(sc.dtypes.bool_), (squared, doubled), sx)
  seed = sc.const(np.ones((3, 3)))
  forward = sc.to_dense(sc.jvp(y, x, seed))
  reverse = sc.vjp((sc.to_dense(y),), (x,), (seed,))[0]
  fun = ConcreteFunction._from_exprs("sparse_cond", [s, x], [sc.to_dense(y), forward, reverse], ["s", "x"], ["y", "fwd", "rev"])
  xv = np.zeros((3, 3))
  xv[pattern.rows, pattern.cols] = [1.0, 2.0, 3.0, 4.0]
  for flag in (0.0, 1.0):
    got = fun(flag, xv)
    np.testing.assert_allclose(got[0], 2 * xv if flag else xv * xv)
    derivative = np.zeros((3, 3))
    derivative[pattern.rows, pattern.cols] = 2 if flag else 2 * xv[pattern.rows, pattern.cols]
    np.testing.assert_allclose(got[1], derivative)
    np.testing.assert_allclose(got[2], derivative)


def test_custom_branch_rules_preserve_derivatives_outside_primal_support():
  @sc.function(sc.arg("x", 2))
  def masked(x):
    return x * sc.const([0.0, 1.0])

  @sc.function(sc.arg("x", 2), sc.arg("dx", 2))
  def tangent(x, dx):
    return dx

  @sc.function(sc.arg("x", 2))
  def forward(x):
    return masked(x), x

  @sc.function(sc.arg("residual", 2), sc.arg("cot", 2))
  def backward(residual, cot):
    return cot

  ruled = sc.custom_derivative(masked, jvp=tangent, fwd=forward, bwd=backward, sparsity=lambda of, wrt, shape: np.eye(*shape, dtype=bool))
  selector, x = sc.sym("s"), sc.sym("x", 2)
  y = sc.cond(selector.cast(sc.dtypes.bool_), (masked, ruled), x)
  dy = sc.jvp(y, x, sc.const(np.ones(2)))
  dx = sc.vjp((y,), (x,), (sc.const(np.ones(2)),))[0]
  f = ConcreteFunction._from_exprs("custom_cond_ad", [selector, x], [y, dy, dx], ["s", "x"], ["y", "dy", "dx"])
  np.testing.assert_array_equal(sc.jacobian_sparsity(y, x).to_mask(), np.eye(2, dtype=bool))
  for flag in (0.0, 1.0):
    got = f(flag, np.array([2.0, 3.0]))
    np.testing.assert_array_equal(got[0], [0.0, 3.0])
    for result in got[1:]:
      np.testing.assert_array_equal(result, [1.0, 1.0] if flag else [0.0, 1.0])


def test_selected_multioutput_branch_runs_once(capfd):
  @sc.function(sc.arg("x", ()))
  def first(x):
    value = sc.print("first {}", x)
    return value + 1, value * value

  @sc.function(sc.arg("x", ()))
  def second(x):
    value = sc.print("second {}", x)
    return value - 1, value.sin()

  @sc.function(sc.arg("x", ()))
  def f(x):
    return sc.cond(x < 0, (first, second), x)

  np.testing.assert_allclose(f(np.array(2.0)), [3.0, 4.0])
  assert capfd.readouterr().out == "first 2\n"
  np.testing.assert_allclose(f(np.array(-2.0)), [-3.0, np.sin(-2.0)])
  assert capfd.readouterr().out == "second -2\n"


def test_only_selected_derivative_branch_recomputes_its_primal(capfd):
  @sc.function(sc.arg("x", ()))
  def first(x):
    return sc.print("square {}", x) ** 2

  @sc.function(sc.arg("x", ()))
  def second(x):
    return sc.print("log {}", x).log()

  @sc.function(sc.arg("x", ()))
  def derivatives(x):
    y = sc.cond(x < 0, (second, first), x)
    return sc.jvp(y, x, sc.const(1.0)), sc.vjp((y,), (x,), (sc.const(1.0),))[0]

  np.testing.assert_allclose(derivatives(np.array(-2.0)), [-4.0, -4.0])
  assert capfd.readouterr().out == "square -2\nsquare -2\n"
  np.testing.assert_allclose(derivatives(np.array(2.0)), [0.5, 0.5])
  assert capfd.readouterr().out == "log 2\nlog 2\n"


def test_forward_does_not_differentiate_an_unused_loop_result():
  @sc.function(sc.arg("x", ()))
  def step(x):
    return x + 1

  @sc.function(sc.arg("x", ()))
  def branch(x):
    final, _ = sc.scan(step, x, (), length=2)
    return x * x, final

  x = sc.sym("unused_loop_x")
  y, _ = sc.cond(x < 0, (branch, branch), x)
  dy = sc.jvp(y, x, sc.const(1.0))
  f = ConcreteFunction._from_exprs("cond_unused_loop", [x], [dy], ["x"], ["dy"])
  np.testing.assert_allclose(f(np.array(2.0)), [4.0])


def test_csc_input_is_canonical_before_either_branch():
  from scipy import sparse
  from scaly.ir.types import CanonicalPattern

  pattern = CanonicalPattern((3, 3), (0, 1, 2, 2), (0, 1, 0, 2))
  type_ = sc.TensorType((3, 3), pattern=pattern)

  @sc.function(sc.arg("x", type_, layout="csr"), outputs=sc.arg("y", type_, layout="csc"))
  def squared(x):
    return x * x

  @sc.function(sc.arg("x", type_, layout="csc"), outputs=sc.arg("y", type_, layout="csr"))
  def doubled(x):
    return 2 * x

  @sc.function(sc.arg("s", ()), sc.arg("x", type_, layout="csc"))
  def f(s, x):
    y = sc.cond(s.cast(sc.dtypes.bool_), (squared, doubled), x)
    return sc.to_dense(y), sc.to_dense(x) + sc.to_dense(y)

  x = sparse.csc_array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [3.0, 0.0, 4.0]])
  for selector in (1.0, 0.0, 1.0):
    expected = 2 * x.toarray() if selector else x.toarray() ** 2
    got = f(np.array(selector), x)
    np.testing.assert_array_equal(got[0], expected)
    np.testing.assert_array_equal(got[1], x.toarray() + expected)


@pytest.mark.parametrize("nested", [False, True])
def test_sparsity_of_two_conditional_calls_keeps_temporary_nodes_alive(nested):
  @sc.function(sc.arg("i", I64), sc.arg("p", 3), sc.arg("x", 3))
  def f(i, p, x):
    yp, _ = sc.cond(i, (square, sine), p)
    yx, _ = sc.cond(i, (square, sine), x)
    return sc.concat([yp, yx])

  i, p, x = sc.sym("two_cond_i", dtype=sc.dtypes.int64), sc.sym("two_cond_p", 3), sc.sym("two_cond_x", 3)
  y = f(i, p, x) if nested else f.instantiate().outputs[0]
  wrt = x if nested else f.instantiate().inputs[2]
  expected = np.vstack([np.zeros((3, 3), dtype=bool), np.eye(3, dtype=bool)])
  for _ in range(20):
    np.testing.assert_array_equal(sc.jacobian_sparsity(y, wrt).to_mask(), expected)


def test_both_dialects_render_conditional_calls():
  from scaly.passes.lowering.ctx import lower_function
  from scaly.ir.text import format_program

  f = choose().instantiate()
  expressions = sc.format_expr(f.outputs)
  assembly = sc.render_expr_assembly(f)
  program = lower_function(f)
  listing = format_program(program)
  program_assembly = sc.render_program_assembly(program)
  assert "cond [square, sine, affine]" in expressions
  assert 'branches=["square", "sine", "affine"]' in assembly
  assert "if " in listing and "else:" in listing
  assert "prog.if " in program_assembly and "} else {" in program_assembly
