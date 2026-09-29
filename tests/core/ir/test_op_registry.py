"""The expression op registry: every builtin is registered through it, a registered name is what an
``Expr`` holds and interns by, a name registers once, and an unknown name is refused."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly.ad import jvp, jvp_many, vjp
from scaly.ir import program as p
from scaly.ir.expr import Expr, ExprOp, OpDef, define_traits, has_trait, op_def, register_op, registered_ops
from scaly.ir.expr_spec import verify_expr
from scaly.ir.program import ProgramOp, RangeKind
from scaly.ir.spec import Rule, VerifyError
from scaly.ir.types import TensorType
from scaly.passes.expr import simplify

# Registered at import, once per process, as an extension would: before any Expr uses it.
TOY = register_op("test_registry_toy", arity=1, numpy=lambda x: 2.0 * x)


def test_every_builtin_is_registered_in_enum_order() -> None:
  assert registered_ops()[: len(ExprOp)] == tuple(ExprOp)
  assert all(op_def(op).name is op for op in ExprOp)
  assert op_def("add") is op_def(ExprOp.ADD)


def test_an_extension_op_interns_by_its_registered_name() -> None:
  x = sc.sym("x", 3)
  by_name = Expr("test_registry_toy", (x,), x.type)
  assert Expr(TOY.name, (x,), x.type) is by_name
  assert by_name.op == "test_registry_toy" and isinstance(by_name.op, str)
  assert by_name.structural_key()[0] == "test_registry_toy"
  verify_expr(by_name)
  with pytest.raises(VerifyError, match="expects 1 args"):
    verify_expr(Expr(TOY.name, (x, x), x.type))


def test_a_builtin_built_by_name_is_the_same_node() -> None:
  x = sc.sym("x", 2)
  total = x + x
  assert Expr("add", (x, x), total.type) is total
  assert total.op is ExprOp.ADD


def test_a_name_registers_once_and_an_unknown_one_is_refused() -> None:
  with pytest.raises(ValueError, match="already registered"):
    register_op("add", arity=2)
  with pytest.raises(ValueError, match="already registered"):
    register_op("test_registry_toy", arity=1)
  with pytest.raises(ValueError, match="unknown expression op 'nope'"):
    Expr("nope", (), TensorType())
  assert isinstance(TOY, OpDef) and TOY.numpy is not None
  np.testing.assert_allclose(TOY.numpy(np.ones(2)), [2.0, 2.0])


# An op defined wholly outside the compiler, as a library would: ``x ** 3`` elementwise, with the
# rules a derivative, a sparsity query, verification and lowering ask for. It has no multi-seed rule,
# so ``jvp_many`` takes the per-seed default.
def _cube_lower(ctx: Any, node: Expr) -> None:
  out = ctx.alloc_tmp(node)
  name = f"i_{out.attrs['name']}"
  i = p.var(name)
  x = p.load(p.view(ctx.buf_of(node.args[0]), [i]))
  ctx.statements.append(p.for_(p.range_(name, 0, node.size, kind=RangeKind.GLOBAL), [p.store(p.view(out, [i]), p.mul(p.mul(x, x), x))]))


CUBE = register_op(
  "test_registry_cube",
  arity=1,
  numpy=lambda x: x**3,
  jvp=lambda e, d: 3.0 * (e.args[0] * e.args[0]) * d[0],
  vjp=lambda e, cot: (cot * 3.0 * (e.args[0] * e.args[0]),),
  sparsity=lambda e, mask, ncols: mask(e.args[0]),
  verify=(Rule(None, "cube-keeps-shape", lambda e: None if e.shape == e.args[0].shape else "shape changed"),),
  lower=_cube_lower,
)


def _cube(x: sc.Expr) -> sc.Expr:
  return Expr(CUBE.name, (x,), x.type)


def test_an_extension_op_differentiates_reports_sparsity_lowers_and_compiles() -> None:
  x = sc.sym("x", 3)
  seed = sc.sym("seed", 3)
  y = _cube(x)
  fn = sc.Function.from_exprs(
    "registry_cube",
    [x, seed],
    [y, jvp(y, x, seed), vjp([y.sum()], [x], [sc.const(1.0)])[0], jvp_many(y, x, sc.const(np.eye(3)))],
    ["x", "seed"],
    ["y", "fwd", "grad", "many"],
  )
  xv, sv = np.array([0.5, -1.0, 2.0]), np.array([1.0, 2.0, -1.0])
  value, forward, gradient, many = fn((xv, sv))
  np.testing.assert_allclose(value, xv**3)
  np.testing.assert_allclose(forward, 3 * xv**2 * sv)
  np.testing.assert_allclose(gradient, 3 * xv**2)
  np.testing.assert_allclose(many, np.diag(3 * xv**2))
  pattern = sc.jacobian_sparsity(y, x)
  assert list(zip(pattern.rows, pattern.cols, strict=True)) == [(0, 0), (1, 1), (2, 2)]
  np.testing.assert_allclose(sc.jacobian(sc.Function.from_exprs("registry_cube_y", [x], [y], ["x"], ["y"]), "y", "x")(xv), np.diag(3 * xv**2))
  folded = simplify(_cube(sc.const(np.array([2.0]))))
  assert folded.op == ExprOp.CONST and folded.value is not None and folded.value[0] == 8.0


def test_an_extension_op_is_verified_by_its_own_rules() -> None:
  x = sc.sym("x", 3)
  verify_expr(_cube(x))
  with pytest.raises(VerifyError, match="cube-keeps-shape"):
    verify_expr(Expr(CUBE.name, (x,), TensorType((2,))))


def test_an_op_without_a_pattern_rule_is_dense_in_what_it_reads() -> None:
  x = sc.sym("x", 4)
  toy = Expr(TOY.name, (x[1:3],), TensorType((2,)))
  pattern = sc.jacobian_sparsity(toy, x)
  assert list(zip(pattern.rows, pattern.cols, strict=True)) == [(0, 1), (0, 2), (1, 1), (1, 2)]


# An elementwise op that needs no lowering rule: its trait names the program op it computes.
HALF_SINE = register_op(
  "test_registry_half_sine",
  arity=1,
  numpy=lambda x: np.sin(x),
  jvp=lambda e, d: e.args[0].cos() * d[0],
  traits={"elementwise": ProgramOp.SIN, "expensive": True},
)


def test_an_elementwise_trait_lowers_the_op_and_colours_it() -> None:
  x = sc.sym("x", 3)
  fn = sc.Function.from_exprs("registry_half_sine", [x], [Expr(HALF_SINE.name, (x,), x.type)], ["x"], ["y"])
  xv = np.array([0.1, 0.2, 0.3])
  np.testing.assert_allclose(fn(xv), np.sin(xv))
  assert has_trait(HALF_SINE.name, "expensive") and not has_trait(TOY.name, "elementwise")


def test_traits_are_checked() -> None:
  with pytest.raises(ValueError, match="elementwise trait of expression op 'test_registry_half_sine' is already defined"):
    define_traits(HALF_SINE.name, elementwise=ProgramOp.COS)
  with pytest.raises(TypeError, match="unknown op trait 'shiny'"):
    define_traits(TOY.name, shiny=True)
