"""Pin per-operation derivatives against NumPy differences and forward/reverse duality."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pytest
from scipy.special import erf

import scaly as sc
from scaly.function.model import as_concrete
from scaly.function.sugar import _mapped_call
from scaly.ir.expr import COMMON_ELEMENTWISE_BINARY, COMMON_ELEMENTWISE_UNARY, ExprOp
from scaly.solvers.model import SolverDescriptor


@dataclass(frozen=True)
class Case:
  name: str
  op: ExprOp
  values: tuple[np.ndarray, ...]
  build: Callable[..., sc.Expr]
  numpy: Callable[..., np.ndarray]


UNARY = {
  ExprOp.NEG: (lambda x: -x, np.negative),
  ExprOp.SIN: (lambda x: x.sin(), np.sin),
  ExprOp.COS: (lambda x: x.cos(), np.cos),
  ExprOp.TAN: (lambda x: x.tan(), np.tan),
  ExprOp.ASIN: (lambda x: x.asin(), np.arcsin),
  ExprOp.ACOS: (lambda x: x.acos(), np.arccos),
  ExprOp.ATAN: (lambda x: x.atan(), np.arctan),
  ExprOp.SINH: (lambda x: x.sinh(), np.sinh),
  ExprOp.COSH: (lambda x: x.cosh(), np.cosh),
  ExprOp.TANH: (lambda x: x.tanh(), np.tanh),
  ExprOp.ERF: (lambda x: x.erf(), erf),
  ExprOp.EXP: (lambda x: x.exp(), np.exp),
  ExprOp.LOG: (lambda x: x.log(), np.log),
  ExprOp.SQRT: (lambda x: x.sqrt(), np.sqrt),
  ExprOp.ABS: (lambda x: x.abs(), np.abs),
}
BINARY = {
  ExprOp.ADD: (lambda x, y: x + y, np.add),
  ExprOp.SUB: (lambda x, y: x - y, np.subtract),
  ExprOp.MUL: (lambda x, y: x * y, np.multiply),
  ExprOp.DIV: (lambda x, y: x / y, np.divide),
  ExprOp.POW: (lambda x, y: x**y, np.power),
  ExprOp.ATAN2: (sc.atan2, np.arctan2),
}
REFUSED = {
  ExprOp.FLOOR: lambda x: x.floor(),
  ExprOp.CEIL: lambda x: x.ceil(),
  ExprOp.MINIMUM: lambda x: sc.minimum(x, 0.5),
  ExprOp.MAXIMUM: lambda x: sc.maximum(x, 0.5),
}
STRUCTURAL_REFUSED = {ExprOp.ASIN, ExprOp.ACOS, ExprOp.ATAN, ExprOp.ATAN2, ExprOp.ABS}


@sc.function(sc.arg("differential_x", 2), outputs=sc.arg("y"), name="differential_piece")
def _piece(x):
  return x.sin() + x * x


def _scatter_numpy(x):
  result = np.zeros((2, 3))
  np.add.at(result.reshape(-1), [4, 1, 4, 0, 1, 5], x.reshape(-1))
  return result


def _cases() -> list[Case]:
  cases = []
  for op, (build, reference) in UNARY.items():
    for layout, shape in [("scalar", ()), ("tensor", (2, 3)), ("empty", (2, 0))]:
      values = np.linspace(0.2, 0.7, int(np.prod(shape))).reshape(shape)
      if op == ExprOp.ABS and values.size > 1:
        values.reshape(-1)[::2] *= -1
      cases.append(Case(f"{op}-{layout}", op, (values,), build, reference))
  for op, (build, reference) in BINARY.items():
    for layout, shapes in [
      ("scalars", ((), ())),
      ("scalar-left", ((), (2, 3))),
      ("scalar-right", ((2, 3), ())),
      ("unequal-ranks", ((3,), (2, 1, 3))),
      ("broadcast-axes", ((2, 1), (3,))),
      ("empty", ((2, 0), (1, 0))),
    ]:
      values = tuple(np.linspace(0.6 + i, 1.1 + i, int(np.prod(shape))).reshape(shape) for i, shape in enumerate(shapes))
      cases.append(Case(f"{op}-{layout}", op, values, build, reference))
    cases.append(Case(f"{op}-repeated", op, (np.array([0.6, 0.9, 1.3]),), lambda x, f=build: f(x, x), lambda x, f=reference: f(x, x)))
  for shape in [(3,), (2, 3), (2, 0)]:
    x = np.linspace(0.2, 0.7, int(np.prod(shape))).reshape(shape)
    label = "x".join(map(str, shape))
    cases.extend(
      [
        Case(f"input-{label}", ExprOp.INPUT, (x,), lambda x: x, lambda x: x),
        Case(f"const-{label}", ExprOp.CONST, (x,), lambda x: sc.const(np.full(x.shape, 0.4)), lambda x: np.full(x.shape, 0.4)),
        Case(f"sum-{label}", ExprOp.SUM, (x,), lambda x: x.sum(), np.sum),
        Case(f"reshape-{label}", ExprOp.RESHAPE, (x,), lambda x: x.reshape((x.size,)), lambda x: x.reshape(-1)),
      ]
    )
  x = np.arange(6.0).reshape(2, 3) / 7
  cases.extend(
    [
      Case("slice-strided", ExprOp.SLICE, (x,), lambda x: x[1, ::2], lambda x: x[1, ::2]),
      Case("slice-negative", ExprOp.SLICE, (x,), lambda x: x[::-1, ::-1], lambda x: x[::-1, ::-1]),
      Case("slice-empty", ExprOp.SLICE, (x,), lambda x: x[:0, :], lambda x: x[:0, :]),
      Case("slice-scalar", ExprOp.SLICE, (x,), lambda x: x[1, 2], lambda x: x[1, 2]),
      Case("gather-repeated", ExprOp.GATHER, (x,), lambda x: sc.gather(x, [[4, 1], [4, 0]]), lambda x: x.reshape(-1)[[[4, 1], [4, 0]]]),
      Case("gather-empty", ExprOp.GATHER, (x,), lambda x: sc.gather(x, np.empty((0, 2), dtype=int)), lambda x: np.empty((0, 2))),
      Case("scatter-repeated", ExprOp.SCATTER, (x,), lambda x: sc.scatter(x, [[4, 1, 4], [0, 1, 5]], (2, 3)), _scatter_numpy),
      Case("scatter-empty", ExprOp.SCATTER, (np.empty((0,)),), lambda x: sc.scatter(x, [], (2, 3)), lambda x: np.zeros((2, 3))),
    ]
  )
  for rank in [1, 2, 3, 4]:
    shape = tuple(range(2, rank + 2))
    value = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10
    axes = tuple(reversed(range(rank)))
    cases.append(
      Case(f"transpose-rank{rank}", ExprOp.TRANSPOSE, (value,), lambda x, axes=axes: x.transpose(axes), lambda x, axes=axes: x.transpose(axes))
    )
  for shape in [(), (2, 0, 3), (2, 3, 4)]:
    value = np.arange(np.prod(shape), dtype=float).reshape(shape) / 10
    axes = () if not shape else (1, 2, 0)
    cases.append(
      Case(f"transpose-cycle-{shape}", ExprOp.TRANSPOSE, (value,), lambda x, axes=axes: x.transpose(axes), lambda x, axes=axes: x.transpose(axes))
    )
  for axis in [0, 1, 2]:
    cases.append(
      Case(
        f"stack-axis{axis}", ExprOp.STACK, (x,), lambda x, axis=axis: sc.stack([x, x], axis=axis), lambda x, axis=axis: np.stack([x, x], axis=axis)
      )
    )
  for axis in [0, 1]:
    cases.append(
      Case(
        f"concat-axis{axis}",
        ExprOp.CONCAT,
        (x,),
        lambda x, axis=axis: sc.concat([x, x], axis=axis),
        lambda x, axis=axis: np.concatenate([x, x], axis=axis),
      )
    )
  for shape in [(), (2, 0)]:
    value = np.zeros(shape)
    cases.append(Case(f"stack-{shape}", ExprOp.STACK, (value,), lambda x: sc.stack([x, x]), lambda x: np.stack([x, x])))
  for axis in [0, 1]:
    for shape in [(2, 3), (2, 0)]:
      values = (np.full(shape, 0.4), np.full(shape, -0.2))
      cases.extend(
        [
          Case(
            f"stack-distinct-{axis}-{shape}",
            ExprOp.STACK,
            values,
            lambda x, y, axis=axis: sc.stack([x, y], axis=axis),
            lambda x, y, axis=axis: np.stack([x, y], axis=axis),
          ),
          Case(
            f"concat-distinct-{axis}-{shape}",
            ExprOp.CONCAT,
            values,
            lambda x, y, axis=axis: sc.concat([x, y], axis=axis),
            lambda x, y, axis=axis: np.concatenate([x, y], axis=axis),
          ),
        ]
      )
  for name, left, right in [("vv", (3,), (3,)), ("mv", (2, 3), (3,)), ("vm", (3,), (3, 2)), ("mm", (2, 3), (3, 2)), ("empty", (2, 0), (0, 3))]:
    values = tuple(np.arange(np.prod(shape), dtype=float).reshape(shape) / 5 + 0.4 for shape in (left, right))
    cases.append(Case(f"matmul-{name}", ExprOp.MATMUL, values, lambda x, y: x @ y, np.matmul))
  for shape in [(3,), (2, 2)]:
    values = np.arange(np.prod(shape), dtype=float).reshape(shape) / 5 + 0.4
    cases.append(Case(f"matmul-self-rank{len(shape)}", ExprOp.MATMUL, (values,), lambda x: x @ x, lambda x: x @ x))
  cases.extend(
    [
      Case("pow-negative-base-constant-exponent", ExprOp.POW, (np.array([-0.5, -1.4]),), lambda x: x**3, lambda x: x**3),
      Case("call", ExprOp.CALL, (np.array([0.2, -0.7]),), lambda x: _piece(x), lambda x: np.sin(x) + x * x),
      Case("vmap", ExprOp.VMAP, (np.linspace(0.1, 0.8, 6),), lambda x: _mapped_call(_piece, 3, [(x, 0, 2)]), lambda x: np.sin(x) + x * x),
    ]
  )
  return cases


CASES = _cases()


def _structural_refusal(case: Case) -> bool:
  return case.op in STRUCTURAL_REFUSED or case.name == "transpose-rank4"


@dataclass
class Evaluation:
  case: Case
  primal: sc.Function
  single: np.ndarray
  many: dict[int, np.ndarray]
  seeds: tuple[np.ndarray, ...]
  cotangent: np.ndarray
  adjoints: tuple[np.ndarray, ...]
  baked: tuple[np.ndarray, np.ndarray]


@pytest.fixture(scope="module", params=CASES, ids=lambda case: case.name)
def evaluated(request):
  case = request.param
  rng = np.random.default_rng(153)
  seeds = tuple(rng.normal(size=(3, *v.shape)) for v in case.values)
  for seed in seeds:
    seed[1] = 0
  shape = np.asarray(case.numpy(*case.values)).shape
  cotangent = rng.normal(size=shape)
  inputs = sc.group(*(sc.arg(f"differential_arg{i}", v.shape) for i, v in enumerate(case.values)))

  @sc.function(inputs, outputs=sc.arg("y"), name=f"primal_{case.name}")
  def primal(args):
    return case.build(*args)

  @sc.function(
    sc.group(inputs, sc.group(*(sc.arg(f"seed{i}", s.shape) for i, s in enumerate(seeds)))),
    outputs=sc.group(
      *(sc.arg(name) for name in ("single", "many0", "many1", "many3", "baked_single", "baked_many", *(f"adj{i}" for i in range(len(seeds)))))
    ),
    name=f"products_{case.name}",
  )
  def products(args):
    values, directions = args
    y = case.build(*values)
    assert y.op == case.op
    singles = []
    for row in range(3):
      terms = [sc.jvp(y, x, seed[row]) for x, seed in zip(values, directions, strict=True)]
      singles.append(sum(terms[1:], terms[0]))
    many = []
    for count in (0, 1, 3):
      terms = [sc.jvp_many(y, x, seed[:count]) for x, seed in zip(values, directions, strict=True)]
      many.append(sum(terms[1:], terms[0]))
    constants = tuple(sc.const(seed) for seed in seeds)
    baked_rows = []
    for row in range(3):
      terms = [sc.jvp(y, x, sc.const(seed[row])) for x, seed in zip(values, seeds, strict=True)]
      baked_rows.append(sum(terms[1:], terms[0]))
    baked_terms = [sc.jvp_many(y, x, seed) for x, seed in zip(values, constants, strict=True)]
    baked_many = sum(baked_terms[1:], baked_terms[0])
    adjoints = sc.vjp((y,), values, (sc.const(cotangent),))
    return (sc.stack(singles), *many, sc.stack(baked_rows), baked_many, *adjoints)

  with pytest.MonkeyPatch.context() as patch:
    patch.setenv("SCALY_STRICT_JVP_MANY", "0" if _structural_refusal(case) else "1")
    run: Any = products
    result = run((case.values, seeds))
  return Evaluation(
    case,
    primal,
    np.asarray(result[0]),
    dict(zip((0, 1, 3), map(np.asarray, result[1:4]), strict=True)),
    seeds,
    cotangent,
    tuple(result[6:]),
    (np.asarray(result[4]), np.asarray(result[5])),
  )


def _assert_differential(evaluation: Evaluation, tangent: np.ndarray) -> None:
  case = evaluation.case
  np.testing.assert_allclose(evaluation.primal(case.values), case.numpy(*case.values), atol=1e-12, rtol=1e-12)
  expected = []
  step = 1e-5
  for row in range(tangent.shape[0]):
    plus = tuple(x + step * seed[row] for x, seed in zip(case.values, evaluation.seeds, strict=True))
    minus = tuple(x - step * seed[row] for x, seed in zip(case.values, evaluation.seeds, strict=True))
    expected.append((np.asarray(case.numpy(*plus)) - np.asarray(case.numpy(*minus))) / (2 * step))
  reference = np.stack(expected) if expected else np.empty(tangent.shape)
  assert tangent.shape == reference.shape
  np.testing.assert_allclose(tangent, reference, atol=2e-8, rtol=2e-7)
  lhs = np.sum(tangent * evaluation.cotangent, axis=tuple(range(1, tangent.ndim))) if tangent.ndim > 1 else tangent * evaluation.cotangent
  rhs = sum(
    np.sum(seed[: tangent.shape[0]] * adj, axis=tuple(range(1, seed.ndim))) if seed.ndim > 1 else seed[: tangent.shape[0]] * adj
    for seed, adj in zip(evaluation.seeds, evaluation.adjoints, strict=True)
  )
  np.testing.assert_allclose(lhs, rhs, atol=1e-11, rtol=1e-11)


def test_single_seed_differences_and_duality(evaluated):
  _assert_differential(evaluated, evaluated.single)


@pytest.mark.parametrize("nseed", [0, 1, 3])
def test_many_seed_differences_and_duality(evaluated, nseed):
  assert evaluated.many[nseed].shape == (nseed, *np.asarray(evaluated.case.numpy(*evaluated.case.values)).shape)
  _assert_differential(evaluated, evaluated.many[nseed])


def test_baked_seeds_differences_and_duality(evaluated):
  for tangent in evaluated.baked:
    _assert_differential(evaluated, tangent)


def test_migration_structural_matches_stacked_single(evaluated):
  for count in (0, 1, 3):
    np.testing.assert_allclose(evaluated.many[count], evaluated.single[:count], atol=1e-12, rtol=1e-12)


def test_every_expr_op_is_accounted_for():
  assert set(UNARY) | {ExprOp.FLOOR, ExprOp.CEIL} == COMMON_ELEMENTWISE_UNARY
  assert set(BINARY) | {ExprOp.MINIMUM, ExprOp.MAXIMUM} == COMMON_ELEMENTWISE_BINARY
  assert {case.op for case in CASES} | set(REFUSED) | {ExprOp.SOLVER_CALL} == set(ExprOp)


@pytest.mark.parametrize("case", [case for case in CASES if _structural_refusal(case)], ids=lambda case: case.name)
def test_structural_refusals(case, monkeypatch):
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  inputs = tuple(sc.sym(f"refused{i}", value.shape) for i, value in enumerate(case.values))
  y = case.build(*inputs)
  for x in inputs:
    with pytest.raises(NotImplementedError, match=f"{case.op.value}.*forbids the unrolled fallback"):
      sc.jvp_many(y, x, sc.sym("refused_seeds", (3, *x.shape)))
    assert sc.jvp_many(y, x, sc.const(np.empty((0, *x.shape)))).shape == (0, *y.shape)


@pytest.mark.parametrize(("op", "point"), [(op, point) for op in REFUSED for point in (0.5, 0.7, 1.0)] + [(ExprOp.SOLVER_CALL, None)])
@pytest.mark.parametrize("mode", ["single", "many", "reverse"])
def test_active_derivative_refusals(op, mode, point, monkeypatch):
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  x = sc.sym("refused_x", ())
  if op == ExprOp.SOLVER_CALL:
    descriptor = SolverDescriptor("differential_solver", "test", 1, 0, 0, (("p", ()),), (("y", ()),), (), 1)
    y = sc.Expr(op, (x,), sc.TensorType((), diff=False), attrs={"solver": descriptor, "output": 0, "output_name": "y"})
  else:

    @sc.function(sc.arg("refused_x", ()), outputs=sc.arg("y"))
    def primal(x):
      return REFUSED[op](x)

    reference = {
      ExprOp.FLOOR: np.floor,
      ExprOp.CEIL: np.ceil,
      ExprOp.MINIMUM: lambda x: np.minimum(x, 0.5),
      ExprOp.MAXIMUM: lambda x: np.maximum(x, 0.5),
    }
    np.testing.assert_equal(primal(np.array(point)), reference[op](point))
    (x,), (y,) = as_concrete(primal).inputs, as_concrete(primal).outputs
  with pytest.raises(NotImplementedError, match="SOLVER_CALL" if op == ExprOp.SOLVER_CALL else op.value):
    if mode == "single":
      sc.jvp(y, x, sc.const(1.0))
    elif mode == "many":
      sc.jvp_many(y, x, sc.const([1.0, -0.4]))
    else:
      sc.vjp((y,), (x,), (sc.const(1.0),))


def test_abs_zero_uses_x_over_abs_x_convention():
  @sc.function(sc.arg("abs_x", 3), outputs=sc.group(sc.arg("fwd"), sc.arg("rev")))
  def products(x):
    y = x.abs()
    return sc.jvp(y, x, sc.const([1.0, 1.0, 1.0])), sc.vjp((y,), (x,), (sc.const([1.0, 1.0, 1.0]),))[0]

  for result in products(np.array([-0.7, 0.0, 0.6])):
    np.testing.assert_allclose(result, [-1.0, np.nan, 1.0], equal_nan=True)


@pytest.mark.parametrize("op", [ExprOp.SQRT, ExprOp.LOG, ExprOp.POW, ExprOp.ATAN2])
def test_singular_boundary_conventions(op, monkeypatch):
  """Pin the nonfinite entries at points where the derivative is undefined."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "0" if op == ExprOp.ATAN2 else "1")

  @sc.function(sc.group(sc.arg("boundary_x", 2), sc.arg("boundary_p", 2)), outputs=sc.group(sc.arg("single"), sc.arg("many"), sc.arg("reverse")))
  def products(inputs):
    x, p = inputs
    y = {
      ExprOp.SQRT: lambda: x.sqrt(),
      ExprOp.LOG: lambda: x.log(),
      ExprOp.POW: lambda: x**0.5,
      ExprOp.ATAN2: lambda: sc.atan2(x, p - 2.0),
    }[op]()
    rows = [sc.const(row) for row in np.eye(2)]
    return (
      sc.stack([sc.jvp(y, x, row) for row in rows]).T,
      sc.jvp_many(y, x, sc.const(np.eye(2))).T,
      sc.stack([sc.vjp((y,), (x,), (row,))[0] for row in rows]),
    )

  singular, smooth = (np.nan, 0.0) if op == ExprOp.ATAN2 else (np.inf, 1.0 if op == ExprOp.LOG else 0.5)
  single, many, reverse = products((np.array([0.0, 1.0]), np.array([2.0, 2.0])))
  for forward in (single, many):
    np.testing.assert_equal(forward, [[singular, np.nan], [0.0, smooth]])
  np.testing.assert_equal(reverse, [[singular, 0.0], [np.nan, smooth]])
