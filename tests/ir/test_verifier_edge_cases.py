"""``verify_expr`` on the ops added with the control, indexing, factorization and loop work.

Every graph the builders make verifies, and so does every derivative AD builds from it (forward,
multi-seed forward, reverse, Jacobian, Hessian) and every rewrite of it, callee bodies included:
AD and the rewrites build new nodes of these ops, and the verifier is the check that they are
well formed. Each op's rule then rejects a node forged with one thing wrong, naming itself.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

import scaly as sc
from scaly import linalg
from scaly.ad.derivatives import hessian, jacobian
from scaly.ad.forward import jvp, jvp_many
from scaly.ad.reverse import vjp
from scaly.ir.expr import Expr, ExprOp, callees_of, ragged_add, ragged_dot, sparse_ldl_factor, sparse_ldl_solve, topo
from scaly.ir.expr_spec import spec_expr, verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import TensorType, dtypes
from scaly.linalg import SparseLDL, SparseMatrix

NEW_OPS = frozenset(
  {
    ExprOp.ATAN2,
    ExprOp.COPYSIGN,
    ExprOp.LT,
    ExprOp.LE,
    ExprOp.EQ,
    ExprOp.NE,
    ExprOp.AND,
    ExprOp.OR,
    ExprOp.NOT,
    ExprOp.ISFINITE,
    ExprOp.SELECT,
    ExprOp.CAST,
    ExprOp.MAX,
    ExprOp.MIN,
    ExprOp.SEGMENT_MAX,
    ExprOp.SEGMENT_MIN,
    ExprOp.INDEX_ADD,
    ExprOp.INDEX_SET,
    ExprOp.TAKE,
    ExprOp.PUT_ADD,
    ExprOp.PUT,
    ExprOp.RAGGED_ADD,
    ExprOp.RAGGED_DOT,
    ExprOp.CHOLESKY,
    ExprOp.LDL,
    ExprOp.LU,
    ExprOp.SPARSE_LDL,
    ExprOp.SPARSE_LDL_SOLVE,
    ExprOp.TRISOLVE,
    ExprOp.VMAP,
    ExprOp.SCAN,
    ExprOp.WHILE,
  }
)


def _fn(name: str, inputs: list[Expr], outputs: list[Expr]) -> sc.ConcreteFunction:
  return sc.Function._from_exprs(name, inputs, outputs, [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _i64(values: Any) -> Expr:
  return sc.const(np.asarray(values, dtype=np.int64), dtype="int64")


def _spd(x: Expr, n: int) -> Expr:
  a = x.reshape((n, n))
  return a @ a.T + sc.const(float(n) * np.eye(n))


def _elementwise() -> tuple[Expr, Expr]:
  x = sc.sym("x", 4)
  y = sc.atan2(x, x * x + 1.0) + sc.copysign(x, x - 0.5)
  flag = sc.logical_or(sc.logical_and(x <= 1.0, sc.not_equal(x, 0.25)), sc.logical_not(sc.isfinite(x) & sc.equal(x, 2.0)))
  y = sc.where(flag, y, x.sin()) + sc.cast(x < 0.0, "float64")
  # Float32 arithmetic is kept to ``sin``: most derivative rules build float64 zeros and constants.
  return x, sc.cast(sc.cast(y, "float32").sin(), "float64") + sc.maximum(x, 0.5) * sc.minimum(x, -0.5)


def _reductions() -> tuple[Expr, Expr]:
  x = sc.sym("x", 5)
  ids = [0, 2, 0, 2, 1]
  seg = sc.segment_max(x * x, ids, 4) + sc.segment_min(x, ids, 4, fill=0.0)
  return x, seg * (x.max() - x.min()) + sc.norm_inf(x)


def _static_updates() -> tuple[Expr, Expr]:
  x = sc.sym("x", 6)
  y = sc.index_add(x * x, [0, 0, 5], x[:3] * 2.0)
  return x, sc.index_set(y, [1, 4], y[2:4].sin())


def _runtime_updates() -> tuple[Expr, Expr]:
  x = sc.sym("x", (2, 5))
  idx, into = _i64([4, 1, 1, -1]), _i64([0, 3, 7])
  y = sc.put_add(x, idx, sc.take(x, idx, fill=0.5) * 2.0)
  return x, sc.put(y, into, sc.take(y, into).cos())


def _ragged() -> tuple[Expr, Expr]:
  x = sc.sym("x", 6)
  lo, hi = _i64([0, 2, 2, 1]), _i64([2, 2, 5, 3])
  y = ragged_add(x, x * x, lo, hi, x[:4], dst_map=[5, 0, 3, 3, 1], src_map=[4, 3, 2, 1, 0])
  return x, sc.concat([y, ragged_dot(x, y, lo, hi)])


def _dense() -> tuple[Expr, Expr]:
  x = sc.sym("x", 9)
  a, b = _spd(x, 3), x[:3]
  chol = linalg.cho_solve(linalg.cholesky(a), b)
  sym = linalg.ldl_solve(linalg.ldl(a), b)
  tri = linalg.solve_triangular(a, x.reshape((3, 3)), lower=False, trans=True, unit_diagonal=True)
  return x, sc.concat([chol, sym, linalg.solve(a, b, assume="gen"), tri.reshape((9,))])


def _dense_looped() -> tuple[Expr, Expr]:
  with sc.options(dense_unroll=0):
    return _dense()


def _lu_factor() -> tuple[Expr, Expr]:
  # The factorization refuses a derivative of its own (``linalg.solve`` differentiates implicitly),
  # so this graph only checks the value side.
  x = sc.sym("x", 9)
  f = linalg.lu(x.reshape((3, 3)))
  return x, linalg.lu_solve(f, x[:3], trans=True)


PATTERN = np.array([[4.0, 0, 0, 0], [1.0, 5.0, 0, 0], [0, 1.0, 6.0, 0], [1.0, 0, 1.0, 7.0]])


def _sparse() -> tuple[Expr, Expr]:
  mat = SparseMatrix.symbol("kv", PATTERN)
  fact = SparseLDL(mat, schedule="loop", name="ve_sldl")
  b = sc.sym("b", 4)
  return mat.values, fact.solve(b * mat.values[:4])


def _sparse_primitives() -> tuple[Expr, Expr]:
  mat = SparseMatrix.symbol("kp", PATTERN)
  fact = SparseLDL(mat, schedule="loop", name="ve_prim")
  factor = sparse_ldl_factor(mat.values, fact.tables())
  return mat.values, sc.concat([sparse_ldl_solve(factor, mat.values[:4], fact.solve_tables()), fact.values])


def _loops() -> tuple[Expr, Expr]:
  x = sc.sym("x", 6)
  c, u, k = sc.sym("c", 2), sc.sym("u", 2), sc.sym("k", (), dtype="int64")
  step = _fn("ve_step", [c, k, u], [c * u.sin() + sc.cast(k, "float64"), c.sum().reshape((1,))])
  final, ys = sc.scan(step, x[:2], [(x, 0, 2)], length=3, index=True)
  p = sc.sym("p", 2)
  halve = _fn("ve_halve", [c, p], [c * 0.5 + p * 0.1])
  go = _fn("ve_go", [c, p], [sc.norm_inf(c) > p[0]])
  done, count = sc.while_loop(go, halve, final, max_iter=6, params=[x[4:]])
  lane = _fn("ve_lane", [u], [(u * u).sum()])
  mapped = sc.vmap(lane, length=3, inputs=[(x, 0, 2)])
  return x, sc.concat([done, ys, mapped, count.reshape((1,))])


CORPUS: dict[str, Callable[[], tuple[Expr, Expr]]] = {
  "elementwise": _elementwise,
  "reductions": _reductions,
  "static_updates": _static_updates,
  "runtime_updates": _runtime_updates,
  "ragged": _ragged,
  "dense": _dense,
  "dense_looped": _dense_looped,
  "lu_factor": _lu_factor,
  "sparse": _sparse,
  "sparse_primitives": _sparse_primitives,
  "loops": _loops,
}
VALUE_ONLY = {"lu_factor", "sparse_primitives"}


def _verify_deep(outputs: list[Expr]) -> set[ExprOp]:
  """Verify ``outputs`` and every callee body they run, transitively; return the ops seen."""
  ops: set[ExprOp] = set()
  seen: set[int] = set()
  pending = [tuple(outputs)]
  while pending:
    outs = pending.pop()
    verify_expr(outs)
    for node in topo(outs):
      ops.add(ExprOp(node.op))
      for callee in callees_of(node):
        if id(callee) not in seen:
          seen.add(id(callee))
          pending.append(tuple(callee.outputs))
  return ops


def _derivatives(x: Expr, y: Expr) -> list[Expr]:
  flat = y.reshape((y.size,))
  seed, seeds, ct = sc.sym("seed", x.shape), sc.sym("seeds", (3, *x.shape)), sc.sym("ct", y.shape)
  scalar = (flat * flat.sin()).sum()
  return [jvp(y, x, seed), jvp_many(y, x, seeds), *vjp((y,), (x,), (ct,)), jacobian(flat, x), hessian(scalar, x)]


@pytest.mark.parametrize("name", list(CORPUS))
def test_graphs_and_rewrites_verify(name: str) -> None:
  x, y = CORPUS[name]()
  _verify_deep([y])
  _verify_deep([sc.simplify(y)])
  _verify_deep(list(sc.cse_many([y, y * 2.0])))


@pytest.mark.parametrize("name", sorted(set(CORPUS) - VALUE_ONLY))
def test_every_derivative_verifies(name: str) -> None:
  x, y = CORPUS[name]()
  _verify_deep(_derivatives(x, y))


@pytest.mark.parametrize("nonsmooth", ["split", "first"])
def test_tie_conventions_build_verified_derivatives(nonsmooth: str) -> None:
  with sc.options(nonsmooth=nonsmooth):
    x, y = _reductions()
    _verify_deep(_derivatives(x, sc.maximum(y, x[:4]) + sc.minimum(x[:4], 0.0)))


def test_the_corpus_reaches_every_new_op() -> None:
  """A new op must join the corpus above, or its builders and AD rules go unverified here."""
  seen: set[ExprOp] = set()
  for name, build in CORPUS.items():
    x, y = build()
    seen |= _verify_deep([y] if name in VALUE_ONLY else [y, *_derivatives(x, y)])
  assert NEW_OPS <= seen, sorted(op.name for op in NEW_OPS - seen)


def test_every_op_has_a_rule_of_its_own() -> None:
  """``docs/dev/codebase.md``: adding an op includes a verify rule in ``ir/expr_spec.py``."""
  unchecked = {ExprOp.SLICE, ExprOp.EXTERN_CALL}
  assert {op for op in ExprOp if op not in spec_expr.op_rules()} == unchecked


# --- forged nodes --------------------------------------------------------------------------------


def _forge(
  node: Expr, *, args: Any = None, shape: Any = None, dtype: Any = None, attrs: dict[str, Any] | None = None, drop: tuple[str, ...] = ()
) -> Expr:
  """``node`` rebuilt with one thing changed, bypassing the builder's checks."""
  merged = {k: v for k, v in node.attrs.items() if k not in drop} | (attrs or {})
  t = node.type
  type_ = TensorType(t.shape if shape is None else shape, dtype=t.dtype if dtype is None else dtype, diff=t.diff)
  return Expr(node.op, node.args if args is None else tuple(args), type_, attrs=merged)


def _loop_parts() -> dict[str, Any]:
  c, p, k = sc.sym("c", 2), sc.sym("p", 2), sc.sym("k", (), dtype="int64")
  body = _fn("vf_body", [c, p], [c * 0.5 + p])
  body_k = _fn("vf_body_k", [c, k, p], [c * 0.5 + p])
  kf = sc.sym("k", ())
  body_float_k = _fn("vf_body_fk", [c, kf, p], [c * kf + p])
  cond = _fn("vf_cond", [c, p], [c[0] > p[0]])
  not_bool = _fn("vf_nb", [c, p], [c[:1]])
  p32 = _fn("vf_c32", [c, sc.sym("p", 2, dtype="float32")], [c[0] > 0.0])
  final, count = sc.while_loop(cond, body, sc.sym("c0", 2), max_iter=4, params=[sc.sym("pv", 2)])
  u = sc.sym("u", 2)
  step = _fn("vf_step", [c, u], [c + u, c[:1]])
  carry, ys = sc.scan(step, sc.sym("c0", 2), [(sc.sym("us", 6), 0, 2)], length=3)
  lane = _fn("vf_lane", [u], [u.sum()])
  mapped = sc.vmap(lane, length=3, inputs=[(sc.sym("us", 6), 0, 2)])
  return {
    "final": final,
    "count": count,
    "body_k": body_k,
    "body_float_k": body_float_k,
    "not_bool": not_bool,
    "cond32": p32,
    "carry": carry,
    "ys": ys,
    "mapped": mapped,
  }


def _sparse_parts() -> tuple[Expr, Expr]:
  mat = SparseMatrix.symbol("kf", PATTERN)
  fact = SparseLDL(mat, schedule="loop", name="vf_sldl")
  factor = sparse_ldl_factor(mat.values, fact.tables())
  return factor, sparse_ldl_solve(factor, sc.sym("b", 4), fact.solve_tables())


def _forged() -> dict[str, list[Expr]]:
  x, v = sc.sym("x", 5), sc.sym("v", 3)
  x2 = sc.sym("x2", (2, 5))
  x32 = sc.sym("x32", 5, dtype="float32")
  flag = x < 1.0
  take = sc.take(x2, _i64([0, 4, 9]))
  put = sc.put_add(x2, _i64([0, 4, 9]), sc.sym("pv", (2, 3)))
  added = sc.index_add(x, [0, 0, 4], v)
  a = sc.sym("a", (3, 3))
  chol, lu, tri = linalg.cholesky(a), linalg.lu(a), linalg.solve_triangular(a, sc.sym("b", 3))
  factor, solved = _sparse_parts()
  lo, hi = _i64([0, 2]), _i64([2, 3])
  radd, rdot = ragged_add(x, x, lo, hi, sc.sym("s", 2)), ragged_dot(x, x, lo, hi)
  seg = sc.segment_max(x, [0, 1, 0, 1, 2], 3)
  loops = _loop_parts()
  final, carry, mapped = loops["final"], loops["carry"], loops["mapped"]
  return {
    "compare-types": [_forge(x < x32.cast("float64"), args=(x, x32)), _forge(flag, shape=(4,)), _forge(flag, dtype=dtypes.float64)],
    "logical-types": [_forge(~flag, args=(x,)), _forge(flag & flag, args=(flag, sc.sym("f3", 3, dtype="bool")))],
    "isfinite-types": [_forge(sc.isfinite(x), dtype=dtypes.float64)],
    "select-types": [
      _forge(sc.where(flag, x, x), args=(flag, x, x32)),
      _forge(sc.where(flag, x, x), args=(x, x, x)),
      _forge(sc.where(flag, x, x), shape=(1, 5)),
    ],
    "cast-types": [_forge(sc.cast(x, "float32"), shape=(4,))],
    "reduce-output-scalar": [_forge(x.max(), dtype=dtypes.float32)],
    "segment-extremum": [
      _forge(seg, drop=("fill",)),
      _forge(seg, attrs={"indices": np.array([0, 1, 0])}),
      _forge(seg, args=(sc.sym("m", (5, 1)),)),
      _forge(seg, attrs={"indices": np.array([0, 1, 0, 1, -1])}),
    ],
    "index-update": [
      _forge(added, attrs={"indices": np.array([0, 4])}),
      _forge(added, attrs={"indices": np.array([0, 4, 5])}),
      _forge(sc.index_set(x, [0, 1, 4], v), attrs={"indices": np.array([0, 1, 1])}),
      _forge(added, args=(x, v.cast("float32"))),
      _forge(added, args=(x, sc.sym("v", (3, 1)))),
      _forge(added, shape=(6,)),
    ],
    "take-shapes": [
      _forge(take, args=(x2, sc.sym("fi", 3))),
      _forge(take, args=(x2, sc.sym("ii", (3, 1), dtype="int64"))),
      _forge(take, shape=(2, 4)),
      _forge(take, dtype=dtypes.float32),
      _forge(take, args=(sc.sym("s0", ()), _i64([0, 1, 2])), shape=(3,)),
    ],
    "put-shapes": [
      _forge(put, args=(x2, put.args[1], sc.sym("pv", (2, 4)))),
      _forge(put, args=(x2, sc.sym("fi", 3), put.args[2])),
      _forge(put, shape=(2, 6)),
    ],
    "factor-shape": [_forge(chol, shape=(3, 4)), _forge(linalg.ldl(a), args=(sc.sym("r", (3, 4)),))],
    "lu-shape": [_forge(lu, shape=(3, 3)), _forge(lu, args=(sc.sym("r", (4, 3)),))],
    "trisolve-shapes": [
      _forge(tri, drop=("unit",)),
      _forge(tri, args=(a, sc.sym("b", 4)), shape=(4,)),
      _forge(tri, args=(a, sc.sym("b", (3, 2, 1))), shape=(3, 2, 1)),
      _forge(tri, shape=(4,)),
    ],
    "sparse-ldl-tables": [
      _forge(factor, drop=("ck_len",)),
      _forge(factor, shape=(factor.size + 1,)),
      _forge(factor, attrs={"l_ptr": factor.attrs["l_ptr"][:-1]}),
      _forge(factor, attrs={"r_cols": factor.attrs["r_cols"][:-1]}),
    ],
    "sparse-ldl-solve-tables": [
      _forge(solved, drop=("perm",)),
      _forge(solved, args=(factor, sc.sym("b", 5))),
      _forge(solved, args=(sc.sym("f", factor.size + 1), sc.sym("b", 4))),
      _forge(solved, attrs={"l_ptr": solved.attrs["l_ptr"][:-1]}),
    ],
    "ragged-shapes": [
      _forge(radd, args=(x, x, sc.sym("lo", 2), hi, sc.sym("s", 2))),
      _forge(radd, args=(x, x, lo, _i64([2, 3, 4]), sc.sym("s", 2))),
      _forge(radd, args=(x, x, lo, hi, sc.sym("s", 3))),
      _forge(radd, shape=(4,)),
      _forge(rdot, shape=(3,)),
    ],
    "vmap-attrs": [
      _forge(mapped, drop=("strides",)),
      _forge(mapped, attrs={"length": -1}),
      _forge(mapped, attrs={"starts": (0, 0)}),
      _forge(mapped, shape=(4,)),
    ],
    "scan-attrs": [
      _forge(carry, drop=("length",)),
      _forge(carry, attrs={"output": 2}),
      _forge(carry, attrs={"starts": ()}),
      _forge(carry, args=(sc.sym("c0", 3), carry.args[1])),
      _forge(loops["ys"], shape=(4,)),
    ],
    "while-attrs": [
      _forge(final, drop=("cond",)),
      _forge(final, attrs={"cond": loops["not_bool"]}),
      _forge(final, attrs={"index": True}),
      _forge(final, attrs={"index": True, "callee": loops["body_float_k"]}),
      _forge(final, attrs={"cond": loops["cond32"]}),
      _forge(final, args=(final.args[0],)),
      _forge(final, attrs={"output": -1}),
      _forge(loops["count"], shape=(1,)),
    ],
  }


FORGED_RULES = sorted(_forged())


@pytest.mark.parametrize("rule", FORGED_RULES)
def test_forged_nodes_fail_their_own_rule(rule: str) -> None:
  nodes = _forged()[rule]
  for k, node in enumerate(nodes):
    with pytest.raises(VerifyError, match=rf"failed rule '{rule}'"):
      verify_expr(node)
    # The same node inside a larger graph is still found, and named first.
    with pytest.raises(VerifyError, match=rf"failed rule '{rule}'"):
      verify_expr(sc.stack([sc.cast(node, "float64").reshape((node.size,)).sum(), sc.const(float(k))]))


def test_forged_nodes_are_forged_from_verified_ones() -> None:
  """Each forged node differs from a valid one in one respect: without the change it verifies."""
  x2 = sc.sym("x2", (2, 5))
  a = sc.sym("a", (3, 3))
  factor, solved = _sparse_parts()
  loops = _loop_parts()
  lo, hi = _i64([0, 2]), _i64([2, 3])
  x = sc.sym("x", 5)
  verify_expr(
    [
      sc.take(x2, _i64([0, 4, 9])),
      sc.put_add(x2, _i64([0, 4, 9]), sc.sym("pv", (2, 3))),
      sc.index_add(x, [0, 0, 4], sc.sym("v", 3)),
      linalg.cholesky(a),
      linalg.lu(a),
      linalg.solve_triangular(a, sc.sym("b", 3)),
      factor,
      solved,
      ragged_add(x, x, lo, hi, sc.sym("s", 2)),
      ragged_dot(x, x, lo, hi),
      sc.segment_max(x, [0, 1, 0, 1, 2], 3),
      loops["final"],
      loops["count"],
      loops["carry"],
      loops["ys"],
      loops["mapped"],
    ]
  )
