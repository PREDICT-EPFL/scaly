"""``verify_expr`` on the ops added with the control, indexing and loop work.

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
from scaly.ad.derivatives import hessian, jacobian
from scaly.ad.forward import jvp, jvp_many
from scaly.ad.reverse import vjp
from scaly.ir.expr import Expr, ExprOp, callees_of, topo
from scaly.ir.expr_spec import spec_expr, verify_expr
from scaly.ir.spec import VerifyError
from scaly.ir.types import TensorType, dtypes

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
    ExprOp.SEGMENT_REDUCE,
    ExprOp.TAKE,
    ExprOp.PUT_ADD,
    ExprOp.PUT,
    ExprOp.VMAP,
    ExprOp.SCAN,
    ExprOp.WHILE,
  }
)


def _fn(name: str, inputs: list[Expr], outputs: list[Expr]) -> sc.ConcreteFunction:
  return sc.Function.from_exprs(name, inputs, outputs, [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _i64(values: Any) -> Expr:
  return sc.const(np.asarray(values, dtype=np.int64), dtype="int64")


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
  "loops": _loops,
}


def _verify_deep(outputs: list[Expr]) -> set[str]:
  """Verify ``outputs`` and every callee body they run, transitively; return the ops seen."""
  ops: set[str] = set()
  seen: set[int] = set()
  pending = [tuple(outputs)]
  while pending:
    outs = pending.pop()
    verify_expr(outs)
    for node in topo(outs):
      ops.add(node.op)
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


@pytest.mark.parametrize("name", sorted(CORPUS))
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
  seen: set[str] = set()
  for build in CORPUS.values():
    x, y = build()
    seen |= _verify_deep([y, *_derivatives(x, y)])
  assert NEW_OPS <= seen, sorted(NEW_OPS - seen)


def test_every_op_has_a_rule_of_its_own() -> None:
  """``docs/dev/codebase.md``: adding an op includes a verify rule (``OpDef.verify``)."""
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


def _forged() -> dict[str, list[Expr]]:
  x, v = sc.sym("x", 5), sc.sym("v", 3)
  x2 = sc.sym("x2", (2, 5))
  x32 = sc.sym("x32", 5, dtype="float32")
  flag = x < 1.0
  take = sc.take(x2, _i64([0, 4, 9]))
  put = sc.put_add(x2, _i64([0, 4, 9]), sc.sym("pv", (2, 3)))
  added = sc.index_add(x, [0, 0, 4], v)
  seg = sc.segment_max(x, [0, 1, 0, 1, 2], 3)
  summed = sc.scatter(v, [0, 2, 2], (4,))
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
    "segment-reduce": [
      _forge(seg, drop=("fill",)),
      _forge(seg, drop=("reduce",)),
      _forge(seg, attrs={"reduce": "mean"}),
      _forge(seg, attrs={"indices": np.array([0, 1, 0])}),
      _forge(seg, attrs={"indices": np.array([0, 1, 0, 1, -1])}),
      _forge(seg, dtype=dtypes.float32),
      _forge(summed, attrs={"indices": np.array([0, 2, 4])}),
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
      _forge(added, args=(x, added.args[1], sc.sym("v", (3, 1)))),
      _forge(added, args=(x, _i64([[0, 0, 4]]), v)),
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
  loops = _loop_parts()
  x = sc.sym("x", 5)
  verify_expr(
    [
      sc.take(x2, _i64([0, 4, 9])),
      sc.put_add(x2, _i64([0, 4, 9]), sc.sym("pv", (2, 3))),
      sc.index_add(x, [0, 0, 4], sc.sym("v", 3)),
      sc.segment_max(x, [0, 1, 0, 1, 2], 3),
      sc.scatter(sc.sym("v", 3), [0, 2, 2], (4,)),
      loops["final"],
      loops["count"],
      loops["carry"],
      loops["ys"],
      loops["mapped"],
    ]
  )
