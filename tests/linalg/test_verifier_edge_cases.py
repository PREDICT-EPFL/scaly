"""``verify_expr`` on the linear-algebra ops (``linalg.ops``), as ``tests/ir/test_verifier_edge_cases.py``
checks the core's: every graph the builders make verifies, and so does every derivative and
rewrite of it; each op's rule rejects a node forged with one thing wrong, naming itself."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

import scaly as sc
from scaly import linalg
from scaly.ir.expr import Expr
from scaly.ir.expr_spec import spec_expr, verify_expr
from scaly.ir.spec import VerifyError
from scaly.linalg import SparseLDL, SparseMatrix
from scaly.linalg.ops import ragged_add, ragged_dot, sparse_ldl_factor, sparse_ldl_solve
from tests.ir.test_verifier_edge_cases import _derivatives, _forge, _i64, _verify_deep

LINALG_OPS = frozenset({"cholesky", "ldl", "lu", "trisolve", "sparse_ldl", "sparse_ldl_solve", "ragged_add", "ragged_dot"})


def _spd(x: Expr, n: int) -> Expr:
  a = x.reshape((n, n))
  return a @ a.T + sc.const(float(n) * np.eye(n))


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
  with sc.options(linalg=dict(dense_unroll=0)):
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


def _sparse_parts() -> tuple[Expr, Expr]:
  mat = SparseMatrix.symbol("kf", PATTERN)
  fact = SparseLDL(mat, schedule="loop", name="vf_sldl")
  factor = sparse_ldl_factor(mat.values, fact.tables())
  return factor, sparse_ldl_solve(factor, sc.sym("b", 4), fact.solve_tables())


CORPUS: dict[str, Callable[[], tuple[Expr, Expr]]] = {
  "ragged": _ragged,
  "dense": _dense,
  "dense_looped": _dense_looped,
  "lu_factor": _lu_factor,
  "sparse": _sparse,
  "sparse_primitives": _sparse_primitives,
}
VALUE_ONLY = {"lu_factor", "sparse_primitives"}


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


def test_the_corpus_reaches_every_linalg_op() -> None:
  """A new op must join the corpus above, or its builders and AD rules go unverified here."""
  seen: set[str] = set()
  for name, build in CORPUS.items():
    x, y = build()
    seen |= _verify_deep([y] if name in VALUE_ONLY else [y, *_derivatives(x, y)])
  assert LINALG_OPS <= seen, sorted(LINALG_OPS - seen)


def test_every_linalg_op_has_a_rule_of_its_own() -> None:
  rules = spec_expr.op_rules()
  assert all(rules.get(op) for op in LINALG_OPS), sorted(op for op in LINALG_OPS if not rules.get(op))


def _forged() -> dict[str, list[Expr]]:
  x = sc.sym("x", 5)
  a = sc.sym("a", (3, 3))
  chol, lu, tri = linalg.cholesky(a), linalg.lu(a), linalg.solve_triangular(a, sc.sym("b", 3))
  factor, solved = _sparse_parts()
  lo, hi = _i64([0, 2]), _i64([2, 3])
  radd, rdot = ragged_add(x, x, lo, hi, sc.sym("s", 2)), ragged_dot(x, x, lo, hi)
  return {
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
  }


FORGED_RULES = sorted(_forged())


@pytest.mark.parametrize("rule", FORGED_RULES)
def test_forged_nodes_fail_their_own_rule(rule: str) -> None:
  nodes = _forged()[rule]
  for k, node in enumerate(nodes):
    with pytest.raises(VerifyError, match=rf"failed rule '{rule}'"):
      verify_expr(node)
    with pytest.raises(VerifyError, match=rf"failed rule '{rule}'"):
      verify_expr(sc.stack([sc.cast(node, "float64").reshape((node.size,)).sum(), sc.const(float(k))]))


def test_forged_nodes_are_forged_from_verified_ones() -> None:
  """Each forged node differs from a valid one in one respect: without the change it verifies."""
  a = sc.sym("a", (3, 3))
  factor, solved = _sparse_parts()
  lo, hi = _i64([0, 2]), _i64([2, 3])
  x = sc.sym("x", 5)
  verify_expr(
    [
      linalg.cholesky(a),
      linalg.lu(a),
      linalg.solve_triangular(a, sc.sym("b", 3)),
      factor,
      solved,
      ragged_add(x, x, lo, hi, sc.sym("s", 2)),
      ragged_dot(x, x, lo, hi),
    ]
  )
