"""Loops of kind ``VECTOR`` rendered as GNU vector statements (``codegen/c.py``), and the products
whose running sums the lowering keeps in them: the same values in every lane count, and a loop
the vector statements cannot express rendered as the loop it also is."""

from __future__ import annotations

import re

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source
from scaly.codegen.c import _emit_statement
from scaly.ir import program as p
from scaly.ir.program import RangeKind
from scaly.ir.types import dtypes

SHAPES = [(None, 64, 64), (20, 12, 40), (8, 256, 256), (3, 70, 130), (16, 700, 40), (5, 9, 6), (256, 64, 6), (1, 30, 100)]


def _product(m: int | None, k: int, n: int, tag: str) -> sc.Function:
  a, b = sc.sym("a", (k,) if m is None else (m, k)), sc.sym("b", (k, n))
  return sc.Function.from_exprs(f"vector_mm_{tag}_{m}_{k}_{n}", [a, b], [(a @ b).block()], ["a", "b"], ["c"])


@pytest.mark.parametrize(("target", "lanes"), [("apple-m3", 2), ("x86-64-v3", 4), ("x86-64-v4", 8), ("generic", 1)])
def test_products_render_the_targets_vectors(target: str, lanes: int) -> None:
  source = render_c_source(_product(20, 12, 40, target.replace("-", "_")), target=target)
  used = set(re.findall(r"\(double(\d)\*\)", source))
  assert used == (set() if lanes == 1 else {str(lanes)})
  if lanes > 2:  # the wider type is declared where it is used
    assert f"typedef double double{lanes} __attribute__((vector_size({8 * lanes})" in source
  # A store of the vector type may alias anything, the ABI's pointer arrays too: stores to an output
  # go lane by lane, and only a sum in an array on the stack is stored as a vector.
  stored = re.findall(r"\*\(double\d\*\)\((\w+)[^;]*\)\s*=[^=]", source)
  assert all(name not in ("arg", "res", "w") for name in stored) and (bool(stored) == (lanes > 1))
  assert (lanes == 1) or "for (long long q" not in source  # every lane loop became vector statements


def test_every_vector_width_computes_the_same_bits() -> None:
  """Each output is one chain of multiply-adds in order of ``k`` whatever the widths, and a block's
  are fused by construction: two, four and eight lanes give the same bits, where a block on one
  target is a streamed column on another (six columns: a block of four on the M3, none on AVX2).
  One lane leaves its sums to the C compiler's vectorizer, which fuses some and not others."""
  rng = np.random.default_rng(0)
  for m, k, n in [*SHAPES, (None, 256, 6), (4, 256, 6)]:
    a = rng.standard_normal((k,) if m is None else (m, k))
    b = rng.standard_normal((k, n))
    got = {}
    for target in ("apple-m3", "x86-64-v3", "x86-64-v4", "generic"):
      with sc.target(target):
        got[target] = np.asarray(_product(m, k, n, target.replace("-", "_"))._flat_numerical_call(a, b)[0])
    for target in ("x86-64-v3", "x86-64-v4"):
      np.testing.assert_array_equal(got[target], got["apple-m3"], err_msg=f"{target} {m}x{k}x{n}")
    for value in got.values():
      np.testing.assert_allclose(value.reshape(a.shape[:-1] + (n,)), a @ b, rtol=1e-12, atol=1e-12)


def _rendered(*body, lanes: int = 2, start: int = 0, stop: p.ProgramNode | int | None = None) -> str:
  q = p.var("q")
  loop = p.for_(p.range_("q", start, start + lanes if stop is None else stop, kind=RangeKind.VECTOR), [f(q) for f in body])
  lines: list[str] = []
  _emit_statement(loop, {}, lines, 0)
  return "\n".join(lines)


X = p.buffer("x", dtypes.float64, (16,))
Y = p.buffer("y", dtypes.float64, (16,))


def _at(buf: p.ProgramNode, index: p.ProgramNode) -> p.ProgramNode:
  return p.view(buf, [index])


def test_a_vector_body_renders_as_vector_statements() -> None:
  scale = p.load(_at(Y, p.const_int(9)))  # the same in every lane: a scalar the arithmetic broadcasts
  text = _rendered(lambda q: p.store(_at(X, p.add(p.const_int(4), q)), p.add(p.mul(scale, p.load(_at(Y, q))), p.const_float(1.0))))
  assert "for (" not in text
  assert "(*(double2*)(y))" in text and "y[9]" in text and "x[(4 + 0)] = v_[0]; x[(4 + 0) + 1] = v_[1];" in text
  # A temporary that differs between the lanes is a vector; the block keeps it in the loop's scope.
  t = p.var("t", dtypes.float64)
  text = _rendered(
    lambda q: p.assign("t", p.mul(p.load(_at(Y, q)), p.const_float(2.0)), dtypes.float64, declare=True),
    lambda q: p.store(_at(X, q), p.add(t, p.const_float(1.0))),
  )
  assert text.splitlines() == ["{", "  double2 t = ((*(double2*)(y)) * 2.0);", "  { const double2 v_ = (t + 1.0); x[0] = v_[0]; x[1] = v_[1]; }", "}"]
  # An index the lanes share, named because two accesses use it, is an integer beside the vectors.
  base = p.var("base", dtypes.int64)
  text = _rendered(
    lambda q: p.assign("base", p.mul(p.var("row", dtypes.int64), p.const_int(8)), dtypes.int64, declare=True),
    lambda q: p.store(_at(X, p.add(base, q)), p.mul(p.load(_at(Y, p.add(base, q))), p.const_float(2.0))),
  )
  assert text.splitlines()[:2] == ["{", "  int64_t base = (row * 8);"] and "(*(double2*)(y + (base + 0)))" in text and "for (" not in text
  # The lanes' own index, named the same way: its value at the first lane, the accesses at stride 1.
  at = p.var("at", dtypes.int64)
  text = _rendered(
    lambda q: p.assign("at", p.add(p.mul(p.var("row", dtypes.int64), p.const_int(8)), q), dtypes.int64, declare=True),
    lambda q: p.store(_at(X, at), p.add(p.load(_at(X, at)), p.load(_at(Y, q)))),
    start=2,
  )
  assert (
    text.splitlines()[:2] == ["{", "  int64_t at = ((row * 8) + 2);"]
    and "(*(double2*)(x + at)) + (*(double2*)(y + 2))" in text
    and "for (" not in text
  )
  # Lanes 6 and 7: the vector starts at the first lane's element.
  text = _rendered(lambda q: p.store(_at(X, q), p.load(_at(Y, q))), start=6)
  assert text == "{ const double2 v_ = (*(double2*)(y + 6)); x[6] = v_[0]; x[7] = v_[1]; }"


@pytest.mark.parametrize(
  "why",
  [
    "stride 2",
    "stride 2, constant first",
    "stride -1",
    "stored at two places",
    "stored at one place",
    "a call on a vector",
    "read at another index",
    "three lanes",
    "a run-time bound",
    "the lane as a value",
    "an index named per lane",
    "an index not affine in the lane",
  ],
)
def test_other_bodies_render_as_loops(why: str) -> None:
  if why == "stride 2":
    text = _rendered(lambda q: p.store(_at(X, q), p.load(_at(Y, p.mul(q, p.const_int(2))))))
  elif why == "stride -1":  # x[q] = y[7 - q]: the lanes reversed
    text = _rendered(lambda q: p.store(_at(X, q), p.load(_at(Y, p.sub(p.const_int(7), q)))))
  elif why == "stride 2, constant first":
    text = _rendered(lambda q: p.store(_at(X, q), p.load(_at(Y, p.mul(p.const_int(2), q)))))
  elif why == "stored at two places":  # x[q] then x[q + 1]: the next lane overwrites what this one stored
    text = _rendered(
      lambda q: p.store(_at(X, q), p.load(_at(Y, q))),
      lambda q: p.store(_at(X, p.add(q, p.const_int(1))), p.const_float(0.0)),
    )
  elif why == "stored at one place":  # every lane writes x[5]: the last lane's value stays
    text = _rendered(lambda q: p.store(_at(X, p.const_int(5)), p.load(_at(Y, q))))
  elif why == "a call on a vector":
    text = _rendered(lambda q: p.store(_at(X, q), p.ProgramNode(p.ProgramOp.SIN, (p.load(_at(Y, q)),), dtype=dtypes.float64)))
  elif why == "read at another index":  # x[q + 1] = x[q]: a lane reads what the one before it writes
    text = _rendered(lambda q: p.store(_at(X, p.add(q, p.const_int(1))), p.load(_at(X, q))))
  elif why == "three lanes":
    text = _rendered(lambda q: p.store(_at(X, q), p.load(_at(Y, q))), lanes=3)
  elif why == "a run-time bound":
    text = _rendered(lambda q: p.store(_at(X, q), p.load(_at(Y, q))), stop=p.var("n"))
  elif why == "an index named per lane":  # at = 2 * q: the temporary differs between the lanes
    at = p.var("at", dtypes.int64)
    text = _rendered(
      lambda q: p.assign("at", p.mul(q, p.const_int(2)), dtypes.int64, declare=True),
      lambda q: p.store(_at(X, q), p.load(_at(Y, at))),
    )
  elif why == "an index not affine in the lane":
    at = p.var("at", dtypes.int64)
    text = _rendered(
      lambda q: p.assign("at", p.mul(q, q), dtypes.int64, declare=True),
      lambda q: p.store(_at(X, q), p.load(_at(Y, at))),
    )
  else:  # the lane number itself, in arithmetic with a vector: not an access
    lane = lambda q: p.ProgramNode(p.ProgramOp.CAST, (q,), dtype=dtypes.float64)  # noqa: E731
    text = _rendered(lambda q: p.store(_at(X, q), p.add(p.load(_at(Y, q)), lane(q))))
  assert text.startswith("for (long long q = 0;")
