"""The indexing ops at constant indices against NumPy: ``segment_reduce`` in its three reductions and
``put``/``put_add`` at constant indices.

``scatter``, ``segment_sum``, ``segment_max`` and ``segment_min`` are one op, ``segment_reduce``, and
``index_add``/``index_set`` are ``put_add``/``put`` whose indices are a constant. Each is checked
built by its builder and, where the op allows more than the builders make (a sum with a fill, an
extremum over a matrix of values), built as a node: values under both lowering hints and folded,
the Jacobian in forward and reverse mode and its structural pattern, all against NumPy. A put at
constant indices must agree bit for bit with the same put at run-time indices, lower without range
checks, and update a loop carry in place through the flat view of a matrix.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_source
from scaly.ir.expr import Expr, ExprOp
from scaly.ir.types import TensorType
from scaly.passes import lowering
from scaly.passes.lowering import lower_function, main_proc

UFUNCS = {"add": np.add, "max": np.maximum, "min": np.minimum}


def _fn(name: str, inputs: Sequence[Expr], outputs: Sequence[Expr]) -> sc.Function:
  return sc.Function.from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _segment(reduce: str, values: Expr, ids: np.ndarray, shape: tuple[int, ...], fill: float) -> Expr:
  """A ``segment_reduce`` node as the IR allows it, past what the builders make."""
  return Expr(ExprOp.SEGMENT_REDUCE, (values,), TensorType(shape), attrs={"reduce": reduce, "indices": ids, "fill": fill})


def _np_segment(reduce: str, values: np.ndarray, ids: np.ndarray, shape: tuple[int, ...], fill: float) -> np.ndarray:
  out = np.full(int(np.prod(shape)), fill)
  UFUNCS[reduce].at(out, ids.reshape(-1), values.reshape(-1))
  return out.reshape(shape)


def _np_segment_jacobian(reduce: str, values: np.ndarray, ids: np.ndarray, size: int, fill: float) -> np.ndarray:
  """d out / d values away from ties: every value of a bin for a sum, the winner for an extremum,
  none where the fill wins."""
  jac = np.zeros((size, values.size))
  out = _np_segment(reduce, values, ids, (size,), fill)
  for k, (v, b) in enumerate(zip(values.reshape(-1), ids.reshape(-1), strict=True)):
    jac[b, k] = 1.0 if reduce == "add" or v == out[b] else 0.0
  return jac


def _np_put(base: np.ndarray, idx: np.ndarray, values: np.ndarray, *, add: bool) -> np.ndarray:
  out = base.copy()
  for j, i in enumerate(idx):
    if 0 <= i < base.shape[-1]:
      out[..., i] = out[..., i] + values[..., j] if add else values[..., j]
  return out


def _np_put_jacobians(shape: tuple[int, ...], idx: np.ndarray, width: int, *, add: bool) -> tuple[np.ndarray, np.ndarray]:
  """d out / d base and d out / d values of a put along the last axis: the base where no lane lands
  (everywhere for a sum), and in each row the lane that lands last (every lane for a sum)."""
  n = shape[-1]
  rows = int(np.prod(shape[:-1], dtype=np.int64))
  d_base, d_values = np.eye(rows * n), np.zeros((rows * n, rows * width))
  for b in range(rows):
    for j, i in enumerate(idx):
      if 0 <= i < n:
        if not add:
          d_values[b * n + i] = 0.0
          d_base[b * n + i] = 0.0
        d_values[b * n + i, b * width + j] = 1.0
  return d_base, d_values


def _three_jacobians(out: Expr, wrt: Expr, inputs: Sequence[Expr], point: Sequence[np.ndarray], name: str) -> list[np.ndarray]:
  """The Jacobian of ``out`` in ``wrt`` by batched forward mode, by one ``vjp`` per output entry and
  by one ``jvp`` per input entry."""
  flat = out.reshape((out.size,))
  w, t = sc.sym(f"{name}_w", (out.size,)), sc.sym(f"{name}_t", wrt.shape)
  (rev,) = sc.vjp((flat,), (wrt,), (w,))
  fun = _fn(name, [*inputs, w, t], [sc.jacobian(flat, wrt), rev, sc.jvp(flat, wrt, t)])
  zw, zt = np.zeros(out.size), np.zeros(wrt.shape)
  dense = fun._flat_numerical_call(*point, zw, zt)[0].reshape(out.size, wrt.size)
  rows = np.array([fun._flat_numerical_call(*point, np.eye(out.size)[i], zt)[1].reshape(-1) for i in range(out.size)])
  cols = np.array([fun._flat_numerical_call(*point, zw, np.eye(wrt.size)[j].reshape(wrt.shape))[2] for j in range(wrt.size)]).T
  return [dense, rows, cols]


def _pattern(expr: Expr, wrt: Expr) -> np.ndarray:
  sp = sc.jacobian_sparsity(expr, wrt)
  dense = np.zeros(sp.shape, dtype=bool)
  dense[sp.rows, sp.cols] = True
  return dense


# --- segment_reduce ------------------------------------------------------------------------------

IDS = np.array([[3, 0, 3], [5, 3, 1], [0, 0, 3], [1, 5, 3]])  # bins 2 and 4 empty, 3 five times
CASES = [  # reduction, fill, output shape
  ("add", 0.0, (6,)),
  ("add", 0.625, (2, 3)),
  ("max", -np.inf, (6,)),
  ("max", 0.25, (3, 2)),
  ("min", np.inf, (6,)),
  ("min", -0.25, (6,)),
]


def _segment_cases(v: Expr) -> list[Expr]:
  """Each case as the builder makes it where one does, and as a node over the matrix of values."""
  built = [
    sc.scatter(v.reshape((12,)), IDS.reshape(-1), (6,)),
    sc.segment_sum(v, IDS, 6),
    sc.segment_max(v, IDS, 6),
    sc.segment_min(v, IDS, 6, fill=-0.25),
  ]
  return built + [_segment(reduce, v, IDS.reshape(-1), shape, fill) for reduce, fill, shape in CASES]


def _segment_numpy(vv: np.ndarray) -> list[np.ndarray]:
  built = [_np_segment("add", vv, IDS, (6,), 0.0)] * 2 + [_np_segment("max", vv, IDS, (6,), -np.inf), _np_segment("min", vv, IDS, (6,), -0.25)]
  return built + [_np_segment(reduce, vv, IDS, shape, fill) for reduce, fill, shape in CASES]


@pytest.mark.parametrize("hint", ["scalar", "block"])
def test_segment_reduce_values_match_numpy(hint: str) -> None:
  v = sc.sym("v", (4, 3))
  outs = [e.scalar() if hint == "scalar" else e.block() for e in _segment_cases(v)]
  sc.verify_expr(outs)
  fn = _fn(f"ix_seg_{hint}", [v], outs)
  vv = np.random.default_rng(1).integers(-24, 25, size=(4, 3)) / 8.0  # sums exact in any order
  for got, want in zip(fn._flat_numerical_call(vv), _segment_numpy(vv), strict=True):
    np.testing.assert_array_equal(got, want)


def test_segment_reduce_of_constants_folds_to_numpy() -> None:
  vv = np.random.default_rng(2).standard_normal((4, 3))
  for got, want in zip(_segment_cases(sc.const(vv)), _segment_numpy(vv), strict=True):
    folded = sc.simplify(got)
    assert folded.op == ExprOp.CONST
    np.testing.assert_array_equal(folded.value, want)


@pytest.mark.parametrize("case", range(len(CASES)), ids=[f"{r}_{f}_{'x'.join(map(str, s))}" for r, f, s in CASES])
def test_segment_reduce_derivatives_match_numpy(case: int) -> None:
  """Away from ties, three ways, with the pattern exactly the bins; the gradient of values shaped
  unlike their ids comes back shaped like the values."""
  reduce, fill, shape = CASES[case]
  v = sc.sym("v", (4, 3))
  out = _segment(reduce, v, IDS.reshape(-1), shape, fill)
  vv = np.random.default_rng(3).standard_normal((4, 3))
  want = _np_segment_jacobian(reduce, vv, IDS, out.size, fill)
  for got in _three_jacobians(out, v, [v], [vv], f"ix_segd_{case}"):
    np.testing.assert_array_equal(got, want)
  assert (_pattern(out, v) == (IDS.reshape(1, -1) == np.arange(out.size)[:, None])).all()
  weights = np.arange(1.0, out.size + 1.0)
  grad = sc.gradient((out * sc.const(weights.reshape(shape))).sum(), v)
  assert grad.shape == v.shape
  (got,) = _fn(f"ix_segg_{case}", [v], [grad])._flat_numerical_call(vv)
  np.testing.assert_array_equal(got, (weights @ want).reshape(4, 3))


# --- put and put_add at constant indices ---------------------------------------------------------

IDX = np.array([4, 1, 4, -1, 6, 0])  # 4 twice (the later lane lands), two lanes outside [0, 5)


def _put_cases(x: Expr, m: Expr, s: Expr, b: Expr, v: Expr, w: Expr, idx: Expr) -> list[Expr]:
  return [
    sc.index_add(x, [2, 2, 0, 4], v[:4]),
    sc.index_set(x, [4, 1, 0], v[:3]),
    sc.index_add(m, [[5, 0], [3, 5]], v[:4].reshape((2, 2))),  # the flat view of a matrix
    sc.index_set(m, [1, 4], v[4:]),
    sc.index_add(s, [0], v[:1]),  # a scalar base
    sc.put_add(b, idx, w),
    sc.put(b, idx, w * 2.0),
  ]


def _put_numpy(xv: np.ndarray, mv: np.ndarray, sv: np.ndarray, bv: np.ndarray, vv: np.ndarray, wv: np.ndarray, iv: np.ndarray) -> list[np.ndarray]:
  flat = mv.reshape(-1)
  return [
    _np_put(xv, np.array([2, 2, 0, 4]), vv[:4], add=True),
    _np_put(xv, np.array([4, 1, 0]), vv[:3], add=False),
    _np_put(flat, np.array([5, 0, 3, 5]), vv[:4], add=True).reshape(2, 3),
    _np_put(flat, np.array([1, 4]), vv[4:], add=False).reshape(2, 3),
    np.asarray(sv + vv[0]),
    _np_put(bv, iv, wv, add=True),
    _np_put(bv, iv, wv * 2.0, add=False),
  ]


def _cases_of(args, idx: Expr) -> list[Expr]:
  x, m, s, b, v, w = args
  return _put_cases(x, m, s, b, v, w, idx)


def _numpy_of(values, iv: np.ndarray) -> list[np.ndarray]:
  xv, mv, sv, bv, vv, wv = values
  return _put_numpy(xv, mv, sv, bv, vv, wv, iv)


def _put_inputs() -> tuple[list[Expr], list[np.ndarray]]:
  inputs = [sc.sym("x", 5), sc.sym("m", (2, 3)), sc.sym("s", ()), sc.sym("b", (2, 3, 5)), sc.sym("v", 6), sc.sym("w", (2, 3, 6))]
  rng = np.random.default_rng(4)
  return inputs, [rng.integers(-24, 25, size=e.shape) / 8.0 for e in inputs]


@pytest.mark.parametrize("hint", ["scalar", "block"])
def test_constant_puts_match_numpy(hint: str) -> None:
  """Under both hints; a function of constant puts alone still expands to scalars, which a run-time
  index would stop."""
  inputs, point = _put_inputs()
  outs = [e.scalar() if hint == "scalar" else e.block() for e in _cases_of(inputs, sc.const(IDX, dtype="int64"))]
  sc.verify_expr(outs)
  fn = _fn(f"ix_put_{hint}", inputs, outs)
  if hint == "scalar":
    assert main_proc(lower_function(fn)).attrs["scalarize_mode"] == "procedure"
  for got, want in zip(fn._flat_numerical_call(*point), _numpy_of(point, IDX), strict=True):
    np.testing.assert_array_equal(got, want)


def test_index_updates_are_puts_at_constant_indices() -> None:
  x, m, v = sc.sym("x", 5), sc.sym("m", (2, 3)), sc.sym("v", 2)
  for update, op in ((sc.index_add, ExprOp.PUT_ADD), (sc.index_set, ExprOp.PUT)):
    node = update(x, [3, 1], v)
    assert node.op == op and node.args[1].op == ExprOp.CONST
    np.testing.assert_array_equal(node.args[1].value, [3, 1])
    viewed = update(m, [3, 1], v)
    assert viewed.op == ExprOp.RESHAPE and viewed.args[0].op == op and viewed.args[0].shape == (6,)
  assert sc.index_add(x, [0, 2], v) is sc.put_add(x, sc.const(np.array([0, 2]), dtype="int64"), v)


def test_constant_puts_lower_without_range_checks() -> None:
  """No lane of a constant put is tested against the axis or sent to a scratch slot: the lanes that
  drop are dropped when the code is generated."""
  b, w, i = sc.sym("b", (2, 3, 5)), sc.sym("w", (2, 3, 6)), sc.sym("i", 6, dtype="int64")
  constant = render_c_source(_fn("ix_put_c", [b, w], [sc.put_add(b, sc.const(IDX, dtype="int64"), w)]))
  runtime = render_c_source(_fn("ix_put_r", [b, w, i], [sc.put_add(b, i, w)]))
  assert "?" in runtime and "?" not in constant


def test_constant_puts_of_constants_fold_to_numpy() -> None:
  _, point = _put_inputs()
  for got, want in zip(_cases_of([sc.const(p) for p in point], sc.const(IDX, dtype="int64")), _numpy_of(point, IDX), strict=True):
    folded = sc.simplify(got)
    assert folded.op == ExprOp.CONST
    np.testing.assert_array_equal(folded.value, want)


@pytest.mark.parametrize("add", [True, False])
def test_constant_put_derivatives_match_numpy(add: bool) -> None:
  """In the base and the values, three ways, with the pattern exactly what lands: repeated,
  overwritten and dropped lanes on three rows."""
  b, w = sc.sym("b", (3, 5)), sc.sym("w", (3, 6))
  out = (sc.put_add if add else sc.put)(b, sc.const(IDX, dtype="int64"), w)
  rng = np.random.default_rng(5)
  point = [rng.standard_normal((3, 5)), rng.standard_normal((3, 6))]
  d_base, d_values = _np_put_jacobians((3, 5), IDX, 6, add=add)
  for wrt, want, tag in ((b, d_base, "b"), (w, d_values, "w")):
    for got in _three_jacobians(out, wrt, [b, w], point, f"ix_putd_{int(add)}_{tag}"):
      np.testing.assert_array_equal(got, want)
    assert (_pattern(out, wrt) == (want != 0)).all()


@pytest.mark.parametrize("add", [True, False])
@pytest.mark.parametrize("indices", [[3, 0, 3, 7], [4, 0, 2, 1]], ids=["repeated_dropped", "every_lane_lands"])
def test_constant_puts_agree_with_run_time_puts_bit_for_bit(add: bool, indices: list[int]) -> None:
  """The constant path, and its gather-based adjoint where every lane lands, give the bits the
  run-time path gives: values and gradients."""
  b, w, i = sc.sym("b", (2, 5)), sc.sym("w", (2, 4)), sc.sym("i", 4, dtype="int64")
  iv = np.array(indices)

  def outs(idx: Expr) -> list[Expr]:
    z = (sc.put_add if add else sc.put)(b.sin(), idx, w * w)
    f = (z * z.cos()).sum()
    return [z, sc.gradient(f, b), sc.gradient(f, w)]

  tag = f"{int(add)}_{indices[0]}"
  constant = _fn(f"ix_bits_c_{tag}", [b, w], outs(sc.const(iv, dtype="int64")))
  runtime = _fn(f"ix_bits_r_{tag}", [b, w, i], outs(i))
  rng = np.random.default_rng(6)
  bv, wv = rng.standard_normal((2, 5)), rng.standard_normal((2, 4))
  for a, r in zip(constant._flat_numerical_call(bv, wv), runtime._flat_numerical_call(bv, wv, iv.astype(float)), strict=True):
    np.testing.assert_array_equal(a, r)


def _run_both(monkeypatch: pytest.MonkeyPatch, build, point, *, expect_in_place: bool) -> list[np.ndarray]:
  fn = build("ip")
  assert ("_inplace_raw" in render_c_source(fn)) == expect_in_place
  got = fn._flat_numerical_call(*point)
  monkeypatch.setattr(lowering, "DONATE_CARRIES", False)
  ref = build("two")._flat_numerical_call(*point)
  monkeypatch.setattr(lowering, "DONATE_CARRIES", True)
  for a, r in zip(got, ref, strict=True):
    np.testing.assert_array_equal(a, r)
  return got


def test_a_matrix_carry_updated_through_its_flat_view_stays_in_place(monkeypatch: pytest.MonkeyPatch) -> None:
  """``index_add`` then ``index_set`` on a ``(2, 3)`` carry: each a put on the flat view, reshaped
  back, which the update chain looks through. The second reads entry 0 and writes entry 5."""
  c, u = sc.sym("c", (2, 3)), sc.sym("u", 2)

  def build(tag: str) -> sc.Function:
    u1 = sc.index_add(c, [0, 4], u)
    body = _fn(f"ix_flat_body_{tag}", [c, u], [sc.index_set(u1, [5], u1[0, :1] * 0.5)])
    c0, us = sc.sym("c0", (2, 3)), sc.sym("us", 8)
    (fin,) = sc.scan(body, c0, [(us, 0, 2)], length=4)
    return _fn(f"ix_flat_{tag}", [c0, us], [fin])

  c0, us = np.random.default_rng(7).integers(-24, 25, size=(2, 3)) / 8.0, np.arange(8.0) / 4.0
  (got,) = _run_both(monkeypatch, build, (c0, us), expect_in_place=True)
  want = c0.copy().reshape(-1)
  for k in range(4):
    want[[0, 4]] += us[2 * k : 2 * k + 2]
    want[5] = want[0] * 0.5
  np.testing.assert_array_equal(got, want.reshape(2, 3))
