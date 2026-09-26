"""The step number as a loop-body input: ``scan(..., index=True)`` and ``while_loop(..., index=True)``.

The body's second input is an ``int64`` scalar counting the steps from zero. It must compute what
an explicit table of step numbers sliced one entry per step computes, in value and in every
derivative, while the generated C computes the entry from the loop counter and stores no table.
"""

from __future__ import annotations

import re
import time

import numpy as np
import pytest

import scaly as sc
from scaly.ad import finite_difference
from scaly.ad.derivatives import gradient, hessian, jacobian
from scaly.codegen import render_c_module
from scaly.function.sugar import step_numbers
from scaly.ir.expr import ExprOp, topo

N = 7


def _fn(name, inputs, outputs):
  return sc.Function._from_exprs(name, list(inputs), list(outputs), [str(x.name) for x in inputs], [f"o{k}" for k in range(len(outputs))])


def _body(name: str = "tv_step") -> sc.Function:
  """A time-varying step: the step number enters through a float time and through a branch."""
  c, k, u = sc.sym("c", 2), sc.sym("k", (), dtype="int64"), sc.sym("u", 1)
  t = k.cast("float64")
  first = sc.where(sc.equal(k, 0), 2.0, 1.0)  # ``k == 0`` would be Python identity, not a comparison
  nxt = sc.stack([c[0] + 0.1 * t * c[1] + first * u[0], c[1] * (1.0 + 0.01 * t) + (c[0] * u[0]).sin()])
  return sc.Function._from_exprs(name, [c, k, u], [nxt, sc.stack([t * c[0]])], ["c", "k", "u"], ["n", "y"])


def _numpy(x0: np.ndarray, us: np.ndarray, *, steps: np.ndarray | None = None) -> float:
  c, s = x0.copy(), 0.0
  steps = np.arange(len(us)) if steps is None else steps
  for i, u in zip(steps, us, strict=True):
    s += (i * c[0]) ** 2
    c = np.array([c[0] + 0.1 * i * c[1] + (2.0 if i == 0 else 1.0) * u, c[1] * (1 + 0.01 * i) + np.sin(c[0] * u)])
  return float(c @ c + s)


def _objective(*, table: bool = False, length: int = N):
  x0, us = sc.sym("x0", 2), sc.sym("us", length)
  if table:
    fin, ys = sc.scan(_body(), x0, [(sc.const(np.arange(length), dtype="int64"), 0, 1), (us, 0, 1)], length=length)
  else:
    fin, ys = sc.scan(_body(), x0, [(us, 0, 1)], length=length, index=True)
  return x0, us, sc.stack([sc.sumsqr(fin) + sc.sumsqr(ys)])[0]


POINT = (np.array([0.3, -0.2]), np.linspace(-0.5, 0.5, N))


def _tables(src: str) -> list[str]:
  return re.findall(r"static const int64_t \w+\[\d+\] = \{[^}]*\}", src)


def test_scan_index_value_matches_numpy() -> None:
  x0, us, v = _objective()
  got = _fn("si_val", [x0, us], [v])._flat_numerical_call(*POINT)[0]
  np.testing.assert_allclose(got, _numpy(*POINT), rtol=1e-13)


@pytest.mark.parametrize("which", [0, 1])
@pytest.mark.parametrize("kind", ["jacobian", "gradient", "hessian"])
def test_scan_index_derivatives_match_table_and_fd(monkeypatch: pytest.MonkeyPatch, which: int, kind: str) -> None:
  """Every derivative equals the explicit-table version to rounding and finite differences to truncation."""
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  results = []
  for table in (False, True):
    x0, us, v = _objective(table=table)
    wrt = (x0, us)[which]
    d = {"jacobian": lambda: jacobian(v, wrt), "gradient": lambda: gradient(v, wrt), "hessian": lambda: hessian(v, wrt)}[kind]()
    results.append(_fn(f"si_{kind}_{which}_{int(table)}", [x0, us], [d])._flat_numerical_call(*POINT)[0])
  np.testing.assert_allclose(results[0], results[1], rtol=1e-12, atol=1e-13)

  def value(z: np.ndarray) -> np.ndarray:
    pt = list(POINT)
    pt[which] = z
    return np.array([_numpy(*pt)])

  if kind == "hessian":
    x0, us, v = _objective()
    grad = _fn("si_g_fd", [x0, us], [gradient(v, (x0, us)[which])])

    def value(z: np.ndarray) -> np.ndarray:  # noqa: F811 - the gradient is the function differenced here
      pt = list(POINT)
      pt[which] = z
      return grad._flat_numerical_call(*pt)[0].reshape(-1)

  fd = finite_difference(value, POINT[which])
  np.testing.assert_allclose(results[0].reshape(-1), np.asarray(fd).reshape(-1), rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("kind", ["value", "gradient", "hessian"])
def test_scan_index_stores_no_table(kind: str) -> None:
  """The step number is arithmetic on the loop counter, forwards and in the backward scan, where it
  counts down; no integer table reaches the C."""
  x0, us, v = _objective(length=40)
  out = {"value": v, "gradient": gradient(v, us), "hessian": hessian(v, us)}[kind]
  src = str(render_c_module(_fn(f"si_c_{kind}", [x0, us], [out])).source)
  assert not _tables(src), _tables(src)
  assert "int64_t" in src  # the body still takes the step number
  if kind != "value":
    assert re.search(r"\(39 - k_\w+\)", src), "the backward scan counts its step number down"


def test_scan_index_backward_walk_counts_steps() -> None:
  """A sliced input walked backwards (a negative stride) does not change the step number."""
  c, k, u = sc.sym("c", 1), sc.sym("k", (), dtype="int64"), sc.sym("u", 1)
  body = sc.Function._from_exprs(
    "bw_step", [c, k, u], [c * 0.5 + k.cast("float64") * u[0], sc.stack([k.cast("float64")])], ["c", "k", "u"], ["n", "y"]
  )
  x0, us = sc.sym("x0", 1), sc.sym("us", 5)
  fin, ys = sc.scan(body, x0, [(us, 4, -1)], length=5, index=True)
  got_fin, got_ys = _fn("bw", [x0, us], [fin, ys])._flat_numerical_call(np.array([1.0]), np.arange(5.0))
  ref = 1.0
  for i in range(5):
    ref = ref * 0.5 + i * (4 - i)
  np.testing.assert_allclose(got_fin, [ref])
  np.testing.assert_array_equal(got_ys, np.arange(5.0))


def test_scan_index_zero_length() -> None:
  x0, us, v = _objective(length=0)
  np.testing.assert_allclose(_fn("si_zero", [x0, us], [v])._flat_numerical_call(POINT[0], np.zeros(0))[0], POINT[0] @ POINT[0])


def test_scan_non_affine_integer_table_is_read_per_step() -> None:
  """An integer table that is not affine in the step still reaches the body entry by entry."""
  steps = np.array([3, 1, 4, 1, 5, 9, 2])
  x0, us = sc.sym("x0", 2), sc.sym("us", N)
  fin, ys = sc.scan(_body(), x0, [(sc.const(steps, dtype="int64"), 0, 1), (us, 0, 1)], length=N)
  v = sc.stack([sc.sumsqr(fin) + sc.sumsqr(ys)])[0]
  got = _fn("si_perm", [x0, us], [v, gradient(v, us)])._flat_numerical_call(*POINT)
  np.testing.assert_allclose(got[0], _numpy(*POINT, steps=steps), rtol=1e-13)
  fd = finite_difference(lambda z: np.array([_numpy(POINT[0], z, steps=steps)]), POINT[1])
  np.testing.assert_allclose(got[1].reshape(-1), np.asarray(fd).reshape(-1), rtol=1e-6, atol=1e-7)


def test_scan_index_sparsity_matches_table_and_keeps_the_periodic_walk() -> None:
  """The index brings no dependence, so the pattern is the table version's, and a long loop whose
  only moving slice is the index still stops at the first repeated pattern."""
  x0, us, v = _objective()
  x0t, ust, vt = _objective(table=True)
  np.testing.assert_array_equal(sc.jacobian_sparsity(v.reshape((1,)), us).to_mask(), sc.jacobian_sparsity(vt.reshape((1,)), ust).to_mask())
  c, k = sc.sym("c", 3), sc.sym("k", (), dtype="int64")
  body = sc.Function._from_exprs("sp_step", [c, k], [sc.stack([c[0] + k.cast("float64"), c[0] * c[1], c[2]])], ["c", "k"], ["n"])
  x = sc.sym("x", 3)
  (fin,) = sc.scan(body, x, [], length=200_000, index=True)
  start = time.perf_counter()
  mask = sc.jacobian_sparsity(fin, x).to_mask()
  assert time.perf_counter() - start < 5.0
  np.testing.assert_array_equal(mask, [[1, 0, 0], [1, 1, 0], [0, 0, 1]])


def test_scan_index_validation() -> None:
  c, u = sc.sym("c", 1), sc.sym("u", 1)
  plain = sc.Function._from_exprs("no_idx", [c, u], [c + u], ["c", "u"], ["n"])
  with pytest.raises(ValueError, match="int64 scalar"):
    sc.scan(plain, sc.sym("x", 1), [(sc.sym("us", 3), 0, 1)], length=3, index=True)
  k2 = sc.sym("k", (2,), dtype="int64")
  wrong = sc.Function._from_exprs("wide_idx", [c, k2], [c], ["c", "k"], ["n"])
  with pytest.raises(ValueError, match="int64 scalar"):
    sc.scan(wrong, sc.sym("x", 1), [], length=3, index=True)
  k = sc.sym("k", (), dtype="int64")
  good = sc.Function._from_exprs("idx_u", [c, k, u], [c + u], ["c", "k", "u"], ["n"])
  with pytest.raises(ValueError, match="1 sliced inputs after the carry and the index, got 0"):
    sc.scan(good, sc.sym("x", 1), [], length=3, index=True)
  nodes = sc.scan(good, sc.sym("x", 1), [(sc.sym("us", 3), 0, 1)], length=3, index=True)
  assert nodes[0].args[1] is step_numbers(3)


# --- while_loop ---------------------------------------------------------------------------------

MAX_ITER = 40


def _while_parts():
  c, k = sc.sym("c", 2), sc.sym("k", (), dtype="int64")
  damp = 1.0 / (1.0 + 0.5 * k.cast("float64"))
  body = sc.Function._from_exprs("dn_step", [c, k], [sc.stack([c[0] - (c[0] * c[0] - c[1]) / (2.0 * c[0]) * damp, c[1]])], ["c", "k"], ["n"])
  cc = sc.sym("cc", 2)
  cond = sc.Function._from_exprs("dn_go", [cc], [(cc[0] * cc[0] - cc[1]).abs() > 1e-9], ["cc"], ["go"])
  a = sc.sym("a", 2)
  fin, n = sc.while_loop(cond, body, a, max_iter=MAX_ITER, index=True)
  return a, fin, n


def _while_numpy(a: np.ndarray) -> np.ndarray:
  c, i = a.copy(), 0
  while i < MAX_ITER and abs(c[0] ** 2 - c[1]) > 1e-9:
    c = np.array([c[0] - (c[0] ** 2 - c[1]) / (2 * c[0]) / (1 + 0.5 * i), c[1]])
    i += 1
  return np.array([c[0] ** 2 * c[1], i])


@pytest.mark.parametrize("start", [np.array([1.3, 2.0]), np.array([1.41, 2.0]), np.array([3.0, 0.5])])
def test_while_index_value_and_count(start: np.ndarray) -> None:
  a, fin, n = _while_parts()
  got = _fn("wi_val", [a], [sc.stack([fin[0] * fin[0] * fin[1], n])])._flat_numerical_call(start)[0]
  np.testing.assert_allclose(got, _while_numpy(start), rtol=1e-12)


@pytest.mark.parametrize("kind", ["jacobian", "gradient", "hessian"])
def test_while_index_derivatives(monkeypatch: pytest.MonkeyPatch, kind: str) -> None:
  monkeypatch.setenv("SCALY_STRICT_JVP_MANY", "1")
  a, fin, _ = _while_parts()
  v = fin[0] * fin[0] * fin[1]
  start = np.array([1.3, 2.0])
  if kind == "jacobian":
    got = _fn("wi_j", [a], [jacobian(v.reshape((1,)), a)])._flat_numerical_call(start)[0]
    fd = finite_difference(lambda z: _while_numpy(z)[:1], start)
  elif kind == "gradient":
    got = _fn("wi_g", [a], [gradient(v, a)])._flat_numerical_call(start)[0]
    fd = finite_difference(lambda z: _while_numpy(z)[:1], start)
  else:
    got = _fn("wi_h", [a], [hessian(v, a)])._flat_numerical_call(start)[0]
    grad = _fn("wi_hg", [a], [gradient(v, a)])
    fd = finite_difference(lambda z: grad._flat_numerical_call(z)[0].reshape(-1), start)
  np.testing.assert_allclose(got.reshape(-1), np.asarray(fd).reshape(-1), rtol=1e-6, atol=1e-7)


def test_while_index_forward_and_reverse_agree() -> None:
  a, fin, _ = _while_parts()
  got = _fn("wi_fr", [a], [jacobian(fin, a), sc.stack([gradient(fin[k], a) for k in range(2)])])._flat_numerical_call(np.array([1.3, 2.0]))
  np.testing.assert_allclose(got[0], got[1], rtol=1e-12, atol=1e-14)


def test_while_index_passes_the_counter() -> None:
  a, fin, _ = _while_parts()
  src = str(render_c_module(_fn("wi_c", [a], [fin, gradient(fin[0], a)])).source)
  assert not _tables(src)
  assert re.search(r"dn_step\w*\(const double\* \w+, const int64_t\* k", src)


def test_while_index_validation() -> None:
  c, k = sc.sym("c", 1), sc.sym("k", (), dtype="int64")
  cc = sc.sym("cc", 1)
  cond = sc.Function._from_exprs("wv_go", [cc], [cc[0] > 0.0], ["cc"], ["go"])
  indexed = sc.Function._from_exprs("wv_idx", [c, k], [c - 1.0], ["c", "k"], ["n"])
  plain = sc.Function._from_exprs("wv_plain", [c], [c - 1.0], ["c"], ["n"])
  with pytest.raises(ValueError, match="int64 scalar"):
    sc.while_loop(cond, plain, sc.sym("x", 1), max_iter=3, index=True)
  with pytest.raises(ValueError, match="body takes the carry and 0 params"):
    sc.while_loop(cond, indexed, sc.sym("x", 1), max_iter=3)
  fin, _ = sc.while_loop(cond, indexed, sc.sym("x", 1), max_iter=3, index=True)
  assert fin.op == ExprOp.WHILE and len(topo([fin])) == 2


def test_scan_uniform_constant_slice_is_passed_as_a_value() -> None:
  """The cotangent of a summed stacked output is a table of ones read one entry per step; the
  backward scan passes the one instead of storing the table."""
  x0, us = sc.sym("x0", 2), sc.sym("us", 40)
  fin, ys = sc.scan(_body(), x0, [(us, 0, 1)], length=40, index=True)
  v = sc.sumsqr(fin) + ys.sum()
  src = str(render_c_module(_fn("si_ones", [x0, us], [gradient(v, us)])).source)
  assert not re.search(r"static const double \w+\[40\]", src)
  point = (POINT[0], np.linspace(-0.5, 0.5, 40))
  rev, fwd = _fn("si_ones_run", [x0, us], [gradient(v, us), jacobian(v.reshape((1,)), us)])._flat_numerical_call(*point)
  np.testing.assert_allclose(rev.reshape(-1), fwd.reshape(-1), rtol=1e-11, atol=1e-12 * np.abs(fwd).max())


def test_scan_varying_float_table_is_still_read() -> None:
  """A float table whose entries differ along the walk stays a table, read one entry per step."""
  c, t = sc.sym("c", 1), sc.sym("t", ())
  body = sc.Function._from_exprs("ft_step", [c, t], [c * 0.5 + t], ["c", "t"], ["n"])
  x0 = sc.sym("x0", 1)
  times = np.array([0.5, 1.5, -2.0, 4.0, 0.25])
  (fin,) = sc.scan(body, x0, [(sc.const(times), 0, 1)], length=5)
  fn = _fn("ft", [x0], [fin])
  ref = 1.0
  for v in times:
    ref = ref * 0.5 + v
  np.testing.assert_allclose(fn._flat_numerical_call(np.array([1.0]))[0], [ref])
  assert re.search(r"static const double \w+\[5\]", str(render_c_module(fn).source))


def test_where_rejects_a_python_bool_condition() -> None:
  """``k == 0`` between an expression and a number is Python identity: ``where`` refuses the bool."""
  k = sc.sym("k", (), dtype="int64")
  with pytest.raises(TypeError, match="sc.equal"):
    sc.where(k == 0, 2.0, 1.0)  # noqa: SIM300 - the identity comparison is the point


@pytest.mark.parametrize("table", [[0.0, -0.0, 0.0], [-0.0, -0.0, -0.0]])
def test_scan_float_table_keeps_the_sign_of_zero(table: list[float]) -> None:
  """``-0.0`` and ``0.0`` compare equal but are different values: ``1 / z`` tells them apart."""
  c, z = sc.sym("c", ()), sc.sym("z", ())
  body = sc.Function._from_exprs(f"nz_step{len(set(map(str, table)))}", [c, z], [c, 1.0 / z], ["c", "z"], ["n", "y"])
  x0 = sc.sym("x0", ())
  _, ys = sc.scan(body, x0, [(sc.const(np.array(table)), 0, 1)], length=3)
  (got,) = _fn(f"nz{len(set(map(str, table)))}", [x0], [ys])._flat_numerical_call(np.array(1.0))
  with np.errstate(divide="ignore"):
    np.testing.assert_array_equal(got, 1.0 / np.array(table))
