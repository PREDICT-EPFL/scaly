"""Coefficients and data as expressions: runtime tables, decision variables, points known now."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import BSpline as SciBSpline

import scaly as sc
from scaly import interp
from scaly.interp.fit import DENSE_FIT
from scaly.passes.lowering import lower_function

from .helpers import inside_points, numbers, scale
from .test_spline import random_knots

EXTRAPS = ["extend", "linear", "clamp", "periodic", "fill"]


def fn_of(name: str, inputs: dict[str, sc.Expr], outputs: dict[str, sc.Expr]) -> sc.Function:
  return sc.Function._from_exprs(name, list(inputs.values()), list(outputs.values()), list(inputs), list(outputs))


def well_spaced(rng: np.random.Generator, n: int, lo: float = 0.0, hi: float = 1.0) -> np.ndarray:
  g = np.linspace(lo, hi, n)
  g[1:-1] += rng.uniform(-0.2, 0.2, n - 2) * (g[1] - g[0])
  return g


@pytest.mark.parametrize("strategy", ["auto", "basis"])
def test_expression_coefficients_evaluate_as_the_numbers_do(strategy: str) -> None:
  rng = np.random.default_rng(1)
  knots = (random_knots(rng, 3, 5, repeat=False), random_knots(rng, 2, 4, repeat=False))
  c = rng.normal(size=(knots[0].size - 4, knots[1].size - 3, 2))
  cs = sc.sym("c", c.shape)
  f = interp.BSpline(knots, cs, (3, 2), strategy=strategy)  # ty: ignore[invalid-argument-type]
  assert f.strategy == "basis" and not f.constant
  ref = interp.BSpline(knots, c, (3, 2))
  pts = inside_points(tuple(ax.edges for ax in f.axes), rng, 60)
  point, batch = sc.sym("p", 2), sc.sym("b", pts.shape)
  fn = fn_of("expr_coeffs", {"p": point, "b": batch, "c": cs}, {"one": f(point), "many": f(batch), "deriv": f.derivative(1, axis=1)(batch)})
  one, many, deriv = fn((pts[3], pts, c))
  np.testing.assert_allclose(one, ref.to_scipy()(pts[3]), rtol=0, atol=1e-13 * scale(c))
  np.testing.assert_allclose(many, ref.to_scipy()(pts), rtol=0, atol=1e-13 * scale(c))
  np.testing.assert_allclose(deriv, ref.to_scipy()(pts, nu=(0, 1)), rtol=0, atol=1e-11 * scale(c))
  with pytest.raises(ValueError, match="constant coefficients"):
    f.to_scipy()
  with pytest.raises(ValueError, match="strategy='pp'"):
    interp.BSpline(knots, cs, (3, 2), strategy="pp")


def test_the_jacobian_in_the_coefficients_is_the_basis() -> None:
  """At a symbolic point the row is dense in structure (any cell may be selected), and its values are
  the basis functions at the point; at points known now ``at()`` has exactly the basis's pattern."""
  rng = np.random.default_rng(2)
  t = random_knots(rng, 3, 8)
  n = t.size - 4
  c = sc.sym("c", n)
  f = interp.BSpline(t, c, 3)
  pts = inside_points((f.axes[0].edges,), rng, 20)
  x = sc.sym("x", pts.size)
  jac = sc.jacobian(f(x), c)
  assert sc.jacobian_sparsity(f(x), c).nnz == pts.size * n
  B = SciBSpline.design_matrix(pts, t, 3).toarray()
  np.testing.assert_allclose(fn_of("coeff_jac", {"x": x, "c": c}, {"j": jac})((pts, rng.normal(size=n))), B, rtol=0, atol=1e-14)
  at = f.at(pts)
  pattern = sc.jacobian_sparsity(at, c)
  structure = SciBSpline.design_matrix(pts, t, 3).tocoo()  # its stored entries, a value of 0 at a knot included
  assert sorted(zip(pattern.rows, pattern.cols, strict=True)) == sorted(zip(structure.row.tolist(), structure.col.tolist(), strict=True))
  cv = rng.normal(size=n)
  np.testing.assert_allclose(fn_of("at_values", {"c": c}, {"v": at})(cv), B @ cv, rtol=0, atol=1e-14)


@pytest.mark.parametrize("extrap", EXTRAPS)
def test_basis_and_at_follow_the_extrapolation(extrap: str) -> None:
  rng = np.random.default_rng(3)
  knots = (random_knots(rng, 3, 4, 0.0, 1.0, repeat=False), random_knots(rng, 1, 3, -1.0, 1.0, repeat=False))
  c = rng.normal(size=(knots[0].size - 4, knots[1].size - 2))
  f = interp.BSpline(knots, c, (3, 1), extrap=extrap, fill=-3.0)  # ty: ignore[invalid-argument-type]
  pts = np.column_stack([rng.uniform(-0.5, 1.5, 200), rng.uniform(-1.5, 1.5, 200)])
  x = sc.sym("x", pts.shape)
  np.testing.assert_allclose(numbers(f.at(pts)), fn_of(f"extrap_{extrap}", {"x": x}, {"y": f(x)})(pts), rtol=0, atol=1e-12)
  B = f.basis(pts)
  inside = np.all((pts >= [0.0, -1.0]) & (pts <= [1.0, 1.0]), axis=1)
  np.testing.assert_allclose((B @ c.reshape(-1))[inside], f.to_scipy()(pts[inside]), rtol=0, atol=1e-12)
  if extrap == "fill":
    assert np.all(B.toarray()[~inside] == 0.0)
  cs = sc.sym("c", c.shape)
  g = interp.BSpline(knots, cs, (3, 1), extrap=extrap, fill=-3.0)  # ty: ignore[invalid-argument-type]
  np.testing.assert_allclose(fn_of(f"at_{extrap}", {"c": cs}, {"y": g.at(pts)})(c), numbers(f.at(pts)), rtol=0, atol=1e-12)


def test_basis_equals_scipys_design_matrix() -> None:
  rng = np.random.default_rng(4)
  t = random_knots(rng, 4, 6)
  f = interp.BSpline(t, np.zeros(t.size - 5), 4)
  pts = inside_points((f.axes[0].edges,), rng, 100)
  np.testing.assert_allclose(f.basis(pts).toarray(), SciBSpline.design_matrix(pts, t, 4).toarray(), rtol=0, atol=1e-13)
  with pytest.raises(ValueError, match="finite"):
    f.basis(np.array([0.0, np.nan]))


DATA_KINDS = ["nearest", "zoh", "linear", "cubic", "spline", "smooth_linear", "pchip", "akima", "makima", "steffen"]


@pytest.mark.parametrize("kind", DATA_KINDS)
def test_expression_data_fit_as_the_numbers_do(kind: str) -> None:
  """The fit in the graph equals the one done now; for the kinds linear in the data the Jacobian in
  the data is the fit of the identity evaluated at the points, for the others finite differences."""
  rng = np.random.default_rng(len(kind))
  g = well_spaced(rng, 11)
  y = np.sin(4 * g) + 0.3 * rng.normal(size=g.size)
  ys = sc.sym("y", g.size)
  f = interp.interpolant(g, ys, kind=kind)  # ty: ignore[invalid-argument-type]
  ref = interp.interpolant(g, y, kind=kind)  # ty: ignore[invalid-argument-type]
  pts = np.concatenate([inside_points((ref.axes[0].edges,), rng, 30), [-0.2, 1.3]])
  x = sc.sym("x", pts.size)
  fn = fn_of(f"data_{kind}", {"x": x, "y": ys}, {"v": f(x), "j": sc.jacobian(f(x), ys)})
  value, jac = fn((pts, y))
  np.testing.assert_allclose(value, fn_of(f"num_{kind}", {"x": x}, {"v": ref(x)})(pts), rtol=0, atol=1e-13)
  if kind in ("pchip", "akima", "makima", "steffen"):
    step = 1e-6
    fd = np.stack([(fn((pts, y + step * e))[0] - fn((pts, y - step * e))[0]) / (2 * step) for e in np.eye(g.size)], axis=1)
    np.testing.assert_allclose(jac, fd, rtol=0, atol=1e-7)
  else:
    identity = interp.interpolant(g, np.eye(g.size), kind=kind)  # ty: ignore[invalid-argument-type]
    np.testing.assert_allclose(jac, fn_of(f"eye_{kind}", {"x": x}, {"v": identity(x)})(pts), rtol=0, atol=1e-12)


@pytest.mark.parametrize("bc", ["not-a-knot", "natural", "clamped", "periodic"])
def test_a_large_cubic_solves_its_slopes_in_scans(bc: str) -> None:
  rng = np.random.default_rng(len(bc))
  n = DENSE_FIT + 44
  g = well_spaced(rng, n, 0.0, 10.0)
  y = np.column_stack([np.sin(g), np.cos(0.7 * g) + 0.1 * rng.normal(size=n)])
  if bc == "periodic":
    y[-1] = y[0]
  ys = sc.sym("y", y.shape)
  f = interp.interpolant(g, ys, kind="cubic", bc=bc)  # ty: ignore[invalid-argument-type]
  assert f.axes[0].n == 2 * n  # the Hermite form: doubled knots
  ref = interp.interpolant(g, y, kind="cubic", bc=bc)  # ty: ignore[invalid-argument-type]
  pts = np.concatenate([rng.uniform(0.0, 10.0, 300), g[::7]])  # between the sites, where the slopes matter
  x = sc.sym("x", pts.size)
  fn = fn_of(f"tridiag_{bc}", {"x": x, "y": ys}, {"v": f(x), "g": sc.gradient(f(x)[:, 1].sum(), ys)})
  value, grad = fn((pts, y))
  np.testing.assert_allclose(value, ref.to_scipy()(pts), rtol=0, atol=1e-12)
  basis = np.eye(n)
  if bc == "periodic":  # only closed data fit: move the first and last values together
    basis = basis[1:-1]
    basis = np.vstack([np.eye(n)[0] + np.eye(n)[-1], basis])
    grad = np.vstack([grad[:1] + grad[-1:], grad[1:-1]])
  columns = np.stack([interp.interpolant(g, row, kind="cubic", bc=bc).to_scipy()(pts) for row in basis], axis=0)  # ty: ignore[invalid-argument-type]
  np.testing.assert_allclose(grad[:, 1], columns.sum(axis=1), rtol=0, atol=1e-11)
  np.testing.assert_allclose(grad[:, 0], 0.0, atol=0)


@pytest.mark.parametrize("kind", ["pchip", "akima", "makima", "steffen"])
def test_shape_preserving_data_differentiate_in_reverse_through_plateaus(kind: str) -> None:
  """A plateau (zero secants) and a straight run (equal secants) are where the slope formulas divide
  by zero in the branch they do not take; reverse mode multiplies that branch's derivative by zero,
  which must not be 0 * inf. The gradient in the data is finite, and where the fit has a kink in a
  datum it lies between the one-sided finite differences (it is one of them, or their mean under
  ``nonsmooth="split"``)."""
  g = np.arange(10.0)
  y = np.array([0.0, 1.0, 1.0, 1.0, 2.0, 3.0, 4.0, 3.0, 5.0, 5.5])
  ys = sc.sym("y", g.size)
  f = interp.interpolant(g, ys, kind=kind)  # ty: ignore[invalid-argument-type]
  pts = np.linspace(0.3, 8.7, 13)
  x = sc.sym("x", pts.size)
  weights = np.linspace(1.0, 2.0, pts.size)
  fn = fn_of(f"plateau_{kind}", {"x": x, "y": ys}, {"v": f(x), "g": sc.gradient((f(x) * sc.const(weights)).sum(), ys)})
  value, grad = fn((pts, y))
  assert np.all(np.isfinite(grad))
  step = 1e-7
  ahead = np.array([((fn((pts, y + step * e))[0] - value) * weights).sum() / step for e in np.eye(g.size)])
  behind = np.array([((value - fn((pts, y - step * e))[0]) * weights).sum() / step for e in np.eye(g.size)])
  assert np.all(grad >= np.minimum(ahead, behind) - 1e-5) and np.all(grad <= np.maximum(ahead, behind) + 1e-5)
  smooth = np.abs(ahead - behind) < 1e-5
  assert smooth.sum() >= 2
  np.testing.assert_allclose(grad[smooth], ahead[smooth], rtol=0, atol=1e-5)


def test_constant_maps_along_a_later_axis_of_a_3d_array() -> None:
  """A fit along the last of three axes, and a derivative spline along it: the axis is moved to the
  front for the product and back after, a permutation that is not its own inverse."""
  rng = np.random.default_rng(9)
  g = (well_spaced(rng, 4), well_spaced(rng, 5), well_spaced(rng, 6))
  v = rng.normal(size=(4, 5, 6))
  vs = sc.sym("v", v.shape)
  kinds = ("linear", "linear", "cubic")
  f, ref = interp.interpolant(g, vs, kind=kinds), interp.interpolant(g, v, kind=kinds)
  pts = inside_points(g, rng, 30)
  x = sc.sym("x", pts.shape)
  fn = fn_of("later_axis", {"x": x, "v": vs}, {"y": f(x), "d": f.derivative(1, axis=2)(x)})
  y, d = fn((pts, v))
  np.testing.assert_allclose(y, ref.to_scipy()(pts), rtol=0, atol=1e-12)
  np.testing.assert_allclose(d, ref.to_scipy()(pts, nu=(0, 0, 1)), rtol=0, atol=1e-11)


def test_expression_data_in_2d_and_a_smoothing_fit() -> None:
  rng = np.random.default_rng(5)
  g = (well_spaced(rng, 7), well_spaced(rng, 6, -1.0, 1.0))
  v = rng.normal(size=(7, 6))
  vs = sc.sym("v", v.shape)
  f = interp.interpolant(g, vs, kind=("cubic", "linear"))
  ref = interp.interpolant(g, v, kind=("cubic", "linear"))
  pts = inside_points(g, rng, 40)
  x = sc.sym("x", pts.shape)
  np.testing.assert_allclose(fn_of("data_2d", {"x": x, "v": vs}, {"y": f(x)})((pts, v)), ref.to_scipy()(pts), rtol=0, atol=1e-13)
  sites = np.sort(rng.uniform(0.0, 1.0, 40))
  data = np.sin(5 * sites) + 0.05 * rng.normal(size=40)
  ds = sc.sym("d", 40)
  for method in ("pspline", "cubic"):
    sm, num = interp.smoothing(sites, ds, lam=1e-4, method=method), interp.smoothing(sites, data, lam=1e-4, method=method)
    p = sc.sym("p", 25)
    q = np.linspace(sites[0], sites[-1], 25)
    np.testing.assert_allclose(fn_of(f"smooth_{method}", {"p": p, "d": ds}, {"y": sm(p)})((q, data)), num.to_scipy()(q), rtol=0, atol=1e-12)
  with pytest.raises(ValueError, match="give lam"):
    interp.smoothing(sites, ds)


def test_calculus_of_expression_coefficients() -> None:
  rng = np.random.default_rng(6)
  t = random_knots(rng, 3, 6, 0.0, 2.0)
  c = rng.normal(size=t.size - 4)
  cs = sc.sym("c", c.size)
  for extrap in ("linear", "clamp", "fill"):
    f, ref = interp.BSpline(t, cs, 3, extrap=extrap, fill=0.0), interp.BSpline(t, c, 3, extrap=extrap, fill=0.0)
    anti = f.antiderivative()
    x = sc.sym("x", 7)
    xs = np.linspace(-0.5, 2.5, 7)
    got = fn_of(f"calc_{extrap}", {"x": x, "c": cs}, {"a": anti(x), "i": sc.stack([f.integrate(-0.5, 1.7), f.integrate(0.2, 2.9)])})((xs, c))
    np.testing.assert_allclose(got[0], fn_of(f"calc_ref_{extrap}", {"x": x}, {"a": ref.antiderivative()(x)})(xs), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(got[1], [ref.integrate(-0.5, 1.7), ref.integrate(0.2, 2.9)], rtol=0, atol=1e-12)
  filled = interp.BSpline(t, cs, 3, extrap="fill", fill=2.5)
  np.testing.assert_allclose(
    fn_of("fill_integral", {"c": cs}, {"i": filled.integrate(-0.5, 2.5)})(c),
    interp.BSpline(t, c, 3, extrap="fill", fill=2.5).integrate(-0.5, 2.5),
    rtol=1e-13,
  )
  with pytest.raises(ValueError, match="constant coefficients"):
    interp.BSpline(t, cs, 3).inverse()


def test_a_broadcast_tables_fit_runs_once_per_call() -> None:
  """A map whose body fits a table passed to every point: hoisting moves the fit (the constant map's
  product) out of the loop into a prologue that runs once."""
  g = np.linspace(0.0, 1.0, 9)
  x, data = sc.sym("x"), sc.sym("data", 9)
  body = sc.Function._from_exprs("interp_stage", [x, data], [interp.interpolant(g, data, kind="cubic")(x)], ["x", "data"], ["y"])
  xs, table = sc.sym("xs", 16), sc.sym("table", 9)
  fn = sc.Function._from_exprs("interp_map", [xs, table], [sc.vmap(body, 16, {"x": (xs, 0, 1), "data": (table, 0, 0)})], ["xs", "table"], ["y"])
  prog = lower_function(fn)
  names = [pr.attrs["name"] for pr in prog.args[: int(prog.attrs["proc_count"])]]
  assert names == ["interp_stage_hoist_1", "interp_stage_hoisted_1", "interp_map"]
  prologue, hoisted = prog.args[:2]
  ops = lambda proc: {n.op.value for stmt in proc.args[int(proc.attrs["param_count"]) :] for n in _walk(stmt)}
  assert "mul" in ops(prologue) and "load" in ops(hoisted)
  values = np.sin(3 * g)
  pts = np.linspace(-0.1, 1.1, 16)
  np.testing.assert_allclose(fn((pts, values)), evaluate_numeric(interp.interpolant(g, values, kind="cubic"), pts), rtol=1e-13, atol=1e-14)


def _walk(node):  # type: ignore[no-untyped-def]
  yield node
  for arg in node.args:
    yield from _walk(arg)


def evaluate_numeric(f: interp.BSpline, pts: np.ndarray) -> np.ndarray:
  x = sc.sym("x", pts.size)
  return fn_of(f"num_eval_{f.digest}", {"x": x}, {"y": f(x)})(pts)


ca = pytest.importorskip("casadi")


@pytest.mark.parametrize("dims", [(9,), (6, 5)])
@pytest.mark.parametrize("method", ["linear", "bspline"])
def test_the_data_jacobian_matches_casadi_inlined(method: str, dims: tuple[int, ...]) -> None:
  """CasADi differentiates its parametric interpolant in the data only when inlined; then the two
  Jacobians agree."""
  rng = np.random.default_rng(len(dims))
  g = tuple(well_spaced(rng, n, 0.0, 1.0 + d) for d, n in enumerate(dims))
  v = rng.normal(size=dims)
  vs = sc.sym("v", dims)
  f = interp.interpolant(g if len(g) > 1 else g[0], vs, kind="linear" if method == "linear" else "cubic")
  pts = inside_points(g, rng, 12)
  pts = pts if pts.ndim > 1 else pts[:, None]
  x = sc.sym("x", pts.shape if len(g) > 1 else (pts.shape[0],))
  ours = fn_of(f"ca_jac_{method}_{len(dims)}", {"x": x, "v": vs}, {"j": sc.jacobian(f(x), vs).reshape((pts.shape[0], v.size))})
  jac = ours((pts if len(g) > 1 else pts[:, 0], v))
  xs, data = ca.MX.sym("x", len(g)), ca.MX.sym("v", v.size)
  itp = ca.interpolant("par", method, [list(a) for a in g], 1, {"inline": True})
  J = ca.Function("J", [xs, data], [ca.jacobian(itp(xs, data), data)])
  theirs = np.stack([np.asarray(J(p, np.ravel(v, order="F"))).reshape(-1) for p in pts])
  fortran = np.ravel(np.arange(v.size).reshape(dims), order="F")  # CasADi's data order
  np.testing.assert_allclose(jac[:, fortran], theirs, rtol=0, atol=1e-12 * scale(theirs))


def _fn(inputs: dict[str, sc.Expr], outputs: dict[str, sc.Expr], name: str) -> sc.Function:
  return sc.Function._from_exprs(name, list(inputs.values()), list(outputs.values()), list(inputs), list(outputs))


@pytest.mark.parametrize(("kind", "sites"), [("cubic", 9), ("cubic", DENSE_FIT), ("cubic", DENSE_FIT + 5), ("spline", 9), ("spline", 30)])
def test_a_periodic_fit_of_expression_data_is_the_numeric_fit(kind: str, sites: int) -> None:
  """The identity fitted for the dense map reads the last value as the first, so every column
  closes up as a periodic fit needs; the numbers' fit of closed data is the reference."""
  x = np.linspace(0.0, 2.0 * np.pi, sites)
  y = np.sin(x) + 0.3 * np.cos(3 * x)
  y[-1] = y[0]
  kw = {"bc": "periodic"} | ({"degree": 3} if kind == "spline" else {})
  v = sc.sym("v", sites)
  f, ref = interp.interpolant(x, v, kind=kind, **kw), interp.interpolant(x, y, kind=kind, **kw)  # ty: ignore[invalid-argument-type]
  pts = np.array([0.3, 2.0, 5.9, 7.0, -1.0])
  p = sc.sym("p", pts.shape)
  np.testing.assert_allclose(
    _fn({"p": p, "v": v}, {"y": f(p)}, f"periodic_expr_{kind}_{sites}")((pts, y)), ref.to_scipy()(np.mod(pts, 2.0 * np.pi)), rtol=0, atol=1e-12
  )


def test_a_large_expression_cubic_along_a_later_axis() -> None:
  rng = np.random.default_rng(3)
  g = (np.linspace(0.0, 1.0, 3), np.linspace(0.0, 10.0, DENSE_FIT + 10))
  v = rng.normal(size=(3, DENSE_FIT + 10))
  vs = sc.sym("v", v.shape)
  f, ref = interp.interpolant(g, vs, kind=("linear", "cubic")), interp.interpolant(g, v, kind=("linear", "cubic"))
  pts = np.column_stack([rng.uniform(0.0, 1.0, 40), rng.uniform(0.0, 10.0, 40)])
  x = sc.sym("x", pts.shape)
  np.testing.assert_allclose(_fn({"x": x, "v": vs}, {"y": f(x)}, "later_axis_scan")((pts, v)), ref.to_scipy()(pts), rtol=0, atol=1e-11)
