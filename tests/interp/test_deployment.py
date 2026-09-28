"""What deployment needs: single precision, tables passed in at run time, their layout."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly import interp
from scaly.codegen import render_c_module, render_c_source

from .helpers import inside_points


@pytest.mark.parametrize("strategy", ["pp", "basis"])
@pytest.mark.parametrize("extrap", ["linear", "clamp", "extend", "fill"])
def test_float32_tables_evaluate_to_single_precision(strategy: str, extrap: str) -> None:
  g = (np.linspace(0.0, 1.0, 12), np.linspace(-1.0, 1.0, 9))
  v = np.sin(3 * g[0])[:, None] * np.cos(2 * g[1])[None, :] + 0.5
  kw = {"kind": ("cubic", "linear"), "strategy": strategy, "extrap": extrap, "fill": 0.25}
  single, double = interp.interpolant(g, v, dtype="float32", **kw), interp.interpolant(g, v, **kw)  # ty: ignore[invalid-argument-type]
  assert single.dtype == sc.dtypes.float32 and "dtype=float32" in repr(single)
  rng = np.random.default_rng(1)
  pts = np.concatenate([inside_points(g, rng, 60), [[-0.3, 0.2], [1.4, -1.6]]])
  x32, x64 = sc.sym("x", pts.shape, dtype="float32"), sc.sym("x", pts.shape)
  f32 = sc.Function.from_exprs(f"single_{strategy}_{extrap}", [x32], [single(x32)], ["x"], ["y"])
  f64 = sc.Function.from_exprs(f"double_{strategy}_{extrap}", [x64], [double(x64)], ["x"], ["y"])
  want = f64(pts.astype(np.float32).astype(np.float64))  # the same, rounded, points
  got = f32(pts.astype(np.float32))
  scale = np.abs(want).max()
  # Outside both axes the continuation sums four rounded terms, which may cancel.
  np.testing.assert_allclose(got[:-2], want[:-2], rtol=0, atol=1e-6 * scale)
  np.testing.assert_allclose(got[-2:], want[-2:], rtol=0, atol=1e-5 * scale)
  assert "static const float" in render_c_source(f32) and "static const double" not in render_c_source(f32)


def test_pack_gives_the_coefficient_buffer_in_c_order() -> None:
  t = (np.array([0.0, 0.0, 0.5, 1.0, 1.0]), np.array([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]))
  c = sc.sym("c", (3, 3, 2))
  f = interp.BSpline(t, c, (1, 2))
  values = np.arange(18.0).reshape(3, 3, 2)
  assert np.array_equal(f.pack(values), values.reshape(-1))
  fn = f.function()
  assert [e.shape for e in fn.inputs] == [(2,), (18,)]
  point = np.array([0.3, 0.6])
  want = interp.BSpline(t, values, (1, 2)).to_scipy()(point)
  np.testing.assert_allclose(fn(point, f.pack(values)), want.reshape(2), rtol=1e-14)
  with pytest.raises(ValueError, match="shape"):
    f.pack(np.zeros((3, 3)))


def test_a_table_passed_in_at_run_time_compiles_and_runs_from_c(tmp_path) -> None:  # type: ignore[no-untyped-def]
  """AOT: a 6 x 5 lookup table is an input of the generated function, filled by a C caller in the
  row-major order the header states, and gives what the JIT gives."""
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required")
  g = (np.linspace(0.0, 1.0, 6), np.linspace(-1.0, 1.0, 5))
  x, table = sc.sym("x", 2), sc.sym("table", (6, 5))
  fn = sc.Function.from_exprs("lut", [x, table], [interp.interpolant(g, table, kind="cubic")(x)], ["x", "table"], ["y"])
  module = render_c_module(fn)
  assert "lut_table_t;  // 6 x 5, row-major (C order)" in module.header
  values = np.add.outer(np.linspace(0.0, 1.0, 6) ** 2, np.sin(np.linspace(-1.0, 1.0, 5)))
  point = np.array([0.37, -0.21])
  want = float(fn((point, values)))
  (tmp_path / module.header_name).write_text(module.header)
  (tmp_path / module.source_name).write_text(module.source)
  fill = ", ".join(repr(float(v)) for v in values.reshape(-1))
  (tmp_path / "main.c").write_text(
    f"""
#include <math.h>
#include "lut.h"
int main(void) {{
  static lut_workspace_t workspace;
  lut_x_t x = {{{{0.37, -0.21}}}};
  lut_table_t table = {{{{{fill}}}}};
  lut_y_t y = {{{{0}}}};
  int err = lut_call(&x, &table, &y, &workspace);
  if (err) return err;
  return fabs(y.data[0] - ({want!r})) > 1e-14 ? 10 : 0;
}}
"""
  )
  exe = tmp_path / "main"
  subprocess.run([cc, "-std=c11", str(tmp_path / "main.c"), str(tmp_path / module.source_name), "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)


def test_a_huge_constant_table_warns_and_an_expression_does_not() -> None:
  g = np.linspace(0.0, 1.0, (1 << 20) + 2)
  with pytest.warns(UserWarning, match="tabulated values"):
    interp.interpolant(g, np.sin(g), kind="zoh")
  import warnings

  with warnings.catch_warnings():
    warnings.simplefilter("error")
    interp.interpolant(g[:1000], np.sin(g[:1000]))
    interp.interpolant(g, sc.sym("table", g.size), kind="zoh")


def test_a_float64_point_into_a_float32_spline() -> None:
  from scipy.interpolate import make_interp_spline

  g = np.linspace(0.0, 1.0, 7)
  f = interp.interpolant(g, np.sin(3 * g), kind="cubic", dtype="float32")
  x = sc.sym("x", 5)
  got = sc.Function.from_exprs("f64_point_f32", [x], [f(x)], ["x"], ["y"])(np.linspace(0.05, 0.95, 5))
  np.testing.assert_allclose(got, make_interp_spline(g, np.sin(3 * g), k=3)(np.linspace(0.05, 0.95, 5)), rtol=0, atol=1e-6)


def test_every_fit_honours_its_dtype() -> None:
  x = np.linspace(0.0, 1.0, 50)
  y = np.sin(3 * x)
  assert interp.smoothing(x, y, dtype="float32").dtype.name == "float32"
  assert interp.smoothing(x, y, method="cubic", dtype="float32").dtype.name == "float32"
  assert interp.interpolant(x, y, kind="pchip", dtype="float32").dtype.name == "float32"
