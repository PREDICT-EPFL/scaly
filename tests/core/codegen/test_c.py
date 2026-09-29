from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest

import scaly as sc


def test_compiled_erf_matches_math_erf() -> None:
  x = sc.sym("x", 9)
  f = sc.Function.from_exprs("compiled_erf", [x], [x.erf()], ["x"], ["y"])
  values = np.array([-6.0, -4.5, -2.0, -0.25, 0.0, 0.25, 2.0, 4.5, 6.0])

  np.testing.assert_allclose(f(values), [math.erf(float(value)) for value in values], rtol=1e-14, atol=1e-15)


def test_c_api_header_exposes_pointer_abi_and_typed_buffers() -> None:
  x = sc.sym("x", 2)
  f = sc.Function.from_exprs("f", [x], [x + 1], ["x"], ["y"])
  from scaly.codegen import render_c_api_header

  header = render_c_api_header(f)
  assert "#define SCALY_SUCCESS 0" in header
  assert "#define SCALY_ERR_NULL_ABI 1" in header
  assert "#define SCALY_ERR_NULL_WORK 2" in header
  assert "#define SCALY_ERR_NULL_RESULT 3" in header
  assert "#define SCALY_ERR_NULL_INPUT 4" in header
  assert "#define f_SZ_ARG 1" in header
  assert "#define f_SZ_RES 1" in header
  assert "#define f_SZ_IW 0" in header
  assert "#define f_SZ_W 0" in header
  assert 'extern "C" {' in header
  assert "int f(const double** arg, double** res, int* iw, double* w, int mem);" in header
  assert "_sz_" not in header and "_mem" not in header
  assert "typedef struct { SCALY_ALIGNAS(16) double data[2]; } f_x_t;" in header
  assert "typedef struct { SCALY_ALIGNAS(16) double data[2]; } f_y_t;" in header
  assert "typedef struct { SCALY_ALIGNAS(16) double data[f_SZ_W > 0 ? f_SZ_W : 1]; } f_workspace_t;" in header
  assert "static inline int f_call(const f_x_t* x, f_y_t* y, f_workspace_t* workspace)" in header
  assert "casadi" not in header

  bare = render_c_api_header(f, typed_buffers=False)
  assert "f_x_t" not in bare and "f_call" not in bare and "SCALY_ALIGNAS" not in bare


def test_c_api_header_exposes_sparse_output_metadata() -> None:
  x = sc.sym("x", 3)
  y = sc.stack([x[0], x[2]])
  f = sc.sparse_jacobian(sc.Function.from_exprs("f", [x], [y], ["x"], ["y"]), "y", "x", name="f_spjac")
  from scaly.codegen import render_c_api_header

  header = render_c_api_header(f)
  assert "typedef struct { SCALY_ALIGNAS(16) double data[2]; } f_spjac_spjac_y_x_t;" in header
  assert "#define f_spjac_spjac_y_x_NNZ 2" in header
  assert "#define f_spjac_spjac_y_x_NROW 2" in header
  assert "#define f_spjac_spjac_y_x_NCOL 3" in header
  assert "static const int f_spjac_spjac_y_x_rows[2] = {0, 1};" in header
  assert "static const int f_spjac_spjac_y_x_cols[2] = {0, 2};" in header
  assert "static const int f_spjac_spjac_y_x_csr_row_ptr[3] = {0, 1, 2};" in header
  assert "static const int f_spjac_spjac_y_x_csr_col_ind[2] = {0, 2};" in header
  assert "static const int f_spjac_spjac_y_x_csc_col_ptr[4] = {0, 1, 1, 2};" in header
  assert "static const int f_spjac_spjac_y_x_csc_row_ind[2] = {0, 1};" in header


def test_c_source_executes_scalar_subset_through_universal_abi(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = sc.sym("x", 2)
  a = sc.const(np.array([[2.0, -1.0], [0.5, 3.0]]))
  y = sc.concat([(a @ x).sin(), x.gather([1, 0])])
  f = sc.Function.from_exprs("f", [x], [y, y.sum()], ["x"], ["y", "s"])
  from scaly.codegen import render_c_source

  src = tmp_path / "f.c"
  lib_path = tmp_path / ("libf.dylib" if sys.platform == "darwin" else "libf.so")
  src.write_text(render_c_source(f))
  cmd = [cc, "-fPIC", str(src), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  assert not hasattr(lib, "f_sz_w") and not hasattr(lib, "f_alloc_mem")

  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.f.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_int]
  lib.f.restype = ctypes.c_int

  xv = np.array([0.25, -0.75])
  x_buf = (ctypes.c_double * 2)(*xv)
  y_buf = (ctypes.c_double * 4)()
  s_buf = (ctypes.c_double * 1)()
  w_buf = (ctypes.c_double * 1)()
  args = (c_double_p * 1)(ctypes.cast(x_buf, c_double_p))
  res = (c_double_p * 2)(ctypes.cast(y_buf, c_double_p), ctypes.cast(s_buf, c_double_p))

  assert lib.f(args, res, None, w_buf, 0) == 0
  expected = np.concatenate([np.sin(np.array([[2.0, -1.0], [0.5, 3.0]]) @ xv), xv[[1, 0]]])
  np.testing.assert_allclose(np.array(y_buf), expected)
  np.testing.assert_allclose(np.array(s_buf), expected.sum())

  assert lib.f(None, res, None, w_buf, 0) == 1
  assert lib.f(args, res, None, None, 0) == 0
  bad_res = (c_double_p * 2)(c_double_p(), ctypes.cast(s_buf, c_double_p))
  assert lib.f(args, bad_res, None, w_buf, 0) == 3
  bad_args = (c_double_p * 1)(c_double_p())
  assert lib.f(bad_args, res, None, w_buf, 0) == 4


def test_c_source_column_slice_is_not_contiguous(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = sc.sym("x", (5, 7))
  y = sc.sym("y", (5, 1, 7))
  f = sc.Function.from_exprs("g", [x, y], [x[:, 1], x[2, :], y[:, 0, :]], ["x", "y"], ["col1", "row2", "row2d"])
  from scaly.codegen import render_c_source

  src = tmp_path / "g.c"
  lib_path = tmp_path / ("libg.dylib" if sys.platform == "darwin" else "libg.so")
  src.write_text(render_c_source(f))
  cmd = [cc, "-fPIC", str(src), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.g.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_int]
  lib.g.restype = ctypes.c_int

  xv = np.arange(35, dtype=np.float64).reshape(5, 7)
  yv = np.arange(35, dtype=np.float64).reshape(5, 1, 7) + 100.0
  x_buf = (ctypes.c_double * 35)(*xv.reshape(-1))
  y_buf = (ctypes.c_double * 35)(*yv.reshape(-1))
  col1_buf = (ctypes.c_double * 5)()
  row2_buf = (ctypes.c_double * 7)()
  row2d_buf = (ctypes.c_double * 35)()
  args = (c_double_p * 2)(ctypes.cast(x_buf, c_double_p), ctypes.cast(y_buf, c_double_p))
  res = (c_double_p * 3)(ctypes.cast(col1_buf, c_double_p), ctypes.cast(row2_buf, c_double_p), ctypes.cast(row2d_buf, c_double_p))
  assert lib.g(args, res, None, None, 0) == 0
  np.testing.assert_allclose(np.array(col1_buf), xv[:, 1])
  np.testing.assert_allclose(np.array(row2_buf), xv[2, :])
  np.testing.assert_allclose(np.array(row2d_buf).reshape(5, 7), yv[:, 0, :])


def test_c_source_lowers_call_nodes_through_internal_raw_function(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = sc.sym("x", 2)
  inner = sc.Function.from_exprs("inner", [x], [x * x, x.sum()], ["x"], ["sq", "sum"])
  z = sc.sym("z", 2)
  inner_sq, inner_sum = inner(z + 1.0)
  outer = sc.Function.from_exprs("outer", [z], [inner_sq + inner_sum], ["z"], ["y"])
  from scaly.codegen import render_c_source

  src = tmp_path / "outer.c"
  lib_path = tmp_path / ("libouter.dylib" if sys.platform == "darwin" else "libouter.so")
  source = render_c_source(outer)
  assert source.index("void inner_raw(") < source.index("int outer(")
  src.write_text(source)
  cmd = [cc, "-fPIC", str(src), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.outer.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_int]
  lib.outer.restype = ctypes.c_int

  zv = np.array([0.25, -0.75])
  z_buf = (ctypes.c_double * 2)(*zv)
  y_buf = (ctypes.c_double * 2)()
  w_buf = (ctypes.c_double * 1)()
  args = (c_double_p * 1)(ctypes.cast(z_buf, c_double_p))
  res = (c_double_p * 1)(ctypes.cast(y_buf, c_double_p))

  assert lib.outer(args, res, None, w_buf, 0) == 0
  actual = zv + 1.0
  np.testing.assert_allclose(np.array(y_buf), actual * actual + actual.sum())


def test_c_module_executes_sparse_jacobian_factory_output(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = sc.sym("x", 4)
  y = sc.stack([x[0], x[2:4].sum(), x[1] * x[3]])
  f = sc.Function.from_exprs("f", [x], [y], ["x"], ["y"])
  spjf = sc.sparse_jacobian(f, "y", "x", name="f_spjac")
  from scaly.codegen import render_c_module

  module = render_c_module(spjf)
  assert "#define f_spjac_spjac_y_x_NNZ 5" in module.header
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  lib_path = tmp_path / ("libspjac.dylib" if sys.platform == "darwin" else "libspjac.so")
  cmd = [cc, "-fPIC", str(source), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.f_spjac.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_int]
  lib.f_spjac.restype = ctypes.c_int

  xv = np.array([2.0, 3.0, 5.0, 7.0])
  x_buf = (ctypes.c_double * 4)(*xv)
  values_buf = (ctypes.c_double * 5)()
  w_buf = (ctypes.c_double * max(module.workspace_size, 1))()
  args = (c_double_p * 1)(ctypes.cast(x_buf, c_double_p))
  res = (c_double_p * 1)(ctypes.cast(values_buf, c_double_p))

  assert lib.f_spjac(args, res, None, w_buf, 0) == 0
  np.testing.assert_allclose(np.array(values_buf), np.array([1.0, 1.0, 1.0, xv[3], xv[1]]))


def test_c_header_typed_buffers_compile_and_run_from_c(tmp_path) -> None:
  """A C11 caller: the workspace is a struct it owns, and every buffer type is 16-byte aligned."""
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = sc.sym("x", 3)
  f = sc.Function.from_exprs("f", [x], [x.sin() + 2.0, x.sum()], ["x"], ["y", "s"])
  from scaly.codegen import render_c_module

  module = render_c_module(f)
  assert module.header_name == "f.h"
  assert module.source_name == "f.c"
  assert '#include "f.h"' in module.source
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  main = tmp_path / "main.c"
  exe = tmp_path / "main"
  main.write_text(
    """
#include <math.h>
#include <stdalign.h>
#include "f.h"

_Static_assert(alignof(f_x_t) == 16, "aligned");
_Static_assert(alignof(f_workspace_t) == 16, "aligned");
_Static_assert(sizeof(((f_x_t*)0)->data) == 3 * sizeof(double), "three doubles");

int main(void) {
  static f_workspace_t workspace;
  f_x_t x = {{0.25, -0.75, 1.5}};
  f_y_t y = {{0}};
  f_s_t s = {{0}};
  int err = f_call(&x, &y, &s, &workspace);
  if (err) return err;
  if (fabs(y.data[0] - (sin(0.25) + 2.0)) > 1e-12) return 10;
  if (fabs(y.data[2] - (sin(1.5) + 2.0)) > 1e-12) return 11;
  if (fabs(s.data[0] - 1.0) > 1e-12) return 12;
  return f_call(&x, &y, &s, NULL);
}
"""
  )

  subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", str(main), str(source), "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)


def test_c_header_typed_buffers_compile_and_run_from_cpp(tmp_path) -> None:
  cc = shutil.which("cc")
  cxx = shutil.which("c++")
  if cc is None or cxx is None:
    pytest.skip("cc and c++ are required for generated C++ wrapper smoke test")

  x = sc.sym("x", 2)
  nlp = sc.Function.from_exprs("nlp", [x], [x[0] * x[0], x * x], ["x"], ["f", "g"])
  hess = sc.lagrangian_hessian(nlp, "x", name="h")
  from scaly.codegen import render_c_module

  module = render_c_module(hess)
  assert "h_lam_f_t" in module.header
  assert "h_lam_g_t" in module.header
  assert "h_hess_gamma_x_x_t" in module.header
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  cpp = tmp_path / "main.cpp"
  obj = tmp_path / "h.o"
  exe = tmp_path / "main"
  cpp.write_text(
    """
#include <cmath>
#include "h.h"

static_assert(alignof(h_x_t) == 16, "aligned");

int main() {
  h_x_t x = {{2.0, 3.0}};
  h_lam_f_t lam_f = {{1.5}};
  h_lam_g_t lam_g = {{0.25, -0.5}};
  h_hess_gamma_x_x_t hess = {};
  h_workspace_t workspace;
  int err = h_call(&x, &lam_f, &lam_g, &hess, &workspace);
  if (err) return err;
  if (std::fabs(hess.data[0] - 3.5) > 1e-12) return 10;
  if (std::fabs(hess.data[1]) > 1e-12) return 11;
  if (std::fabs(hess.data[2]) > 1e-12) return 12;
  if (std::fabs(hess.data[3] + 1.0) > 1e-12) return 13;
  return 0;
}
"""
  )

  subprocess.run([cc, "-c", str(source), "-o", str(obj)], check=True)
  subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(cpp), str(obj), "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)


def test_scalarized_stores_coalesce_into_vector_accesses() -> None:
  from scaly.codegen.c import render_program_c_source

  x = sc.sym("x", 7)
  f = sc.Function.from_exprs("coalesced", [x], [(x * 2.0 + 1.0).scalar()], ["x"], ["y"])
  source = render_program_c_source(f)
  assert "*(double2*)(res[0]) = (double2){" in source
  assert "*(double2*)(res[0] + 4) = (double2){" in source
  assert "res[0][6] = " in source
  values = np.arange(7.0) + 0.5

  assert np.array_equal(f(values), values * 2.0 + 1.0)


def test_store_run_stays_scalar_when_not_contiguous_or_reading_its_own_target(tmp_path) -> None:
  from scaly.codegen.c import _render_raw_callee
  from scaly.ir import program as p
  from scaly.ir.program import ProgramNode, ProgramOp
  from scaly.ir.types import dtypes
  from scaly.passes.program.coalesce_stores import coalesce_stores
  from scaly.passes.program.prepare_scalar import prepare_scalar_expressions

  x = p.buffer("x", dtypes.float64, (4,))
  y = p.buffer("y", dtypes.float64, (5,))

  def at(buf, i):
    return p.view(buf, [p.const_int(i)])

  a = ProgramNode(ProgramOp.BUFFER, (), {**y.attrs, "name": "a", "alias_of": "y", "alias_offset": 0}, y.dtype)
  body = [
    a,
    p.store(at(y, 0), p.load(at(x, 0))),
    p.store(at(y, 2), p.load(at(x, 2))),
    p.store(at(y, 3), p.load(at(y, 2))),
    p.store(at(y, 4), p.load(at(a, 3))),
  ]
  proc = p.proc("gaps", [x, y], body)
  optimized = prepare_scalar_expressions(coalesce_stores(p.program([proc]))).args[0]
  assert all(stmt.op != ProgramOp.STORE_PAIR for stmt in optimized.args[optimized.attrs["param_count"] :])
  source = "\n".join(_render_raw_callee(optimized))

  assert "double2" not in source
  assert "y[0] = x[0];" in source and "y[2] = x[2];" in source and "y[3] = y[2];" in source and "y[4] = a[3];" in source

  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")
  src = tmp_path / "gaps.c"
  lib_path = tmp_path / ("libgaps.dylib" if sys.platform == "darwin" else "libgaps.so")
  src.write_text("#include <stddef.h>\n" + source + "\nvoid gaps_test(double* x, double* y) { gaps_raw(x, y, NULL); }\n")
  cmd = [cc, "-fPIC", str(src), "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)
  lib = ctypes.CDLL(str(lib_path))
  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.gaps_test.argtypes = [c_double_p, c_double_p]
  xv = (ctypes.c_double * 4)(1, 2, 3, 4)
  yv = (ctypes.c_double * 5)()
  lib.gaps_test(xv, yv)
  assert list(yv) == [1, 0, 3, 3, 3]


def test_renderer_spells_explicit_paired_store() -> None:
  from scaly.codegen.c import _render_raw_callee
  from scaly.ir import program as p
  from scaly.ir.types import dtypes

  out = p.buffer("out", dtypes.float64, (2,))
  proc = p.proc("paired", [out], [p.store_pair(p.view(out, [p.const_int(0)]), p.const_float(1), p.const_float(2))])
  source = "\n".join(_render_raw_callee(proc))
  assert "*(double2*)(out) = (double2){1.0, 2.0};" in source
