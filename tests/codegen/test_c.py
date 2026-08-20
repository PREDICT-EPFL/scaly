from __future__ import annotations

import ctypes
import math
import shutil
import subprocess
import sys

import numpy as np
import pytest

import alloy as al


def test_compiled_erf_matches_math_erf() -> None:
  x = al.sym("x", 9)
  f = al.Function("compiled_erf", [x], [x.erf()], ["x"], ["y"])
  values = np.array([-6.0, -4.5, -2.0, -0.25, 0.0, 0.25, 2.0, 4.5, 6.0])

  np.testing.assert_allclose(f(values), [math.erf(float(value)) for value in values], rtol=1e-14, atol=1e-15)


def test_c_api_header_exposes_universal_and_typed_buffers() -> None:
  x = al.sym("x", 2)
  f = al.Function("f", [x], [x + 1], ["x"], ["y"])
  from alloy.codegen import render_c_api_header

  header = render_c_api_header(f)
  assert "#define ALLOY_SUCCESS 0" in header
  assert "#define ALLOY_ERR_NULL_ABI 1" in header
  assert "#define ALLOY_ERR_NULL_WORK 2" in header
  assert "#define ALLOY_ERR_NULL_RESULT 3" in header
  assert "#define ALLOY_ERR_NULL_INPUT 4" in header
  assert "#define f_SZ_ARG 1" in header
  assert "#define f_SZ_RES 1" in header
  assert "#define f_SZ_IW 0" in header
  assert "#define f_SZ_W 0" in header
  assert 'extern "C" {' in header
  assert "int f(const double** arg, double** res, int* iw, double* w, void* mem);" in header
  assert "int f_sz_arg(void);" in header
  assert "int f_sz_res(void);" in header
  assert "int f_sz_iw(void);" in header
  assert "int f_sz_w(void);" in header
  assert "void* f_alloc_mem(void);" in header
  assert "int f_init_mem(void* mem);" in header
  assert "void f_free_mem(void* mem);" in header
  assert "typedef struct { double data[2]; } f_x_in;" in header
  assert "typedef struct { double data[2]; } f_y_out;" in header
  assert 'static_assert(sizeof(f_x_in) == sizeof(double) * 2, "f_x_in size mismatch");' in header
  assert 'static_assert(sizeof(f_y_out) == sizeof(double) * 2, "f_y_out size mismatch");' in header
  assert "static inline int f_call(const f_x_in& in_x, f_y_out& out_y)" in header


def test_c_api_header_exposes_sparse_output_metadata() -> None:
  x = al.sym("x", 3)
  y = al.stack([x[0], x[2]])
  f = al.spjacobian(al.Function("f", [x], [y], ["x"], ["y"]), "x", "y", name="f_spjac")
  from alloy.codegen import render_c_api_header

  header = render_c_api_header(f)
  assert "typedef struct { double data[2]; } f_spjac_spjac_y_x_out;" in header
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

  x = al.sym("x", 2)
  a = al.const(np.array([[2.0, -1.0], [0.5, 3.0]]))
  y = al.concat([(a @ x).sin(), x.gather([1, 0])])
  f = al.Function("f", [x], [y, y.sum()], ["x"], ["y", "s"])
  from alloy.codegen import render_c_source

  src = tmp_path / "f.c"
  lib_path = tmp_path / ("libf.dylib" if sys.platform == "darwin" else "libf.so")
  src.write_text(render_c_source(f))
  cmd = [cc, "-fPIC", str(src), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  assert lib.f_sz_arg() == 1
  assert lib.f_sz_res() == 2
  assert lib.f_sz_iw() == 0
  assert lib.f_sz_w() == 0
  lib.f_alloc_mem.restype = ctypes.c_void_p
  lib.f_init_mem.argtypes = [ctypes.c_void_p]
  lib.f_init_mem.restype = ctypes.c_int
  lib.f_free_mem.argtypes = [ctypes.c_void_p]
  lib.f_free_mem.restype = None
  mem = lib.f_alloc_mem()
  assert mem is None
  assert lib.f_init_mem(mem) == 0
  lib.f_free_mem(mem)

  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.f.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_void_p]
  lib.f.restype = ctypes.c_int

  xv = np.array([0.25, -0.75])
  x_buf = (ctypes.c_double * 2)(*xv)
  y_buf = (ctypes.c_double * 4)()
  s_buf = (ctypes.c_double * 1)()
  w_buf = (ctypes.c_double * max(lib.f_sz_w(), 1))()
  args = (c_double_p * 1)(ctypes.cast(x_buf, c_double_p))
  res = (c_double_p * 2)(ctypes.cast(y_buf, c_double_p), ctypes.cast(s_buf, c_double_p))

  assert lib.f(args, res, None, w_buf, None) == 0
  expected = np.concatenate([np.sin(np.array([[2.0, -1.0], [0.5, 3.0]]) @ xv), xv[[1, 0]]])
  np.testing.assert_allclose(np.array(y_buf), expected)
  np.testing.assert_allclose(np.array(s_buf), expected.sum())

  assert lib.f(None, res, None, w_buf, None) == 1
  assert lib.f(args, res, None, None, None) == 0
  bad_res = (c_double_p * 2)(c_double_p(), ctypes.cast(s_buf, c_double_p))
  assert lib.f(args, bad_res, None, w_buf, None) == 3
  bad_args = (c_double_p * 1)(c_double_p())
  assert lib.f(bad_args, res, None, w_buf, None) == 4


def test_c_source_column_slice_is_not_contiguous(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = al.sym("x", (5, 7))
  y = al.sym("y", (5, 1, 7))
  f = al.Function("g", [x, y], [x[:, 1], x[2, :], y[:, 0, :]], ["x", "y"], ["col1", "row2", "row2d"])
  from alloy.codegen import render_c_source

  src = tmp_path / "g.c"
  lib_path = tmp_path / ("libg.dylib" if sys.platform == "darwin" else "libg.so")
  src.write_text(render_c_source(f))
  cmd = [cc, "-fPIC", str(src), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.g.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_void_p]
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
  assert lib.g(args, res, None, None, None) == 0
  np.testing.assert_allclose(np.array(col1_buf), xv[:, 1])
  np.testing.assert_allclose(np.array(row2_buf), xv[2, :])
  np.testing.assert_allclose(np.array(row2d_buf).reshape(5, 7), yv[:, 0, :])


def test_c_source_lowers_call_nodes_through_internal_raw_function(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = al.sym("x", 2)
  inner = al.Function("inner", [x], [x * x, x.sum()], ["x"], ["sq", "sum"])
  z = al.sym("z", 2)
  inner_sq, inner_sum = inner.call([z + 1.0])
  outer = al.Function("outer", [z], [inner_sq + inner_sum], ["z"], ["y"])
  from alloy.codegen import render_c_source

  src = tmp_path / "outer.c"
  lib_path = tmp_path / ("libouter.dylib" if sys.platform == "darwin" else "libouter.so")
  source = render_c_source(outer)
  assert source.index("void inner_raw(") < source.index("int outer(")
  src.write_text(source)
  cmd = [cc, "-fPIC", str(src), "-lm", "-o", str(lib_path)]
  cmd.insert(1, "-dynamiclib" if sys.platform == "darwin" else "-shared")
  subprocess.run(cmd, check=True)

  lib = ctypes.CDLL(str(lib_path))
  assert lib.outer_sz_arg() == 1
  assert lib.outer_sz_res() == 1
  assert lib.outer_sz_w() == 0

  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.outer.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_void_p]
  lib.outer.restype = ctypes.c_int

  zv = np.array([0.25, -0.75])
  z_buf = (ctypes.c_double * 2)(*zv)
  y_buf = (ctypes.c_double * 2)()
  w_buf = (ctypes.c_double * max(lib.outer_sz_w(), 1))()
  args = (c_double_p * 1)(ctypes.cast(z_buf, c_double_p))
  res = (c_double_p * 1)(ctypes.cast(y_buf, c_double_p))

  assert lib.outer(args, res, None, w_buf, None) == 0
  actual = zv + 1.0
  np.testing.assert_allclose(np.array(y_buf), actual * actual + actual.sum())


def test_c_module_executes_sparse_jacobian_factory_output(tmp_path) -> None:
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for generated C smoke test")

  x = al.sym("x", 4)
  y = al.stack([x[0], x[2:4].sum(), x[1] * x[3]])
  f = al.Function("f", [x], [y], ["x"], ["y"])
  spjf = al.spjacobian(f, "x", "y", name="f_spjac")
  from alloy.codegen import render_c_module

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
  lib.f_spjac.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_void_p]
  lib.f_spjac.restype = ctypes.c_int

  xv = np.array([2.0, 3.0, 5.0, 7.0])
  x_buf = (ctypes.c_double * 4)(*xv)
  values_buf = (ctypes.c_double * 5)()
  w_buf = (ctypes.c_double * max(lib.f_spjac_sz_w(), 1))()
  args = (c_double_p * 1)(ctypes.cast(x_buf, c_double_p))
  res = (c_double_p * 1)(ctypes.cast(values_buf, c_double_p))

  assert lib.f_spjac(args, res, None, w_buf, None) == 0
  np.testing.assert_allclose(np.array(values_buf), np.array([1.0, 1.0, 1.0, xv[3], xv[1]]))


def test_c_api_header_typed_cpp_wrapper_compiles_and_runs(tmp_path) -> None:
  cc = shutil.which("cc")
  cxx = shutil.which("c++")
  if cc is None or cxx is None:
    pytest.skip("cc and c++ are required for generated C++ wrapper smoke test")

  x = al.sym("x", 2)
  f = al.Function("f", [x], [x.sin() + 2.0], ["x"], ["y"])
  from alloy.codegen import render_c_module

  module = render_c_module(f)
  assert module.header_name == "f.h"
  assert module.source_name == "f.c"
  assert '#include "f.h"' in module.source
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  cpp = tmp_path / "main.cpp"
  obj = tmp_path / "f.o"
  exe = tmp_path / "main"
  cpp.write_text(
    """
#include <cmath>
#include "f.h"

int main() {
  f_x_in x = {{0.25, -0.75}};
  f_y_out y = {};
  int err = f_call(x, y);
  if (err) return err;
  if (std::fabs(y.data[0] - (std::sin(0.25) + 2.0)) > 1e-12) return 10;
  if (std::fabs(y.data[1] - (std::sin(-0.75) + 2.0)) > 1e-12) return 11;
  return 0;
}
"""
  )

  subprocess.run([cc, "-c", str(source), "-o", str(obj)], check=True)
  subprocess.run([cxx, "-std=c++17", str(cpp), str(obj), "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)


def test_c_api_header_typed_cpp_wrapper_handles_factory_names(tmp_path) -> None:
  cc = shutil.which("cc")
  cxx = shutil.which("c++")
  if cc is None or cxx is None:
    pytest.skip("cc and c++ are required for generated C++ wrapper smoke test")

  x = al.sym("x", 2)
  nlp = al.Function("nlp", [x], [x[0] * x[0], x * x], ["x"], ["f", "g"])
  hess = al.lagrangian_hessian(nlp, "x", ["f", "g"], name="h")
  from alloy.codegen import render_c_module

  module = render_c_module(hess)
  assert "h_lam_f_in" in module.header
  assert "h_lam_g_in" in module.header
  assert "h_hess_gamma_x_x_out" in module.header
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

int main() {
  h_x_in x = {{2.0, 3.0}};
  h_lam_f_in lam_f = {{1.5}};
  h_lam_g_in lam_g = {{0.25, -0.5}};
  h_hess_gamma_x_x_out hess = {};
  int err = h_call(x, lam_f, lam_g, hess);
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
  subprocess.run([cxx, "-std=c++17", str(cpp), str(obj), "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)
