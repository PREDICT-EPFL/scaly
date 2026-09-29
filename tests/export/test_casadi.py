"""The CasADi 3.8 compatible layer (the ``casadi`` adapter): ``casadi.external`` loads the library and
agrees with the JIT, the six symbols acados resolves are exported, ``_sparsity_out``
decodes to the pattern, compact sparse outputs arrive in compressed-column order, and a dense
matrix input or output is rejected at render time."""

from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_api_header, render_c_module, render_c_source, workspace_size
from scaly.export.casadi import CASADI_QUERIES

casadi = pytest.importorskip("casadi")

ACADOS_SYMBOLS = ("", "_work", "_sparsity_in", "_sparsity_out", "_n_in", "_n_out")


def _spjac() -> sc.Function:
  x = sc.sym("x", 4)
  p = sc.sym("p", 2)
  y = sc.stack([x[0] * p[0], x[2:4].sum(), x[1] * x[3] + p[1]])
  return sc.sparse_jacobian(sc.Function.from_exprs("f", [x, p], [y], ["x", "p"], ["y"]), "y", "x", name="f_spjac")


def _build(tmp_path, fun: sc.Function, adapters: tuple[str, ...] = ()):
  cc = shutil.which("cc")
  if cc is None:
    pytest.skip("cc is required for the CasADi compatibility smoke test")
  module = render_c_module(fun, adapters=(*adapters, "casadi"))
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  lib_path = tmp_path / f"lib{fun.name}{'.dylib' if sys.platform == 'darwin' else '.so'}"
  cmd = [
    cc,
    "-dynamiclib" if sys.platform == "darwin" else "-shared",
    "-fPIC",
    "-Wall",
    "-Wextra",
    "-Werror",
    str(source),
    "-lm",
    "-o",
    str(lib_path),
  ]
  subprocess.run(cmd, check=True)
  return module, lib_path


def test_casadi_header_declares_the_query_set() -> None:
  fun = _spjac()
  header = render_c_api_header(fun, adapters=("casadi",))
  assert "#ifndef casadi_int\n#define casadi_int long long int\n#endif" in header
  for suffix in CASADI_QUERIES:
    assert f" f_spjac_{suffix}(" in header, suffix
  assert "int f_spjac_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);" in header
  # The header's tables describe the compressed-column order the entry gathers into.
  sp = fun.output_sparsities[0]
  assert sp is not None and sp.to_csc()[2] != tuple(range(sp.nnz)), "the fixture must need a gather to be meaningful"
  assert f"static const int f_spjac_spjac_y_x_csc_val_perm[{sp.nnz}] = {{{', '.join(str(k) for k in range(sp.nnz))}}};" in header
  assert workspace_size(fun, adapters=("casadi",)) == workspace_size(fun) + sp.nnz
  assert "casadi_int" not in render_c_api_header(fun)

  cpp = render_c_api_header(fun, adapters=("cpp", "casadi"))
  assert 'extern "C" casadi_int f_spjac_n_in(void);' in cpp
  assert "constexpr std::array<int, nnz> csc_val_perm = {0, 1, 2, 3, 4};" in cpp


def test_casadi_source_gathers_into_compressed_column_order() -> None:
  source = render_c_source(_spjac(), adapters=("casadi",))
  assert "double* spjac_y_x_native = w + 0;" in source
  assert "res[0][k] = spjac_y_x_native[spjac_y_x_csc_val_perm[k]];" in source
  assert "casadi_int f_spjac_n_in(void) { return 2; }" in source
  assert 'case 1: return "p";' in source


def test_casadi_external_loads_and_matches_the_jit(tmp_path) -> None:
  fun = _spjac()
  module, lib_path = _build(tmp_path, fun)
  ext = casadi.external("f_spjac", str(lib_path))
  assert ext.n_in() == 2 and ext.n_out() == 1
  assert ext.name_in() == ["x", "p"] and ext.name_out() == ["spjac_y_x"]
  assert ext.size_in(0) == (4, 1) and ext.size_out(0) == (3, 4)

  sp = fun.output_sparsities[0]
  assert sp is not None
  rows, cols = ext.sparsity_out(0).get_triplet()
  assert set(zip(rows, cols)) == set(zip(sp.rows, sp.cols))

  xv = np.array([2.0, 3.0, 5.0, 7.0])
  pv = np.array([1.5, -0.5])
  native = np.asarray(fun((xv, pv)))
  dense = np.zeros(sp.shape)
  dense[list(sp.rows), list(sp.cols)] = native
  np.testing.assert_allclose(np.array(ext(xv, pv).full()), dense)

  # Through ctypes, the way acados reaches it: the six symbols resolve and the buffer is CSC-ordered.
  lib = ctypes.CDLL(str(lib_path))
  for suffix in ACADOS_SYMBOLS:
    assert hasattr(lib, f"f_spjac{suffix}"), suffix
  lib.f_spjac_sparsity_out.restype = ctypes.POINTER(ctypes.c_longlong)
  lib.f_spjac_sparsity_out.argtypes = [ctypes.c_longlong]
  encoded = lib.f_spjac_sparsity_out(0)
  nrow, ncol = encoded[0], encoded[1]
  col_ptr = [encoded[2 + k] for k in range(ncol + 1)]
  row_ind = [encoded[3 + ncol + k] for k in range(col_ptr[-1])]
  assert (nrow, ncol) == sp.shape
  csc_col_ptr, csc_row_ind, perm = sp.to_csc()
  assert (tuple(col_ptr), tuple(row_ind)) == (csc_col_ptr, csc_row_ind)
  assert not lib.f_spjac_sparsity_out(1)

  sizes = [ctypes.c_longlong() for _ in range(4)]
  assert lib.f_spjac_work(*(ctypes.byref(s) for s in sizes)) == 0
  assert [s.value for s in sizes] == [2, 1, 0, module.workspace_size]

  c_double_p = ctypes.POINTER(ctypes.c_double)
  lib.f_spjac.argtypes = [ctypes.POINTER(c_double_p), ctypes.POINTER(c_double_p), ctypes.POINTER(ctypes.c_int), c_double_p, ctypes.c_int]
  x_buf = (ctypes.c_double * 4)(*xv)
  p_buf = (ctypes.c_double * 2)(*pv)
  out = (ctypes.c_double * sp.nnz)()
  w = (ctypes.c_double * max(module.workspace_size, 1))()
  args = (c_double_p * 2)(ctypes.cast(x_buf, c_double_p), ctypes.cast(p_buf, c_double_p))
  res = (c_double_p * 1)(ctypes.cast(out, c_double_p))
  assert lib.f_spjac(args, res, None, w, lib.f_spjac_checkout()) == 0
  lib.f_spjac_release(0)
  np.testing.assert_allclose(np.array(out), native[list(perm)])


def test_casadi_external_dense_vector_function_without_gather(tmp_path) -> None:
  x = sc.sym("x", 3)
  fun = sc.Function.from_exprs("g", [x], [x.sin(), x.sum()], ["x"], ["y", "s"])
  module, lib_path = _build(tmp_path, fun, adapters=("cpp",))
  assert module.header_name == "g.hpp" and module.workspace_size == workspace_size(fun)
  ext = casadi.external("g", str(lib_path))
  xv = np.array([0.25, -0.75, 1.5])
  y, s = ext(xv)
  np.testing.assert_allclose(np.array(y.full()).ravel(), np.sin(xv))
  np.testing.assert_allclose(float(s), xv.sum())
  assert ext.sparsity_out(1).shape == (1, 1)


def test_casadi_rejects_dense_matrix_buffers() -> None:
  m = sc.sym("m", (2, 3))
  fun = sc.Function.from_exprs("dense", [m], [m * 2.0], ["m"], ["n"])
  with pytest.raises(ValueError, match="column-major"):
    render_c_module(fun, adapters=("casadi",))
  with pytest.raises(ValueError, match="input 'm'"):
    render_c_api_header(fun, adapters=("casadi",))
  row = sc.sym("row", (1, 3))
  render_c_module(sc.Function.from_exprs("row_ok", [row], [row * 2.0], ["row"], ["n"]), adapters=("casadi",))
