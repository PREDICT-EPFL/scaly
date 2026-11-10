"""The CasADi 3.8 compatible layer (``casadi=True``): ``casadi.external`` loads the library and
agrees with the JIT, the six symbols acados resolves are exported, ``_sparsity_out``
decodes to the pattern, and dense matrices and compact sparse outputs arrive in column-major order."""

from __future__ import annotations

import ctypes
import subprocess
import sys

import casadi
import numpy as np
import pytest

from scaly.function.model import as_concrete
import scaly as sc
from scaly.codegen import render_c_api_header, render_c_module, workspace_size
from scaly.codegen.casadi import CASADI_QUERIES
from tests.compilers import c_compiler, cxx_compiler

ACADOS_SYMBOLS = ("", "_work", "_sparsity_in", "_sparsity_out", "_n_in", "_n_out")


def _spjac() -> sc.Function:
  @sc.function(sc.group(sc.arg("x", 4), sc.arg("p", 2)), outputs=sc.arg("y"))
  def f(inputs: tuple[sc.Expr, sc.Expr]) -> sc.Expr:
    x, p = inputs
    return sc.stack([x[0] * p[0], x[2:4].sum(), x[1] * x[3] + p[1]])

  return sc.sparse_jacobian(f, "y", "x", name="f_spjac")


def _build(tmp_path, fun: sc.Function, **kwargs):
  cc = c_compiler()
  module = render_c_module(fun, casadi=True, **kwargs)
  (tmp_path / module.header_name).write_text(module.header)
  source = tmp_path / module.source_name
  source.write_text(module.source)
  lib_path = tmp_path / f"lib{fun.name}{'.dylib' if sys.platform == 'darwin' else '.so'}"
  cmd = [
    *cc,
    "-dynamiclib" if sys.platform == "darwin" else "-shared",
    "-fPIC",
    "-Wall",
    "-Wextra",
    "-Werror",
    *(["-Wno-unused-parameter"] if module.backends else []),
    str(source),
    *module.link_flags,
    "-lm",
    "-o",
    str(lib_path),
  ]
  subprocess.run(cmd, check=True)
  return module, lib_path


def test_casadi_header_declares_the_query_set() -> None:
  fun = _spjac()
  header = render_c_api_header(fun, casadi=True)
  assert "#ifndef casadi_int\n#define casadi_int long long int\n#endif" in header
  for suffix in CASADI_QUERIES:
    assert f" f_spjac_{suffix}(" in header, suffix
  assert "int f_spjac_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);" in header
  # The header's tables describe the compressed-column order the entry gathers into.
  sp = as_concrete(fun).output_sparsities[0]
  assert sp is not None and sp.to_csc()[2] != tuple(range(sp.nnz)), "the fixture must need a gather to be meaningful"
  assert f"static const int f_spjac_spjac_y_x_csc_val_perm[{sp.nnz}] = {{{', '.join(str(k) for k in range(sp.nnz))}}};" in header
  assert workspace_size(fun, casadi=True) == workspace_size(fun) + sp.nnz
  assert "casadi_int" not in render_c_api_header(fun)

  cpp = render_c_api_header(fun, lang="cpp", casadi=True)
  assert 'extern "C" casadi_int f_spjac_n_in(void);' in cpp
  assert "constexpr std::array<int, nnz> csc_val_perm = {0, 1, 2, 3, 4};" in cpp


def test_casadi_external_loads_and_matches_the_jit(tmp_path) -> None:
  fun = _spjac()
  module, lib_path = _build(tmp_path, fun)
  ext = casadi.external("f_spjac", str(lib_path))
  assert ext.n_in() == 2 and ext.n_out() == 1
  assert ext.name_in() == ["x", "p"] and ext.name_out() == ["spjac_y_x"]
  assert ext.size_in(0) == (4, 1) and ext.size_out(0) == (3, 4)

  sp = as_concrete(fun).output_sparsities[0]
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
  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y"), sc.arg("s")), name="g")
  def fun(x: sc.Expr) -> tuple[sc.Expr, sc.Expr]:
    return (x.sin(), x.sum())

  module, lib_path = _build(tmp_path, fun, lang="cpp")
  assert module.header_name == "g.hpp" and module.workspace_size == workspace_size(fun)
  ext = casadi.external("g", str(lib_path))
  xv = np.array([0.25, -0.75, 1.5])
  y, s = ext(xv)
  np.testing.assert_allclose(np.array(y.full()).ravel(), np.sin(xv))
  np.testing.assert_allclose(float(s), xv.sum())
  assert ext.sparsity_out(1).shape == (1, 1)


@pytest.mark.parametrize("shape", [(0,), (0, 3), (3, 0), (0, 0)])
@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_casadi_external_empty_dense_leaves(tmp_path, shape, side, lang):
  if side == "input":
    fun = sc.function(sc.arg("z", shape), outputs=sc.arg("s"), name="empty_dense")(lambda z: z.sum())
  else:
    fun = sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("z", shape), sc.arg("s")), name="empty_dense")(
      lambda x: (x[:0].reshape(shape), x.sum())
    )

  module, lib_path = _build(tmp_path, fun, lang=lang)
  ext = casadi.external(fun.name, str(lib_path))
  dims = (0, 1) if len(shape) == 1 else shape
  x = np.array([2.0, -3.0, 5.0])
  if side == "input":
    assert ext.size_in(0) == dims
    assert ext.nnz_in(0) == 0
    assert ext.size_out(0) == (1, 1)
    expected = np.empty(shape).sum()
    np.testing.assert_allclose(float(ext(casadi.DM.zeros(*dims))), expected)
  else:
    assert ext.size_in(0) == (3, 1)
    assert ext.size_out(0) == dims
    assert ext.nnz_out(0) == 0
    assert ext.size_out(1) == (1, 1)
    z, s = ext(x)
    np.testing.assert_array_equal(z.full(), np.empty(dims))
    expected = x.sum()
    np.testing.assert_allclose(float(s), expected)

  lib = ctypes.CDLL(str(lib_path))
  entry = lib.empty_dense
  ptr = ctypes.POINTER(ctypes.c_double)
  entry.argtypes = [ctypes.POINTER(ptr), ctypes.POINTER(ptr), ctypes.POINTER(ctypes.c_int), ptr, ctypes.c_int]
  entry.restype = ctypes.c_int
  scalar = (ctypes.c_double * 1)()
  args = (ptr * 1)(None if side == "input" else x.ctypes.data_as(ptr))
  res = (ptr * ext.n_out())(*([None] if side == "output" else []), ctypes.cast(scalar, ptr))
  work = (ctypes.c_double * max(module.workspace_size, 1))()
  assert entry(args, res, None, work, 0) == 0
  np.testing.assert_allclose(scalar[0], expected)
  assert entry(None, res, None, work, 0) == 1
  assert entry(args, None, None, work, 0) == 1
  res[ext.n_out() - 1] = None
  assert entry(args, res, None, work, 0) == 3
  if side == "output":
    args[0] = None
    assert entry(args, res, None, work, 0) == 4


@pytest.mark.parametrize("lang", ["c", "cpp"])
@pytest.mark.parametrize("shape", [(2, 3), (3, 2), (2, 2), (1, 3), (3, 1)])
def test_casadi_dense_matrices_match_numpy(tmp_path, lang, shape):
  weights = np.arange(np.prod(shape), dtype=float).reshape(shape) + 0.5

  @sc.function(sc.arg("m", shape), outputs=sc.group(sc.arg("n"), sc.arg("t"), sc.arg("s")), name="dense")
  def fun(m):
    return m * weights, m.T, (m * weights).sum()

  module, lib_path = _build(tmp_path, fun, lang=lang)
  ext = casadi.external(fun.name, str(lib_path))
  matrix = np.arange(np.prod(shape), dtype=float).reshape(shape) - 2
  n, t, s = ext(matrix)
  assert ext.sparsity_in(0) == casadi.Sparsity.dense(*shape)
  assert ext.sparsity_out(0) == casadi.Sparsity.dense(*shape)
  np.testing.assert_allclose(n.full(), matrix * weights)
  np.testing.assert_allclose(t.full(), matrix.T)
  np.testing.assert_allclose(float(s), (matrix * weights).sum())
  np.testing.assert_allclose(fun(matrix)[0], matrix * weights)
  assert module.header == render_c_api_header(fun, lang=lang, casadi=True)
  assert module.workspace_size == workspace_size(fun, casadi=True)


def test_casadi_column_major_cpp_buffer_indexing(tmp_path):
  @sc.function(sc.arg("x", (2, 3)), name="matrix_cpp")
  def fun(x):
    return x

  module, _ = _build(tmp_path, fun, lang="cpp")
  client = tmp_path / "client.cpp"
  client.write_text(
    f'#include "{module.header_name}"\n'
    "static_assert(matrix_cpp::x_t::offset(1, 2) == 5);\n"
    "static_assert(matrix_cpp::x_t::offset(0, 1) == 2);\n"
    "int main() {\n"
    "  matrix_cpp::x_t x{};\n"
    "  for (int k = 0; k < 6; ++k) x.data[k] = k;\n"
    "  const auto& cx = x;\n"
    "  return x(0, 1) == 2 && cx(1, 0) == 1 && x.ptr()[4] == 4 ? 0 : 1;\n"
    "}\n"
  )
  executable = tmp_path / "client"
  subprocess.run([*cxx_compiler(), "-std=c++17", "-pedantic-errors", "-Wall", "-Wextra", "-Werror", str(client), "-o", str(executable)], check=True)
  subprocess.run([str(executable)], check=True)


def test_casadi_dense_matrix_result_from_vector(tmp_path):
  @sc.function(sc.arg("x", 6), name="dense_result")
  def fun(x):
    return x.reshape((2, 3))

  _, lib_path = _build(tmp_path, fun)
  ext = casadi.external(fun.name, str(lib_path))
  np.testing.assert_array_equal(ext(np.arange(6)).full(), np.arange(6).reshape((2, 3)))


@pytest.mark.solver("piqp")
def test_casadi_dense_solver_parameters_use_a_lowered_function_wrapper(tmp_path):
  weights = np.arange(6, dtype=float).reshape((2, 3)) + 1

  @sc.problem(vars=sc.arg("x", 2), params=sc.arg("p", (2, 3)))
  def problem(x, p):
    target = (p * weights).sum()
    return sc.ProblemSpec(minimize=(x[0] - target) ** 2 + x[1] ** 2, eq=(x[1],), ineq=(sc.bounded(x[:1], lo=-1000.0, hi=1000.0),))

  solve = sc.solver(problem, "piqp", name="dense_solver")
  for render in (render_c_module, render_c_api_header, workspace_size):
    with pytest.raises(ValueError, match="solver entry.*Function wrapper"):
      render(solve.function, casadi=True)

  @sc.function(sc.arg("x0", 2), sc.arg("lam0", 2), sc.arg("lam_eq", 1), sc.arg("lam_ineq", 1), sc.arg("p", (2, 3)), name="dense_solver_external")
  def wrapped(x0, lam0, lam_eq, lam_ineq, p):
    return solve.function(x0, lam0, lam_eq, lam_ineq, p)

  _, lib_path = _build(tmp_path, wrapped)
  ext = casadi.external(wrapped.name, str(lib_path))
  p = np.arange(6, dtype=float).reshape((2, 3))
  result = ext(np.zeros(2), np.zeros(2), np.zeros(1), np.zeros(1), p)
  np.testing.assert_allclose(result[0].full().ravel(), [(p * weights).sum(), 0])


@pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "empty"])
@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_casadi_sparse_leaves_match_casadi_sparsity(tmp_path, empty, lang):
  from scipy import sparse

  pattern = sc.SparsityPattern((3, 4), (), ()) if empty else sc.SparsityPattern((3, 4), (0, 0, 1, 2), (1, 3, 0, 1))
  pattern, _ = pattern.canonicalize()
  type_ = sc.TensorType(pattern.shape, pattern=pattern)

  @sc.function(sc.arg("a", type_), sc.arg("x", 4), outputs=sc.group(sc.arg("b", type_), sc.arg("y", 3)), name="sparse_external")
  def fun(a, x):
    return a * 2.0, a @ x

  module, lib_path = _build(tmp_path, fun, lang=lang)
  client = tmp_path / ("header.c" if lang == "c" else "header.cpp")
  client.write_text(f'#include "{module.header_name}"\nint main(void) {{ return 0; }}\n')
  compiler, standard = (c_compiler(), "-std=c11") if lang == "c" else (cxx_compiler(), "-std=c++17")
  subprocess.run(
    [*compiler, standard, "-pedantic-errors", "-Wall", "-Wextra", "-Werror", "-c", str(client), "-o", str(client.with_suffix(".o"))], check=True
  )
  ext = casadi.external(fun.name, str(lib_path))
  expected_sparsity = casadi.Sparsity.triplet(*pattern.shape, list(pattern.rows), list(pattern.cols))
  assert ext.sparsity_in(0) == expected_sparsity
  assert ext.sparsity_out(0) == expected_sparsity
  assert ext.sparsity_in(1) == casadi.Sparsity.dense(4, 1)
  assert ext.sparsity_out(1) == casadi.Sparsity.dense(3, 1)
  values = np.arange(pattern.nnz, dtype=float)  # Includes an explicit stored zero.
  reference = sparse.coo_array((values, (pattern.rows, pattern.cols)), shape=pattern.shape).tocsc()
  matrix = casadi.DM(expected_sparsity, reference.data)
  x = np.array([2.0, -3.0, 5.0, 7.0])
  b, y = ext(matrix, x)
  np.testing.assert_allclose(b.full(), reference.toarray() * 2)
  np.testing.assert_allclose(y.full().ravel(), reference @ x)
  assert module.workspace_size == workspace_size(fun)
  assert "_native" not in module.body


@pytest.mark.parametrize("side", ["input", "output"])
@pytest.mark.parametrize("layout", ["csr", "coo", "batch"])
def test_casadi_rejects_sparse_leaves_it_cannot_describe(side, layout):
  pattern = sc.SparsityPattern((3, 4), (0, 1), (1, 0))
  pattern, _ = pattern.canonicalize()
  shape = (2, *pattern.shape) if layout == "batch" else pattern.shape
  order = sc.SparsityPattern(pattern.shape, (1, 0), (0, 1)) if layout == "coo" else "csr" if layout == "csr" else "csc"
  leaf = sc.arg("a", sc.TensorType(shape, pattern=pattern), layout=order)
  if side == "input":
    fun = sc.function(leaf, name="bad_sparse_external")(lambda a: sc.values(a).sum())
  else:
    fun = sc.function(sc.arg("v", int(np.prod(shape[:-2])) * pattern.nnz), outputs=leaf, name="bad_sparse_external")(
      lambda v: sc.pack(v.reshape((*shape[:-2], pattern.nnz)), pattern)
    )
  reason = "batch axes" if layout == "batch" else f"must declare CSC layout; got {layout!r}"
  with pytest.raises(ValueError, match=f"{side} 'a'.*{reason}"):
    render_c_module(fun, casadi=True)
