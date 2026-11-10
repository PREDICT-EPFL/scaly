"""Sparse typed interfaces pass declared-order values and publish one local pattern per leaf."""

from math import prod
import re
import subprocess

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen.aot import render_c_api_header, write_module
from scaly.codegen.casadi import CASADI_QUERIES
from scaly.ir.types import CanonicalPattern
from tests.compilers import c_compiler, cxx_compiler


PATTERN = CanonicalPattern((3, 4), (0, 0, 1, 2, 2), (1, 3, 0, 1, 2))
COORDINATES = sc.SparsityPattern(PATTERN.shape, (2, 0, 2, 1, 0), (2, 3, 1, 0, 1))


def _array(header: str, prefix: str, name: str, lang: str) -> tuple[int, ...]:
  expression = rf"{prefix}_{name}\[\d+\] = \{{([^}}]*)\}}" if lang == "c" else rf" {name} = \{{([^}}]*)\}}"
  match = re.search(expression, header)
  assert match is not None, (prefix, name, header)
  return tuple(int(value) for value in match[1].split(",") if value.strip())


def _coordinates(header: str, prefix: str, format_: str, lang: str) -> tuple[np.ndarray, np.ndarray]:
  table = lambda name: np.array(_array(header, prefix, name, lang), dtype=int)
  if format_ == "coo":
    return table("rows"), table("cols")
  if format_ == "csr":
    return np.repeat(np.arange(PATTERN.shape[0]), np.diff(table("csr_row_ptr"))), table("csr_col_ind")
  return table("csc_row_ind"), np.repeat(np.arange(PATTERN.shape[1]), np.diff(table("csc_col_ptr")))


@pytest.mark.parametrize("lang", ["c", "cpp"])
@pytest.mark.parametrize("layout", ["csc", "csr", COORDINATES], ids=["csc", "csr", "coo"])
@pytest.mark.parametrize("batch", [(), (2, 3), (0,)])
@pytest.mark.parametrize("empty", [False, True], ids=["nonempty", "empty"])
def test_sparse_headers_compile_and_run_declared_layouts(lang, layout, batch, empty, tmp_path):
  pattern = CanonicalPattern(PATTERN.shape, (), ()) if empty else PATTERN
  if empty and isinstance(layout, sc.SparsityPattern):
    layout = sc.SparsityPattern(pattern.shape, (), ())
  type_ = sc.TensorType((*batch, *pattern.shape), pattern=pattern)
  out_layout = "csr" if layout == "csc" else COORDINATES if layout == "csr" else "csc"
  if empty and isinstance(out_layout, sc.SparsityPattern):
    out_layout = sc.SparsityPattern(pattern.shape, (), ())

  @sc.function(sc.arg("a", type_, layout=layout), outputs=sc.arg("b", type_, layout=out_layout), name="sparse_abi")
  def fun(a):
    return a * 2.0

  module = write_module(fun, tmp_path, lang=lang)
  header = module.header
  assert header == render_c_api_header(fun, lang=lang)
  assert not re.search(r"\w+\[0\] =", header)
  assert "val_perm" not in header
  count = prod(batch) * pattern.nnz
  if lang == "c":
    for ident in ("a", "b"):
      assert f"double data[{max(count, 1)}]; }} sparse_abi_{ident}_t;" in header
      for name, value in (
        ("NROW", pattern.shape[0]),
        ("NCOL", pattern.shape[1]),
        ("NNZ", pattern.nnz),
        ("NDIM", type_.ndim),
        ("STORAGE_SIZE", count),
      ):
        assert f"#define sparse_abi_{ident}_{name} {value}\n" in header
      assert _array(header, f"sparse_abi_{ident}", "shape", lang) == type_.shape
    local_a, local_b = header, header
  else:
    dims = ", ".join(map(str, type_.storage_shape))
    assert f"using a_t = Buffer<double, {dims}>;" in header
    assert f"using b_t = Buffer<double, {dims}>;" in header
    local_a = header.split("namespace a {", 1)[1].split("\n}", 1)[0]
    local_b = header.split("namespace b {", 1)[1].split("\n}", 1)[0]
    for local in (local_a, local_b):
      for name, value in (("nrow", pattern.shape[0]), ("ncol", pattern.shape[1]), ("nnz", pattern.nnz), ("ndim", type_.ndim)):
        assert f"constexpr int {name} = {value};" in local
      assert f"constexpr std::size_t storage_size = {count};" in local
      assert _array(local, "", "shape", lang) == type_.shape
  format_ = "coo" if isinstance(layout, sc.SparsityPattern) else layout
  out_format = "coo" if isinstance(out_layout, sc.SparsityPattern) else out_layout
  if not empty:
    rows, cols = _coordinates(local_a, "sparse_abi_a", format_, lang)
    out_rows, out_cols = _coordinates(local_b, "sparse_abi_b", out_format, lang)
    canonical = sparse.coo_array((np.ones(pattern.nnz), (pattern.rows, pattern.cols)), shape=pattern.shape)
    for format_name, actual_rows, actual_cols in ((format_, rows, cols), (out_format, out_rows, out_cols)):
      reference = canonical.tocsc() if format_name == "csc" else canonical.tocsr()
      reference_rows, reference_cols = (
        (reference.indices, np.repeat(np.arange(pattern.shape[1]), np.diff(reference.indptr)))
        if format_name == "csc"
        else (np.repeat(np.arange(pattern.shape[0]), np.diff(reference.indptr)), reference.indices)
      )
      if format_name == "coo":
        reference_rows, reference_cols = COORDINATES.rows, COORDINATES.cols
      np.testing.assert_array_equal(actual_rows, reference_rows)
      np.testing.assert_array_equal(actual_cols, reference_cols)
    assert len(rows) == pattern.nnz
    # Reconstruct the caller's matrix exclusively from the published tables.
    values = np.arange(1.0, count + 1.0).reshape(-1, pattern.nnz)
    if count:
      values[0, 0] = 0.0  # An explicit stored zero must still occupy its slot.
    references = [sparse.coo_array((v, (rows, cols)), shape=pattern.shape).toarray() * 2 for v in values]
    expected = np.array([matrix[out_rows, out_cols] for matrix in references]).reshape(-1)
    assert set(zip(rows, cols)) == set(zip(pattern.rows, pattern.cols))
    # A batched leaf publishes each local index array once, never once per batch element.
    table = "rows" if format_ == "coo" else f"{format_}_{'row_ind' if format_ == 'csc' else 'col_ind'}"
    assert len(_array(local_a, "sparse_abi_a", table, lang)) == pattern.nnz
  else:
    values, expected = np.empty(0), np.empty(0)
    for format_name, local, prefix in ((format_, local_a, "sparse_abi_a"), (out_format, local_b, "sparse_abi_b")):
      if format_name != "coo":
        name = f"{format_name}_{'col_ptr' if format_name == 'csc' else 'row_ptr'}"
        assert _array(local, prefix, name, lang) == (0,) * (pattern.shape[1 if format_name == "csc" else 0] + 1)
  initializer = ", ".join(map(str, values.reshape(-1))) or "0"
  if lang == "c":
    declarations = f"sparse_abi_a_t a = {{{{{initializer}}}}}; sparse_abi_b_t b = {{0}}; sparse_abi_workspace_t w;"
    call = "sparse_abi_call(&a, &b, &w)"
  else:
    declarations = f"sparse_abi::a_t a = {{{{{initializer}}}}}; sparse_abi::b_t b = {{}}; sparse_abi::workspace_t w;"
    call = "sparse_abi::call(a, b, w)"
  main = tmp_path / ("main.c" if lang == "c" else "main.cpp")
  main.write_text(
    f'#include <stdio.h>\n#include "{module.header_name}"\n'
    f"int main(void) {{ {declarations}\n"
    f"  int status = {call}; if (status) return status;\n"
    f'  for (int k = 0; k < {count}; ++k) printf("%.17g\\n", b.data[k]);\n  return 0;\n}}\n'
  )
  obj, exe = tmp_path / "kernel.o", tmp_path / "main"
  subprocess.run([*c_compiler(), "-c", str(tmp_path / module.source_name), "-o", str(obj)], check=True)
  compiler, standard = (c_compiler(), "-std=c11") if lang == "c" else (cxx_compiler(), "-std=c++17")
  subprocess.run([*compiler, standard, "-pedantic-errors", "-Wall", "-Wextra", "-Werror", str(main), str(obj), "-lm", "-o", str(exe)], check=True)
  result = subprocess.run([str(exe)], check=True, text=True, capture_output=True)
  np.testing.assert_allclose(np.fromstring(result.stdout, sep="\n"), expected)


@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_sparse_headers_split_shared_names_and_avoid_alias_collisions(lang, tmp_path):
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)

  @sc.function(sc.arg("x_t", type_), sc.arg("x", type_), outputs=sc.arg("x", type_), name="names")
  def fun(a, b):
    return a + b

  module = write_module(fun, tmp_path, lang=lang)
  if lang == "c":
    assert "#define names_x_t_NNZ 5" in module.header
    assert "#define names_x_in_NNZ 5" in module.header
    assert "#define names_x_out_NNZ 5" in module.header
  else:
    for name in ("x_t", "x_in", "x_out"):
      assert f"namespace {name} {{" in module.header
  main = tmp_path / "use.cpp"
  main.write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
  subprocess.run([*cxx_compiler(), "-std=c++17", "-pedantic-errors", "-fsyntax-only", str(main)], check=True)


@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_dense_headers_allow_internal_sparse_calls(lang):
  inner = sc.function(sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN)), name="sparse_header_inner")(lambda x: x.sin())
  outer = sc.function(sc.arg("x", PATTERN.nnz), name="dense_header_outer")(lambda x: sc.values(inner(sc.pack(x, PATTERN))))
  header = render_c_api_header(outer, lang=lang)
  assert outer.name in header
  np.testing.assert_allclose(outer(np.arange(5.0)), np.sin(np.arange(5.0)))


@pytest.mark.parametrize("lang", ["c", "cpp"])
@pytest.mark.parametrize("empty", [False, True])
def test_declared_sparse_and_derivative_metadata_coexist(lang, empty, tmp_path):
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)

  @sc.function(sc.arg("spjac_y:x", type_), sc.arg("x", 2), outputs=sc.arg("y"), name="mixed_metadata")
  def fun(a, x):
    return sc.values(a).sum() + (sc.const(np.zeros(2)) if empty else x * x).sum()

  derivative = sc.sparse_jacobian(fun, "y", "x", name="mixed_sparse")
  module = write_module(derivative, tmp_path, lang=lang)
  if not empty:
    assert "csc_val_perm" in module.header and "csr_val_perm" in module.header
  assert not re.search(r"\w+\[0\] =", module.header)
  main = tmp_path / "main.cpp"
  main.write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
  subprocess.run([*cxx_compiler(), "-std=c++17", "-pedantic-errors", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(main)], check=True)


@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_sparse_output_named_after_function_compiles(lang, tmp_path):
  @sc.function(sc.arg("x", PATTERN.nnz), name="selfsparse")
  def selfsparse(x):
    return sc.pack(x, PATTERN)

  module = write_module(selfsparse, tmp_path, lang=lang)
  if lang == "cpp":
    assert "namespace selfsparse_ {" in module.header
  main = tmp_path / "main.cpp"
  main.write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
  subprocess.run([*cxx_compiler(), "-std=c++17", "-pedantic-errors", "-fsyntax-only", str(main)], check=True)


@pytest.mark.parametrize("name", ["std", *(f"f_{query}" for query in CASADI_QUERIES)])
@pytest.mark.parametrize("side", ["input", "output"])
def test_cpp_sparse_metadata_avoids_declared_function_and_library_names(name, side, tmp_path):
  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)
  if side == "input":
    fun = sc.function(sc.arg(name, type_), name="f")(lambda a: sc.values(a))
  else:
    fun = sc.function(sc.arg("v", PATTERN.nnz), outputs=sc.arg(name, type_), name="f")(lambda v: sc.pack(v, PATTERN))
  module = write_module(fun, tmp_path, lang="cpp", casadi=True)
  main = tmp_path / "main.cpp"
  main.write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
  subprocess.run([*cxx_compiler(), "-std=c++17", "-pedantic-errors", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(main)], check=True)


@pytest.mark.parametrize("name", ["solver_names_with_options", "solver_qp_stats", "solver_qp_default_options"])
def test_cpp_sparse_metadata_avoids_solver_declarations(name, tmp_path):
  from tests.solvers.problem_helpers import build_qp

  type_ = sc.TensorType(PATTERN.shape, pattern=PATTERN)

  @sc.function(sc.arg(name, type_), sc.arg("mu", 2), outputs=sc.arg("b", type_), name="solver_names")
  def fun(a, mu):
    qp = build_qp(P=sc.const(np.eye(2)), c=-mu, name="solver_qp")
    return a * qp(mu)[0][0]

  module = write_module(fun, tmp_path, lang="cpp")
  main = tmp_path / "main.cpp"
  main.write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
  subprocess.run([*cxx_compiler(), "-std=c++17", "-pedantic-errors", "-Wall", "-Wextra", "-Werror", "-fsyntax-only", str(main)], check=True)
