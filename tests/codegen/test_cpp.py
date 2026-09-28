"""The C++ header (the ``cpp`` adapter): a ``Buffer`` per input and output with the ``Expr`` shape, a
namespace per function, ``constexpr`` sparsity tables, and ``call`` against a caller-owned
workspace, all compiled against the same C kernel the C header declares."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import render_c_module


def _spjac() -> sc.Function:
  x = sc.sym("x", 4)
  y = sc.stack([x[0], x[2:4].sum(), x[1] * x[3]])
  return sc.sparse_jacobian(sc.Function._from_exprs("f", [x], [y], ["x"], ["y"]), "y", "x", name="f_spjac")


def test_cpp_header_declares_namespace_buffers_and_constexpr_tables() -> None:
  traj = sc.sym("traj", (3, 2))
  f = sc.Function._from_exprs("roll", [traj], [traj.sin()], ["traj"], ["out"])
  module = render_c_module(f, adapters=("cpp",))
  assert module.header_name == "roll.hpp" and module.source_name == "roll.c"
  assert "#include" not in module.source.replace(module.body, "")
  header = module.header
  assert "#ifndef SCALY_BUFFER_HPP" in header
  assert "namespace roll {" in header
  assert 'extern "C" int roll(const double** arg, double** res, int* iw, double* w, int mem);' in header
  assert "using traj_t = Buffer<double, 3, 2>;" in header
  assert "using out_t = Buffer<double, 3, 2>;" in header
  assert "using workspace_t = Buffer<double, roll_SZ_W>;" in header
  assert "inline int call(const traj_t& traj, out_t& out, workspace_t& workspace)" in header
  assert "typedef struct" not in header and "roll_call" not in header

  spjac = render_c_module(_spjac(), adapters=("cpp",)).header
  assert "namespace spjac_y_x {" in spjac
  assert "constexpr int nnz = 5;" in spjac
  assert "constexpr std::array<int, nnz> csc_val_perm = {0, 3, 1, 2, 4};" in spjac
  assert "constexpr std::array<int, ncol + 1> csc_col_ptr = {0, 1, 2, 3, 5};" in spjac
  assert "static const int" not in spjac


def test_cpp_header_compiles_and_runs(tmp_path) -> None:
  cc = shutil.which("cc")
  cxx = shutil.which("c++")
  if cc is None or cxx is None:
    pytest.skip("cc and c++ are required for the generated C++ header smoke test")

  traj = sc.sym("traj", (3, 2))
  roll = render_c_module(sc.Function._from_exprs("roll", [traj], [traj.sin()], ["traj"], ["out"]), adapters=("cpp",))
  spjac = render_c_module(_spjac(), adapters=("cpp",))
  for module in (roll, spjac):
    (tmp_path / module.header_name).write_text(module.header)
    (tmp_path / module.source_name).write_text(module.source)
  main = tmp_path / "main.cpp"
  main.write_text(
    """
#include <cmath>
#include <memory>
#include "roll.hpp"
#include "f_spjac.hpp"

static_assert(roll::traj_t::ndim == 2, "kept the Expr rank");
static_assert(roll::traj_t::shape[0] == 3 && roll::traj_t::shape[1] == 2, "kept the Expr shape");
static_assert(roll::traj_t::size == 6, "flattened size");
static_assert(alignof(roll::traj_t) == 16, "aligned");
static_assert(roll::sz_arg == 1 && roll::sz_res == 1, "constexpr sizes");
static_assert(f_spjac::spjac_y_x::nnz == 5, "constexpr nnz");

int main() {
  roll::traj_t traj = {};
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j) traj(i, j) = 0.1 * static_cast<double>(i) + static_cast<double>(j);
  roll::out_t out = {};
  auto workspace = std::make_unique<roll::workspace_t>();
  if (int err = roll::call(traj, out, *workspace)) return err;
  for (std::size_t i = 0; i < 3; ++i)
    for (std::size_t j = 0; j < 2; ++j)
      if (std::fabs(out(i, j) - std::sin(traj(i, j))) > 1e-12) return 10;

  f_spjac::x_t x = {{2.0, 3.0, 5.0, 7.0}};
  f_spjac::spjac_y_x_t values = {};
  f_spjac::workspace_t w;
  if (int err = f_spjac::call(x, values, w)) return err;
  // Column-major walk through the CSC tables: (0,0)=1, (2,1)=x3, (1,2)=1, (1,3)=1, (2,3)=x1.
  const double expected_csc[] = {1.0, 7.0, 1.0, 1.0, 3.0};
  const int expected_rows[] = {0, 2, 1, 1, 2};
  namespace sp = f_spjac::spjac_y_x;
  for (int k = 0; k < sp::nnz; ++k) {
    if (sp::csc_row_ind[k] != expected_rows[k]) return 20;
    if (std::fabs(values(sp::csc_val_perm[k]) - expected_csc[k]) > 1e-12) return 21;
  }
  return 0;
}
"""
  )
  objs = []
  for module in (roll, spjac):
    obj = tmp_path / (module.source_name + ".o")
    subprocess.run([cc, "-c", str(tmp_path / module.source_name), "-o", str(obj)], check=True)
    objs.append(str(obj))
  exe = tmp_path / "main"
  subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", str(main), *objs, "-lm", "-o", str(exe)], check=True)
  subprocess.run([str(exe)], check=True)


def test_cpp_header_and_c_header_compile_the_same_kernel() -> None:
  x = sc.sym("x", 2)
  f = sc.Function._from_exprs("same", [x], [x * 2.0], ["x"], ["y"])
  assert render_c_module(f, adapters=()).body == render_c_module(f, adapters=("cpp",)).body
  with pytest.raises(ValueError, match="unknown output adapter 'rust'; available: casadi, cpp"):
    render_c_module(f, adapters=("rust",))
  with pytest.raises(TypeError, match="sequence of names"):
    render_c_module(f, adapters="cpp")
  np.testing.assert_allclose(f(np.array([1.0, 2.0])), [2.0, 4.0])


def test_headers_survive_buffer_names_that_collide_with_the_wrapper(tmp_path) -> None:
  """An input named after the function, the workspace, or an entry parameter must not shadow them."""
  cc = shutil.which("cc")
  cxx = shutil.which("c++")
  if cc is None or cxx is None:
    pytest.skip("cc and c++ are required for the generated header smoke test")
  f, workspace, arg = sc.sym("f", 2), sc.sym("workspace", 2), sc.sym("arg", 2)
  fun = sc.Function._from_exprs("f", [f, workspace, arg], [f + workspace + arg], ["f", "workspace", "arg"], ["res"])
  c = render_c_module(fun)
  cpp = render_c_module(fun, adapters=("cpp",))
  assert (
    "static inline int f_call(const f_f__t* f_, const f_workspace__t* workspace_, const f_arg__t* arg_, f_res__t* res_, f_workspace_t* workspace)"
    in c.header
  )
  assert "inline int call(const f__t& f_, const workspace__t& workspace_, const arg__t& arg_, res__t& res_, workspace_t& workspace)" in cpp.header
  for module in (c, cpp):
    (tmp_path / module.header_name).write_text(module.header)
  (tmp_path / c.source_name).write_text(c.source)
  (tmp_path / "use.c").write_text(
    '#include "f.h"\nint use(void) { f_f__t a = {{1, 2}}; f_workspace__t b = {{3, 4}}; f_arg__t d = {{5, 6}}; f_res__t r; return f_call(&a, &b, &d, &r, NULL); }\n'
  )
  (tmp_path / "use.cpp").write_text(
    '#include "f.hpp"\nint use() { f::f__t a = {{1, 2}}; f::workspace__t b = {{3, 4}}; f::arg__t d = {{5, 6}}; f::res__t r; f::workspace_t w; return f::call(a, b, d, r, w); }\n'
  )
  subprocess.run([cc, "-std=c11", "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "use.c"), "-o", str(tmp_path / "use_c.o")], check=True)
  subprocess.run([cxx, "-std=c++17", "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "use.cpp"), "-o", str(tmp_path / "use_cpp.o")], check=True)
  subprocess.run([cc, "-c", str(tmp_path / c.source_name), "-o", str(tmp_path / "f.o")], check=True)


def test_headers_split_names_shared_by_an_input_and_an_output(tmp_path) -> None:
  """A solver Function takes a warm start and returns the solution under the same names, and one of
  its buffers may be empty; both headers must still compile."""
  cc = shutil.which("cc")
  cxx = shutil.which("c++")
  if cc is None or cxx is None:
    pytest.skip("cc and c++ are required for the generated header smoke test")
  w, lam, z0 = sc.sym("w", 3), sc.sym("lam", 0), sc.sym("z0", 2)
  fun = sc.Function._from_exprs("solve", [w, lam, z0], [w + z0[0], lam], ["w", "lam", "z0"], ["w", "lam"])
  c = render_c_module(fun)
  cpp = render_c_module(fun, adapters=("cpp",))
  assert "typedef struct { SCALY_ALIGNAS(16) double data[1]; } solve_lam_in_t;" in c.header
  assert (
    "static inline int solve_call(const solve_w_in_t* w_in, const solve_lam_in_t* lam_in, const solve_z0_t* z0, solve_w_out_t* w_out, solve_lam_out_t* lam_out, solve_workspace_t* workspace)"
    in c.header
  )
  assert (
    "inline int call(const w_in_t& w_in, const lam_in_t& lam_in, const z0_t& z0, w_out_t& w_out, lam_out_t& lam_out, workspace_t& workspace)"
    in cpp.header
  )
  for module in (c, cpp):
    (tmp_path / module.header_name).write_text(module.header)
  (tmp_path / c.source_name).write_text(c.source)
  (tmp_path / "use.c").write_text(
    '#include "solve.h"\nint use(void) { solve_w_in_t a = {{1, 2, 3}}; solve_lam_in_t l = {{0}}; solve_z0_t z = {{5, 6}}; solve_w_out_t r; solve_lam_out_t lo; return solve_call(&a, &l, &z, &r, &lo, NULL); }\n'
  )
  (tmp_path / "use.cpp").write_text(
    '#include "solve.hpp"\nint use() { solve::w_in_t a = {{1, 2, 3}}; solve::lam_in_t l = {}; solve::z0_t z = {{5, 6}}; solve::w_out_t r; solve::lam_out_t lo; solve::workspace_t ws; return solve::call(a, l, z, r, lo, ws); }\n'
  )
  subprocess.run(
    [cc, "-std=c11", "-pedantic", "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "use.c"), "-o", str(tmp_path / "use_c.o")], check=True
  )
  subprocess.run(
    [cxx, "-std=c++17", "-pedantic", "-Wall", "-Wextra", "-Werror", "-c", str(tmp_path / "use.cpp"), "-o", str(tmp_path / "use_cpp.o")], check=True
  )
  subprocess.run([cc, "-c", str(tmp_path / c.source_name), "-o", str(tmp_path / "solve.o")], check=True)


def test_parameters_named_like_keywords_compile_from_c_and_cpp(tmp_path) -> None:
  """Parameter names become C names by default; a C or C++ keyword gets a trailing underscore."""
  cxx = shutil.which("c++")
  if cxx is None:
    pytest.skip("c++ is required to compile the generated headers")

  @sc.function(2, 2, output="delete")
  def keywords(new, this):
    return new * this

  for lang in ("c", "cpp"):
    module = render_c_module(keywords, adapters=("cpp",) if lang == "cpp" else ())
    (tmp_path / module.header_name).write_text(module.header)
    main = tmp_path / f"main_{lang}.cpp"
    main.write_text(f'#include "{module.header_name}"\nint main() {{ return 0; }}\n')
    subprocess.run([cxx, "-std=c++17", "-fsyntax-only", str(main)], check=True, cwd=tmp_path)
  assert "new_" in render_c_module(keywords, adapters=("cpp",)).header
