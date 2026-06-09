from __future__ import annotations

import ctypes
import shutil
import subprocess
import sys

import numpy as np
import pytest

import alloy as al
from alloy.expr import topo


def test_elementwise_eval_and_topological_order() -> None:
  x = al.sym("x", 3)
  y = (x.sin() + x * x).sum()
  f = al.Function("f", [x], [y], ["x"], ["y"])

  np.testing.assert_allclose(f(np.array([1.0, 2.0, 3.0])), np.sin([1.0, 2.0, 3.0]).sum() + 14.0)

  nodes = topo(f.outputs)
  loc = {e.id: i for i, e in enumerate(nodes)}
  assert nodes[-1].op == al.Ops.SUM
  assert [e.op for e in nodes].count(al.Ops.INPUT) == 1
  assert all(loc[arg.id] < loc[e.id] for e in nodes for arg in e.args)


def test_common_ops_contains_modeling_basics() -> None:
  for op in [
    al.Ops.SIN,
    al.Ops.COS,
    al.Ops.TAN,
    al.Ops.ATAN2,
    al.Ops.SINH,
    al.Ops.COSH,
    al.Ops.TANH,
    al.Ops.EXP,
    al.Ops.LOG,
    al.Ops.SQRT,
    al.Ops.TRANSPOSE,
    al.Ops.SLICE,
    al.Ops.GATHER,
    al.Ops.SCATTER,
    al.Ops.CONCAT,
    al.Ops.MATMUL,
    al.Ops.CALL,
  ]:
    assert op in al.COMMON_OPS
  assert al.Ops.SIN.value == "sin"


def test_binary_nonlinear_method_helpers_eval() -> None:
  x = al.sym("x", 3)
  y = al.sym("y", 3)
  atan = x.atan2(y)
  mn = x.minimum(y)
  mx = x.maximum(y)
  f = al.Function("binary_helpers", [x, y], [atan, mn, mx], ["x", "y"], ["atan", "min", "max"])
  xv = np.array([0.5, -1.0, 2.0])
  yv = np.array([1.5, 2.0, -0.25])

  atan_v, mn_v, mx_v = f(xv, yv)
  np.testing.assert_allclose(atan_v, np.arctan2(xv, yv))
  np.testing.assert_allclose(mn_v, np.minimum(xv, yv))
  np.testing.assert_allclose(mx_v, np.maximum(xv, yv))
  assert atan.type.diff
  assert not mn.type.diff
  assert not mx.type.diff


def test_type_shapes_reject_negative_dimensions_and_mismatched_sparsity() -> None:
  for make in [lambda: al.sym("bad", -1), lambda: al.sym("bad", (2, -1)), lambda: al.TensorType((-1,))]:
    try:
      _ = make()
    except ValueError as e:
      assert "negative dimensions" in str(e)
    else:  # pragma: no cover
      raise AssertionError("negative shape should fail")

  try:
    _ = al.SparsityType((-1, 2), (), ())
  except ValueError as e:
    assert "negative dimensions" in str(e)
  else:  # pragma: no cover
    raise AssertionError("negative sparsity shape should fail")

  try:
    _ = al.TensorType((2, 2), sparsity=al.SparsityType.dense((2, 3)))
  except ValueError as e:
    assert "does not match sparsity shape" in str(e)
  else:  # pragma: no cover
    raise AssertionError("mismatched tensor sparsity shape should fail")


def test_structural_transpose_concat_vec_eval_and_ad() -> None:
  x = al.sym("x", (2, 2))
  y = al.concat([x.T, x + 1.0], axis=1).vec()
  f = al.Function("f", [x], [y], ["x"], ["y"])
  jf = al.jacobian(f, "x", "y")
  xv = np.array([[1.0, 2.0], [3.0, 4.0]])

  np.testing.assert_allclose(f(xv), np.concatenate([xv.T, xv + 1.0], axis=1).reshape(8))
  np.testing.assert_allclose(
    jf(xv),
    np.array(
      [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
      ]
    ),
  )


def test_slice_split_eval_and_ad() -> None:
  x = al.sym("x", 4)
  left, right = al.split(x, [2, 2])
  y = al.stack([x[0], x[2:4].sum(), al.concat([left, right])[3]])
  f = al.Function("f", [x], [y], ["x"], ["y"])
  jf = al.jacobian(f, "x", "y")
  xv = np.array([1.0, 2.0, 3.0, 4.0])

  np.testing.assert_allclose(f(xv), np.array([1.0, 7.0, 4.0]))
  np.testing.assert_allclose(jf(xv), np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 1.0], [0.0, 0.0, 0.0, 1.0]]))

  try:
    _ = al.split(x, [1, 2])
  except ValueError as e:
    assert "do not sum to axis length 4" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid split should fail")

  try:
    _ = al.split(x, [-1, 5])
  except ValueError as e:
    assert "cannot contain negative entries" in str(e)
  else:  # pragma: no cover
    raise AssertionError("negative split size should fail")


def test_gather_scatter_eval_and_ad() -> None:
  x = al.sym("x", 5)
  y = al.scatter(x.gather([3, 1, 4]), [0, 2, 3], 5)
  f = al.Function("f", [x], [y], ["x"], ["y"])
  jf = al.jacobian(f, "x", "y")
  xv = np.array([10.0, 11.0, 12.0, 13.0, 14.0])

  np.testing.assert_allclose(f(xv), np.array([13.0, 0.0, 11.0, 14.0, 0.0]))
  np.testing.assert_allclose(
    jf(xv),
    np.array(
      [
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 1.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
      ]
    ),
  )

  try:
    _ = al.gather(x, [5])
  except IndexError as e:
    assert "indices must be in [0, 5)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("out-of-range gather should fail")

  try:
    _ = al.scatter(al.sym("v", 2), [1, 1], 3)
  except ValueError as e:
    assert "scatter indices must be unique" in str(e)
  else:  # pragma: no cover
    raise AssertionError("duplicate scatter should fail")


def test_dot_sumsqr_and_norm_2() -> None:
  x = al.sym("x", (2, 2))
  f = al.Function("f", [x], [al.dot(x, x.T), x.sumsqr(), al.norm_2(x)], ["x"], ["dot", "sumsqr", "norm"])
  xv = np.array([[1.0, 2.0], [3.0, 4.0]])

  dot_val, sumsqr_val, norm_val = f(xv)
  np.testing.assert_allclose(dot_val, np.dot(xv.reshape(-1), xv.T.reshape(-1)))
  np.testing.assert_allclose(sumsqr_val, np.sum(xv * xv))
  np.testing.assert_allclose(norm_val, np.linalg.norm(xv.reshape(-1)))

  try:
    _ = al.dot(al.sym("a", 2), al.sym("b", 3))
  except ValueError as e:
    assert "dot size mismatch" in str(e)
  else:  # pragma: no cover
    raise AssertionError("dot size mismatch should fail")


def test_shape_checks_for_structural_ops() -> None:
  x = al.sym("x", (2, 3))
  y = al.sym("y", (4, 2))

  try:
    _ = x @ y
  except ValueError as e:
    assert "cannot matmul shapes (2, 3) and (4, 2)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid matmul should fail")

  try:
    _ = x.transpose((0, 0))
  except ValueError as e:
    assert "not a permutation" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid transpose should fail")

  try:
    _ = al.concat([x, al.sym("z", (2, 4))], axis=0)
  except ValueError as e:
    assert "cannot concat shapes" in str(e)
  else:  # pragma: no cover
    raise AssertionError("invalid concat should fail")


def test_mixed_lowering_hints_survive_semantic_graph() -> None:
  x = al.sym("x", 3)
  scalar_region = (x.sin() + x * x).scalar()
  block_region = (al.const(np.eye(3)) @ x).block()
  opaque_region = (x + 1.0).opaque()
  f = al.Function("mixed", [x], [scalar_region + block_region + opaque_region], ["x"], ["y"])

  lowerings = [e.lowering for e in topo(f.outputs) if e.lowering != "auto"]
  assert "scalar" in lowerings
  assert "block" in lowerings
  assert "opaque" in lowerings


def test_function_call_node_eval() -> None:
  x = al.sym("x", 2)
  inner = al.Function("inner", [x], [x.sin()], ["x"], ["y"])
  z = al.sym("z", 2)
  (inner_z,) = inner.call([z])
  outer = al.Function("outer", [z], [inner_z + 1.0], ["z"], ["out"])

  np.testing.assert_allclose(outer(np.array([0.1, 0.2])), np.sin([0.1, 0.2]) + 1.0)
  assert any(e.op == al.Ops.CALL for e in topo(outer.outputs))


def test_function_call_normalizes_raw_constant_args() -> None:
  x = al.sym("x", 2)
  inner = al.Function("inner", [x], [x + 1.0], ["x"], ["y"])
  (inner_const,) = inner.call([[1.0, 2.0]])
  outer = al.Function("outer", [], [inner_const], [], ["out"])

  np.testing.assert_allclose(outer(), np.array([2.0, 3.0]))
  assert not inner_const.type.diff


def test_function_signature_and_call_shape_errors() -> None:
  x = al.sym("x", 2)

  try:
    _ = al.Function("bad", [x], [x], [], ["y"])
  except ValueError as e:
    assert "expected 1 input names, got 0" in str(e)
  else:  # pragma: no cover
    raise AssertionError("input name arity mismatch should fail")

  y = al.sym("y", 2)
  try:
    _ = al.Function("missing", [x], [x + y], ["x"], ["z"])
  except ValueError as e:
    assert "undeclared symbolic inputs: ['y']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("undeclared graph input should fail")

  f = al.Function("f", [x], [x], ["x"], ["y"])
  try:
    _ = f.call([al.sym("z", 3)])
  except ValueError as e:
    assert "call argument 'x' has shape (3,), expected (2,)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("call shape mismatch should fail")


def test_debug_printing_uses_stable_topological_names() -> None:
  x = al.sym("x", 2)
  text = ((x + 1.0) * x).debug()

  assert "%0 = input x : float64(2,)" in text
  assert "add(%0, %1)" in text
  assert text.endswith("outputs %3")


def test_structural_equality_collapses_to_identity() -> None:
  # Construction-time interning: two ``Expr``s with the same structural key are the same
  # Python object, so structural equality is the same as ``is`` equality.
  x0 = al.sym("x", 2)
  x1 = al.sym("x", 2)
  y = al.sym("y", 2)

  assert x0 is x1
  assert x0.id == x1.id
  assert x0.structurally_equal(x1)
  assert x0.structural_hash() == x1.structural_hash()
  assert x0 is not y
  assert not x0.structurally_equal(y)


def test_differentiability_metadata_propagates_through_exprs() -> None:
  x = al.sym("x", 3)
  p = al.sym("p", 3, diff=False)
  c = al.const([1.0, 2.0, 3.0])

  assert x.type.diff
  assert not p.type.diff
  assert not c.type.diff
  assert (x + c).type.diff
  assert not (p + c).type.diff
  assert x.reshape((3, 1)).T.type.diff
  assert x.gather([2, 0]).type.diff
  assert al.scatter(p.gather([1, 2]), [0, 2], 3).type.diff is False
  assert al.stack([p, c]).type.diff is False
  assert al.concat([x[:1], p[:1]]).type.diff
  assert not x.floor().type.diff
  assert not al.minimum(x, p).type.diff

  u = al.sym("u", 3)
  inner = al.Function("inner", [u], [u * u], ["u"], ["y"])
  (diff_call,) = inner.call([x])
  (const_call,) = inner.call([c])
  assert diff_call.type.diff
  assert not const_call.type.diff


def test_cse_merges_equivalent_subgraphs() -> None:
  x = al.sym("x", 2)
  y = al.cse((x + 1.0) * (x + 1.0))

  assert y.op == al.Ops.MUL
  assert y.args[0] is y.args[1]
  np.testing.assert_allclose(al.Function("cse_eval", [x], [y], ["x"], ["y"])(np.array([2.0, 3.0])), np.array([9.0, 16.0]))

  a, b = x[0], x[1]
  z = al.cse(al.stack([a * b, b * a]))
  assert z.args[0] is z.args[1]


def test_simplify_rewrites_algebraic_identities_and_folds_constants() -> None:
  x = al.sym("x", 3)
  y = al.simplify(((x + 0.0) * 1.0).reshape((3,)))
  assert y is x

  c = al.simplify((al.const([1.0, 2.0]) + al.const([3.0, 4.0])).sum())
  assert c.op == al.Ops.CONST
  assert c.value is not None
  np.testing.assert_allclose(c.value, 10.0)

  z = al.simplify(x * 0.0)
  assert z.op == al.Ops.CONST
  assert z.value is not None
  np.testing.assert_allclose(z.value, np.zeros(3))

  m = al.simplify(al.const(np.zeros((2, 3))) @ al.sym("v", 3))
  assert m.op == al.Ops.CONST
  assert m.value is not None
  np.testing.assert_allclose(m.value, np.zeros(2))

  q = al.sym("q", 2)
  np.testing.assert_allclose(al.Function("simp_sub", [q], [al.simplify(q - q)], ["q"], ["y"])(np.array([2.0, 3.0])), np.zeros(2))
  np.testing.assert_allclose(al.Function("simp_div", [q], [al.simplify(q / q)], ["q"], ["y"])(np.array([2.0, 3.0])), np.ones(2))
  np.testing.assert_allclose(al.Function("simp_cse", [q], [al.simplify(al.cse(q + q))], ["q"], ["y"])(np.array([2.0, 3.0])), np.array([4.0, 6.0]))


def test_jit_reports_input_errors() -> None:
  x = al.sym("x", 2)
  y = al.sym("y", 2)
  out = ((x + 2.0) * y).sum()
  f = al.Function("f", [x, y], [out], ["x", "y"], ["out"])
  env = {"x": np.array([1.0, 3.0]), "y": np.array([4.0, 5.0])}

  np.testing.assert_allclose(f(**env), ((env["x"] + 2.0) * env["y"]).sum())

  try:
    _ = f(x=env["x"])
  except TypeError as e:
    assert "missing keyword inputs: ['y']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("missing keyword input should fail")

  try:
    _ = f(x=env["x"], y=env["y"], z=env["x"])
  except TypeError as e:
    assert "unexpected keyword inputs: ['z']" in str(e)
  else:  # pragma: no cover
    raise AssertionError("extra keyword input should fail")

  try:
    f(x=env["x"].reshape(1, 2), y=env["y"])
  except ValueError as e:
    assert "input 'x' has shape (1, 2), expected (2,)" in str(e)
  else:  # pragma: no cover
    raise AssertionError("shape mismatch should fail")


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
