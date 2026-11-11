"""External C kernels compile separately, compose in graphs and accept Function derivative rules."""

from dataclasses import FrozenInstanceError, replace
import json
import sys

import numpy as np
import pytest

import scaly as sc
from scaly.codegen.aot import render_c_module, write_module
from scaly.codegen.jit import get_compiled
from scaly.ir.expr import callees_of, invocation_key
from scaly.ir.program import ProgramOp, walk_program

PAIR_SOURCE = """#include "offset.h"
#ifndef SCALE
#define SCALE 1.0
#endif
void scaly_ext_pair(const double* x, double* squared, double* total, double* w) {
  total[0] = 0.0;
  for (int i = 0; i < 3; ++i) {
    w[i] = SCALE * x[i];
    squared[i] = w[i] * x[i];
    total[0] += x[i] + OFFSET;
  }
}
"""


def pair_library(**changes):
  return sc.CLibrary("pair", sources={"pair.c": PAIR_SOURCE}, headers={"offset.h": "#define OFFSET 0.25\n"}, **changes)


def pair(library=None):
  return sc.extern(
    library or pair_library(), "scaly_ext_pair", inputs=(sc.arg("x", 3),), outputs=sc.group(sc.arg("squared", 3), sc.arg("total", ())), workspace=3
  )


def host(library=None):
  external = pair(library)

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("squared", 3), sc.arg("total", ())), name="external_host")
  def evaluate(x):
    return external(x)

  return evaluate


def test_external_source_two_results_share_one_invocation_and_match_numpy():
  external = pair()
  x = sc.sym("x", 3)
  squared, total = external(x)
  assert squared.op == total.op == sc.ExprOp.EXTERN
  assert callees_of(squared) == callees_of(total) == (external,)
  assert invocation_key(squared) == invocation_key(total)
  module = render_c_module(host(), lanes=1)
  calls = [n for n in walk_program(module.program) if n.op == ProgramOp.CALL and n.attrs.get("external")]
  assert len(calls) == 1
  assert module.workspace_size >= 3
  assert module.body.count("scaly_ext_pair(") == 2  # declaration and invocation
  assert "SCALE" not in module.body
  value = np.array([0.2, -0.7, 1.1])
  actual = host()(value)
  np.testing.assert_allclose(actual[0], value**2)
  np.testing.assert_allclose(actual[1], (value + 0.25).sum())
  np.testing.assert_allclose(external(value)[0], value**2)


def test_external_call_nests_maps_and_allocates_unused_outputs():
  external = pair()

  @sc.function(sc.arg("x", 3), outputs=sc.arg("squared", 3))
  def square(x):
    return external(x.sin())[0]

  batch = np.arange(12.0).reshape(4, 3) / 7
  np.testing.assert_allclose(sc.vmap(square, 4)(batch), np.sin(batch) ** 2)


@pytest.mark.skipif(sys.platform not in {"linux", "darwin"}, reason="system libm link acceptance is Linux/macOS")
def test_named_system_math_library_is_resolved_and_linked():
  library = sc.CLibrary(
    "math_kernel",
    sources={"math.c": "#include <math.h>\nvoid scaly_ext_sine(const double* x, double* y, double* w) { (void)w; y[0] = sin(x[0]); }"},
    link=("m",),
    versions={"m": "system"},
  )
  sine = sc.extern(library, "scaly_ext_sine", inputs=(sc.arg("x", ()),), outputs=sc.arg("y", ()))
  np.testing.assert_allclose(sine(np.array(0.7)), np.sin(0.7))


def test_library_content_and_flags_change_jit_key(tmp_path, monkeypatch):
  monkeypatch.setenv("SCALY_CACHE_DIR", str(tmp_path))
  original = pair_library()
  libraries = [
    original,
    replace(original, sources={"pair.c": PAIR_SOURCE.replace("w[i] * x[i]", "2 * w[i] * x[i]")}),
    replace(original, compile_flags=("-DSCALE=3",)),
    replace(original, headers={"offset.h": "#define OFFSET 1.0\n"}),
    replace(original, defines={"SCALE": "4"}),
    replace(original, versions={"pair": "2"}),
  ]
  compiled = [get_compiled(host(library)) for library in libraries]
  assert len({item.cache_key for item in compiled}) == len(libraries)
  assert get_compiled(host(pair_library())).cache_key == compiled[0].cache_key
  value = np.array([0.2, -0.7, 1.1])
  for item, scale in zip(compiled[:3], (1, 2, 3), strict=True):
    np.testing.assert_allclose(item.run([value])[0], scale * value**2)
  with pytest.raises(FrozenInstanceError):
    original.name = "changed"


@pytest.mark.parametrize("mode", ["forward", "reverse", "jacobian"])
def test_active_derivative_names_custom_derivative(mode):
  x = sc.sym("x", 3)
  y = pair()(x)[0]
  with pytest.raises(NotImplementedError, match="scaly_ext_pair.*sc.custom_derivative"):
    if mode == "forward":
      sc.jvp(y, x, sc.const(np.ones(3)))
    elif mode == "reverse":
      sc.vjp([y], [x], [sc.const(np.ones(3))])
    else:
      sc.jacobian(y, x)


@pytest.mark.parametrize("declared", [False, True])
def test_wrapper_custom_derivative_matches_finite_differences(declared):
  external = pair()

  @sc.function(sc.arg("x", 3), outputs=sc.arg("y", 3))
  def square(x):
    return external(x)[0]

  @sc.function(sc.arg("x", 3), sc.arg("dx", 3), outputs=sc.arg("dy", 3))
  def forward(x, dx):
    return 2 * x * dx

  @sc.function(sc.arg("x", 3), outputs=sc.group(sc.arg("y", 3), sc.arg("residual", 3)))
  def primal(x):
    return square(x), x

  @sc.function(sc.arg("residual", 3), sc.arg("weight", 3), outputs=sc.arg("dx", 3))
  def backward(x, weight):
    return 2 * x * weight

  ruled = sc.custom_derivative(
    square, jvp=forward, fwd=primal, bwd=backward, sparsity=(lambda of, wrt, shape: np.eye(*shape, dtype=bool)) if declared else None
  )
  x = sc.sym("x", 3)
  pattern = sc.jacobian_sparsity(ruled(x), x)
  assert pattern.nnz == (3 if declared else 9)
  value = np.array([0.2, -0.7, 1.1])
  eps = 1e-6
  finite = np.column_stack([(ruled(value + eps * d) - ruled(value - eps * d)) / (2 * eps) for d in np.eye(3)])
  np.testing.assert_allclose(sc.jacobian(ruled)(value), finite, atol=1e-8)
  np.testing.assert_allclose(sc.gradient(sc.function(sc.arg("x", 3))(lambda x: ruled(x).sum()))(value), finite.sum(axis=0), atol=1e-8)


def test_aot_exports_library_source_headers_and_build_manifest(tmp_path):
  module = write_module(host(pair_library(compile_flags=("-DSCALE=2",), defines={"EXTRA": "1"})), tmp_path, lanes=1)
  manifest = json.loads((tmp_path / "external_host.build.json").read_text())
  assert manifest["source"] == module.source_name
  assert len(manifest["libraries"]) == 1
  library = manifest["libraries"][0]
  assert library["name"] == "pair"
  assert library["compile_flags"] == ["-DSCALE=2"]
  assert library["defines"] == {"EXTRA": "1"}
  assert library["link"] == []
  assert (tmp_path / library["sources"][0]).read_text() == PAIR_SOURCE
  assert (tmp_path / library["headers"][0]).read_text() == "#define OFFSET 0.25\n"
  assert library["include_dirs"][0] == (tmp_path / library["sources"][0]).parent.relative_to(tmp_path).as_posix()


def test_extern_sparse_leaves_keep_their_declared_storage_order():
  from scipy import sparse

  pattern = sc.SparsityPattern((2, 3), (0, 0, 1), (0, 2, 1))
  type_ = sc.TensorType((2, 3), pattern=pattern.canonicalize()[0])
  library = sc.CLibrary(
    "sparse",
    sources={"sparse.c": "void sparse_scale(const double* x, double* y, double* w) { (void)w; for(int i=0;i<3;++i) y[i] = (i+1)*x[i]; }"},
  )
  scale = sc.extern(library, "sparse_scale", inputs=(sc.arg("x", type_, layout="csc"),), outputs=sc.arg("y", type_, layout="csc"))

  @sc.function(sc.arg("x", type_, layout="csr"), outputs=sc.arg("y", type_, layout="csr"))
  def wrapped(x):
    return scale(x)

  value = sparse.coo_array(([2.0, 3.0, 5.0], (pattern.rows, pattern.cols)), shape=pattern.shape)
  result = wrapped(value)
  assert isinstance(result, sparse.sparray)
  np.testing.assert_allclose(sparse.coo_array(result).toarray(), [[2, 0, 9], [0, 10, 0]])


@pytest.mark.parametrize("declaration", ["input_hole", "output_hole", "workspace", "symbol", "link", "path"])
def test_extern_rejects_invalid_declarations(declaration):
  with pytest.raises((ValueError, TypeError)):
    if declaration == "input_hole":
      sc.extern(pair_library(), "foo", inputs=(sc.arg("x"),), outputs=sc.arg("y", 3))
    elif declaration == "output_hole":
      sc.extern(pair_library(), "foo", inputs=(sc.arg("x", 3),), outputs=sc.arg("y"))
    elif declaration == "workspace":
      sc.extern(pair_library(), "foo", inputs=(sc.arg("x", 3),), outputs=sc.arg("y", 3), workspace=-1)
    elif declaration == "symbol":
      sc.extern(pair_library(), "bad name", inputs=(sc.arg("x", 3),), outputs=sc.arg("y", 3))
    elif declaration == "link":
      pair_library(link=("-lm",))
    else:
      sc.CLibrary("escape", sources={"../escape.c": ""})


def test_extern_verifier_checks_concrete_types_and_symbol_conflicts():
  from scaly.ir.expr import Expr
  from scaly.ir.expr_spec import verify_expr

  x = sc.sym("x", 3)
  external = pair()
  good = external(x)[0]
  verify_expr(good)
  bad = Expr(sc.ExprOp.EXTERN, (sc.sym("wrong", 2),), good.type, attrs=good.attrs)
  with pytest.raises(sc.VerifyError, match="shape"):
    verify_expr(bad)
  other = pair(replace(external.library, compile_flags=("-DSCALE=2",)))

  @sc.function(sc.arg("x", 3))
  def conflicting(x):
    return external(x)[0] + other(x)[0]

  with pytest.raises(ValueError, match="conflicting external declarations"):
    render_c_module(conflicting)


def test_inactive_external_derivatives_and_zero_input_call():
  x = sc.sym("x", 3)
  y = pair()(sc.stop_gradient(x))[0]
  assert not np.any(sc.jvp(y, x, sc.const(np.ones(3))).value)
  assert not np.any(sc.vjp([y], [x], [sc.const(np.ones(3))])[0].value)
  constant = sc.extern(
    sc.CLibrary("constant", sources={"constant.c": "void external_constant(double* y, double* w) { (void)w; y[0]=2; }"}),
    "external_constant",
    inputs=(),
    outputs=sc.arg("y", ()),
  )
  np.testing.assert_allclose(constant(), 2)


@pytest.mark.parametrize("field", ["include_dirs", "compile_flags", "link"])
def test_library_rejects_a_bare_string_for_a_sequence_field(field):
  with pytest.raises(TypeError, match=field):
    pair_library(**{field: "-O3" if field == "compile_flags" else "inc"})


def test_custom_derivative_of_an_extern_asks_for_a_wrapping_function():
  with pytest.raises(TypeError, match="wrap.*Function"):
    sc.custom_derivative(pair(), sparsity=lambda of, wrt, shape: np.eye(*shape, dtype=bool))
