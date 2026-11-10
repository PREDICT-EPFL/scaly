"""Refuse typed sparse headers while keeping native sparse execution available."""

import numpy as np
import pytest
from scipy import sparse

import scaly as sc
from scaly.codegen.aot import render_c_api_header, render_c_module, write_module
from scaly.codegen.cpp import render_cpp_header
from scaly.ir.types import CanonicalPattern


PATTERN = CanonicalPattern((300, 400), (0, 10, 299), (2, 20, 399))


@pytest.mark.parametrize("lang", ["c", "cpp"])
@pytest.mark.parametrize("sparse_side", ["input", "output"])
def test_sparse_headers_refuse_instead_of_exposing_logical_buffers(lang, sparse_side, tmp_path):
  leaf = sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN), layout="csr")
  function = (
    sc.function(leaf, name="sparse_input_header")(lambda x: sc.values(x))
    if sparse_side == "input"
    else sc.function(sc.arg("x", PATTERN.nnz), name="sparse_output_header")(lambda x: sc.pack(x, PATTERN))
  )
  module = render_c_module(function, lang=lang)
  assert function.name in module.body
  with pytest.raises(NotImplementedError, match="stored-pattern leaves"):
    _ = module.header
  with pytest.raises(NotImplementedError, match="stored-pattern leaves"):
    render_c_api_header(function, lang=lang)
  with pytest.raises(NotImplementedError, match="stored-pattern leaves"):
    write_module(function, tmp_path, lang=lang)
  assert not list(tmp_path.iterdir())
  if lang == "cpp":
    instance = function.instantiate()
    with pytest.raises(NotImplementedError, match="stored-pattern leaves"):
      render_cpp_header(instance, (), 0, casadi=False, sparsities=instance.output_sparsities)
  result = function(np.arange(3.0))
  if sparse_side == "input":
    np.testing.assert_array_equal(result, np.arange(3.0))
  else:
    assert isinstance(result, sparse.csc_array)
    assert result.shape == PATTERN.shape and result.nnz == PATTERN.nnz
    np.testing.assert_array_equal(result.data, np.arange(3.0))


@pytest.mark.parametrize("lang", ["c", "cpp"])
def test_dense_headers_allow_internal_sparse_calls(lang):
  inner = sc.function(sc.arg("x", sc.TensorType(PATTERN.shape, pattern=PATTERN)), name="sparse_header_inner")(lambda x: x.sin())
  outer = sc.function(sc.arg("x", PATTERN.nnz), name="dense_header_outer")(lambda x: sc.values(inner(sc.pack(x, PATTERN))))
  header = render_c_api_header(outer, lang=lang)
  assert outer.name in header
  np.testing.assert_allclose(outer(np.arange(3.0)), np.sin(np.arange(3.0)))
