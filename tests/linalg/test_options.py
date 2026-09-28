"""The ``linalg`` option namespace: validated like any other, read when a node is built, never part of
a derivative's name, and declared by importing ``scaly.linalg``, which ``sc.options`` does on first use."""

from __future__ import annotations

import dataclasses
import subprocess
import sys

import numpy as np
import pytest

import scaly as sc
from scaly.ad.forward import options_tag
from scaly.codegen import render_c_source
from scaly.linalg import LinalgOptions


@pytest.fixture(autouse=True)
def _restore_default():
  yield
  sc.set_options(linalg=dataclasses.asdict(LinalgOptions()))


def test_linalg_options_are_validated_at_the_call() -> None:
  for bad in (-1, 2.5, True, "8"):
    for name in ("dense_unroll", "sparse_unroll"):
      with pytest.raises(ValueError, match="non-negative integer"):
        sc.set_options(linalg={name: bad})
  for name in ("dense_unroll", "sparse_unroll"):
    with sc.options(linalg={name: 0}) as inside:
      assert getattr(inside.namespace("linalg"), name) == 0
  with pytest.raises(TypeError, match=r"unknown option linalg.'unroll'"):
    sc.set_options(linalg={"unroll": 3})
  assert sc.get_options().namespace("linalg") == LinalgOptions()


def test_linalg_options_do_not_name_derivative_helpers_apart() -> None:
  with sc.options(linalg=dict(dense_unroll=0, sparse_unroll=3)):
    assert options_tag() == ""


def test_the_namespace_loads_with_its_package_on_first_use() -> None:
  """``import scaly`` leaves ``scaly.linalg`` unloaded; ``sc.options(linalg=...)`` loads it."""
  code = (
    "import sys, scaly as sc\n"
    "assert 'scaly.linalg' not in sys.modules\n"
    "with sc.options(linalg=dict(dense_unroll=0)) as inside:\n"
    "  assert inside.namespace('linalg').dense_unroll == 0\n"
    "assert 'scaly.linalg' in sys.modules and sc.linalg.LinalgOptions().dense_unroll == 8\n"
  )
  proc = subprocess.run([sys.executable, "-c", code], check=False, capture_output=True, text=True)
  assert proc.returncode == 0, proc.stderr


def test_dense_unroll_is_decided_when_the_node_is_built() -> None:
  a, b = sc.sym("a", (4, 4)), sc.sym("b", 4)
  with sc.options(linalg=dict(dense_unroll=0)):
    looped = [sc.linalg.cholesky(a), sc.linalg.ldl(a), sc.linalg.solve_triangular(a, b)]
  unrolled = [sc.linalg.cholesky(a), sc.linalg.ldl(a), sc.linalg.solve_triangular(a, b)]
  assert all(not e.attrs["unroll"] for e in looped) and all(e.attrs["unroll"] for e in unrolled)
  fns = {tag: sc.Function.from_exprs(f"du_{tag}", [a, b], outs, ["a", "b"], ["l", "d", "x"]) for tag, outs in (("loop", looped), ("flat", unrolled))}
  for k, (loop_op, flat_op) in enumerate(zip(looped, unrolled, strict=True)):
    for tag, op, has_loop in (("loop", loop_op, True), ("flat", flat_op, False)):
      src = render_c_source(sc.Function.from_exprs(f"du_{tag}{k}", [a, b], [op], ["a", "b"], ["o"]))
      assert ("for (" in src.split(f"int du_{tag}{k}")[1]) == has_loop
  m = np.random.default_rng(3).standard_normal((4, 4))
  av, bv = m @ m.T + 4 * np.eye(4), np.arange(4.0)
  for got, ref in zip(fns["loop"]._flat_numerical_call(av, bv), fns["flat"]._flat_numerical_call(av, bv), strict=True):
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=1e-14)
  with sc.options(linalg=dict(dense_unroll=16)):
    assert sc.linalg.cholesky(sc.sym("big", (12, 12))).attrs["unroll"]
