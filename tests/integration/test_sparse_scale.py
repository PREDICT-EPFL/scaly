"""Compiler work over represented entries of a matrix with a large logical shape."""

import json
import os
import subprocess
import sys
import threading
import time
import tracemalloc

import numpy as np
import pytest

import scaly as sc
from scaly.ad.forward import jvp
from scaly.ad.reverse import vjp
from scaly.ad.sparse import sparse_jacobian_colored
from scaly.ad.sparsity import jacobian_sparsity
from scaly.function.concrete import ConcreteFunction
from scaly.ir.expr import zeros_like
from scaly.ir.expr_spec import verify_expr
from scaly.passes.expr import simplify_cse_fixpoint
from scaly.passes.lowering import lower_function

_TIME_LIMIT = 180
_MEMORY_LIMIT = 2 * 1024**3


def _peak_rss() -> int:
  if sys.platform != "win32":
    import resource

    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(peak if sys.platform == "darwin" else peak * 1024)
  import ctypes
  from ctypes import wintypes

  class Counters(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("faults", wintypes.DWORD)] + [
      (name, ctypes.c_size_t) for name in ("peak_rss", "rss", "peak_paged", "paged", "peak_nonpaged", "nonpaged", "pagefile", "peak_pagefile")
    ]

  counters = Counters()
  counters.cb = ctypes.sizeof(counters)
  process = ctypes.windll.kernel32.GetCurrentProcess
  process.restype = wintypes.HANDLE
  query = ctypes.windll.psapi.GetProcessMemoryInfo
  query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
  assert query(process(), ctypes.byref(counters), counters.cb)
  return counters.peak_rss


def _workload(n: int, entries: int) -> None:
  x = sc.sym("scale_x")
  positions = np.arange(entries, dtype=np.int64)
  indices = (positions // 10) * n + positions % 10
  matrix = sc.scatter(x * sc.const(np.ones(entries)), indices, (n, n))
  assert matrix.support is not None and len(matrix.support) == entries
  zero = zeros_like(matrix)
  assert zero.support == frozenset()
  zero.structural_key()
  moved = matrix.T[::-1, ::2]
  pattern = jacobian_sparsity(moved, x)
  assert pattern.nnz == (entries + 19) // 20 * 10
  tangent = simplify_cse_fixpoint(jvp(matrix, x, sc.const(1.0)))
  assert tangent.shape == matrix.shape and tangent.support is not None and len(tangent.support) == entries
  jac = sparse_jacobian_colored(matrix, x)
  assert jac.sparsity.nnz == entries and jac.coloring_width == 1
  selected = matrix[-1:, :10].sum()
  adjoint = simplify_cse_fixpoint(vjp((selected,), (x,), (sc.const(1.0),))[0])
  assert adjoint.shape == ()
  vector = sc.scatter(sc.const([1.0]), [0], (n,))
  product = matrix @ vector
  assert jacobian_sparsity(product, x).nnz == entries // 10
  assert (sc.sym("scale_unknown", (n, n)) @ vector).support is None
  row = sc.sym("scale_row", (1, n))
  row_adjoint = vjp((row + zero,), (row,), (matrix,))[0]
  source = sc.sym("scale_slice", (n, n))
  slice_adjoint = vjp((source[:, ::2],), (source,), (moved,))[0]
  outputs = (matrix, tangent, moved, product, adjoint, zero, jac.values, row_adjoint, slice_adjoint)
  for output in outputs:
    verify_expr(output)
  names = ["matrix", "tangent", "moved", "product", "adjoint", "zero", "jac", "row_adj", "slice_adj"]
  fn = ConcreteFunction._from_exprs("sparse_scale", [x], outputs, ["x"], names)
  program = lower_function(fn)
  assert program


def test_sparse_matrix_compiler_resource_gate() -> None:
  result = subprocess.run(
    [sys.executable, __file__],
    capture_output=True,
    text=True,
    timeout=_TIME_LIMIT,
    env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"},
  )
  assert result.returncode == 0, result.stdout + result.stderr
  measurement = json.loads(result.stdout)
  assert measurement["seconds"] < _TIME_LIMIT
  assert measurement["peak_bytes"] < _MEMORY_LIMIT
  assert measurement["peak_rss"] < _MEMORY_LIMIT


@pytest.mark.parametrize("index", [(slice(None, None, -1), slice(1, None, 2)), (2, slice(None, None, -1))])
def test_sparse_assembly_movement_and_derivatives_match_numpy(index) -> None:
  x = sc.sym("scale_reference", 6)
  positions = np.array([0, 4, 7, 16, 22, 22])
  matrix = sc.scatter(x * x, positions, (5, 5))
  moved = matrix.T[index]
  coefficient = np.array([0.0, 2.0, 0.0, 3.0, 0.0])
  product = matrix @ sc.const(coefficient)
  seed = np.arange(1.0, 7.0)
  tangent = jvp(moved, x, sc.const(seed))
  cot = np.arange(1.0, moved.size + 1).reshape(moved.shape)
  adjoint = vjp((moved,), (x,), (sc.const(cot),))[0]
  row = sc.sym("reference_row", (1, 5))
  row_adjoint = vjp((row + sc.const(np.zeros((5, 5))),), (row,), (matrix,))[0]
  source = sc.sym("reference_slice", (5, 5))
  slice_adjoint = vjp((source.T[index],), (source,), (moved,))[0]
  outputs = [moved, product, tangent, adjoint, row_adjoint, slice_adjoint]
  fn = ConcreteFunction._from_exprs("scale_reference", [x], outputs, ["x"], ["moved", "product", "tangent", "adjoint", "row_adj", "slice_adj"])
  values = np.linspace(0.5, 1.5, 6)
  dense = np.zeros((5, 5))
  np.add.at(dense.reshape(-1), positions, values * values)
  delta = np.zeros((5, 5))
  np.add.at(delta.reshape(-1), positions, 2 * values * seed)
  lifted = np.zeros((5, 5))
  lifted[index] = cot
  projection = np.zeros((5, 5))
  projection[index] = dense.T[index]
  expected = (
    dense.T[index],
    dense @ coefficient,
    delta.T[index],
    2 * values * lifted.T.reshape(-1)[positions],
    dense.sum(axis=0)[None, :],
    projection.T,
  )
  for actual, reference in zip(fn(values), expected, strict=True):
    np.testing.assert_allclose(actual, reference, rtol=1e-13, atol=1e-13)
  basis = np.zeros((25, 6), dtype=bool)
  basis[positions, np.arange(6)] = True
  np.testing.assert_array_equal(jacobian_sparsity(moved, x).to_mask(), basis.reshape(5, 5, 6).transpose(1, 0, 2)[index].reshape(-1, 6))
  expected_product = np.stack([(basis[:, i].reshape(5, 5) @ (coefficient != 0)) for i in range(6)], axis=1)
  np.testing.assert_array_equal(jacobian_sparsity(product, x).to_mask(), expected_product)


if __name__ == "__main__":
  tracemalloc.start()

  def limit_memory() -> None:
    while True:
      if tracemalloc.get_traced_memory()[1] >= _MEMORY_LIMIT or _peak_rss() >= _MEMORY_LIMIT:
        os.write(2, b"sparse compiler gate exceeded its memory limit\n")
        os._exit(91)
      time.sleep(0.01)

  threading.Thread(target=limit_memory, daemon=True).start()
  started = time.monotonic()
  _workload(100_000, 1_000_000)
  print(json.dumps({"seconds": time.monotonic() - started, "peak_bytes": tracemalloc.get_traced_memory()[1], "peak_rss": _peak_rss()}))
