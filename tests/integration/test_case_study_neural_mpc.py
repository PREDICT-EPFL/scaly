"""The neural MPC case study's acados drop-in, on a tiny network, as acados builds and calls it.

`examples/case_studies/neural_mpc` replaces acados' generated `expl_vde_forw` by Scaly's: rendered with
the `casadi` adapter, compiled with `casadi_int` defined as `int` (acados' width), with a zero-length parameter
input (acados' empty `p`) and the sensitivity matrix passed as a flat column-major vector. This checks
that path through the C entry point and the CasADi query functions, against NumPy, so the study is not
the only thing exercising it.
"""

from __future__ import annotations

import ctypes
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import scaly as sc
from scaly.codegen import write_module

NX, NU = 2, 1
RNG = np.random.default_rng(3)
W1, B1 = RNG.standard_normal((3, NX)), RNG.standard_normal(3)
W2, B2 = RNG.standard_normal((NX, 3)), RNG.standard_normal(NX)


def f_np(x, u):
  return np.array([x[1], u[0]]) + W2 @ np.tanh(W1 @ x + B1) + B2


def jac_np(x, u):
  h = np.tanh(W1 @ x + B1)
  jx = np.array([[0.0, 1.0], [0.0, 0.0]]) + W2 @ ((1 - h * h)[:, None] * W1)
  return jx, np.array([[0.0], [1.0]])


@sc.function(
  sc.L("x", NX), sc.L("Sx", NX * NX), sc.L("Sp", NX * NU), sc.L("u", NU), sc.L("p", 0), output=sc.G("f", "Sx_dot", "Sp_dot"), name="tiny_vde_forw"
)
def vde_forw(x, Sx, Sp, u, p):
  f = sc.concat([x[1:2], u]) + sc.const(W2) @ (sc.const(W1) @ x + sc.const(B1)).tanh() + sc.const(B2)
  jx, ju = sc.jacobian(f, x), sc.jacobian(f, u)
  sx = Sx.reshape((NX, NX)).T
  return f, (jx @ sx).T.reshape((NX * NX,)), jx @ Sp + ju.reshape((NX * NU,))


@pytest.mark.skipif(shutil.which("cc") is None, reason="needs a C compiler on PATH")
def test_vde_forw_through_the_casadi_layer_as_acados_builds_it(tmp_path: Path) -> None:
  module = write_module(vde_forw, tmp_path, adapters=("casadi",))
  lib = tmp_path / "vde.so"
  subprocess.run(["cc", "-O2", "-Dcasadi_int=int", "-shared", "-fPIC", "-o", str(lib), str(tmp_path / module.source_name)], check=True)
  so = ctypes.CDLL(str(lib))
  ip = ctypes.POINTER(ctypes.c_int)
  so.tiny_vde_forw_sparsity_in.restype = ip
  so.tiny_vde_forw_sparsity_out.restype = ip
  assert so.tiny_vde_forw_n_in() == 5 and so.tiny_vde_forw_n_out() == 3
  assert [so.tiny_vde_forw_sparsity_in(i)[0] for i in range(5)] == [NX, NX * NX, NX * NU, NU, 0]
  assert so.tiny_vde_forw_sparsity_in(4)[1] == 1
  sizes = [ctypes.c_int() for _ in range(4)]
  so.tiny_vde_forw_work(*(ctypes.byref(s) for s in sizes))
  sz_w = max(sizes[3].value, 1)

  x, u = np.array([0.3, -0.7]), np.array([0.4])
  sx = RNG.standard_normal((NX, NX))
  sp = RNG.standard_normal((NX, NU))
  inputs = [x, sx.T.reshape(-1), sp.reshape(-1), u, np.zeros(1)]  # Sx column-major, as acados passes it
  dp = ctypes.POINTER(ctypes.c_double)
  arg = (dp * 5)(*(a.ctypes.data_as(dp) for a in inputs))
  outs = [np.zeros(NX), np.zeros(NX * NX), np.zeros(NX * NU)]
  res = (dp * 3)(*(o.ctypes.data_as(dp) for o in outs))
  work = np.zeros(sz_w)
  assert so.tiny_vde_forw(arg, res, None, work.ctypes.data_as(dp), 0) == 0

  jx, ju = jac_np(x, u)
  np.testing.assert_allclose(outs[0], f_np(x, u), rtol=1e-14, atol=1e-14)
  np.testing.assert_allclose(outs[1].reshape(NX, NX).T, jx @ sx, rtol=1e-13, atol=1e-13)
  np.testing.assert_allclose(outs[2], (jx @ sp + ju).reshape(-1), rtol=1e-13, atol=1e-13)
