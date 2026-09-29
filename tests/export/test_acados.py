"""The acados drop-in: the explicit-integrator functions of a model with two controls and a parameter
against NumPy (the forward sensitivities column-major, as acados passes them), the one-control form,
and the drop-in's files: the header inlined, ``casadi_int`` defined, the C accepted by the compiler."""

from __future__ import annotations

import shutil
import subprocess

import numpy as np
import pytest

import scaly as sc
from scaly import export

NX, NU, NP = 3, 2, 1


def xdot(x, u, p):
  return sc.stack([x[1] * u[0], -p[0] * x[0].sin() + u[1], x[0] * x[2] - u[0] * u[1]])


def _numpy(x, u, p):
  f = np.array([x[1] * u[0], -p[0] * np.sin(x[0]) + u[1], x[0] * x[2] - u[0] * u[1]])
  jx = np.array([[0.0, u[0], 0.0], [-p[0] * np.cos(x[0]), 0.0, 0.0], [x[2], 0.0, x[0]]])
  ju = np.array([[x[1], 0.0], [0.0, 1.0], [-u[1], -u[0]]])
  return f, jx, ju


RNG = np.random.default_rng(5)
X, U, P = RNG.standard_normal(NX), RNG.standard_normal(NU), np.array([1.7])


def test_the_three_functions_against_numpy_with_two_controls() -> None:
  fns = export.acados_functions(xdot, NX, NU, name="acados_two", n_p=NP)
  assert sorted(fns) == ["acados_two_expl_ode_fun", "acados_two_expl_vde_adj", "acados_two_expl_vde_forw"]
  f, jx, ju = _numpy(X, U, P)
  np.testing.assert_allclose(fns["acados_two_expl_ode_fun"](X, U, P), f, rtol=1e-14)
  sx, sp = RNG.standard_normal((NX, NX)), RNG.standard_normal((NX, NU))
  f_out, sx_dot, sp_dot = fns["acados_two_expl_vde_forw"](X, sx.ravel(order="F"), sp.ravel(order="F"), U, P)
  np.testing.assert_allclose(f_out, f, rtol=1e-14)
  np.testing.assert_allclose(sx_dot, (jx @ sx).ravel(order="F"), rtol=1e-13, atol=1e-14)
  np.testing.assert_allclose(sp_dot, (jx @ sp + ju).ravel(order="F"), rtol=1e-13, atol=1e-14)
  lam = RNG.standard_normal(NX)
  np.testing.assert_allclose(fns["acados_two_expl_vde_adj"](X, lam, U, P), np.r_[jx.T @ lam, ju.T @ lam], rtol=1e-13, atol=1e-14)


def test_one_control_keeps_the_column_form() -> None:
  one = export.acados_functions(lambda x, u, p: xdot(x, sc.concat([u, u]), p), NX, 1, name="acados_one", n_p=NP)
  _, jx, ju2 = _numpy(X, np.r_[U[0], U[0]], P)
  ju = ju2.sum(axis=1, keepdims=True)
  sx, sp = RNG.standard_normal((NX, NX)), RNG.standard_normal((NX, 1))
  _, _, sp_dot = one["acados_one_expl_vde_forw"](X, sx.ravel(order="F"), sp.ravel(), U[:1], P)
  np.testing.assert_allclose(sp_dot, (jx @ sp + ju).ravel(), rtol=1e-13, atol=1e-14)


def test_the_dropin_writes_self_contained_sources(tmp_path) -> None:
  fns = export.acados_functions(xdot, NX, NU, name="acados_drop", n_p=NP)
  written = export.install_dropin(fns, tmp_path / "acados_drop_model")
  assert sorted(p.name for p in written) == sorted(f"{k}.c" for k in fns)
  for path in written:
    text = path.read_text()
    assert text.startswith("#define casadi_int int\n") and '#include "acados_drop' not in text
  compiler = shutil.which("cc")
  if compiler is None:
    pytest.skip("no C compiler")
  for path in written:
    subprocess.run([compiler, "-fsyntax-only", "-std=c11", str(path)], check=True)
