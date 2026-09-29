"""The integrator methods: every registered method builds the map its shorthand builds, options
pass through, and what a method cannot do it refuses, naming why."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy.integrate import solve_ivp

import scaly as sc
from scaly import integrators as si
from scaly.codegen import render_c_module

MU = 1.5


@sc.function(2, 1, output="xdot")
def van_der_pol(x: sc.Expr, u: sc.Expr) -> sc.Expr:
  return sc.stack([x[1], MU * (1 - x[0] * x[0]) * x[1] - x[0] + u[0]])


# Each registered method with its default options, and the shorthand call that builds the same map.
SHORTHANDS: dict[str, Any] = {
  **{
    name: (lambda name: lambda f, dt: si.explicit(f, name, dt=dt))(name)
    for name in ("euler", "heun", "midpoint", "ralston", "rk3", "ssprk3", "rk4", "rk38", "bs32", "dopri5", "tsit5")
  },
  **{
    name: (lambda name: lambda f, dt: si.implicit(f, name, dt=dt))(name)
    for name in ("backward_euler", "implicit_midpoint", "trapezoidal", "sdirk2", "sdirk3")
  },
  "gauss_legendre": lambda f, dt: si.implicit(f, "gauss_legendre", stages=2, dt=dt),
  "radau_iia": lambda f, dt: si.implicit(f, "radau_iia", stages=3, dt=dt),
  "lobatto_iiia": lambda f, dt: si.implicit(f, "lobatto_iiia", stages=3, dt=dt),
  "lobatto_iiic": lambda f, dt: si.implicit(f, "lobatto_iiic", stages=3, dt=dt),
  "adaptive": lambda f, dt: si.adaptive(f, dt=dt),
  "stormer_verlet": lambda f, dt: si.symplectic(f, "stormer_verlet", split=1, dt=dt),
  "symplectic_euler": lambda f, dt: si.symplectic(f, "symplectic_euler", split=1, dt=dt),
}


def test_every_method_is_registered() -> None:
  assert sorted(si.REGISTRY.installed()) == sorted(SHORTHANDS)
  assert isinstance(si.REGISTRY.auto(si.ODE(van_der_pol)), si.RK4)


@pytest.mark.parametrize("key", sorted(SHORTHANDS))
def test_a_method_builds_its_shorthands_map(key: str) -> None:
  cls = si.REGISTRY.get(key)
  method = cls(split=1) if key in ("stormer_verlet", "symplectic_euler") else cls()
  built = si.solver(si.ODE(van_der_pol, dt=0.1), method)
  assert built.name == f"van_der_pol_{method.label}"
  assert render_c_module(built).body == render_c_module(SHORTHANDS[key](van_der_pol, 0.1)).body


def test_options_pass_through() -> None:
  ode, free = si.ODE(van_der_pol, dt=0.2), si.ODE(van_der_pol)
  pairs = [
    (si.solver(ode, si.RK4(steps=6)), si.rk4(van_der_pol, dt=0.2, steps=6)),
    (
      si.solver(ode, si.RadauIIA(2, newton=sc.roots.Newton(tol=1e-12, rtol=1e-12, max_iter=20))),
      si.implicit(van_der_pol, "radau_iia", stages=2, dt=0.2, tol=1e-12, newton="full"),
    ),
    (
      si.solver(free, si.SDIRK3(steps=2, newton=sc.roots.Newton(tol=None, max_iter=5, simplified=True))),
      si.implicit(van_der_pol, "sdirk3", dt=None, steps=2, newton_iters=5),
    ),
    (si.solver(free, si.Adaptive(si.Tsit5(), rtol=1e-8, atol=1e-10)), si.adaptive(van_der_pol, "tsit5", rtol=1e-8, atol=1e-10)),
    (si.solver(ode, "tsit5"), si.explicit(van_der_pol, "tsit5", dt=0.2)),
  ]
  for built, shorthand in pairs:
    assert built.name == shorthand.name
    assert render_c_module(built).body == render_c_module(shorthand).body


def test_the_adaptive_tsit5_pair_follows_its_tolerance() -> None:
  step = si.solver(si.ODE(van_der_pol, name="vdp"), si.Adaptive(si.Tsit5(), rtol=1e-10, atol=1e-12))
  assert step.name == "vdp_tsit5_adaptive"
  x0, u = np.array([2.0, 0.0]), np.array([0.3])
  got = step(x0, u, np.array(3.0))
  exact = solve_ivp(lambda t, x: [x[1], MU * (1 - x[0] ** 2) * x[1] - x[0] + u[0]], (0, 3.0), x0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
  np.testing.assert_allclose(got, exact, rtol=1e-7)


def test_a_package_adds_a_method_by_subclassing() -> None:
  class Kutta3(si.ExplicitRK):
    name = "integrators.kutta3"
    table = si.TABLEAUS["rk3"]

  built = si.solver(si.ODE(van_der_pol, dt=0.1), Kutta3(steps=2))
  np.testing.assert_array_equal(
    built(np.array([1.0, 0.5]), np.array([0.2])), si.explicit(van_der_pol, "rk3", dt=0.1, steps=2)(np.array([1.0, 0.5]), np.array([0.2]))
  )


def test_what_a_method_refuses() -> None:
  ode = si.ODE(van_der_pol, dt=0.1)
  with pytest.raises(ValueError, match="needs split, the number of positions in the state"):
    si.solver(ode, si.StormerVerlet())
  with pytest.raises(ValueError, match="embedded explicit pair"):
    si.Adaptive(si.RK4())
  with pytest.raises(ValueError, match="needs at least 2 stages"):
    si.LobattoIIIA(1)
  with pytest.raises(ValueError, match="stage matrix's own solve"):
    si.RadauIIA(newton=sc.roots.Newton(linear="cholesky"))
  with pytest.raises(ValueError, match="steps must be a positive integer"):
    si.RK4(steps=0)
  with pytest.raises(TypeError, match="must be an sc.Function"):
    si.ODE(lambda x: x)  # ty: ignore[invalid-argument-type]
  with pytest.raises(ValueError, match="dt must be positive"):
    si.ODE(van_der_pol, dt=-1.0)


def test_shooting_takes_a_method_or_its_name() -> None:
  x, u, xn = np.array([1.5, -0.3]), np.array([0.4]), np.array([1.4, -0.2])
  for method in (si.RK4(steps=2), "rk4"):
    interval = si.MultipleShooting(method).interval(van_der_pol, dt=0.1)
    assert interval.fn.name == "van_der_pol_rk4_shooting"
  np.testing.assert_allclose(interval.fn(x, u, xn), si.rk4(van_der_pol, dt=0.1)(x, u) - xn, rtol=1e-15)
