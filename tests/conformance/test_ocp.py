"""The DiscreteOCP suite (``scaly.testing.conformance.ocp``) over every installed OCP method: on each
reference problem it takes, the direct method's optimum to its own tolerance, and a problem it
refuses is refused by ``sc.ocp.solver`` too. A method installed but not listed fails: a method is
supported once it passes here."""

from __future__ import annotations

from typing import Any

import pytest

import scaly as sc
from scaly import ocp
from scaly.testing.conformance import ocp as suite

# The method with the options its tolerance below needs, and that tolerance on the controls.
METHODS: dict[str, tuple[Any, float]] = {
  "direct": (ocp.Direct(sc.opt.IPOPT(options={"tol": 1e-10})), 1e-6),
  "ilqr": (ocp.ILQR(mu=1e-8, mu_min=1e-12), 1e-6),
  "tinyadmm": (ocp.TinyADMM(rho=2.0, abs_pri_tol=1e-11, abs_dua_tol=1e-11, max_iter=50000), 1e-6),
  "altro": (ocp.ALTRO(projected_newton=True), 1e-4),
  "scvx": (ocp.SCvx(), 1e-4),
}


def test_every_installed_method_is_listed() -> None:
  assert sorted(ocp.REGISTRY.installed()) == sorted(METHODS)


@pytest.mark.method("opt.ipopt")
@pytest.mark.parametrize("problem_name", list(suite.problems()))
@pytest.mark.parametrize("method_name", list(METHODS))
def test_the_method_reaches_the_direct_optimum(method_name: str, problem_name: str) -> None:
  method, tolerance = METHODS[method_name]
  problem, _ = suite.problems()[problem_name]
  support = method.supports(problem)
  if not support:
    with pytest.raises(ValueError, match="cannot solve this problem"):
      ocp.solver(problem, method)
    pytest.skip(f"{method_name} refuses {problem_name}: {', '.join(support.reasons)}")
  suite.check(method, problem_name, tolerance, label=method_name)
