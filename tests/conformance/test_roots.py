"""The Root and LeastSquares suite (``scaly.testing.conformance.roots``) over every installed roots
method, on each problem it takes; a problem it refuses is refused by ``sc.roots.solver`` too. A
method installed but not listed fails."""

from __future__ import annotations

from typing import Any

import pytest

import scaly as sc
from scaly.testing.conformance import roots as suite

METHODS: dict[str, Any] = {
  "newton": sc.roots.Newton(),
  "newton_bisection": sc.roots.NewtonBisection(),
  "gauss_newton": sc.roots.GaussNewton(),
  "levenberg_marquardt": sc.roots.LevenbergMarquardt(),
}


def test_every_installed_method_is_listed() -> None:
  assert sorted(sc.roots.REGISTRY.installed()) == sorted(METHODS)


@pytest.mark.parametrize("problem_name", list(suite.problems()))
@pytest.mark.parametrize("method_name", list(METHODS))
def test_the_method_solves_what_it_takes(method_name: str, problem_name: str) -> None:
  method = METHODS[method_name]
  problem, _, _ = suite.problems()[problem_name]
  support = method.supports(problem)
  if not support:
    with pytest.raises((ValueError, TypeError)):
      sc.roots.solver(problem, method)
    pytest.skip(f"{method_name} refuses {problem_name}: {', '.join(support.reasons)}")
  suite.check(method, problem_name, label=method_name)
