"""The QP suite (``scaly.testing.conformance.qp``) over every installed opt method: each is listed
here, at the options it conforms with, or with the reason it is held to another suite; a method
installed but not listed fails."""

from __future__ import annotations

from typing import Any

import pytest

import scaly as sc
from scaly.testing.conformance import qp

METHODS: dict[str, Any] = {
  "ipm": "opt.ipm",
  "piqp": "opt.piqp",
  "ipopt": "opt.ipopt",
  "sqp": None,  # an NLP method: its QP subproblems stop at 1e-6, leaving stationarity at 1.5e-6, and at tol=1e-8 HS35 ends in NUMERICS
}
CONFORMING = [pytest.param(name, marks=pytest.mark.method(f"opt.{name}"), id=f"opt.{name}") for name, method in METHODS.items() if method is not None]


def test_every_installed_method_is_listed() -> None:
  """A method installed here and missing from the table fails; a plugin listed but not installed is fine."""
  assert set(sc.opt.REGISTRY.installed()) <= set(METHODS), sorted(set(sc.opt.REGISTRY.installed()) - set(METHODS))
  assert "ipm" in sc.opt.REGISTRY.installed()


@pytest.mark.parametrize("method", CONFORMING)
@pytest.mark.parametrize("name", qp.SOLVABLE)
def test_a_solvable_qp(name: str, method: str) -> None:
  qp.check_solves(METHODS[method], name)


@pytest.mark.parametrize("method", CONFORMING)
@pytest.mark.parametrize("name", qp.UNSOLVABLE)
def test_a_qp_with_no_solution_is_not_reported_solved(name: str, method: str) -> None:
  qp.check_refuses(METHODS[method], name)
