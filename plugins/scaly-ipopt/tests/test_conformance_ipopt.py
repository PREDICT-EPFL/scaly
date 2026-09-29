"""The QP suite (``scaly.testing.conformance.qp``) on ``opt.ipopt``, as a plugin runs it in its own
tests: the contract on every solvable problem, and no problem without a solution reported solved."""

from __future__ import annotations

import pytest

from scaly.testing.conformance import qp

pytestmark = pytest.mark.method("opt.ipopt")


@pytest.mark.parametrize("name", qp.SOLVABLE)
def test_a_solvable_qp(name: str) -> None:
  qp.check_solves("opt.ipopt", name)


@pytest.mark.parametrize("name", qp.UNSOLVABLE)
def test_a_qp_with_no_solution_is_not_reported_solved(name: str) -> None:
  qp.check_refuses("opt.ipopt", name)
