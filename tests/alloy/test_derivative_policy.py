"""Phase 11 first slice: SolverDescriptor's ``derivative_policy`` field.

Today only the framework is wired (enum + descriptor field + public export).
AD doesn't yet honor ``USER_PROVIDED`` / ``IMPLICIT`` — the policy framework
gives Phase 11 follow-ups a stable target without changing today's
non-differentiable semantics.
"""

from __future__ import annotations

import alloy as al
from alloy.solvers.solver_function import DerivativePolicy, SolverDescriptor


def test_derivative_policy_default_is_opaque() -> None:
  desc = SolverDescriptor(
    name="dummy",
    backend="piqp",
    n=2,
    n_eq=0,
    n_ineq=0,
    input_signature=(("x0", (2,)),),
    output_signature=(("x", (2,)),),
    param_names=(),
  )
  assert desc.derivative_policy == DerivativePolicy.OPAQUE
  assert desc.derivative_function is None


def test_derivative_policy_values_exposed() -> None:
  assert DerivativePolicy.OPAQUE.value == "opaque"
  assert DerivativePolicy.USER_PROVIDED.value == "user_provided"
  assert DerivativePolicy.IMPLICIT.value == "implicit"


def test_derivative_policy_is_part_of_public_alloy_surface() -> None:
  assert al.DerivativePolicy is DerivativePolicy
