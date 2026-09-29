# Testing

What a test of scaly, or of a package that adds a method to it, needs. `scaly.testing` is not part
of the `scaly` namespace: import it by its full name.

## The `method` marker

Installed as a pytest plugin with scaly. `@pytest.mark.method("opt.piqp")` marks a test that needs
a method which may be missing, an external solver above all. The test is skipped where the method is
not installed or its library does not load, and under `SCALY_REQUIRE_METHODS=1` the run fails
instead.

::: scaly.testing.plugin.method_available

## Conformance suites

One suite per problem class: its reference problems and the contract a method meets on them. A
package that adds a method runs the suite in its own tests, as scaly's `tests/conformance/` does for
every method it ships.

::: scaly.testing.conformance.qp.check_solves

::: scaly.testing.conformance.qp.check_refuses

::: scaly.testing.conformance.ocp.check

::: scaly.testing.conformance.roots.check

## Reference problems

::: scaly.testing.qp.maros_meszaros

::: scaly.testing.qp.random_qp

::: scaly.testing.qp.mpc_qp

::: scaly.testing.helpers.build_qp

::: scaly.testing.helpers.build_nlp

## Hyper-dual numbers

::: scaly.testing.hyperdual.lagrangian_hessian_np
