# Differentiation

These functions construct derivatives directly from symbolic expressions. For the usual modelling
interface, including derivatives of a named `Function`, see [Building functions](functions.md).
The [derivatives guide](../guide/derivatives.md) explains gradients, Jacobians, and Hessians.

## Modes

A Jacobian-vector product, `jvp`, computes a directional derivative without constructing the full
Jacobian. A vector-Jacobian product, `vjp`, propagates output weights back to the inputs.
`jvp_many` handles several directions together.

::: scaly.ad.forward.jvp

::: scaly.ad.forward.jvp_many

::: scaly.ad.reverse.vjp

## Holding a value fixed in derivatives

::: scaly.stop_gradient

## Whole derivatives

::: scaly.ad.derivatives.jacobian

::: scaly.ad.derivatives.gradient

::: scaly.ad.derivatives.hessian

::: scaly.ad.derivatives.finite_difference

## Sparsity

A sparsity pattern identifies entries that may be nonzero. Coloring groups derivative directions
that can be evaluated together without mixing their results. See the
[sparsity guide](../guide/sparsity.md) for the storage format and examples.

::: scaly.ad.sparsity.jacobian_sparsity

::: scaly.ad.sparsity.column_coloring

::: scaly.ad.sparsity.star_coloring

::: scaly.ad.sparsity.color_groups

## Sparse derivatives

::: scaly.ad.sparse.SparseJacobian

::: scaly.ad.sparse.sparse_jacobian

::: scaly.ad.sparse.sparse_hessian

::: scaly.ad.sparse.sparse_jacobian_colored

::: scaly.ad.sparse.sparse_jacobian_reference
