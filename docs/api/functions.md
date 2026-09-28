# Building functions

Declare the shape and structure of a function with `L`, `G`, and `@function`. The derivative
wrappers, such as `gradient`, then create functions from that declaration. See the
[functions guide](../guide/functions.md) for examples and the
[derivatives guide](../guide/derivatives.md) for choosing a derivative.

## Declared trees

::: scaly.function.tree.Tree

::: scaly.function.tree.L

::: scaly.function.tree.G

## The decorator

::: scaly.function.api.function

## Derivative requests

Pass these request objects to `Function.factory` when you need several outputs or derivatives in
one function. For a single derivative, the named wrappers below are usually more convenient.

::: scaly.function.factory.DerivSpec

::: scaly.function.factory.Jac

::: scaly.function.factory.Grad

::: scaly.function.factory.Hess

::: scaly.function.factory.SpJac

::: scaly.function.factory.SpHess

::: scaly.function.factory.Fwd

::: scaly.function.factory.Adj

## Named wrappers

The common derivative functions accept either an Expr and an Expr input, or a Function, an output
name, and an input name.

::: scaly.function.api.jacobian

::: scaly.function.api.gradient

::: scaly.function.api.hessian

::: scaly.function.api.sparse_jacobian

::: scaly.function.api.sparse_hessian

::: scaly.function.api.forward

::: scaly.function.api.adjoint

::: scaly.function.api.lagrangian_hessian

::: scaly.function.api.sparse_lagrangian_hessian
