# Building functions

## The decorator

::: alloy.function.api.function

## Derivative requests

Typed requests passed to `Function.factory`.

::: alloy.function.factory.DerivSpec

::: alloy.function.factory.Jac

::: alloy.function.factory.Grad

::: alloy.function.factory.Hess

::: alloy.function.factory.SpJac

::: alloy.function.factory.SpHess

::: alloy.function.factory.Fwd

::: alloy.function.factory.Adj

## Named wrappers

The common derivative functions accept either an Expr and an Expr input, or a Function, an output
name, and an input name.

::: alloy.function.api.jacobian

::: alloy.function.api.gradient

::: alloy.function.api.hessian

::: alloy.function.api.sparse_jacobian

::: alloy.function.api.sparse_hessian

::: alloy.function.api.forward

::: alloy.function.api.adjoint

::: alloy.function.api.lagrangian_hessian

::: alloy.function.api.sparse_lagrangian_hessian
