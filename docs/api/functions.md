# Building functions

## Declared trees

::: scaly.function.tree.Tree

::: scaly.function.tree.L

::: scaly.function.tree.G

## The decorator

::: scaly.function.api.function

## Derivative requests

Typed requests passed to `Function.factory`.

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

## Functions with an extern body

A Function whose C is written by someone other than the compiler, a solver for instance, reaches
lowering, code generation and the JIT only through this protocol.

::: scaly.function.extern.extern_function

::: scaly.function.extern.ExternCallee

::: scaly.function.extern.ExternSource

::: scaly.function.extern.ExternRenderCtx

::: scaly.function.extern.BuildRequirements

::: scaly.function.extern.ExternState

::: scaly.function.extern.extern_functions
