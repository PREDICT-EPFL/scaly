# Core

The expression graph and the types on it.

## Expressions

::: alloy.ir.expr.Expr
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^(lowering|with_lowering|scalar|block|opaque|sym|const)$"

::: alloy.ir.expr.Expr.sym
    options:
      show_source: false
      show_signature: false
      heading: "sym(name, shape=None, *, dtype=dtypes.float64, diff=True)"

::: alloy.ir.expr.Expr.const
    options:
      show_source: false
      show_signature: false
      heading: "const(value, *, dtype=None)"

::: alloy.ir.expr.ExprOp

::: alloy.ir.expr.OpInfo

## Functions

::: alloy.function.model.Function


## Types

::: alloy.ir.types.TensorType

::: alloy.ir.types.DType

::: alloy.ir.types.dtypes

::: alloy.ir.types.ScalarType

::: alloy.ir.types.SparsityType

::: alloy.ir.types.DeviceSpec

::: alloy.ir.types.BackendSupport

::: alloy.ir.types.as_dtype

::: alloy.ir.types.backend_supports

## Builders

::: alloy.ir.expr.dot

::: alloy.ir.expr.sumsqr

::: alloy.ir.expr.norm_2

::: alloy.ir.expr.stack
    options:
      show_source: false

::: alloy.ir.expr.concat
    options:
      show_source: false

::: alloy.ir.expr.split

::: alloy.ir.expr.vec

::: alloy.ir.expr.gather
    options:
      show_source: false

::: alloy.ir.expr.scatter
    options:
      show_source: false

::: alloy.ir.expr.atan2

::: alloy.ir.expr.minimum

::: alloy.ir.expr.maximum

::: alloy.function.sugar.vmap
    options:
      show_source: false

## Verification and rewriting

::: alloy.ir.expr_spec.verify_expr

::: alloy.ir.spec.Spec

::: alloy.ir.spec.Rule

::: alloy.ir.spec.VerifyError

::: alloy.ir.match.Pattern

::: alloy.ir.match.PatternMatcher

::: alloy.ir.match.rewrite

::: alloy.passes.expr.simplify

::: alloy.passes.expr.cse

::: alloy.passes.expr.cse_many

## Text

::: alloy.ir.expr.format_expr

::: alloy.ir.text.render_expr_assembly

::: alloy.ir.text.render_program_assembly
