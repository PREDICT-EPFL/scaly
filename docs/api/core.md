# Core

The expression graph and the types on it.

## Expressions

::: scaly.ir.expr.Expr
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^(lowering|with_lowering|scalar|block|opaque|sym|const)$"

::: scaly.ir.expr.Expr.sym
    options:
      show_source: false
      show_signature: false
      heading: "sym(name, shape=None, *, dtype=dtypes.float64, diff=True)"

::: scaly.ir.expr.Expr.const
    options:
      show_source: false
      show_signature: false
      heading: "const(value, *, dtype=None)"

::: scaly.ir.expr.ExprOp

::: scaly.ir.expr.OpDef

::: scaly.ir.expr.register_op

::: scaly.ir.expr.op_def

::: scaly.ir.expr.registered_ops

## Functions

::: scaly.function.model.Function

::: scaly.function.model.ConcreteFunction

::: scaly.function.model.NotConcrete


## Types

::: scaly.ir.types.TensorType

::: scaly.ir.types.DType

::: scaly.ir.types.dtypes

::: scaly.ir.types.ScalarType

::: scaly.ir.types.SparsityType

::: scaly.ir.types.DeviceSpec

::: scaly.ir.types.BackendSupport

::: scaly.ir.types.as_dtype

::: scaly.ir.types.backend_supports

## Builders

::: scaly.ir.expr.dot

::: scaly.ir.expr.sumsqr

::: scaly.ir.expr.norm_2

::: scaly.ir.expr.stack
    options:
      show_source: false

::: scaly.ir.expr.concat
    options:
      show_source: false

::: scaly.ir.expr.split

::: scaly.ir.expr.vec

::: scaly.ir.expr.gather
    options:
      show_source: false

::: scaly.ir.expr.scatter
    options:
      show_source: false

::: scaly.ir.expr.atan2

::: scaly.ir.expr.minimum

::: scaly.ir.expr.maximum

::: scaly.function.sugar.vmap
    options:
      show_source: false

## Verification and rewriting

::: scaly.ir.expr_spec.verify_expr

::: scaly.ir.spec.Spec

::: scaly.ir.spec.Rule

::: scaly.ir.spec.VerifyError

::: scaly.ir.match.Pattern

::: scaly.ir.match.PatternMatcher

::: scaly.ir.match.rewrite

::: scaly.passes.expr.simplify

::: scaly.passes.expr.cse

::: scaly.passes.expr.cse_many

## Text

::: scaly.ir.expr.format_expr

::: scaly.ir.text.render_expr_assembly

::: scaly.ir.text.render_program_assembly
