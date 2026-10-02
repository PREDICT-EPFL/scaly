# Core

`Expr` represents a calculation whose inputs are not yet known. `Function` gives that calculation
named inputs and outputs so that you can evaluate it, differentiate it, or generate C code.
The [functions guide](../guide/functions.md) shows how to use both.

The first sections cover modelling. Verification, rewriting, and text rendering are lower-level
interfaces for inspecting or extending the compiler.

## Expressions

::: scaly.ir.expr.Expr
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^(lowering|with_lowering|opaque|sym|const)$"

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

::: scaly.ir.expr.OpInfo

## Functions

::: scaly.function.model.Function
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^with_device$"


Concrete instances are returned by `Function.instantiate`; their resolved graph metadata is
available for inspection and export.

::: scaly.function.concrete.ConcreteFunction
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^with_device$"


## Types

::: scaly.ir.types.TensorType

::: scaly.ir.types.DType

::: scaly.ir.types.dtypes
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^float32$"

::: scaly.ir.types.SparsityPattern

::: scaly.ir.types.as_dtype

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

## PyTorch weights

::: scaly.utils.torch_state_dict.load_torch_state_dict

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
