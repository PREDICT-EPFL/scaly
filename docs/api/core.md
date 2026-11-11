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
      heading: "const(value, *, dtype=None, pattern=None)"

::: scaly.ir.expr.ExprOp
    options:
      show_source: false
      filters:
        - "!^(EXTERN|LOOP)$"

::: scaly.ir.expr.OpInfo
    options:
      show_source: false
      show_signature: false
      filters:
        - "!^_"

## Functions

<!-- The curated entries below exclude CLibrary and extern until interface review. -->

::: scaly.function.model.Function
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^with_device$"


`Function.instantiate` returns a concrete instance, which holds the resolved shapes and
expression graph for inspection and export.

::: scaly.function.concrete.ConcreteFunction
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^with_device$"
        - "!^role$"


## Types

::: scaly.ir.types.TensorType
    options:
      show_source: false
      show_signature: false
      filters:
        - "!^_"

::: scaly.ir.types.DType

::: scaly.ir.types.dtypes
    options:
      show_source: false
      filters:
        - "!^_"
        - "!^(float32|uint8)$"

::: scaly.ir.types.SparsityPattern
    options:
      show_source: false
      filters:
        - "!^_"

::: scaly.ir.types.CanonicalPattern
    options:
      show_source: false
      filters:
        - "!^_"

::: scaly.ir.types.as_dtype

## Builders

::: scaly.ir.einsum.einsum
    options:
      show_source: false


::: scaly.ir.expr.diag
    options:
      show_source: false

::: scaly.ir.expr.matmul
    options:
      show_source: false

::: scaly.ir.expr.dot
    options:
      show_source: false

::: scaly.ir.expr.sumsqr

::: scaly.ir.expr.reduce_max

::: scaly.ir.expr.reduce_min

::: scaly.ir.expr.argmax

::: scaly.ir.expr.argmin

::: scaly.ir.expr.norm_inf

::: scaly.ir.expr.norm_1

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

::: scaly.ir.expr.segment_sum
    options:
      show_source: false

::: scaly.ir.expr.atan2

::: scaly.ir.expr.minimum

::: scaly.ir.expr.maximum

::: scaly.ir.expr.where

::: scaly.ir.expr.cast

::: scaly.ir.expr.equal

::: scaly.ir.expr.not_equal

::: scaly.ir.expr.isfinite

::: scaly.ir.expr.print_
    options:
      show_source: false
      show_signature: false
      heading: "print(fmt, *values)"
      toc_label: "print"
      separate_signature: false

::: scaly.function.sugar.vmap
    options:
      show_source: false

## Sparse matrices

These builders convert between sparse and dense tensors. The
[sparse matrices guide](../guide/sparse_matrices.md) shows them in use.

::: scaly.ir.expr.pack

::: scaly.ir.expr.values

::: scaly.ir.expr.to_dense

::: scaly.ir.expr.from_dense

::: scaly.ir.expr.sparsify

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
