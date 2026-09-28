# Extensions

`scaly.ext` is what a package outside the compiler builds on: new expression ops and their rules,
lowerings, program passes, option namespaces, Functions whose body is C from elsewhere, output
adapters, and Functions built from expressions. A package checks the version it was written against
when it is imported:

```python
from scaly.ext import require_ext_api

require_ext_api(1, "my-package")
```

## Version

::: scaly.utils.ext_api.EXT_API_VERSION

::: scaly.utils.ext_api.require_ext_api

## Expression ops

An op registers once, from the module that provides its builder, with the rules and traits the
compiler asks for; `OpDef` lists each one's signature and default.

::: scaly.ir.expr.register_op

::: scaly.ir.expr.OpDef

::: scaly.ir.expr.define_rules

::: scaly.ir.expr.define_traits

::: scaly.ir.expr.has_trait

## Derivative and sparsity rules

A rule skips what a zero tangent or cotangent contributes, gives a ``NoAdjoint`` for an argument
it has no cotangent for, and builds a Jacobian pattern from its arguments' with the mask helpers.
``scaly.linalg.ops`` registers its factorizations, solves and ragged runs with nothing more.

::: scaly.ad.forward.is_zero_const

::: scaly.ad.forward.zeros_many

::: scaly.ad.forward.JVPManyUnsupported

::: scaly.ad.reverse.NoAdjoint

::: scaly.ad.sparsity.incidence

::: scaly.ad.sparsity.mask_compose

::: scaly.ad.sparsity.mask_or

::: scaly.ad.sparsity.empty_mask

## Lowering

::: scaly.passes.lowering.LowerCtx
    options:
      members:
        - buf_of
        - alloc_tmp
        - bind
        - output_buffer
        - new_private
        - new_alias
        - new_const_index
        - emit
        - fresh_id
        - fresh_name
        - copy_loop
        - blocked_sum
        - lane_loops

::: scaly.passes.lowering.PositionRanges

## Program passes

::: scaly.passes.program.insert_after

::: scaly.passes.program.insert_before

::: scaly.passes.program.pipeline

## Option namespaces

::: scaly.utils.options.register_option_namespace

::: scaly.utils.options.OptionNamespace

## Building Functions

::: scaly.function.model.Function.from_exprs

::: scaly.function.model.Function.lift

::: scaly.codegen.jit.load_library
