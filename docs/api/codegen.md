# Code generation

## Rendering

::: scaly.codegen.aot.CModule

::: scaly.codegen.aot.render_c_module

::: scaly.codegen.aot.render_c_source

::: scaly.codegen.aot.render_c_api_header

::: scaly.codegen.aot.write_module

::: scaly.codegen.aot.workspace_size

## The target processor

::: scaly.ir.target.Target

::: scaly.ir.target.target

::: scaly.ir.target.set_target

::: scaly.ir.target.get_target

::: scaly.ir.target.host_target

## The ABI

::: scaly.codegen.abi.c_api_signature

::: scaly.codegen.abi.BufferType

## Output adapters

::: scaly.codegen.adapter.register_adapter

::: scaly.codegen.adapter.Adapter

::: scaly.codegen.adapter.HeaderSpec

::: scaly.codegen.adapter.EntryHook

::: scaly.codegen.adapter.get_adapter

::: scaly.codegen.adapter.available_adapters

The adapters themselves, the C++ header and the CasADi layer, are in [Export](export.md).

## Compiling and caching

::: scaly.codegen.jit.CompiledFunction

::: scaly.codegen.jit.invalidate_cache

::: scaly.codegen.jit.JitError

::: scaly.codegen.jit.JitUnavailable

## Toolchain

::: scaly.codegen.toolchain.find_c_compiler

::: scaly.codegen.toolchain.cache_root
