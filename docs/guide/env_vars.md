# Environment variables

There are multiple environment variables you can use to configure alloy's behavior, all referenced in ``alloy.utils.env.alloy_env_vars()`.

| Variable | Effect |
| --- | --- |
| `ALLOY_CC` | the C compiler to use (default: `cc` from `PATH`) |
| `ALLOY_CC_OPT` | the optimization flag the JIT passes to that compiler (default: `-O2`). Set it when a benchmark compiles a baseline at a different level, so both sides of a comparison get the same one. The JIT always adds `-march=native` (`-mcpu=native` on AArch64) and `-fno-math-errno`, since it compiles for the machine it runs on; the whole flag set is part of the cache key, so switching flags does not reuse an artifact |
| `ALLOY_CACHE_DIR` | where the just-in-time path (JIT) caches compiled artifacts — hence the `jit` in the default, `$XDG_CACHE_HOME/alloy/jit` or `~/.cache/alloy/jit`. Deleting it is always safe; see [Generating C](codegen.md) |
| `ALLOY_BUILD_SOLVERS` | `auto`, `skip` or `required` — how `uv sync` treats the vendored solver build |
| `ALLOY_VIZ_DIR` | where the visualizer writes recordings |
| `ALLOY_STRICT_JVP_MANY` | raise instead of falling back when multi-seed forward AD meets an unsupported operation |
| `ALLOY_SOLVER_INCLUDE_DIR`, `ALLOY_SOLVER_LIB_DIR`, `ALLOY_PIQP_LIB`, `ALLOY_IPOPT_LIB` | point solver discovery at explicit paths, mainly for debugging a packaging layout |
| `ALLOY_SOLVER_SYSTEM_FALLBACK` | opt in to system library discovery when debugging a non-vendored install; the supported path remains the vendored one |
