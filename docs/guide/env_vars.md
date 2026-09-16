# Environment variables

Every environment variable scaly reads is listed here and in `scaly.utils.env.scaly_env_vars()`.

| Variable | Effect |
| --- | --- |
| `SCALY_CC` | the C compiler to use (default: `cc` from `PATH`) |
| `SCALY_CC_OPT` | the optimization flag the JIT passes to that compiler (default: `-O2`). Set it when a benchmark compiles a baseline at a different level, so both sides of a comparison get the same one. The JIT always adds `-march=native` (`-mcpu=native` on AArch64) and `-fno-math-errno`, since it compiles for the machine it runs on. The whole flag set is part of the cache key, so switching flags does not reuse an artifact |
| `SCALY_CACHE_DIR` | where the just-in-time (JIT) path caches compiled artifacts, hence the `jit` in the default, `$XDG_CACHE_HOME/scaly/jit` or `~/.cache/scaly/jit`. Deleting it is always safe; see [Code generation](codegen.md) |
| `SCALY_BUILD_SOLVERS` | `auto`, `skip` or `required`, how `uv sync` treats the vendored solver build |
| `SCALY_VIZ_DIR` | where the visualizer writes recordings |
| `SCALY_STRICT_JVP_MANY` | raise instead of falling back when multi-seed forward AD meets an unsupported operation |
| `SCALY_SOLVER_INCLUDE_DIR`, `SCALY_SOLVER_LIB_DIR`, `SCALY_PIQP_LIB`, `SCALY_IPOPT_LIB` | point solver discovery at explicit paths, mainly for debugging a packaging layout |
| `SCALY_SOLVER_SYSTEM_FALLBACK` | opt in to system library discovery when debugging a non-vendored install; the supported path is the vendored one |
