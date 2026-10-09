# Environment variables

Scaly's environment settings select the compiler used for Python evaluation,
locate cached artifacts and visualization recordings, and configure native
solver discovery.
They apply to the Python process that reads them. Setting them before starting
that process avoids a mixture of old and new settings in already loaded
functions.

For example, this command reports the toolchain with Clang selected:

```bash
SCALY_CC=clang uv run scaly_toolchain
```

The report includes the resolved compiler path, cache location, and available
solver libraries. These settings do not change how your application's build
system compiles exported C.

## Compiler selection and optimization

| Variable | Default | Meaning |
| --- | --- | --- |
| `SCALY_CC` | `zig cc` from the `toolchain` extra, then `CC`, then `cc` from `PATH` | Compiler executable for numerical evaluation |
| `SCALY_CC_OPT` | `-O2` | One optimization flag passed to that compiler |
| `SCALY_VECTOR_LIBM` | Native host detection | `none` for scalar math calls or `glibc` for supported vector math functions |

An explicit `SCALY_CC` takes precedence over the `toolchain` extra and `CC`. Its value is an executable
name or path, such as `clang` or `/usr/bin/clang`, rather than a shell command
with additional flags. If the selected executable cannot be found, Scaly does
not silently try another compiler.

`SCALY_CC_OPT=-O3` selects a different optimization level. It is a single
compiler argument, not a space-separated list of flags. Scaly also adds
`-march=native`, or `-mcpu=native` on AArch64, and `-fno-math-errno`.

The Python compiler detects a fixed lane width for the native CPU. On supported x86-64 hosts
with glibc 2.35 or newer, it also selects the libmvec vector math library. `SCALY_VECTOR_LIBM=none`
forces scalar math calls. `SCALY_VECTOR_LIBM=glibc` requests libmvec explicitly and requires a
compatible target. Vector and scalar math implementations can produce different rounded results.
`scaly_toolchain` reports the detected native build recipe.

The build recipe, the compiler and its flags, and the CPU that `-march=native` resolves to
contribute to the cache key. Changing the optimization level or pointing `SCALY_CC` at another
compiler therefore produces a different artifact when the function is compiled again, and several
machines can share one cache directory. An already loaded function continues using its current
library. See [compilation and caching](codegen.md#compilation-and-caching-in-python).

## Cache directory

`SCALY_CACHE_DIR` holds generated source and compiled function libraries. It
defaults to `$XDG_CACHE_HOME/scaly/jit`, otherwise `~/.cache/scaly/jit`. A path
beginning with `~` expands to the user's home directory. Compiled artifacts can
be deleted and are rebuilt on a later compilation.

## Visualization recordings

`SCALY_VIZ_DIR` holds the recordings that [`scaly.viz`](../api/viz.md) writes
for marked functions. It defaults to `$XDG_CACHE_HOME/scaly/viz`, otherwise
`~/.cache/scaly/viz`.

## Native solver builds

`SCALY_BUILD_SOLVERS` controls the PIQP and IPOPT plugin build hooks when
installing from source. It does not rebuild libraries in an installed wheel.

| Value | Behavior |
| --- | --- |
| `auto` | Default. Build native libraries. Editable installs may skip unavailable toolchains, while wheel builds require them. |
| `skip`, `0`, or `false` | Skip native solver builds |
| `required`, `1`, or `true` | Require the native build even for an editable install |

`SCALY_SOLVER_CACHE` sets the directory where the build hooks keep finished
builds, shared by every source checkout on the machine. It defaults to
`$XDG_CACHE_HOME/scaly/solvers`, otherwise `~/.cache/scaly/solvers`.

Skipping a build does not provide a Python solver alternative. Solver calls
still need the native libraries. This setting is mainly relevant to source
checkouts and custom packaging. [Installation](installation.md) covers normal
package installation, and [Contributing](../dev/contributing.md#setup) lists
the source-build requirements.

## Custom solver-library locations

Installed plugins normally supply the required headers and libraries. The
following settings select custom builds or help diagnose library discovery:

| Variable | Meaning |
| --- | --- |
| `SCALY_SOLVER_INCLUDE_DIR` | Directory containing solver C headers |
| `SCALY_SOLVER_LIB_DIR` | Directory containing solver shared libraries |
| `SCALY_<NAME>_LIB` | Exact library path, such as `SCALY_PIQP_LIB` or `SCALY_IPOPT_LIB` |

The include and library overrides add locations before the plugin's locations.
A custom library still needs matching headers and its own native dependencies.
An exact library path alone does not supply those headers.
`uv run scaly_toolchain` shows the resolved locations and whether the libraries
can be loaded.
