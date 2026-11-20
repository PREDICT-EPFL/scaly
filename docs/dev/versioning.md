# Versioning and releases

Scaly is in the `0.x.y` series. Patch releases preserve the public interfaces of that release line,
while minor releases may contain breaking changes. This policy covers the Python API, generated C, and
official solver plugins.

## Initial versions

The first non-prerelease version is `0.1.0`. There is no separate `0.0.x` series to signal
that the project is young.

The project stays in the `0.x.y` range while its public interfaces are still expected to change
substantially. `1.0.0` comes when we are ready to maintain the documented public API through
explicit deprecations with a migration period. Feature completeness by itself is not the criterion.

Within the pre-1.0 series, patch releases stay within a compatible release line. A new minor version
may open a new compatibility line and contain breaking changes.

## Core and plugin versions

The core package and each solver plugin are versioned independently. A release of one package does
not require publishing unchanged packages to keep version numbers aligned. Their initial versions
may happen to be equal, and a coordinated compatibility change may update several versions in the
same commit, but equality has no compatibility meaning.

Compatibility is enforced in two places:

- Each plugin declares the supported core range in its package dependencies, `scaly>=0.1.0,<0.2`
  for the `0.1.x` compatibility line. The exclusive upper bound also excludes every `0.2`
  pre-release.
- The solver registry checks the plugin protocol version at runtime. A breaking change to the plugin
  contract bumps that protocol version and requires coordinated releases of the affected official
  plugins.

## What a release may change

Generated code has three compatibility concerns.

### Exported C symbols

The generated interface includes the typed entry `<name>`, the input and
output types, the layout of `<name>_workspace_t`, and the pointer-array
entry `<name>_abi`. Each input and output buffer holds values of its own C
type. CasADi exports have a separate five-argument interface under
`<name>` with a byte workspace and size queries. The solver statistics
accessor `<name>_stats` and derived output names such as `f_spjac_y_x`
follow the same compatibility rules.

A patch release preserves these interfaces. A minor release may change them
and records the change in the release notes. Private C procedures may change
in any release.

A fully declared Function keeps its declared name. `@sc.function()` on a body without parameters
counts as fully declared, so its body runs at the decorator and its instance takes the plain name.
An instance of a function with open input shapes is named `<name>_<shapes>_t<nesting>`, where the shapes are the bound open shapes in
declaration order, written `3x4` for a matrix and `s` for a scalar, and the nesting is a six-digit
hexadecimal digest of the parameter tree. An input whose dtype is left open adds `f32`, `f64`,
`i32`, `i64` or `b` after its shape, if its shape is open, for float32, float64, int32, int64 or
bool. These tokens appear only when some open dtype binds a dtype other than float64, so a binding
of float64 throughout keeps the name it would have without them. A bare function includes all of its input shapes. The
name never contains `__`, which C++ reserves. The declaration alone decides whether an instance
name carries this suffix, never the order of calls, and a derivative given an explicit name
inherits the suffix of its source instance. Two instances of one function that would share a name
raise an error, as do two distinct graphs in one module whose names sanitize to the same C
identifier. Give independently declared functions distinct names.

### Sparsity tables

The `<prefix>_NNZ`, `_NROW` and `_NCOL` macros and the index tables in the
generated header take their prefix from the input or output name, a derived output's included, so
their names follow the symbol rule above. A sparse output's values follow its declared layout, and
the header publishes that layout's tables, as
[the generated interface](../how_it_works/generated_interface.md#sparse-outputs) describes. The
order rule is part of the interface. The tables' contents are not: a sharper derivative analysis
may drop an entry from a derivative's pattern, so a consumer reads the tables rather than copying
them. Whenever generated output changes,
`_JIT_CACHE_VERSION` in `src/scaly/codegen/jit.py` is bumped so a cached library from an older
version is never reused.

### Cached solver modules

Cached solver modules depend on the names and contents of readable files in their native include
and library directories. These come from installed plugins, `SCALY_SOLVER_INCLUDE_DIR`,
`SCALY_SOLVER_LIB_DIR`, and the parent directories of `SCALY_<NAME>_LIB` overrides. Scaly reads
each directory once per process, across all solver backends. A changed file selects a new cache
entry even when the generated C and plugin version stay the same. Pointing an override at a system
directory such as `/usr/lib` reads every readable file beneath it.

Keep native solver files fixed while Python runs, and stop Python before replacing them. Replacing
them in a live process can store a new build under the old cache key, leaving an incorrect entry
on disk after that process exits.

### Plugin protocol

`SOLVER_PLUGIN_PROTOCOL_VERSION` and `SCALY_SOLVER_STATS_VERSION` version the
contract between core and solver plugins. [Solver plugins](solver_plugins.md#versioning) lists what
each covers and its history. A protocol bump is a minor release of `scaly` and a coordinated release
of every official plugin, which raise their lower bound on `scaly` in the same commit. A patch
release never bumps either constant.

## Git tags

Every published package version gets exactly one package-qualified Git tag, for example:

```text
scaly-v0.1.0
scaly-piqp-v0.1.0
scaly-ipopt-v0.1.0
```

There is no additional repository-wide release tag. Several package tags may point to the same
commit when that commit releases several packages, and packages that are not released receive no
new tag.
