# Versioning and releases

Scaly is in the `0.x.y` series. Patch releases preserve the public interfaces of that release line,
while minor releases may contain breaking changes. This policy covers the Python API, generated C, and
official solver plugins.

## Initial versions

The first non-prerelease version is planned as `0.1.0`. Installable wheels published before then are Python
pre-releases of that version: `0.1.0a1`, `0.1.0a2`, and so on, with a release candidate such as
`0.1.0rc1` while that release is being prepared. There is no separate `0.0.x` series to signal
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

- Each plugin declares the supported core range in its package dependencies, `scaly>=0.1.0a1,<0.2`
  for the `0.1.x` compatibility line. Naming a pre-release on the lower bound lets installers pick
  `0.1.0b1` or `0.1.0rc1` without `--pre`. The exclusive upper bound also excludes every `0.2`
  pre-release.
- The solver registry checks the plugin protocol version at runtime. A breaking change to the plugin
  contract bumps that protocol version and requires coordinated releases of the affected official
  plugins.

## What a release may change

Generated code has three compatibility concerns.

### Exported C symbols

The stable interface is the universal pointer signature of the exported entry
`<name>` and, for solver modules, the `<name>_stats` accessor, together with the derived-output
names `{kind}_{of}_{wrt}` such as `f_spjac_y_x`. A patch release changes none of these. A minor
release may rename, reorder or remove them, and the release notes list the change. The typed C
structs, the C++ `Buffer` aliases and the CasADi query functions are sugar over the pointer
signature and follow the same rule. The `static` `_raw` bodies
are internal and may change in any release.

A fully shaped Function keeps its declared name. Instances with shape holes use
`<name>_<shapes>_t<nesting>`: `3x4` for a matrix, `s` for a scalar, and a six-digit
hexadecimal digest of the parameter nesting. Only open input shapes contribute shape tokens;
a bare function includes all input shapes. An empty shape-token list omits its separator.
Mangling never introduces `__`. The declaration determines whether to mangle, independently
of call history, and explicit derivative names inherit the source specialization suffix.
A specialization-name collision raises. Distinct reachable graphs that sanitize to the same C
identifier also raise; give independently declared functions distinct names.

### Sparsity tables

The `<prefix>_NNZ`, `_NROW` and `_NCOL` macros and the index tables in the
generated header take their prefix from the derived-output name, so their names follow the symbol
rule above. No release promises their contents. The coordinate order is an artifact of lowering,
and a coloring or `vmap` change may reorder it. A consumer stays correct by reading values through
the `<prefix>_csr_val_perm` and `<prefix>_csc_val_perm` tables, as
[the generated interface](../how_it_works/generated_interface.md#sparse-outputs) describes. Whenever generated output changes,
`_JIT_CACHE_VERSION` in `src/scaly/codegen/jit.py` is bumped so a cached library from an older
version is never reused.

### Plugin protocol

`SOLVER_PLUGIN_PROTOCOL_VERSION` and `SCALY_SOLVER_STATS_VERSION` version the
contract between core and solver plugins. [Solver plugins](solver_plugins.md#versioning) lists what
each covers and its history. A protocol bump is a minor release of `scaly` and a coordinated release
of every official plugin, which raise their lower bound on `scaly` in the same commit. A patch
release never bumps either constant.

## Git tags

The repository has no tags yet. From the first release on, every published package version gets
exactly one package-qualified Git tag, for example:

```text
scaly-v0.1.0
scaly-piqp-v0.1.0
scaly-ipopt-v0.1.0
```

There is no additional repository-wide release tag. Several package tags may point to the same
commit when that commit releases several packages, and packages that are not released receive no
new tag.
