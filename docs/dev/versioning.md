# Versioning and releases

The versioning and release policy for the distributions built from this repository and the official
solver plugins. `scaly` 0.1.0a1 was published as one distribution; these rules apply from the first
release that ships the split.

## Initial versions

The first paper-worthy release will be `0.1.0`. Installable wheels published before then are Python
pre-releases of that version: `0.1.0a1`, `0.1.0a2`, and so on, with a release candidate such as
`0.1.0rc1` while the paper release is being frozen. There is no separate `0.0.x` series to signal
that the project is young.

The project stays in the `0.x.y` range while its public interfaces are still expected to change
substantially. `1.0.0` comes when we are ready to maintain the documented public API through
explicit deprecations with a migration period. Feature completeness by itself is not the criterion.

Within the pre-1.0 series, patch releases stay within a compatible release line. A new minor version
may open a new compatibility line and contain breaking changes.

## Lockstep distributions and independent plugins

The repository builds seven first-party distributions from one source tree: `scaly-core`,
`scaly-numerics`, `scaly-control`, `scaly-tools`, `scaly-experimental`, `scaly-testing` and the
`scaly` metapackage ([The codebase](codebase.md) lists what each ships). They release in lockstep.
`distributions.toml` holds one `version`; every lockstep distribution is released at that version,
and each pins the first-party distributions it depends on to it exactly (`scaly-control` depends on
`scaly-numerics==X.Y.Z`). A lockstep release publishes all seven, changed or not, so any set of them
installed together is at one version.

`scaly-experimental` makes no stability promise: its modules (`scaly.nn`, `scaly.geometry`, the
ALTRO and SCvx methods) may change or go in any release, and importing one gives an
`sc.ExperimentalWarning`. Promoting a module is one line in `distributions.toml`: it moves to the
distribution of its namespace and from then on follows that distribution's promise.

The solver plugins (`scaly-piqp`, `scaly-ipopt`, `scaly-sqp`) version independently. A release of
one does not require publishing the others, and version equality with the lockstep distributions has
no compatibility meaning. Each plugin declares:

- the range of `scaly-numerics` it supports, the distribution that ships `scaly.opt`:
  `scaly-numerics>=0.1.0a1,<0.2` for the `0.1.x` compatibility line (`>=X.Y,<X.(Y+1)` before 1.0).
  Naming a pre-release on the lower bound lets installers pick `0.1.0b1` or `0.1.0rc1` without
  `--pre`; the exclusive upper bound also excludes every `0.2` pre-release. CI runs each plugin
  against the oldest and the newest release in its range;
- the method API it implements, which the method registry checks when the plugin registers; and
- the upstream version of the solver it vendors, in its metadata and its changelog.

The `scaly` metapackage offers the plugins as extras with ranges, not pins (`scaly[piqp]`,
`scaly[solvers]`, `scaly[all]`), so a plugin release reaches users without a lockstep release.

## Method-API versions

Method APIs are versioned per problem class (`opt.QP` API 1, `opt.NLP` API 1, `ocp.DiscreteOCP`
API 1, ...): the class declares its `method_api`, and a method, built in or from a plugin, states the
`api` it implements. Looking a method up refuses one whose `api` is not its class's, naming both, so
a plugin built against another method API fails when it is first used rather than inside a solve.

## What a release may change

Three generated interfaces reach other people's builds, and each carries a promise.

Exported C symbols. The stable interface is the universal pointer signature of the exported entry
`<name>` and, for solver modules, the `<name>_stats` accessor, together with the derived-output
names `{kind}_{of}_{wrt}` such as `f_spjac_y_x`. A patch release changes none of these. A minor
release may rename, reorder or remove them, and the release notes list the change. The typed C
structs, the C++ `Buffer` aliases and the CasADi query functions are sugar over the pointer
signature and follow the same rule. The `static` `_raw` bodies
are internal and may change in any release.

Sparsity tables. The `<prefix>_NNZ`, `_NROW` and `_NCOL` macros and the index tables in the
generated header take their prefix from the derived-output name, so their names follow the symbol
rule above. Their contents are stable in no release. The coordinate order is an artifact of lowering,
and a coloring or `VMAP` change may reorder it. A consumer stays correct by reading values through
the `<prefix>_csr_val_perm` and `<prefix>_csc_val_perm` tables, as
[the generated interface](../how_it_works/generated_interface.md#sparse-outputs) describes. Whenever generated output changes,
`_JIT_CACHE_VERSION` in `src/scaly/codegen/jit.py` is bumped so a cached library from an older
version is never reused.

Extension API. `EXT_API_VERSION` (`scaly.ext`) versions what any package outside the compiler
builds on, and every such package checks it at import; it is also part of every JIT cache key.

Method API. `METHOD_API` (`scaly.opt.method`, the method API of `sc.opt.NLP`, which continues the
solver plugin protocol's numbering) and `SCALY_SOLVER_STATS_VERSION` version the contract between
`scaly.opt` and the solver plugins; [Solver plugins](solver_plugins.md#versioning) lists what each
covers and its history. A protocol bump is a minor lockstep release and a coordinated release of every
official plugin, which raise their lower bound on `scaly-numerics` in the same commit. A patch release
never bumps either constant.

## Git tags

A lockstep release gets one tag, `vX.Y.Z`, which names the version of all seven lockstep
distributions. A plugin release gets a package-qualified tag, `scaly-piqp-v0.1.0`. Several tags may
point to one commit when it releases a lockstep version and plugins together. Tags are created by
the release script only and are not moved or reused after publication.

## Release process

`scripts/release.py` makes a lockstep release:

```bash
uv run scripts/release.py 0.2.0
```

1. It sets `version` in `distributions.toml`, rewrites every manifest from the table (the pins move
   with it) and relocks.
2. It builds the wheel and sdist of every lockstep distribution and every solver plugin into `dist/`.
3. It runs the release check: a clean environment gets `scaly[experimental,solvers]` from those
   wheels alone, with the third-party packages the examples declare, and runs the conformance
   suites, every example and notebook, and the examples' lint there, every solver required to load.
4. With `--tag`, once the check passes on a committed tree, it tags `vX.Y.Z`.

Pushing the tag runs the release job in CI: it checks the tag names the version in
`distributions.toml`, runs the same build and check (`scripts/release.py --check-only`) and attaches
the wheels to the run, with the plugins run against the oldest and newest `scaly-numerics` their
ranges allow. Publishing stays a deliberate step after that: `uv publish dist/*`, then the GitHub
Release for the tag. A plugin releases on its own: its version in its manifest, its wheel built by
`uv build --package scaly-<name>`, `uv run scripts/isolation.py scaly-<name>` against the workspace,
its tag, and `uv publish`.
