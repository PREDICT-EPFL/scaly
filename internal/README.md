# Internal notes

Material kept in the repository but deliberately **not** published to the documentation site.
Zensical builds everything under `docs/`, so anything that should stay unpublished lives here
instead.

- [GitHub Issues](https://github.com/PREDICT-EPFL/scaly/issues) is the actionable list, with planning
  metadata in the public [scaly roadmap Project](https://github.com/orgs/PREDICT-EPFL/projects/2).
  [Tracking work](../docs/dev/issue_tracking.md) defines the contributor and agent workflow.
- [`notes/`](notes/) holds design studies and investigation records. Dated records preserve
  evidence and rejected approaches; they do not describe current behavior unless explicitly
  maintained. The maintained notes include
  [`notes/core_compiler_roadmap.md`](notes/core_compiler_roadmap.md), the design and order of the core
  compiler work, [`notes/refactorings.md`](notes/refactorings.md), which explains open refactoring issues,
  [`notes/user_guide_writing.md`](notes/user_guide_writing.md), the agreed user-guide writing brief,
  [`notes/benchmark_protocol.md`](notes/benchmark_protocol.md), the full benchmark measurement protocol,
  and [`notes/vendored_solvers.md`](notes/vendored_solvers.md), the solver-build constraints and license survey.

References to retired tasks that were completed before the GitHub import remain historical
identifiers. They have no new issue number; find their original descriptions in git history.
Devrush's identifiers belong to its separate frozen backlog and must never be mapped to main's issues.

For how scaly works now, see [`docs/how_it_works/architecture.md`](../docs/how_it_works/architecture.md) and [`docs/dev/codebase.md`](../docs/dev/codebase.md).

## What is in `notes/`

| Note | What it records |
| --- | --- |
| `core_compiler_roadmap.md` | compiler decisions, release scope, and implementation order, linked to GitHub issues |
| `small_core_static_sparsity_2026_09_30.md` | the investigation behind the 2026-10-01 roadmap rewrite: the closed opset, sparse patterns and support, TACO-style lowering |
| `refactorings.md` | refactorings we have decided on but not yet carried out, one `#` section each |
| `refactorings_landed_2026_09.md` | refactoring designs moved out of `refactorings.md` once they landed in September 2026 |
| `benchmark_protocol.md` | the benchmark measurement protocol and the reasoning behind each rule |
| `program_ir_migration.md` | the migration that introduced the program dialect and the single lowering path |
| `native_toolchain_exploration.md` | the conda-prefix and delocate experiments behind the vendored-solver build |
| `vendored_solvers.md` | vendored PIQP and IPOPT build constraints, packaging history, and license survey |
| `macos_clang_call_miscompile.md` | the Apple-clang miscompile of inlined callee bodies, and the fix |
| `fuzzing.md` | notes and ideas on fuzz testing of the compiler core |
| `compiler_maintenance_context.md` | implementation details moved out of the public compiler explanations |
| `release_workflow_design.md` | how `ci.yml` builds release artifacts and `release.yml` publishes them |
| `documentation_api_review.md` | API gaps reproduced while rewriting the user documentation |
| `user_guide_writing.md` | maintained writing principles agreed during the user-guide review |
| `generated_interface_2026_09_18.md` | completed interface design, including native and CasADi storage-order decisions |
| `optimization_cleanup_2026_09_10.md` | completed optimization review and controlled before/after measurements |
| `c77_c79_implementation.md` | completed range and lane work, original acceptance limits, and deferred performance evidence |
| `perf_2026_09_07/`, `perf_2026_09_22/` | dated kernel investigations and their reproduction scripts |
| `windows_support_2026_10_02.md` | Windows toolchain investigation supporting issue #84 |
