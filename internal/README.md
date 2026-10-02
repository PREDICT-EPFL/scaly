# Internal notes

Material kept in the repository but deliberately **not** published to the documentation site.
Zensical builds everything under `docs/`, so anything that should stay unpublished lives here
instead.

- [`todo.md`](todo.md) — the actionable list. The benchmark suite's build-out history is in
  `notes/benchmark-buildout.md`.
- [`notes/`](notes/) — mostly frozen records: design studies, migration plans and investigation
  write-ups that explain how the current design was arrived at. They are dated and superseded by
  definition; read them for the reasoning, not for how anything works today. The maintained exceptions are
  [`notes/core_compiler_roadmap.md`](notes/core_compiler_roadmap.md), the design and order of the core
  compiler work, [`notes/refactorings.md`](notes/refactorings.md), which tracks planned refactorings,
  [`notes/user_guide_writing.md`](notes/user_guide_writing.md), the agreed user-guide writing brief, and
  [`notes/benchmark_protocol.md`](notes/benchmark_protocol.md), the full benchmark measurement protocol.

For how scaly works now, see [`docs/how_it_works/architecture.md`](../docs/how_it_works/architecture.md) and [`docs/dev/codebase.md`](../docs/dev/codebase.md).

## What is in `notes/`

| Note | What it records |
| --- | --- |
| `core_compiler_roadmap.md` | the core compiler features before GPU support: decisions, order of work, one design paragraph per todo item |
| `small_core_static_sparsity_2026_09_30.md` | the investigation behind the 2026-10-01 roadmap rewrite: the closed opset, sparse patterns and support, TACO-style lowering |
| `refactorings.md` | refactorings we have decided on but not yet carried out, one `#` section each |
| `refactorings_landed_2026_09.md` | refactoring designs moved out of `refactorings.md` once they landed in September 2026 |
| `benchmark_protocol.md` | the benchmark measurement protocol and the reasoning behind each rule |
| `program_ir_migration.md` | the migration that introduced the program dialect and the single lowering path |
| `native_toolchain_exploration.md` | the conda-prefix and delocate experiments behind the vendored-solver build |
| `vendored_solvers.md` | known issues in the vendored PIQP and IPOPT builds |
| `macos_clang_call_miscompile.md` | the Apple-clang miscompile of inlined callee bodies, and the fix |
| `fuzzing.md` | notes and ideas on fuzz testing of the compiler core |
| `compiler_maintenance_context.md` | implementation details moved out of the public compiler explanations |
| `release_workflow_design.md` | how `ci.yml` builds release artifacts and `release.yml` publishes them |
| `documentation_api_review.md` | API gaps reproduced while rewriting the user documentation |
| `user_guide_writing.md` | maintained writing principles agreed during the user-guide review |
