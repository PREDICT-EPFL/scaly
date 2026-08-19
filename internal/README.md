# Internal notes

Material kept in the repository but deliberately **not** published to the documentation site.
Zensical builds everything under `docs/`, so anything that should stay unpublished lives here
instead.

- [`roadmap.md`](roadmap.md) — the library roadmap: phases, status, exit criteria. The
  benchmark and paper roadmap is `BENCHMARKS.md` at the repository root.
- [`notes/`](notes/) — mostly frozen records: design studies, migration plans and investigation
  write-ups that explain how the current design was arrived at. They are dated and superseded by
  definition; read them for the reasoning, not for how anything works today. The exception is
  [`notes/refactorings.md`](notes/refactorings.md), which is forward-looking and maintained.

For how alloy works now, see [`docs/how_it_works/architecture.md`](../docs/how_it_works/architecture.md).

## What is in `notes/`

| Note | What it records |
| --- | --- |
| `refactorings.md` | refactorings we have decided on but not yet carried out, one `#` section each |
| `program_ir_migration.md` | the migration that introduced the program dialect and the single lowering path |
| `naming.md` | how the project got its name, and the alternatives considered |
| `native_toolchain_exploration.md` | the conda-prefix and delocate experiments behind the vendored-solver build |
| `vendored_solvers.md` | known issues in the vendored PIQP and IPOPT builds |
| `macos_clang_call_miscompile.md` | the Apple-clang miscompile of inlined callee bodies, and the fix |
