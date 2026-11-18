# Keep unsafe constant casts in C

Issue #365 follows the maintainer's 2026-11-18 decision in the rewritten description.
The base and pull request base are main. The branch is independent.

Read the compiler roadmap's dtype section and per-item rules, compiler maintenance
context, repository guidance, and ~/.scaly-suite/RULES.md. Scope is the constant-folding
rule, the sc.cast docstring, the cast rule in the functions guide, and focused tests.
The mean, renaming, shape-helper and leaf-dtype tasks are excluded.

- [x] Inspect the issue, prerequisites, checkout, design and implementation.
- [x] Add focused regression tests and reproduce the failure.
- [x] Guard unsafe float-to-integer folds and document the C conversion rule.
- [x] Prove the tests catch a regression and keep all generated-C snapshots identical.
- [x] Run the complete merge checks detached through ~/.scaly-suite/run.
- [ ] Open the draft pull request and get the requested Opus 5.5 high cross-review.
- [ ] Address findings, check any fix, remove this temporary plan, and mark ready.

The phases run in order. Stop immediately if a generated-C snapshot changes.
Never merge, queue, push to main, create tags or regenerate snapshots.

Focused verification: 74 passed, including the unmodified generated-C snapshot gate
and the operation-coverage check.
Disabling the guard makes the new operand-range assertion fail for int32.
Current NumPy already refuses undefined conversions on this host, so the C checks also
pin that behavior independently of the host's conversion warnings. No devrush test matches
this requirement; its float-to-index tests concern clamping before the cast.

The first full run found the prohibition on naming individual elementwise ops in
generic passes. The guard now uses the source and destination dtypes, which identify
float-to-integer conversions, and preserves the uniform constant's compact storage.
No coverage exemption or unrelated file changes are needed. The complete merge checks
passed, including 5136 tests with 16 skipped, benchmark smoke and documentation build.
