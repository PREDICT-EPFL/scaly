# Keep unsafe constant casts in C

Issue #365 follows the maintainer's 2026-11-18 decision in the rewritten description.
The base and pull request base are main. The branch is independent.

Read the compiler roadmap's dtype section and per-item rules, compiler maintenance
context, repository guidance, and ~/.scaly-suite/RULES.md. Scope is the constant-folding
rule, the sc.cast docstring, the cast rule in the functions guide, and focused tests.
The mean, renaming, shape-helper and leaf-dtype tasks are excluded.

- [x] Inspect the issue, prerequisites, checkout, design and implementation.
- [ ] Add focused regression tests and reproduce the failure.
- [ ] Guard unsafe float-to-integer folds and document the C conversion rule.
- [ ] Prove the tests catch a regression and keep all generated-C snapshots identical.
- [ ] Run the complete merge checks detached through ~/.scaly-suite/run.
- [ ] Open the draft pull request and get the requested Opus 5.5 high cross-review.
- [ ] Address findings, check any fix, remove this temporary plan, and mark ready.

The phases run in order. Stop immediately if a generated-C snapshot changes.
Never merge, queue, push to main, create tags or regenerate snapshots.
