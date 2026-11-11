# Batched MATMUL, issue #67

Authorized by the section 7 coordinator under internal/notes/compiler_run_2026_11_10/plan.md and ~/.scaly-suite/RULES.md. Work stays in this thread; delegate only one cross-review and its delta check.

## Context and current state

- Branch t3code/67-batched-matmul is clean apart from this plan and starts at #401's reviewed head, on t3code/66-axis-reductions.
- #401 was still open with auto-merge enabled at initial inspection.
- Read the run plan, suite rules, implement, unslop, write-plan, inspect-issues and update-issues skills. Consulted gh stack --help.
- Read architecture, codebase, conventions, contributing, tracking and compiler maintenance context, but combined output was truncated; re-read relevant sections before edits.
- Dense builders, verifier, lowering, AD, sparsity masks and transpose/ones folds implemented.
- NumPy differential cases, AD harness extension, mask finite differences and schedule checks pass.
- Batched sparse products explicitly deferred to #403, with the issue number in the error; unbatched sparse paths unchanged.
- First full suite found two structural expectations: preserved unbatched reverse rules to avoid extra copies, updated the leading-seed access-map test for the batched node. Rerun pending.
- Existing C snapshots pass byte-identically to the #66 base. Guide/API updated. Source committed and restacked onto the rebased #66 after #30. No pushes yet.

## Remaining work

- [ ] Read #67 body/comments/dependencies and #296 via REST with bounded output; inspect relevant roadmap decisions and implementation. Record work started in tracker.
- [ ] Establish gh stack tracking over #66. When #401 merges, sync/rebase through gh stack; do not rebase merely for movement on main.
- [ ] Add failing NumPy differential tests for vector promotion, rank 3–5 batches, broadcasting, empty and singleton axes. Add AD harness and sparsity-pattern tests, including one batched seed MATMUL.
- [ ] Implement inference/builders/verifier, batch-outer accumulator lowering preserving reduction order, forward/reverse AD with REDUCE unbroadcast, masks and transpose folds. Keep supported sparse cases; either extend existing CSR loop directly or file source:agent limitation and name issue in clear error. Do not edit ir/einsum.py or sparse-sparse products.
- [ ] Update published guide/API using write-docs skill. Verify ones folds and snapshots; gather operation counts, source size and C compile time if snapshots change. Runtime measurements deferred to Phase 4.
- [ ] Run focused tests with at most -n 2 and required checks. Full suite only detached through ~/.scaly-suite/run, poll its exit file.
- [ ] Commit and push review-ready state, open PR against #66 branch until merged, link PR immediately with T3 tool.
- [ ] Run cross-review skill once and its delta check only, apply findings, post verdict promptly in PR description for #68 coordinator. Push all fixes together after checks.
- [ ] Queue once with gh pr merge N --squash --auto after verdict and green checks. Watch PR using T3, fix conflicts/ejections and requeue as needed. Report criteria evidence, verdict, follow-ups and queued/merged state.

All GitHub reads use gh api REST. Push only review/merge states. No direct main push, tags or releases. Delete this temporary plan on completion.

## Restacked validation

All pre-merge checks passed through the detached suite runner: 4,581 passed, 55 skipped, benchmark smoke, documentation build and private-name check. Focused differential/AD/snapshot run passed 1,454 tests. C snapshots remain byte-identical to the #66 base. No source files were changed during this run.

A previous run began within the source-digest clock slack after an edit, making JIT index checks decline their keys; the isolated JIT index tests passed on retry. The restacked full run passed those checks too.

Next: first push and draft PR, one independent cross-review and its delta check, remove this plan, ready/queue the PR. Keep #65's rules and reduction work untouched. Link the stack with gh stack so #68 can extend it.

## Cross-review fixes

Opus requested two fixes: restore known-zero support for batched products, and count emitted batch loops rather than unit axes when scheduling. Both reproductions failed before the fixes and pass afterwards. Support now broadcasts only known entries and uses a sparse block-diagonal Boolean product; singleton batch dimensions no longer take row loops outside the reduction. Focused checks passed 1,730 tests, with 180 support/forward/schedule tests also passing.

Tests have moved into their owning AD, IR, rewrite and lowering files, sharing shape cases in tests/matmul_cases.py. The guide/API prose was tightened. #407 owns the non-blocking cost of enumerating every output/contraction pair in batched Jacobian masks. Full suite and the one delta check remain before the final consolidated push; remove this plan after the delta check.
