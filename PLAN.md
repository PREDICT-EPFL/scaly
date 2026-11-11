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
- Existing C snapshots pass byte-identically to the #66 base. Guide/API updated. No commits or pushes yet.

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
