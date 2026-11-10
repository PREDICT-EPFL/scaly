# Sparse numerical typing (#288)

The issue and the roadmap's Signatures and templates section decide the numerical leaf union.
The compiler-run plan in the coordinator's worktree authorizes queueing after review and green
checks. Shared suite rules are in ~/.scaly-suite/RULES.md. Runtime boundary changes and window
order are out of scope. Preserve dense declaration types and keep sparse names unpublished.

- [x] Read the issue, dependency #42, parent #37, architecture, code map, conventions and run rules.
- [x] Add static positive and negative cases. The old annotations produced 34 diagnostics.
- [x] Fix default TensorType declarations and inferred Expr outputs conservatively. Preserve
      dense types through explicit declarations where inference cannot distinguish sparsity.
- [ ] Run strict types, focused runtime coverage, formatting and lint. Prove static gates fail
      under a typing regression. Run the detached full pre-merge hook through the shared slots.
- [ ] Commit, push, open a draft PR and get the required cross-review. Fix findings and confirm
      only the delta, if necessary. Rebase for overlapping siblings when needed.
- [ ] Remove this plan, finish green checks, mark ready and queue once. Report criterion evidence,
      typing decisions, departures, open points and follow-ups.

Project metadata cannot currently be read or changed through GraphQL because the shared quota
is exhausted. The REST issue comment records ownership without changing unrelated metadata.

Validation: strict ty, Ruff lint and formatting pass; 143 focused tests pass. Three
perturbations (default overload, inferred result and an expected error) each fail the static gate.
The detached full pre-merge hook is waiting in /tmp/scaly-288/premerge.log; its exit file
is /tmp/scaly-288/premerge.exit. No runtime implementation or generated snapshots changed.
