# Dense leaf layouts for CasADi

Issue #103 extends the leaf layout field introduced by #42. The sparse stack is the
starting base. Follow the unattended compiler run plan, the roadmap's Sparse tensors
section and decision 4, and docs/dev/versioning.md. Zero-size dense leaves belong to #290.

- [x] Read the issue, parent, design, implementation and local instructions.
- [x] Add compiled NumPy differential tests and demonstrate the existing refusal.
- [x] Add dense row-major and column-major declarations, preserving layouts through
      trees, numerical calls, lowering, calls and maps. Select column-major at CasADi export.
- [x] Update the finished dense interface documentation and prove the tests detect a bad order.
- [x] After #289 merges, fetch and rebase onto origin/main past the sparse stack base.
- [ ] Run all merge checks, keeping existing C snapshots identical.
- [ ] Commit, push, open a draft PR, obtain the required cross-review and confirm any fixes.
- [ ] Remove this plan, mark ready, and queue with gh pr merge --squash after green checks.

Hard-to-undo decisions to report: dense layout keyword spellings and their ABI buffer
order, CasADi's export-time selection, and header metadata and typed indexing for that order.
No roadmap departure is planned.

Opaque solver entries bypass normal lowering and cannot yet convert dense matrix
leaf orders. Direct CasADi exports retain an explicit refusal; an ordinary Function
wrapper is compiled and checked against NumPy. Follow-up #324 owns direct opaque
boundary support, including explicit dense layouts in solver calls.

Focused validation: 154 sparse/dense boundary and CasADi tests passed, followed by
34 dense tests with C++ buffer indexing and scalar/vector mapping regressions.
Ruff and ty pass. Replacing DenseLayout.permutation with the identity makes the
compiled non-square CasADi differential test fail with incorrect matrix entries.
The original implementation failed seven new matrix tests with the documented refusal.

Use ~/.scaly-suite/run detached for full checks on the state intended for push.
Read ~/.scaly-suite/RULES.md for current run-wide rules. Focused tests use at most two workers.
Use REST for GitHub reads while GraphQL quota is exhausted. Queue once with
gh pr merge --squash --auto after the verdict; use a PR watch for CI wakes.
