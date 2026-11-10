# Dense leaf layouts for CasADi

Issue #103 extends the leaf layout field introduced by #42. The sparse stack is the
starting base. Follow the unattended compiler run plan, the roadmap's Sparse tensors
section and decision 4, and docs/dev/versioning.md. Zero-size dense leaves belong to #290.

- [x] Read the issue, parent, design, implementation and local instructions.
- [x] Add compiled NumPy differential tests and demonstrate the existing refusal.
- [x] Add dense row-major and column-major declarations, preserving layouts through
      trees, numerical calls, lowering, calls and maps. Select column-major at CasADi export.
- [x] Update the finished dense interface documentation and prove the tests detect a bad order.
- [ ] After #289 merges, fetch and rebase onto origin/main past the sparse stack base.
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

Use the run-wide suite lock and detached runner. Full checks must run on the state
intended for push, with PYTEST_XDIST_AUTO_NUM_WORKERS=8. Focused tests use at most two workers.
