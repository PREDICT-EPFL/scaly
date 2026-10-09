# Issue #21: intermediate-expression derivatives

The decided design is the #21 paragraph in `internal/notes/core_compiler_roadmap.md`, under One AD engine. The branch is based on `t3code/issue-20-elementwise-table` (PR #169), sixth in the AD stack. No loop IR, custom rules, traversal refactor or elementwise-table changes are in scope.

Read: architecture, codebase, conventions, contributing, compiler maintenance context, AD mode/sparsity/sparse implementations, expression substitution and lowering. Port source: `origin/devrush:tests/core/ad/test_wrt_expressions.py`, Derivatives with respect to an expression that is not an input. Its while-loop problem will use today's body Function and a Python loop. Devrush has no stop-gradient tests in tests/core.

- [x] Inspect issue, parent and prerequisite, verify worktree and stack base, read agreed design.
- [x] Port differential tests for slices, dense/sparse forms, calls/maps, overlapping selections, nested selections and the Newton carry body. See the baseline failures.
- [x] Substitute independent stand-ins before forward/reverse/sparsity analysis and restore the original expressions. Add stop-gradient rules and primal lowering.
- [x] Write and execute guide examples and add API entries using write-docs.
- [ ] Prove perturbations fail; run focused checks, #15 harness, byte-identical snapshots, examples and pre-merge checks.
- [ ] Commit/push, open draft PR on #169, link the stack, get one cross-family review and one delta check, then mark ready and watch.

Tests precede implementation. Documentation follows stable behavior. Validation precedes review. No independent implementation work needs delegation. If lower stack layers change, rebase and rerun checks. A coordinator measurement window restricts work to focused tests while open.

Validation so far: 1,058 focused AD/coverage/snapshot tests passed, guide fences match the stated outputs, both disabled-substitution and identity-stop perturbations fail. Full suite passed 2,427 with 3 skips before adding two mapped C-transparency tests. Those tests and focused sugar/snapshot/coverage checks pass. Final pre-merge rerun remains. Follow-up #170 records mapped affine views that erase the selected node before AD; explicit windows preserve it.
