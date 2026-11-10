# Sparse generated interfaces, issue #43

Stack layer 5 on `t3code/42-sparse-boundary`, PR #287. The PR base stays that branch.
The decided scope is the Sparse tensors section, decision 4 and Rules for every item in
`internal/notes/core_compiler_roadmap.md`. Also read #94, the #37 coordination and #39 to #42
handoffs, the architecture/codebase/conventions/contributing pages, versioning and compiler
maintenance context. No comments currently exist on #43 itself.

The kernel already uses declared sparse layouts. Replace the temporary typed-header refusal
with stored-size buffers and local layout metadata for inputs and outputs. Keep arbitrary-order
derivative metadata and dense snapshots unchanged. Do not publish sparse names in the guide or
API. CasADi requires CSC and two-dimensional leaves. With no reply to the optional question,
proceed with explicit rejection of other layouts and batch axes rather than workspace conversion.

- [x] Inspect the clean linked worktree, prerequisites, tracker and existing interfaces.
- [x] Add differential C/C++ and CasADi tests and observe the current failures.
  Read frozen devrush sparse-signature/export tests for useful problem cases.
- [x] Publish declared-layout metadata and correct typed buffer sizes, handling shared names,
  batches and empty patterns. Build CasADi sparsity from leaf types.
- [x] Prove tests detect wrong tables/value ordering by temporary perturbations.
  Run focused tests, dense snapshots, #41 scale gate and custom derivative regression.
- [x] Run the full pre-merge hook, commit, push and open a draft PR against #42.
- [ ] Apply cross-review with Opus and Sol, fix findings and run short delta checks.
- [ ] Remove this plan, finish the PR evidence and tracker status, mark ready and watch CI.

Focused proof passes: 68 interface/snapshot tests, plus the scale and custom-derivative checks in
the preceding 85-test focused run. Three temporary mutations failed as required: reversed tables
in C and C++, corrupted compiled values in C and C++, and lost CasADi input sparsity. All restored.
The complete pre-merge hook passes: 2,904 passed, 52 skipped, strict types, Ruff, benchmark smoke,
docs without warnings and private-name check. The node-ID baseline has 2,956 tests.

Draft PR #289 is linked in T3. Sol's review found two C++ metadata namespace collisions, with
CasADi query names and std. Both reproduced on inputs and outputs. The allocator now reserves
those declarations, plus emitted solver entry/stats/default-options names. All query names and
three solver declaration cases have compiling regressions. Opus's initial review is still pending.
The full hook on these fixes passes with 2,935 tests and 52 skips. Thirteen existing C++ headers
also match the #42 renderer byte for byte. Commit the fixes and request the low-effort delta check.

Choices for the final report: stored-shape C++ Buffer aliases; local nrow/ncol/nnz with full shape
and storage_size; familiar csc_col_ptr/csc_row_ind, csr_row_ptr/csr_col_ind, rows/cols spellings;
buffer-derived names with scoped collision suffixes, preserving legacy derivative metadata first;
omit empty index arrays but retain compressed pointers and one-value buffer sentinels; allow null
data pointers only on empty sparse leaves; construct CasADi CSC metadata directly from layout
indices without another pattern conversion; reject non-CSC and batched CasADi leaves.
Devrush sources: tests/linalg/test_sparse_signature.py, sparse matvec and self-named sparse-output
header cases, plus tests/export/test_casadi.py and tests/export/test_cpp.py. Solver/KKT examples
need sparse kernels and solves outside this task and were not ported.

Phases are sequential. Reviews can run in parallel after all checks pass. Keep an inventory of
header names, table ordering, batch metadata and CasADi choices for the final report, plus any
reviewer disagreements. No merges, tags, main pushes or `gh stack` commands.
