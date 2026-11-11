# Walk each Function graph once

Implement #400 under the compiler run plan in
`internal/notes/compiler_run_2026_11_10/plan.md`, the Foundations #64 decision in
`internal/notes/core_compiler_roadmap.md`, and `~/.scaly-suite/RULES.md`.
Base and pull request base are main. Own only graph traversal in
`src/scaly/function/digest.py` and its tests. Keep source digest changes in #398
and value cleanup in #399 outside this task. Preserve all digest inputs and
deterministic walk ordering. C snapshots must remain unchanged.

- [x] Read the plan, rules, implementation workflow, project guidance, issue and overlaps.
- [ ] Add failing traversal-count tests for shared many-output graphs and attribute expressions.
- [ ] Make one walk number the input and output graph, skipping known nodes in nested attribute walks.
- [ ] Prove the new tests fail under a perturbation, regenerate the test node-ID baseline,
  run digest and JIT index tests and the requested verify run with two workers.
- [ ] Run all pre-merge checks detached through `~/.scaly-suite/run` on the final state.
- [ ] Commit, push once, open and link a draft PR, get one correctness cross-review,
  address findings and delta-check fixes if needed, and settle review delegates.
- [ ] Remove this temporary plan, mark ready, update #400 and queue squash auto-merge
  after review and green checks. Expect conflict-driven rebases on #398 and #399.

Use operation counts, never timings, as evidence. Report the PR, counts before
and after, validation and open points. GitHub reads and writes use REST except
the explicitly requested merge command. Project field metadata is unavailable
through the normal REST issue representation; do not use GraphQL to obtain it.
