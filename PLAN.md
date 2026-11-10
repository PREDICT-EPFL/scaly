# Finish support analysis after the loop-nest emitter

The maintainer expanded PR #277, implementing #40, after #27 merged. Follow the
three phases in order. The design remains `internal/notes/core_compiler_roadmap.md`,
especially Sparse tensors, Custom derivatives and Rules for every item. Read #40,
#276, the compiler maintenance context and the project contribution rules.

## Progress

- [x] Wait for #39 to contain the fetched main, then restack only this branch's
  commits with `git rebase --onto`.
- [x] Restore the new emitter's empty-domain contract when simplification makes
  an empty constant output. The existing empty-extent tests reproduce the issue.
- [x] Run the full merge hook, push with an explicit force-with-lease and post
  the new head in the conversation before starting the investigation. The hook
  passes 2,716 tests with 52 skips, including byte-identical C snapshots.
- [x] Get the independent investigation of custom-derivative support from
  claudeAgent, claude-fable-5-2, medium effort. Give it the design and both code
  versions without the previous review's conclusion. Its only owned file is
  `internal/notes/support_analysis_2026_11_10/custom_derivative_investigation.md`.
- [x] If the investigation reopens the design, stop and post the evidence on
  #40 for the maintainer. Otherwise apply its outcome and record the conclusion
  in the PR description.
  Fable confirms the existing contract permits rules outside primal support.
  Keep the current AD code: ordinary coefficient pruning remains, and removing
  the blanket result filter is required. Preserve the memo and warn #41 about
  its compressed relation row domain.
- [ ] Fold #276 into this PR on the new loop-nest emitter. Test support-based
  scalar zero skipping, output initialization and numerical agreement with a
  NumPy or unrolled reference. Show the new gates fail under perturbation.
- [ ] Review and measure any changed C snapshots using the serialized procedure.
  Report source size, generation time and function-evaluation time in the PR.
- [ ] Run the full merge checks, then one cross-review of the work added since
  the last verdict and one short check of fixes. Include the restack's empty-copy
  compatibility fix in that review.
- [ ] Update PR evidence and review line, reference #40 and #276, remove this
  temporary plan and return the PR to review. No merge.

The investigation is settled without reopening the design. Lowering may proceed.

## Boundaries

- `gh stack`, merging, main pushes and tags remain forbidden.
- Lowering changes are now authorized. #276 is in this PR's scope, rather than
  deferred. #41 keeps snapshots byte-identical, so this is the sole snapshot
  regeneration in flight after #27.
- #221 still owns argument-specialized CALL support. #41 owns full-size work and
  cache-memory costs. #42 owns sparse leaves and the boundary.
- Preserve stored patterns even when support is smaller. Sparse names stay out
  of the guide and filtered from the API page.
- SELECT, general REDUCE and LOOP are not present on this base. Their rules stay
  with the tasks introducing those ops.
- Implementation steps stay here, with scope, status, evidence and blockers in
  public records. Remove this file when the work is complete.
