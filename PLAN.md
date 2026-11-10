# Array cells for mapped range fusion

Task #183 follows the decided Direction in the issue and the Tensor core and the loop compiler
section of `internal/notes/core_compiler_roadmap.md`. The unattended compiler run plan controls
checks and merging. Only `src/scaly/passes/program/fuse_ranges.py` and its existing test module
change in the completed pull request. #223 owns producer selection and may require a rebase.

- [x] Read the run plan, issue, parent, architecture, codebase rules and relevant implementation.
- [x] Add failing rewrite-count tests for stencil gradients with and without measurements,
      plus differential tests for every rejected-exit case.
- [x] Replace per-index cell objects with trip and interned-value arrays. Keep varying external
      load indices in arrays until folding or final grouping determines which scalar nodes survive.
- [ ] Run focused tests, prove the new gates fail under a perturbation, check C snapshots and
      collect an indicative million-stage compile timing.
- [ ] Commit the final state and run detached pre-merge checks under the shared suite lock.
- [ ] Open the draft pull request, obtain the required cross-review and confirm fixes.
- [ ] Rebase as needed, keep snapshots identical to main, remove this temporary plan, push,
      mark ready and queue the pull request once checks and review pass.

GitHub GraphQL initially rejected requests for rate limits. REST access works. No prerequisite
issues are open. Parent #297's older serialization wording is superseded by the task brief.

Focused validation: 36 fusion and snapshot tests pass, as do Ruff and ty. The original pass
fails the measurement-stencil count test at 270 versus 16,398 calls. A per-position rewrite
perturbation makes both stencil variants fail. Current counts are 12/12 and 22/22.
