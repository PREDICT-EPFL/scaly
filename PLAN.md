# Generated wrapper naming follow-ups

The compiler-run coordinator assigned #343 and #344 after #325's review. #325 has both approvals and entered the merge queue. Follow the plan now on main and `~/.scaly-suite/RULES.md`, including detached full suites and pushing only complete checked states.

The changes belong in separate PRs. #343 changes C++ workspace reference naming in `codegen/cpp.py`. #344 changes C wrapper parameter allocation against complete C typedef names in `codegen/aot.py`. Preserve exported pointer entry names and buffer typedef names.

- [x] Read both issues, their reproductions, dependencies, project metadata and naming implementations. Neither has a parent or blockers.
- [ ] On `t3code/343-cpp-workspace-shadowing`, based on #325, add a failing C++ compile-and-call test for export `workspace` with input and output suffix collisions.
- [ ] Use the existing name allocator for the C++ workspace reference. Prove the regression fails when reverted. Keep C snapshots identical and regenerate test IDs.
- [ ] Run focused checks and the detached full pre-merge runner. Remove this plan before the first PR push, keeping its history available. Obtain the prescribed cross-review, batch any fixes, and queue once reviewed and green.
- [ ] Implement #344 separately from #325's branch, or main if #325 has merged. Compile and run C wrappers where input or output parameter names hide later buffer or workspace typedefs. Preserve the existing typed aliases and pointer entries.
- [ ] Prove both positive and reverted cases, keep snapshots identical unless a deliberate generated change requires regeneration, run the full checks and review, and queue the second PR.

Issue #343 is active; #344 is accepted next work. Do not amend or push #325 while its merge-queue checks run.
