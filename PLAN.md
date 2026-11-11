# In-place lowering, issue #36

Base: the #31 branch merged with origin/t3code/35-index-operands. Design is the #36 paragraph of internal/notes/core_compiler_roadmap.md. The compiler run plan and ~/.scaly-suite/RULES.md govern validation and publication.

- [x] Merge the two prerequisites and resolve the verifier import conflict.
- [x] Add failing differential tests for last use, same index, disjoint indices, conservative refusal and floor reads.
- [x] Implement lowering proofs and scatter reuse, loop procedure variants, and the inout parameter contract across verification and program passes.
- [x] Measure the four-entry update of a 100k carry with counts, buffers, generated C and compile time.
- [x] Run focused checks and the detached full suite through ~/.scaly-suite/run.
- [ ] Cross-review with Opus and sol, fix findings and run one delta check.
- [ ] After #411 and #415 merge, rebase task commits onto origin/main, rerun the suite, open and link the PR. Queue only on instruction.

Focused validation: 237 pass across lowering, program passes and import boundaries; loop and C snapshot checks pass. The floor-read mutation reproduces [2.5, 2.5, 3.5] exactly and is restored. Measurements and generated C are under internal/notes/compiler_run_2026_11_10/in_place/.

The first full suite had 3932 passes, 56 skips and two architecture-check failures. The import cycle and repeated CAST handling were removed; the focused coverage and import checks pass. An unchanged carry also uses one slot without a step copy. The next full suite log is /tmp/scaly36-validation/suite-3.log.

The full merge suite passed with 3935 tests and 56 skips, smoke and docs included. A subsequent focused reproduction found stale zero-support metadata on a reused private buffer under scalarization. The regression failed with 0 instead of 5; reuse now removes that value metadata. A fresh full suite validates the fix.

Final pre-review full merge suite: /tmp/scaly36-validation/suite-4.log, exit 0. 3936 passed, 56 skipped, benchmark smoke passed, documentation build passed, type/format/lint/private-name checks passed. Existing C snapshots remain byte-identical. Cross-review is next.
