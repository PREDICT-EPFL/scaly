# Select masks in mapped adjoints and forward products

Implements #364 and #414 under `internal/notes/compiler_run_2026_11_10/plan.md`,
section 4 (#24), on origin/main. The run plan and `~/.scaly-suite/RULES.md` override
the ordinary implement workflow. Read PR #358 before extending its helper protocol.
Activity comes from select conditions, never cotangent or tangent values.

- [x] Read both issues, #358, the run rules, architecture, conventions and compiler decisions.
- [x] Reproduce both failures with differential regressions, including higher derivatives at a
      zero cotangent and runtime/baked/multiple forward seeds.
- [x] Return masks introduced inside mapped adjoints, preserving helpers without masks.
- [x] Preserve selected forward tangents through downstream singular partials and nested selects.
- [ ] Check sparse/shared/overlapping mapped inputs, finite differences and singular boundaries.
      Include `sqrt(maximum(x, 0))` after #65 lands; its rules are outside this change.
- [x] Update the functions guide and `where` docstring using write-docs.
- [x] Prove regression gates fail under perturbation and generated C without selects is identical.
- [ ] Run the full pre-merge hook detached through the suite queue, then push one reviewed state.
- [ ] Get Opus's thorough cross-review plus the required independent correctness review; fix and
      confirm only review deltas. Delete this temporary plan before making the PR ready.
- [ ] Queue the single PR closing both issues after its verdict, watch through merge and report.

Nearby PRs #368, #361, #405 and #408 can require conflict resolution. Do not rebase merely
because main moved. No changes to LOOP, nonsmooth conventions or other children's scope.

Evidence before full checks: 1,467 focused AD/differential/snapshot tests pass. Restoring the
original AD modules makes 26 of the new cases fail. Six C/header files for inline, called and
mapped gradients without selects are byte-identical to origin/main. Proof files are in
`/tmp/scaly-364-proof-0cqs9ig2/`. Types and Ruff pass. #408 is still open and conflicted,
so the maximum regression needs its final integration after #65 lands. No code from #65 copied.
