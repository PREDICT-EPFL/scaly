# Prevent public uint8 arithmetic, #363

Base `origin/main`, after #30. Section 4, #24, under
`internal/notes/compiler_run_2026_11_10/plan.md`. Follow `~/.scaly-suite/RULES.md`.

The roadmap's scalar vocabulary excludes uint8. The published type table also excludes it.
`passes/program/pack_masks.py` uses uint8 program buffers and loads for packed masks, without
expression tensor types. Reject uint8 in `TensorType` so casts, symbols, constants and typed
Function leaves cannot introduce uint8 expressions. Keep the internal descriptor and program
operations intact. Signed overflow, invalid float-to-integer casts and float32 lowering belong
to other issues.

- [x] Read the task, roadmap, run plan, rules, public API and packed-mask implementation.
- [x] Add failing public-path regressions, including 0 and 255 with scalar and block hints.
- [x] Reject uint8 expression types, update the documented limit and verify internal masks.
- [x] Update collection baseline and prove regressions fail without the guard.
- [x] Run the full pre-merge hook detached through the shared suite runner.
- [ ] Open a draft PR on main after checks, link it, and get Opus cross-review and the required
      additional correctness review for IR changes. Fix findings and check only those fixes.
- [ ] Remove this temporary plan, mark ready and report the result.
- [ ] Wait for the section coordinator to say #361 has merged, then rebase onto main, rerun the
      necessary checks, push once, and queue after the review verdict and green CI.
- [ ] Report when merged. Never push main, create tags/releases or run deployments.

Inspection also checked the frozen devrush dtype tests. They cover float32 metadata and coercion,
not uint8 arithmetic. Existing tests cover NumPy references for both cast lowerings, bit selects
at each lane width, packed-mask conversion and C snapshots.

Draft PR: https://github.com/PREDICT-EPFL/scaly/pull/405. Initial full hook passed,
3,776 passed and 55 skipped, plus smoke, docs and private-name checks.
Sol found an alias bypass through `DType("u1", 8, "uint8_t", is_integer=True,
is_unsigned=True)`. The guard now checks unsigned eight-bit descriptors as well as the canonical
name. Three new alias cases failed before the fix, then all 117 focused checks passed, including
internal masks and C/header snapshots. Full validation of this fix and a Sol delta check remain.
Opus's original review is still pending. Keep all fixes together for the second push.
