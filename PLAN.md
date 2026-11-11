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
- [ ] Add failing public-path regressions, including 0 and 255 with scalar and block hints.
- [ ] Reject uint8 expression types, update the documented limit and verify internal masks.
- [ ] Update collection baseline and prove regressions fail without the guard.
- [ ] Run the full pre-merge hook detached through the shared suite runner.
- [ ] Open a draft PR on main after checks, link it, and get Opus cross-review and the required
      additional correctness review for IR changes. Fix findings and check only those fixes.
- [ ] Remove this temporary plan, mark ready and report the result.
- [ ] Wait for the section coordinator to say #361 has merged, then rebase onto main, rerun the
      necessary checks, push once, and queue after the review verdict and green CI.
- [ ] Report when merged. Never push main, create tags/releases or run deployments.

Inspection also checked the frozen devrush dtype tests. They cover float32 metadata and coercion,
not uint8 arithmetic. Existing tests cover NumPy references for both cast lowerings, bit selects
at each lane width, packed-mask conversion and C snapshots.
