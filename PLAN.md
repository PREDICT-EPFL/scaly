# Cortex-M oracle builds, #207

Implements section #24 under `internal/notes/compiler_run_2026_11_10/plan.md` on main.
The branch started on #368, the typed-entry layer of #30 -> #55 -> #56 -> #207.
The original branch is rebased onto origin/main, dropping those old inherited
commits. The scratch branch t3code/207-cortex-m-trial carries them on #368's
supplied head 11586a7f. Rebase the task commits to main after its merge.
Do not push before that merge is confirmed.
Read `~/.scaly-suite/RULES.md` before tests and whenever notified of an update.

Scope: a float32 oracle with a derivative from an existing example, compiled with
`zig cc -target thumbv7em-none-eabihf -ffreestanding`, no undefined symbols beyond
libm, prints removed, zero frame budget. CI records object flash, static RAM and
caller buffers, with stack limits stated. No hardware execution or timing.

References: #207 body, roadmap release scope, milestone checks and dtype decisions,
`compiler_maintenance_context.md`, project architecture, codebase and conventions.
#314 owns `frame_budget` and has merged. The CI conflict was resolved by keeping
main's JIT verification job and adding the embedded job to the required dependencies.

- [x] Read the brief, rules, tracker, stack contracts and project guidance.
- [x] Reproduce the freestanding build and add the symbol gate with a failing perturbation.
- [x] Add the float32 mapped-stage oracle and CI job with size and memory reporting.
- [x] Write and execute the embedded guide section with write-docs.
- [x] Incorporate merged #314 and drop the old inherited dtype commits by rebasing onto main.
- [x] Trial on #368's updated head supplied by the coordinator.
- [ ] Run focused checks, then the detached full pre-merge checks through the suite runner.
- [ ] Obtain the claude-fable-5-2 high review and confirm fixes with a delta check if needed.
- [ ] Open the PR on main and link it after the coordinator confirms #368 has merged.
- [ ] Rebase onto main once all lower layers merge, retarget, run CI and queue.
- [ ] Report the merged PR, size evidence, limits and follow-ups to the coordinator.

Delete this temporary plan before marking the PR ready.

Findings and evidence:

- Zig 0.16 and pinned 0.17 reject `thumbv7em-none-eabihf`. The equivalent is
  `-target thumb-freestanding-eabihf -mcpu=cortex_m4+vfp4d16sp`, or cortex_m7.
  `readelf -A` confirms v7E-M, VFP register arguments and single precision.
- Zig has no bare-metal math header. `examples/embedded/math.h` declares only
  the example's sinf/cosf, to be supplied by target libm at final link.
- Disable unwind tables to avoid the AEABI unwind-runtime dependency.
- The real float32 zero-budget build passes on #368's supplied head for M4F and
  M7. Each object has text=452, data=4, bss=0: flash=456 bytes and static RAM=4
  bytes. Caller workspace=320 bytes, inputs=32, outputs=96. Only sinf/cosf are
  undefined. No private array remains on the stack. These sizes exclude target
  libm, startup and stack. Reports are under /tmp/scaly-207/typed-proof/.
- The stage's block lowering hint keeps arrays in the example so it exercises
  spilling. Without the hint the entire small oracle scalarizes and needs no w.
  Perturbing frame_budget to 65536 fails the nonempty-workspace assertion.
- The guide command runs with PYTHONPATH=. so the CLI can import examples from
  the checkout. Its object independently confirms the same sizes and symbols.
- The float32 NumPy differential passes and fails after perturbing the stage math.
- `sc.print` rejects float32 in the supplied stack. Filed #373 with source:agent.
  This change tests print removal with a supported boolean diagnostic instead.
- #314 merged. The coordinator supplied #368 at 11586a7f. The task's scratch
  branch trials on that head. Hold the push until #368's merge is confirmed.
- Current main-based checks: Ruff lint/format and ty pass, collection passes with
  6155 tests, and the documentation site builds. The earlier typed-base float32
  differential passed and the numeric perturbation failed as expected.
- Per the coordinator's request, the detached pre-merge run finished on the main-based
  branch: 6145 passed, 5 skipped, 5 failed in 475 s. Four failures are the new float32
  tests and example blocked on #368. The fifth is unchanged upstream test
  test_called_masks_use_stored_coordinates_for_sparse_outputs[False-True] at line 336,
  which requires exact equality for a one-ULP cos difference at x=9. It reproduces
  focused and is reported to the section coordinator. Log and exit status are
  /tmp/scaly-207/main-suite-20261111.{log,exit}. Start no further full run until the
  coordinator supplies #368's new head, then run one full suite on the combined trial.
- Review goes to claude-fable-5-2 at high effort, replacing Opus per the updated
  run-wide rules. No push, PR or review yet. The one-ULP failure is owned by a
  separate thread and is a known la015-only failure until its fix merges.
- The symbol checker rejects malloc injected into the no-budget probe. Both Cortex-M
  probes leave only sinf/cosf undefined; the supported boolean diagnostic compiles
  without stdio or undefined symbols when SCALY_PRINTF is empty.
- Reports record object flash/static RAM and caller-buffer requirements. They exclude
  startup, target libm and stack. Zig caches do not recreate Clang's side-file stack
  reports, so this change does not rely on them.
