# Cortex-M oracle builds, #207

Implements section #24 under `internal/notes/compiler_run_2026_11_10/plan.md` on main.
The branch starts on #368, the typed-entry layer of #30 -> #55 -> #56 -> #207.
Read `~/.scaly-suite/RULES.md` before tests and whenever notified of an update.

Scope: a float32 oracle with a derivative from an existing example, compiled with
`zig cc -target thumbv7em-none-eabihf -ffreestanding`, no undefined symbols beyond
libm, prints removed, zero frame budget. CI records object flash, static RAM and
caller buffers, with stack limits stated. No hardware execution or timing.

References: #207 body, roadmap release scope, milestone checks and dtype decisions,
`compiler_maintenance_context.md`, project architecture, codebase and conventions.
#314 owns `frame_budget`. Incorporate it after merge, never copy its implementation.
Open overlap: #321 edits `.github/workflows/ci.yml` and `ci-platform.yml`.

- [x] Read the brief, rules, tracker, stack contracts and project guidance.
- [x] Reproduce the freestanding build and add the symbol gate with a failing perturbation.
- [x] Add the float32 mapped-stage oracle and CI job with size and memory reporting.
- [ ] Write and execute the embedded guide section with write-docs. Written, site builds.
- [ ] Incorporate merged #314 and rebase once before the first push onto the newest stack base.
- [ ] Run focused checks, then the detached full pre-merge checks through the suite runner.
- [ ] Open the stacked PR, link it, obtain the Opus review and confirm fixes with a delta check.
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
- The no-budget probe compiles with only sinf/cosf undefined. Zero-budget build
  and symbol perturbation await merged #314. Do not treat the probe as final proof.
- The float32 NumPy differential passes and fails after perturbing the stage math.
- `sc.print` rejects float32 in the supplied stack. Filed #373 with source:agent.
  This change tests print removal with a supported boolean diagnostic instead.
- #314 is first in the merge queue, awaiting its combination's checks. Type checking
  and the zero-budget build currently fail only because frame_budget is absent.
