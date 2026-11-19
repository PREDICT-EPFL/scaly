# Typed Function workspace

Implement [#372](https://github.com/PREDICT-EPFL/scaly/issues/372) on
`t3code/372-typed-workspace`. Decision 11 and the
[design comment](https://github.com/PREDICT-EPFL/scaly/issues/372#issuecomment-7006776424)
define the generated C interface.

- [x] Read the design and related changes. Record the start and settle the four interface choices.
- [x] Reproduce the float32 mask frame on main. Pack spills and callee scratch per dtype.
- [x] Render the typed C and C++ entries, the CasADi byte adapter, the solver bridges, and JIT allocation.
- [x] Update the guide, API and existing callers. The mask test reported 8,192 bytes on main and
  4,096 with bool spilling disabled. It reports zero and matches NumPy with the fix.
- [x] Run the Cortex-M4F/M7 builds, the strict-aliasing CasADi and PIQP bridge compile tests,
  and the C, C++ and CasADi guide examples. Regenerate the local snapshots.
- [x] Measure source size, generation, compile time, workspace and frame bytes. See
  `benchmarks/results/workspace-372/{before,after}/measurements.json`.
- [ ] Pass the detached pre-merge checks on la015.
- [ ] Have Opus review the public interface and snapshot diff. Check any fixes at low effort.
- [ ] Stop with the branch local until #69 layer 3 merges.
- [ ] On the coordinator's signal, rebase onto main, regenerate and remeasure the snapshots, rerun
  the full checks, and confirm the review delta. Remove this plan, publish the branch, open and link
  the pull request, and record the handoff on #372.

The lowering check that rejects an all-float32 host calling a float64 extern with scratch remains
with [#474](https://github.com/PREDICT-EPFL/scaly/issues/474#issuecomment-7007249552).
