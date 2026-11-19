# Sparse traversal during lowering

Layer 3 of [#69](https://github.com/PREDICT-EPFL/scaly/issues/69) follows #452, #471 and
#487 on main. The [compiler roadmap](internal/notes/core_compiler_roadmap.md#tensor-core-and-the-loop-compiler)
and the #69 plan fix the rule: fuse a sparse producer only when its consumer walks the same
stored entries. This branch holds the snapshot slot ahead of #372. Do not touch that workspace
interface or the layer 4 criteria.

- [x] Read the tracker handoffs, roadmap, prior evidence and shared-machine rules. Record the start.
- [ ] Add focused differential and structural tests for matching and different traversals and
  a sampled product with explicit masks. Capture their failure on main.
- [ ] Use compact map plans to compare traversals and retain the existing indexed-write and
  carried-loop barriers. Run the focused tests, then show they detect a perturbed gate.
- [ ] Regenerate and audit the snapshots. Compare main and this branch for C size, generation,
  C compilation and workspace. Run the detached pre-merge hook through the shared queue.
- [ ] Open a draft PR against main, get the requested independent review and confirm fixes.
  Remove this plan, mark the PR ready and record the handoff on #69.

The measurements use the same benchmark harness and dependencies on both revisions. They can
run once the focused tests pass. Review waits for the full checks and snapshot audit.
