# Issue 19: one forward traversal

Base: `t3code/issue-18-ad-calls`, stack position 4 of 7. The decided contract is
`internal/notes/core_compiler_roadmap.md`, One AD engine, #19. Read the architecture,
codebase, conventions, contributing and compiler maintenance notes, forward.py,
calls.py, and #15's differential and joint harnesses. Elementwise formula changes,
reverse cleanup, intermediate wrt and custom rules remain in #20–#22.

- [x] Inspect issue, parent, dependencies, stack and snapshot overlap. Checkout is clean
  at the #18 tip; no other open PR regenerates snapshots.
- [x] Add failing differential tests for multiple outputs/independents, deep graphs,
  typed tangents, structural zero terms, broadcast alignment, and folded matmul.
  Port relevant devrush zero-product and joint-input problems against NumPy.
- [x] Record before source size/generation time and native runtime on all four problems.
- [x] Implement iterative seed-axis traversal; route callee bodies through it;
  retain baking, periodic tiles, local coloring, packing; remove fallback and cap.
- [x] Update #15 harness, refusal diagnostics, stale environment documentation.
- [x] Run focused checks, perturb new gates, regenerate and explain C snapshots.
- [x] Run pre-merge hook, collect after measurements; recheck snapshot overlap.
- [x] Commit/push, draft PR against #18, link stack 160 162 165 and PR #166.
- [ ] Cross-review once, fix findings and check fixes once; record criterion evidence.
- [ ] Delete temporary plan, mark ready, Project In review, hand back PR.

Implementation and validation are sequential. The independent cross-review follows
checks. No implementation delegation is needed.

The workspace itself is on la015. Before and after studies are complete: all 60
headline attempts passed, plus ten complete-cell reruns for generation dispersion.
No measurement remains active. `internal/notes/perf_2026_10_09_forward/` retains
data, provenance, comparison script, reports, gate proof and interpretation.
All six kernels are slower; these measured runtime regressions are also maintainer
open points. Full pre-merge hook passed with 2356 tests passing, three skipped and
two expected failures; smoke and docs passed.

The full-suite failures identified an obsolete buffer-name assertion and two
structure gates affected by the new layout. The chain gate now measures forward
mass helper bodies at M=33 and M=65, above the automatic scalarization cutoff.
Restored bodies have 838/837 lines; forcing their scalar expansion has 10644/21428
and fails the gate. Neural MPC fixed workspace is 6144 instead of 0 doubles;
N=100 is 7744 instead of 1600. Both permitted matrix-vector folds were tried and
retain this fixed workspace. Tight gates pin 6144 and 7744 and growth to 1600.
The rewritten chain gate and workspace increase remain explicit maintainer open
points, as the coordinator requested. No acceptance decision is implied.
