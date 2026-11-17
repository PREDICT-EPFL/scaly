# Lowering fusion, layer 2 of #69

Base: origin/main at cd2b6c75. Branch: t3code/69-layer2-consumer-agreement.
This is step 1 of the maintainer's strict six-step sequence, with main as the PR base.
Do not merge, queue, push main, or create tags.

## Context and boundaries

Read #69 and its four-layer plan and schedule, #452, the compiler roadmap's
Tensor core and the loop compiler and Rules for every item, the compiler maintenance
context, the rangeify study of 2026-09-08, the architecture and developer guides,
and ~/.scaly-suite/RULES.md. The implement, inspect-issues, update-issues,
write-plan, cross-review and unslop skills apply.

Layer 1 already chose scatter sums at lowering and deleted combine_scatter_sums,
fuse_elementwise including inline_producer, and fold_arith._constant_fills.
Remaining work is consumer agreement and index divisions in non-elementwise readers.
Constant data is already read directly by LowerCtx.read, with whole-buffer fills
chosen by lowering/movement.py. No program fill or scatter matcher remains.

Do not change linalg, plans._guard_sampled_term, sparse traversal, demanded entries,
lanes, lifetime packing, the KKT gate or #166's criteria. Preserve carried ranges,
invariant hoisting and read-before-overwriting-scatter guards.

## Work, sequential

- [x] Inspect checkout, prerequisites, open PRs and the decided design. No open PRs;
  all four prerequisites closed. Project #2 has no status value for #69.
- [x] Record started status and capture main source, generation, C compilation and
  workspace measurements for chain M=5, race_cars 5, npmpc_jac 100 and npmpc 100,
  plus npmpc_jac 6 workspace. These are diagnostics on a shared machine, no runtime claims.
- [x] Add failing NumPy differential regressions for agreeing two/three-consumer
  reads, independent outputs, and disagreeing coordinates/domains.
- [x] Implement lowering-time agreement and eliminate deferred non-elementwise
  index divisions. Verify narrow tests, retained carried/write guards and perturbed gates.
- [x] Regenerate snapshots and the test node list. Explain every snapshot hunk and
  capture matching after measurements. Keep evidence in internal/notes.
- [x] Run all pre-merge checks through the detached shared suite runner.
- [ ] Commit, push once and open/link a draft PR. Opus 5.5 high reviews code and
  every snapshot line; fix findings and use one Opus low delta check.
- [ ] Remove this plan at completion, finish the PR evidence and verdict, mark ready,
  record review status, and report evidence, measurements, open points and follow-ups.

## Validation requirements

Focused pytest uses at most two workers. Full suites run only through
~/.scaly-suite/run, detached with a log and exit file. Use REST for GitHub reads.
The PR ends Part of #69. Every new gate must fail under a recorded perturbation.
No runtime timing or benchmark study. Follow-up issues use source:agent.

## Current proof and status

The focused lowering, mapped-range, lanes, snapshots, typed-entry and loop-reverse
checks pass 540 cases. A new crossed-group cycle test also passes; the complete
test-node collection now has 5,469 entries. Six benchmark kernel gates pass.
Agreement, coordinate/domain guards, range splitting and group-dependency closure
were each perturbed and failed their gates, then restored. Source and caller
workspace values remain as recorded in internal/notes/lowering_fusion_layer2_2026_11_17/.

The first full hook found one obsolete spilling fixture and a scheduling cycle
between two merged groups, causing one worker to grow to 20 GB. That worker was
terminated. Both causes are fixed. The fixture uses disagreeing coordinates now,
and group planning takes the transitive dependencies of prior merges into account.
The existing reverse-over-reverse test passes under a 3 GB limit. The second full
hook is detached, with log and exit file at benchmarks/results/layer2/pre-merge-final.*.
No PR has been opened and no branch has been pushed yet. Project status is In progress,
verified through REST with the Status field explicitly requested.

The second full hook passed, exit 0: 5,141 passed, 16 skipped, 13 smoke checks,
types, Ruff, docs and private-name checks. Four final source/workspace gates pass.
