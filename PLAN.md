# Sparse function boundary, issue 42

Branch `t3code/42-sparse-boundary`, based on `t3code/41-full-size-work` (PR #286).
This is layer four of the five-layer sparse boundary stack. PR base stays the #41 branch.
The decided contract is `internal/notes/core_compiler_roadmap.md`, decision 4, sparse tensors,
and rules for every item. PR #274 holds the storage-size audit; #42's discussion records the
pattern-copy handoff. #277's custom-rule rows and #286's resource gate remain binding.

## Ownership and dependencies

- Root: IR builders, verification and support; storage-aware lowering; exports and API filtering;
  coverage, collection baseline and complete validation; commits, PR and review.
- Boundary agent: function declarations, instance keys, calls/maps, numerical sparse inputs and
  results, JIT and `ir/layout.py`, with function tests.
- AD agent: differentiation rules and call/map derivative storage audit, with AD tests.
- Agents share the canonical row-major internal stored order. `Tree.layouts` supplies leaf layouts;
  lowering converts between canonical values and declared ABI order. PACK consumes dense storage
  shape and VALUES exposes it. Public layout belongs to a leaf, never its TensorType or pattern.

## Work

- [x] Read project instructions, design, issue discussion and prerequisite PR descriptions.
- [x] Confirm clean linked worktree on the requested stack base.
- [x] Reproducing tests for constants/builders, layouts/numerical boundary, calls/maps and AD.
- [x] PACK/VALUES, sparse constants, explicit dense conversion and support compaction.
- [x] Storage allocations, logical-to-slot maps, declared-layout calls/maps, sparse arithmetic.
- [x] Audit every #42-owned entry in PR #274, including lowering ctx and stored seed windows.
- [x] Numerical SciPy/NumPy proof, empty patterns, batches, unequal patterns and explicit zeros.
- [x] Show new gates fail under perturbation; retain #41 and custom-rule regression gates.
- [x] Full merge hook, byte-identical C snapshots, public-name/API filtering and examples.
- [ ] Push draft PR, link in T3, independent Opus and Sol review, fix and delta checks.
- [ ] Remove plan, mark ready, update issue/project and hand back with PR evidence.

No guide publication until #46/#47. Generated C/C++/CasADi sparse interfaces belong to #43;
pattern holes #44, optimized sparse kernels and sampled adjoint #46, sparse-sparse products #47,
and first-class sparse derivative outputs #48 remain outside this change. Temporary stored-entry
execution paths are identified in the PR. No snapshots will be regenerated without the serialized
procedure and measurements. No merge, default-branch push, tags or gh stack.

## Current evidence

Root builder/lowering tests and boundary/AD tests pass. All seven C/header snapshots stay
byte-identical. The first full suite exposed two scalar-view failures in the mapped-view rewrite;
they are fixed and all four benchmark kernel gates pass. The missing PACK/VALUES differential
classification is fixed, and all 1,097 AD tests pass. Strict types, Ruff, import/public-name/op
coverage checks and docs build pass. Rendered API core/functions contain none of the unpublished
sparse names. Function-composition, mapped-derivative and print examples run.

In-memory perturbations detect logical-shape private allocations, wrong stored-coordinate reads,
lost numerical layout permutations, full-base view enumeration, PACK tangents, VALUES transposes
and sparse mapped logical/slot confusion. No perturbation remains.

Mixed sparse MATMUL lowering enumerates contribution tables in stored order; elementwise lowering
uses stored maps and a zero slot for absent entries. These are temporary #46 paths. Mixed sparse
MATMUL AD uses explicit dense projection before sampling its adjoint, replaced by #46. Two sparse
MATMUL operands still refuse lowering, owned by #47. Other unsupported sparse movement/print ops
still refuse construction. Identity RESHAPE wrappers preserve sparse lowering hints.

Public seeded/custom derivative layouts are complete. The final collection has 2,881 tests.
All 41 boundary tests, 1,097 AD tests, 102 import/public-name/op-coverage tests and required
#41/custom-rule gates pass. The complete merge hook is running. The final full hook must use
the current collection baseline before the draft PR and independent review.

The full hook passed. Final suite: 2,829 passed and 52 optional skips. Draft PR #287 is open
against the requested base and linked in T3. Independent Opus xhigh and Sol xhigh reviews follow.
