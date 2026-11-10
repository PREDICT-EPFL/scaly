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


## Review fixes

Sol found three defects, reproduced by the added AD cases before the fixes:
- A rank-1 broadcast tangent mixed its seed axis with operand axes. Tangents now align before
  stored-coordinate sampling; sparse seed batches retain their shared pattern when reshaped.
- Undeclared custom rules supplied absent logical coordinates to stored-slot conversions.
  Custom relations now use the declared storage domains, while retaining custom rows outside
  primal value support. Direct/mapped tests cover both small and 100,000-square patterns.
- An empty stored map window returned a scalar adjoint. It now returns zeros with the outer
  argument's shape, tested through gradient and Hessian.

All 24 sparse AD cases and 73 AD/custom/snapshot cases pass, and mutations detect all three
regressions. The full hook is running on the fixes. Opus's review is still pending; collect it
before the one delta check for each reviewer. Commit and push these fixes, then remove the plan
only after review work is complete. Never merge or run gh stack.


Opus reviewed the current head and independently confirmed the fixed broadcast defect. Its
remaining findings are API source-listing leaks, sparse generated headers that still claim dense
buffers, and derivative results defaulting to CSC instead of inheriting leaf layout. The source
filters and result-layout fixes are implemented and pass 77 focused cases. Cheap notes addressed:
internal AD helper boundaries now use canonical CSR order, CSC permutation uses NumPy lexsort,
and the column-colored sparse-input Hessian fallback will be named with the temporary paths.
The layout field stays in the instance key because the task explicitly requires it, even though
layouts cannot vary within a fully declared Function today. The static typing limit stays noted.

The header refusal is prepared in untracked `sparse_header_refusal.patch`; `git apply --check`
passes and both bad headers were reproduced. It is NOT applied: the maintainer's brief forbids
editing another batch task's files, and the #274 audit assigns aot.py/cpp.py interfaces to #43.
An asynchronous question asks permission for this narrow refusal. Do not apply it before approval.
All other changes are independently testable; the complete hook is running on them.

The coordinator also requests a final-report-only inventory of all #42 choices the sparse roadmap
section left open or that depart from it, with alternatives/reasons, file/function, undo cost and
logical-size work, plus every reviewer disagreement and its resolution. Do not put that inventory
in the PR description. Main review and Sol agreed on the broadcast defect; no explicit reviewer
disagreement has appeared. Sol's low delta check approved all three earlier fixes.


All non-header Opus fixes pass the full suite: 2,848 passed and 52 optional skips, with 2,900
collection entries. The full hook's benchmark smoke is finishing. The static typing limit now
has follow-up #288 in the 0.2.0 backlog, blocked by #42. No main-review/Sol disagreement has
been observed; the main reviewer independently reproduced Sol's broadcast bug and confirmed
its fix. The complete Opus delta check still waits on the header-ownership answer.


The maintainer approved the minimal C/C++ header refusal as an explicit exception, confirming #43
has not started and will replace the guard. Applied the four added lines to aot.py/cpp.py and
removed the temporary patch. Six codegen boundary cases were added; four failed before the guard.
The tests cover sparse inputs, sparse outputs, both header paths including direct C++ rendering,
write_module without partial files, sparse JIT after refusal, and dense exports with sparse internal
calls. Ruff and strict types pass. The final full hook is running before the Opus delta check.
