# Issue 41: remove work over unstored matrix entries

This is layer 3 of the sparse stack, based on `t3code/40-support`. The implementation follows
the sparse-tensors section and rules in `internal/notes/core_compiler_roadmap.md`, issue #41's
discussion, PR #274's storage audit, and the maintainer's dispatch. Sparse leaves, buffer sizing,
calls/maps, scalar zero skipping and sparse kernels remain with their owning tasks.

- [x] Read the issue, prerequisites, design, project conventions and surrounding implementation.
- [x] Add the isolated 100,000 by 100,000 / 1,000,000-entry gate and small NumPy differential cases.
  Establish failures before fixing each path.
- [x] Make uniform constants, folding/hashing, coordinate maps, support and derivative incidence
  depend on represented entries. Preserve #40's custom-derivative behavior.
- [x] Remove compiler-time full-shape work exposed by lowering, keeping logical buffer sizes for #42.
- [x] Measure repeated gate runs and calibrate fixed cross-platform limits. Check cached-support
  memory with a self-contained growing-sum reproduction.
- [x] Run the merge checks and gate perturbations. C snapshots must stay byte-identical.
- [ ] Commit and push, open a draft PR against `t3code/40-support`, obtain cross-review, address
  findings and check the fixes. Delete this plan before marking ready, then update tracking.

Each phase depends on the preceding implementation; independent checks may run together. Rebase
onto an updated #40 head only when the coordinator requests it, using `git rebase --onto`.

Devrush's sparse-selection and constant-table tests were inspected. They cover sparse leaves,
selection APIs and target alignment, which belong to other tasks. The gate and coordinate-map
cases here use the current scatter/gather API and independent NumPy references.

## Evidence so far

The original gate failed on `zeros_like` allocating 74.5 GiB, then on the shape-only slice
allocation. The final gate additionally exercises compact Jacobian coloring/recovery, sparse
cotangent slice transposes and reverse broadcasting. Small numerical cases cover repeated
scatter coordinates, movement, products, both derivative modes and the added reverse paths.

Eight isolated perturbations were detected: full constant serialization, dense scatter folding,
logical-row CSR conversion, full transpose and slice aranges, erased matmul dependencies, a
one-byte memory limit and a one-millisecond subprocess timeout. Replacing compact support with
frozensets also fails the cache representation test. All perturbations were in memory only.

The first complete merge hook passed 2,725 tests with 52 optional skips, smoke, docs, both Ruff
checks, types and private-name checks. The final merge hook is running after the added coloring,
reverse paths, uniform-lowering numerical test and refreshed node-ID baseline. Snapshot checks
still pass byte-for-byte. Strict Windows and Python 3.12 type checks pass.

#40's custom derivative declaration/filter logic is unchanged. Its surrounding mask annotations,
row storage and composition now use COO. Call/map consumers explicitly request CSR only at their
existing boundary when they need its indexed access; sparse call layouts remain with #42.

The final merge hook passed 2,726 tests with 52 optional skips, benchmark smoke, docs, strict
`ty`, Ruff lint/format and private-name checks. The completed workload's three fresh runs were
29.151, 23.897 and 25.970 seconds, with 352,257,925 to 352,258,083 traced bytes and 777,121,792 to
777,441,280 peak resident bytes. The 180-second / 2-GiB limits provide at least 6.1x time and
2.7x resident-memory headroom, including a run overlapping the full suite. The four-platform CI
run must confirm portability. Coloring and original reverse-broadcast/slice perturbations are
also being checked against this expanded gate. The original matmul enumeration times out in a
separate three-second reproduction that the current implementation passes.

## Review fixes

Sol reproduced lost custom derivatives when cotangent values were compacted by value support,
scalar coordinate-map failures and a signed-zero SUM difference. New tests failed on all seven
custom derivative cases, scalar movement, and uniform-first negative zero before the fixes.
The reverse compaction guard traverses operands and ordinary callees iteratively and refuses to
remove entries if custom rules can contribute derivative paths. The derivative relation row
domain itself remains the rows present in COO incidence, including custom declarations/dense
patterns and all downstream images; it never comes from the differentiated node's value support.
The maintainer's #40 investigation memo was read without edits. Its eight-case composition test
passes. New direct/nested/mapped reverse-map tests and a public Hessian reproduction pass.
Scalar coordinate maps retain one-dimensional arrays; uniform SUM obtains NumPy's additive-zero
behavior before scaling. Windows peak-memory queries now share one ctypes structure type across
the watchdog and main thread, avoiding concurrent argtype changes between distinct class types.
The 197-test focused check passed, including snapshots and the gate. The full suite rerun passes 2,736 tests with 52 skips; the remaining merge-hook checks
are in progress. Opus's independent review is still pending.

## Main review follow-up

Opus reproduced scalar-broadcast full-size expansion, repeated custom-rule ancestry walks,
quadratic concat/stack assembly, small scatter-folding regressions, support unions escaping
compact storage, and constant construction-order dependence. Its randomized 12,600-case check
found no support/incidence value difference from the base. Remaining work:

- [x] Gate scalar scaling and symbolic scalar coefficient differentiation without broadcasting
  coefficient support/masks across the logical matrix.
- [x] Cache custom-rule reachability per immutable node.
- [x] Batch concat/stack coordinates once and enforce compact storage for every set result/map.
- [x] Normalize constant storage at construction, restore small scatter folds and copy context,
  make small support iteration deterministic, and account for dtype/C changes.
- [x] Add failing regressions, run all merge checks and recalibrate the enlarged gate.
- [ ] Push the fixes, request one Opus delta check covering its findings, then finish the PR.

The updated #40 head currently conflicts only in the two temporary PLAN.md files. Source and
node-ID changes merge cleanly. Do not rebase until the coordinator requests it.

Main-review fixes pass 178 focused tests and 6,300 additional randomized support/incidence
comparisons against the base. All C snapshot cases pass. The small scatter gradient,
Hessian and Jacobian sources are byte-identical to the base reproduction again. Perturbations
restoring repeated ancestry walks and per-argument concat/stack unions fail the new work-bound
checks. The full suite passes 2,746 tests with 52 skips; final smoke/docs are finishing.

Three enlarged-gate processes took 48.499, 29.478 and 31.177 seconds. Peaks were 342,661,862 to
342,662,253 traced bytes and 834,031,616 to 834,637,824 resident bytes. The slowest overlapped the
full suite. Existing 180-second / 2-GiB limits retain 3.7x time and 2.5x resident headroom.
Constant storage now normalizes uniform values before interning, and the two generic array
helpers own physical data and uniform detection. Small constant scatter folds remain enabled.
Small support iteration is sorted; set differences/XOR work, and hashing is cached with the same
hash as an equal frozenset. This also serves #40's newly added support-bearing buffer metadata
without editing its files. Reverse zero/one helper dtypes remain float64, matching the base.
After review, removing our completed PLAN.md will naturally clear the current PLAN-only merge
conflict without a rebase or edits to #40's planning file.

The complete merge hook exited successfully after smoke, docs and private-name checks.
