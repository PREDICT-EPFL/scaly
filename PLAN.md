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
- [ ] Measure repeated gate runs and calibrate fixed cross-platform limits. Check cached-support
  memory with a self-contained growing-sum reproduction.
- [ ] Run the merge checks and gate perturbations. C snapshots must stay byte-identical.
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
