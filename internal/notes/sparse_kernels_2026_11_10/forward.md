# Sparse kernel semantics

Issue #46 implements decision 5 of the core compiler roadmap. Sparse names remain
filtered from the published API until #307 publishes the complete sparse unit.

Absent coordinates are structural zeros. A sparse product visits only stored
coordinates, so an absent zero times infinity contributes nothing. A run-time zero
at a stored coordinate is evaluated and may produce NaN. Constant folding and
support analysis retain the existing real-arithmetic zero simplifications. This
applies to elementwise products and mixed matrix products. The forward tests cover
both traversal directions.

Movement patterns are computed from the stored coordinates at graph construction.
Batch axes share the union of their local patterns, including explicit zeros.
A movement with a scalar or vector result has dense storage. Matrix products
with one sparse matrix and a vector also have dense vector results. A dense
matrix operand counts as every coordinate when the product's matrix pattern is
built, independently of its values.

Lowering partitions compact maps by fixed operand membership. Small regions use
straight-line statements. Contiguous regions are dense blocks. Other regular
regions use affine indices, and irregular regions use constant tables. All loop
forms use LowerCtx's loop compiler. No generated code compares sparse indices.
The sparse buffers' physical order is separate from canonical logical slots.
Input permutations enter access tables, computed outputs write their declared
layout directly, and callers that require canonical values obtain a stored-size
conversion. Calls and maps retain their existing boundary conversions.

Mixed products traverse compact CSR rows, or rows of the transposed sparse
operand for dense times sparse. Empty rows in dense vector results are zero.
Loaded row bounds keep the output coordinates outside an ordered scalar
accumulator. They do not use output lanes or interchange with the output range.
There is one contribution per stored entry and dense output column, with no
expanded contribution table. Sparse-sparse products belong to #47.

Roadmap departure: #46 covers SUM over stored entries. Max and min over stored
entries move to #66 because it introduces REDUCE with kind. Its max/min reduction
must include an absent zero wherever the reduction domain contains an absent
coordinate and must preserve the loaded-bound scheduling contract above.

## Forward validation

The new movement cases fail on the original implementation at construction.
Replacing compact-map values with zero fails 14 numerical cases. Restoring the
old mixed-product lowerer fails all four loaded-bound/table gates. Reversing the
output layout map incorrectly fails the declared-layout case. Removing the
widening guards fails the accumulating-scatter, carried-range and loaded-bound
contracts individually. Every perturbation was removed before validation.

The focused compiler checks passed 349 tests, including dense C snapshots,
operation coverage, import layers, public names and existing sparse AD.
n=100, nnz=298, columns=7: 2086 multiplies and 2086 accumulator additions; 697 index-table entries before program optimization; 4992 generated C bytes; C compilation 0.026 s (cc -O3 -fno-math-errno -std=c99 -c).
