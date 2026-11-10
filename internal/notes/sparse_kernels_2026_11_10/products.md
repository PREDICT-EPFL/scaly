# Sparse-sparse products

Issue #47 adds a column accumulator plan beside #46's compact maps. The expression's stored
pattern is still #39's Boolean product. Lowering compresses active columns, matches right-hand
entries to left-hand column bounds, and sorts the requested writes by column. It clears only
the statically known output rows, accumulates each column in order, and writes directly into
the declared output layout. The plan reuses `nest`, `reduce`, `index_at` and stored
reads. Affine tables become index arithmetic; small constant loops can expand in the existing
program passes. There is no scalar-product-pair table and no new intermediate-representation op.

For A of shape (m, k) and B of shape (k, n), scratch space is one m-entry column. Each plan array
has length bounded by A's stored entries, B's stored entries or the requested output entries.
Column pointers describe occupied columns, so empty logical dimensions do not enlarge tables.
The numerical work is the number of contributing scalar products, plus stored-entry clearing
and writing. Duplicate contributions accumulate in increasing shared-coordinate order.

Forward automatic differentiation forms `dA @ B + A @ dB` with stored-pattern tangents.
Reverse mode requests `Cbar @ B.T` and `A.T @ Cbar` on the corresponding operand patterns.
An internal `sampled` attribute on `MATMUL` gives those products exactly the requested type,
without constructing the unsampled Boolean pattern. Forward differentiation preserves this
attribute, including through Hessians; reverse differentiation samples each operand again.
Support uses the declared bound and derivative analysis joins only requested outputs.

`SampledProductPlan` groups requested output rows and stores bounds into each operand's entries.
The kernel scatters a left row's physical stored-slot indices into one dense integer row. It
computes a scalar dot product for each requested coordinate, then clears the mapped slots.
A missing slot gives a zero-trip range around the multiply-add, so an absent zero never
multiplies infinity, while an explicit stored zero still does. No floating cotangent row is
materialized. Scratch is one row of the inner dimension; all tables follow operand and requested
output fill. The former single-use-gather shortcut is removed. Mixed-product lowering and its
differentiation rules are intact. The roadmap's #47 paragraph records this representation.

## Evidence

The new files are `tests/passes/lowering/test_sparse_products.py`, `tests/ad/test_sparse_products.py`
and `tests/integration/test_sparse_products.py`, plus the sampled-pattern verifier case in
`tests/ir/test_sparse_patterns.py`. Their 78 new cases pass. They cover compressed sparse
column and row layouts and a reversed coordinate order, scalar and block lowering, rectangular
and banded patterns, empty rows and columns, empty operands and products, zero inner dimensions,
explicit stored zeros, and structural zeros beside infinity. Two simultaneous forward directions
and both sampled adjoints agree with SciPy, central differences and the adjoint identity.

The weighted-Gram growth gate increases both the dimensions and the row degree. It bounds tables
by six times the combined operand and result fill, bounds all private buffers by stored sizes,
and allows only two stored intermediates plus the two column workspaces before optimization.
The direct product gate permits exactly one private column buffer. A 10,000 by 13,000 operand
times a 13,000 by 17,000 operand has three and two entries respectively; kernel tables have fewer
than 50 entries and scratch is 10,000 entries. The sampled-adjoint gate has a diagonal operand
and a two-column operand: its unsampled adjoint would hold 97 squared entries, but every private
buffer has at most 194 entries. Neither dense matrix nor scalar-product-pair buffers appear.

The repaired reverse gate forbids building that 97-squared Boolean pattern in the first place.
Weighted Gram derivatives with one dense column in G are checked at n=32 and n=64. All product
patterns and buffers follow stored fill, sampled coefficient visits are bounded by 160n across
the gradient and three Hessian directions, and each sampled plan's tables are bounded by six
times operand and requested output fill. A 128-variable diagonal/thin sparse Hessian gate bounds
each intermediate derivative relation by 16n entries, catching an unsampled sparsity join. A
24-variable dense Hessian gate permits seed-by-stored-entry buffers but rejects seed-by-full-
product buffers. A projection with absent coordinates checks both operand gradients, both orders
of second derivatives, and the sparse Hessian against differences.

The ports use only problem definitions from the frozen notebooks:

- `examples/linalg/sparse_kkt_mpc.ipynb`, "The KKT system of linear MPC: assembly, orderings,
  regularization, inertia and sensitivities": the dynamics constraint
  matrix, with its input, current state and previous state blocks. The port checks weighted normal
  equations and `A.T @ A`, gradients in matrix values and weights, a forward Jacobian and a Hessian.
- `examples/linalg/sparse_fem_topology.ipynb`, "Finite elements with sparse matrices as values:
  a conduction tree by topology optimization": small
  linear triangular elements, duplicate assembly entries and the `R K R.T` restriction to free
  nodes. Element matrices are independently computed from affine basis gradients. Values,
  Jacobians and gradients agree with SciPy and differences.

Both ports use main's public expression API. Factorizations, solves, ordering studies, optimization
and plots are outside this kernel task; sparse factorization remains #54.

The first test failed before the change with "sparse-sparse products require sparse product
kernels". Eleven temporary perturbations each failed their targeted gate: replacing multiplication
with addition, clearing to one, ignoring the output permutation, allocating an m-by-n workspace,
inserting scalar-pair tables, inserting logical-column tables, returning zero forward tangents,
doubling a reverse contribution, disabling sampled lowering, and separately corrupting the kernel
under each notebook port. The arithmetic perturbation also fails the stored-zero/infinity and
shared-product gates. All perturbations were removed. The existing sparse-kernel, import and
dense-snapshot checks passed, 209 cases. No C snapshot was regenerated.

Cross-review found that the original gather shortcut avoided the full buffer but still built
the full Boolean pattern, traversed unsampled entries, and lost its buffer saving in forward-
over-reverse. The sampled `MATMUL` mode replaces it. Seven additional in-memory perturbations
fail the repaired gates: restoring the ordinary reverse product, dropping the sample in forward
mode, computing full support, joining unsampled derivative rows, executing an absent factor,
ignoring the declared sample, and corrupting sampled arithmetic. The 126-case focused group and
277 existing support, verifier, sparsity, kernel, import and snapshot checks pass after the fix.

The detached `~/.scaly-suite/run` exited zero after the sampled-mode fix. Its full suite passed with 3,342 tests and 55 solver
skips, followed by benchmark smoke, the documentation build and the private-name check. Types,
formatting and lint passed too. The node-ID baseline includes all 3,397 collected tests.

Cross-review approved the fix. The main review by claude-opus-5-5 at xhigh effort identified the
unsampled-pattern defect; the independent gpt-6.1-sol review at xhigh effort found no numerical
defect. Each model's one low-effort check confirmed the repaired pattern, differentiation and
lowering, with 122 focused cases passing. No issue remains open in #47's scope. The per-seed
code-growth note stays with #225, the temporary root plan is removed before ready, and plain
output-row/write fields replace the misleading reuse of `MapRegion`.

## Counts and generated C

Weighted Gram products use G of shape (2n, n) with d entries per row. The two kernels perform
`2*n*d + 2*n*d*d` multiply-add pairs. Each pair is
one floating multiply and one floating addition in the reference schedule. The resulting stored
pattern has n times (2d - 1) entries. Counts below follow those exact traversals, not timings.
Largest private buffers are stored intermediates of G's size; each kernel's scratch is n entries.

Generated C uses scalar lanes, the generic CPU recipe and the C dialect. Compile time is for
`cc -O3 -fno-math-errno -std=c99 -c` on la015 during the compiler run. It describes this build,
not an isolated timing comparison. No execution timing is claimed.

```text
n    d   stored G  stored result  multiply-add pairs  float operations  table entries  source bytes  largest private  C compile seconds
64   5   640       576            3840                7680              3136           16961         640              0.043
256  5   2560      2304           15360               30720             12544          68081         2560             0.054
128  32  8192      8064           270336              540672            40832          214514        8192             0.074
```

Per-entry seeds, sparse derivative leaves, sparse selection and traversal fusion remain #225,
#48, #359 and #69. Publishing the sparse names remains #307.

The repaired reverse cases use the same scalar C recipe and compile command above. The Gram
case has 2n rows and n columns, with two local entries and one shared dense column per row.
The derivative is with respect to the diagonal weights. Coefficient visits count the sampled
dot-product iterations, before the presence test; they bound its multiply-add count. Largest
private buffers are stored transposes, not dense products.

```text
case       n     sampled result sizes  coefficient visits  scratch row sizes  table entries  source bytes  largest private  C compile seconds
diag/thin  1000  1000                  2000                2                  0              2401          2000             0.032
Gram       1000  5994, 2000            23968               1000, 1000         41965          237659        5994             0.078
diag/thin  2000  2000                  4000                2                  0              2407          4000             0.033
Gram       2000  11994, 4000           47968               2000, 2000         83965          501967        11994            0.112
```

The inherited forward rule still emits one kernel per seed and shares its index tables. Code
size grows with the seed count. #225 already owns replacing those per-seed kernel groups and
stored-slice copies; coloring bounds the seed count in the sparse derivative gates here.
