# Sparse-sparse products

Issue #47 adds a column accumulator plan beside #46's compact maps. The expression's stored
pattern is still #39's Boolean product. Lowering compresses active columns, matches right-hand
entries to left-hand column bounds, and sorts the requested writes by column. It clears only
the statically known output rows, accumulates each column in order, and writes directly into
the declared output layout. The plan reuses `MapRegion`, `nest`, `reduce`, `index_at` and stored
reads. Affine tables become index arithmetic; small constant loops can expand in the existing
program passes. There is no scalar-product-pair table and no new intermediate-representation op.

For A of shape (m, k) and B of shape (k, n), scratch space is one m-entry column. Each plan array
has length bounded by A's stored entries, B's stored entries or the requested output entries.
Column pointers describe occupied columns, so empty logical dimensions do not enlarge tables.
The numerical work is the number of contributing scalar products, plus stored-entry clearing
and writing. Duplicate contributions accumulate in increasing shared-coordinate order.

Forward automatic differentiation already forms `dA @ B + A @ dB` with stored-pattern tangents,
so it needs no new rule. Reverse mode projects `Cbar @ B.T` and `A.T @ Cbar` onto the corresponding
operand patterns. A single-use sparse product read by a gather lowers directly into that gather.
This avoids allocating the unsampled product, including when its Boolean pattern is full.
For a sampled product, all potentially updated accumulator rows are initialized once, then only
the requested rows are cleared between columns. Unrequested rows may retain accumulated values;
they never contribute to a requested row. Shared products and products returned as outputs keep
their ordinary materialization. The new recognition in `ctx.py` and `gather.py` applies only to
products of two sparse operands. Mixed-product lowering and its differentiation rules are intact.

## Evidence

The new files are `tests/passes/lowering/test_sparse_products.py`, `tests/ad/test_sparse_products.py`
and `tests/integration/test_sparse_products.py`. Their 69 cases pass. They cover compressed sparse
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
under each notebook port. All perturbations were removed. The existing sparse-kernel, import and
dense-snapshot checks passed, 209 cases. No C snapshot was regenerated.

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
