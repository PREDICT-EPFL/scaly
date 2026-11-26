# Residual forward rules for the dense solves (#491), 2026-11-23

`measure.py` was run on `main` at the merge of #519 (`measurements-before.json`) and on the branch
(`measurements-after.json`), on an Intel i7-10700 workstation, not the reference machine. Times are
medians of repeated calls of the compiled function, so they compare before with after only.

Jacobian of `x` in `K`, generated C bytes:

| assume | n = 4 before | after | n = 16 before | after |
|---|---|---|---|---|
| pos | 250,581 | 109,513 | 612,068 | 413,861 |
| sym | 311,655 | 121,618 | 915,834 | 434,630 |
| gen | 411,147 | 71,384 | 1,804,554 | 410,755 |

Evaluation time, ms:

| n | solve(pos) Jacobian in K, before | after | cholesky Jacobian in A, before | after |
|---|---|---|---|---|
| 8 | 0.188 | 0.061 | 0.152 | 0.130 |
| 16 | 3.09 | 0.41 | 2.47 | 2.23 |
| 24 | 20.95 | 1.89 | 15.07 | 12.72 |
| 32 | 79.67 | 6.56 | 67.74 | 62.00 |

The Cholesky Jacobian gains little: its rule still does two triangular solves of an n x n matrix
per direction, n^3 work against the n^3/3 of the factorization it no longer repeats.

Factorizations run per evaluation, counted with `tests/linalg/test_entry.py::_runs` at n = 4: a
solution and its Jacobian in K, 17 before and 1 after for every `assume`; the Hessian of a cost in
K, 51 before and 3 after. The remaining two are the forward helpers of the residual results, which
recompute the call they differentiate (#90).
