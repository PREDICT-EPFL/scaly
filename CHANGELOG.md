# Changelog

## Unreleased

`scaly.linalg.solve_triangular` takes a sparse triangular matrix, such as the `L`
of a sparse `ldl`, and `scaly.linalg.permutation` returns the ordering that a
sparse `cholesky` or `ldl` applies, so that one sparse factorization serves
several solves.

`scaly.linalg` takes float32 matrices as well as float64 and computes in the
matrix's dtype, with the pivot test in that dtype's machine epsilon. The status
of `solve`, `cholesky`, `ldl` and `lu_factor` is now int32 instead of int64, so
an output declared for it needs `sc.dtypes.int32`.

Renamed `sc.reduce_max` and `sc.reduce_min` to `sc.max` and `sc.min`, and added
`Expr.max` and `Expr.min`. Renamed `sc.vec` and `Expr.vec` to `sc.ravel` and
`Expr.ravel`. The old names have been removed. Flattening remains row-major.
