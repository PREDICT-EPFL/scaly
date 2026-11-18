# Changelog

## Unreleased

Renamed `sc.reduce_max` and `sc.reduce_min` to `sc.max` and `sc.min`, and added
`Expr.max` and `Expr.min`. Renamed `sc.vec` and `Expr.vec` to `sc.ravel` and
`Expr.ravel`. The old names have been removed. Flattening remains row-major.
