# #54 sparse LDLᵀ: working plan

Deleted in the last commit before the pull request is marked ready.

## Steps

1. [x] `scaly.linalg.sparse`: sparse `cholesky`, `ldl`, and `solve(assume="pos"|"sym")` with
       `ordering=`, as library Functions over `scan`, driven by `symbolic.ldl`'s tables.
       Entry points in `dense.py`/`entry.py` dispatch on the matrix's pattern; `ordering=` with a
       dense matrix raises `TypeError`.
2. [x] Tests: SciPy to 1e-12 on random, banded, arrow and block-tridiagonal patterns under each
       ordering; status on an unusable pivot; first and second derivatives of the solves against
       finite differences (and an unrolled reference).
3. [x] Lowering: prove in-place carries threaded through inner LOOPs (roadmap rule 1, "covers a
       left-looking LDLᵀ"), so `in_place=("L", "D", "w")` holds and no step copies O(n). Snapshots
       byte-identical; differential test.
4. [ ] Column chunks as a `COND` over widths (devrush's direct kernel); measure code size.
       Deferred with the timing gate: the roadmap keeps the widths open until measured, and the
       measurement needs la015.
5. [x] Docs: docstrings and `docs/guide/linalg.md` state the sparse case; API page.
6. [ ] Checks: `wt hook pre-merge --yes`.
7. [ ] Gate: loop structure against devrush's `_lower_sparse_ldl`. Timings need la015 (unreachable
       from this session): leave open on the PR.

## Notes

- Internal factor form: `Lx` (strict lower, CSC of the permuted matrix, `l_ptr`/`l_rows`) and `d`
  (pivots for LDLᵀ, the diagonal of L for Cholesky). The public factor packs `Lx` and the diagonal
  into a sparse value with the pattern strict ∪ diagonal.
- Loop bodies are generic (tables passed as broadcast operands, length = n), so one procedure
  serves every pattern; the top-level Function per (kind, ordering) bakes the tables.
