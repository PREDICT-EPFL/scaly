# #509: scaly.linalg follows the matrix dtype, int32 statuses

Blocked by #511 (dtype holes on Function leaves): the module-level Functions in
`src/scaly/linalg/dense.py` declare leaves without a type, which are always float64.

Done:
- `tests/linalg/test_dtypes.py`: float32 against float64 per routine and batch shape, no float64 in
  the lowered program, the pivot test in float32's epsilon, derivatives against float64 finite
  differences, mixed dtypes refused, Python right-hand sides take the matrix dtype.
- `dense.py`: constants, masks and the pivot epsilon take the matrix dtype; statuses are int32;
  `_dense` takes float32 or float64; `_right_hand_side` coerces by #25's rule.

Remaining, once #511 lands:
- declare the float leaves of every Function in `dense.py` with a dtype hole
- update the int64 statuses and "float64 matrix" checks in `tests/linalg/`
- add a float32 `solve(assume="pos")` to the Cortex-M build in `tests/codegen/test_embedded.py`
- docstrings, `docs/guide/linalg.md` and the changelog
