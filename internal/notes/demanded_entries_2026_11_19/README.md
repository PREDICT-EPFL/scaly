# Demanded entries, #75

The base is main after #505 (`5242034b`). Lowering walks back from the
outputs and records, for each value, the entries its readers load. A value
evaluated where it is read is computed at exactly those entries. A stored dense
value is computed only there when the narrowed loop pays for itself: always for
an axis reduction and a product with a vector operand, for an elementwise value
that calls libm when it skips a third of its entries, and for anything else,
matrix-matrix products and scatters included, only when the narrowed loop needs
no index table the whole loop did not already need. A sum of scatters into
zeros also zeroes only the read entries that its first scatter does not
overwrite, when those need no table. Sparse values and forward helpers that
plan by seed support keep computing all they store. Both open choices are
posted on #75.

The branch was later rebased onto `04a2bf53`. On both sides every measured
source and header kept its SHA-256, and the snapshots stayed byte-identical to
main, so the measurements below stand.

The first cost rule narrowed every product and every elementwise value under
two thirds of its entries. The code review timed it on la015 (indicative, not
a benchmark study): a table-driven 4 x 3 by 3 x 246 product was 3.2 times
slower than computing it whole at 527 of 984 entries and 1.36 times faster at
10%, and `a*b + a` was 1.5 to 2.1 times slower at 30% to 60%, while a narrowed
row sum and a matrix-vector product were about 4 times faster at 54%. The rule
above follows those numbers.

A selected concatenation (layer 4 of #69) now reads a computed part from
storage instead of evaluating it at every gathered entry, which is devrush's
second guard. An unselected part is read at an entry it computes.

## Measurements against main

The before process imports main's `src/` from a source archive, the after
process this checkout. Both use this checkout's harness, problems, plugins and
lockfile. One observation per cell on shared la015, Clang 20.1.8 at
`-O3 -march=native -fno-math-errno -fveclib=libmvec`, ending at the object
file. Generation covers derivative construction, lowering, rendering and
writing the files. No kernel is timed.

| Kernel | C lines, main / #75 | C bytes, main / #75 | Generation ms, main / #75 | C compile ms, main / #75 | Caller workspace doubles, main / #75 |
|---|---:|---:|---:|---:|---:|
| chain M=5 | 7,906 / 7,878 | 790,281 / 788,826 | 9,243.4 / 9,246.6 | 3,168.5 / 3,156.0 | 66,744 / 66,744 |
| race_cars 5 | 1,178 / 1,178 | 73,548 / 73,548 | 812.9 / 821.6 | 226.2 / 227.5 | 0 / 0 |
| npmpc_jac 100 | 1,067 / 1,067 | 64,593 / 64,593 | 945.8 / 908.2 | 515.0 / 505.8 | 0 / 0 |
| npmpc 100 | 1,506 / 1,493 | 134,005 / 133,543 | 1,175.8 / 1,170.5 | 907.1 / 902.7 | 19,744 / 19,744 |
| npmpc_jac 6 | 1,067 / 1,067 | 64,427 / 64,427 | 777.2 / 743.7 | 237.5 / 238.3 | 0 / 0 |
| chain_jac 5 | 2,646 / 2,646 | 157,020 / 157,020 | 3,058.0 / 3,034.1 | 1,040.6 / 1,045.3 | 20,160 / 20,160 |
| KKT assembly | 89 / 89 | 3,730 / 3,730 | 7.4 / 7.4 | 51.1 / 51.1 | 0 / 0 |

race_cars 5, both neural model predictive control (MPC) Jacobians, the paper
chain's sparse Jacobian and the KKT fixture are byte-identical to main. The
#69 gates hold: chain M=5 below 100,000 doubles, race-car `SZ_W` zero, the
neural MPC Jacobian at zero for N=6 and N=100.

`narrowed.json` counts, per kernel, the stored values given a narrowed
domain and the size of that domain. For a sum of scatters the domain bounds
which values it stores and which entries it zeroes, and leaves the code as it
was when both would need a table, as in race_cars 5. In chain M=5 the dense
Hessian assembly zeroes 3 entries instead of 23,544, since its first scatter
stores every other read entry, and the objective's diagonal scatter stores 3 of
its 72 values. In npmpc 100 two products with a vector operand compute 2 of
their 8 entries. The other scattered values reach the read entries at irregular
places, so they stay whole. Removing that work needs the reader to take the
scattered values directly, without the dense buffer, which is a follow-up.

A first version narrowed placements and copies under the two-thirds rule. It
grew chain M=5 to 972,999 bytes (three new index tables, 28,203 entries) and
npmpc 100 to 185,786 bytes, with no work saved worth a table read per value.
That is why copies and placements now narrow only without new tables.

## Snapshots

`uv run python -m tests.test_c_snapshot` regenerated the 21-file corpus.
`snapshot-audit.json` records both digests of every file: all are
byte-identical to main, and again after the rebase. An earlier version changed
`spjac.c` by zeroing a sum through a 12-entry table, which the final rule
rejects.

## Tests and perturbations

`tests/passes/lowering/test_demand.py` compares compiled C with NumPy,
counts floating-point operations on the lowered program, each once per loop
trip, and runs the lowered program's index arithmetic to check that no load
reads an entry no store wrote, including loads a select discards. All 16 cases
fail on main on their counts (for example 1,581 multiplies expected where main
runs 2,952 in devrush's 984-versus-527 case, and 4 sines where main runs 24
behind a repeating gather), and pass here. The two regression tests from the
review, a scalar product no reader uses and a narrowed scatter's sign of zero,
pass on main and fail without their fixes.

`prove.py` applies one perturbation at a time to a copy of `src/`.
`perturbations.json` holds every outcome.

| Perturbation | Fails |
|---|---|
| Never narrow a stored value | 15 of 16 on their counts. The concatenation guard keeps its part whole either way |
| Map a transpose's demand through its own axes | the transpose family, on values |
| Demand a scatter's values that miss the read entries | the scatter family and the sum-of-scatters case, on values and unwritten loads |
| Demand the wrong columns of a product | the one-column product, on values |
| Evaluate a computed part inside a selected concatenation | the concatenation guard |
| Narrow an elementwise value through any table | the concatenation guard |
| Read an unselected concatenation part at entry zero | the concatenation family, on an unwritten load |
| Skip the zeros under a narrowed sum of scatters | the scatter family, the sum-of-scatters case and the sign-of-zero regression |
| Let a repeating gather read a sum in place | the repeating-gather guard |

## Reproduction

```sh
notes=internal/notes/demanded_entries_2026_11_19
scratch=$(mktemp -d)
mkdir "$scratch/main"
git archive 5242034b src | tar -x -C "$scratch/main"
export PYTHONHASHSEED=0 SCALY_VECTOR_LIBM=glibc
PYTHONPATH="$scratch/main/src:." uv run "$notes/measure.py" "$scratch/before"
PYTHONPATH="src:." uv run "$notes/measure.py" "$scratch/after"
PYTHONPATH="src:." uv run "$notes/narrowed.py" "$notes/narrowed.json"
uv run python -m tests.test_c_snapshot
uv run "$notes/audit_snapshots.py" 5242034b > "$notes/snapshot-audit.json"
uv run "$notes/prove.py" "$scratch/main/src" "$scratch/perturbed" > "$notes/perturbations.json"
```
