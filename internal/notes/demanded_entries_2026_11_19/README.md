# Demanded entries, #75

The base is main after #505 (`5242034b`). Lowering walks back from the
outputs and records, for each value, the entries its readers load. A value
evaluated where it is read is computed at exactly those entries. A stored value
is computed only there when the narrowed loop pays for itself: always for a
product or an axis reduction, for an elementwise operation when it skips a
third of the entries or they are evenly spaced, and for a copy or a placement
(a scatter into zeros, or a sum of them) only when it needs no index table that
the whole loop did not already need. A sum of placements also zeroes only the
read entries that its first placement does not overwrite, when those are evenly
spaced. Paths that already compute a known support (stored patterns, forward
helpers specialized to seed supports, reductions over a known support) keep
computing all of it. Both choices are posted on #75.

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
| chain M=5 | 7,906 / 7,878 | 790,281 / 788,826 | 9,243.4 / 9,269.8 | 3,168.5 / 3,135.9 | 66,744 / 66,744 |
| race_cars 5 | 1,178 / 1,178 | 73,548 / 73,548 | 812.9 / 816.8 | 226.2 / 227.2 | 0 / 0 |
| npmpc_jac 100 | 1,067 / 1,067 | 64,593 / 64,593 | 945.8 / 938.3 | 515.0 / 506.9 | 0 / 0 |
| npmpc 100 | 1,506 / 1,493 | 134,005 / 133,543 | 1,175.8 / 1,178.7 | 907.1 / 897.0 | 19,744 / 19,744 |
| npmpc_jac 6 | 1,067 / 1,067 | 64,427 / 64,427 | 777.2 / 739.5 | 237.5 / 235.8 | 0 / 0 |
| chain_jac 5 | 2,646 / 2,646 | 157,020 / 157,020 | 3,058.0 / 3,049.1 | 1,040.6 / 1,035.0 | 20,160 / 20,160 |
| KKT assembly | 89 / 89 | 3,730 / 3,730 | 7.4 / 7.3 | 51.1 / 50.6 | 0 / 0 |

race_cars 5, both neural model predictive control (MPC) Jacobians, the paper
chain's sparse Jacobian and the KKT fixture are byte-identical to main. The
#69 gates hold: chain M=5 below 100,000 doubles, race-car `SZ_W` zero, the
neural MPC Jacobian at zero for N=6 and N=100.

`narrowed.json` counts the stored values computed at fewer than all their
entries. In chain M=5 they are ten sums of placements (13,548 of 25,626
entries) and one placement (3 of 504). The dense Hessian assembly zeroes 3
entries instead of 23,544, since its first placement stores every other
read entry, and the objective's diagonal placement stores 3 of its 72 values.
In npmpc 100, two sums of placements (5,011 of 9,835 entries) and two matrix
products (2 of 8 entries). The other placements' read values sit at irregular
places, so storing only them would need a table, and they stay whole. Removing
that work needs the reader to take the placed values directly, without the
dense buffer, which is a follow-up.

A first version narrowed placements and copies under the two-thirds rule. It
grew chain M=5 to 972,999 bytes (three new index tables, 28,203 entries) and
npmpc 100 to 185,786 bytes, with no work saved worth a table read per value.
That is why copies and placements now narrow only without new tables.

## Snapshots

`uv run python -m tests.test_c_snapshot` regenerated the 21-file corpus.
`snapshot-audit.json` records both digests of every file: all are
byte-identical to main. An earlier version changed `spjac.c` by zeroing the
sum through a 12-entry table, which the final rule rejects.

## Tests and perturbations

`tests/passes/lowering/test_demand.py` compares compiled C with NumPy and
counts floating-point operations on the lowered program, each once per loop
trip. All 13 cases fail on main on their counts (for example 2,952 multiplies
where 527 x 3 are expected in devrush's 984-versus-527 case, and 24 sines where
4 are expected behind a repeating gather), and pass here.

`prove.py` applies one perturbation at a time to a copy of `src/`.
`perturbations.json` holds every outcome.

| Perturbation | Fails |
|---|---|
| Never narrow a stored value | 12 of 13 on their counts. The concatenation guard keeps its part whole either way |
| Map a transpose's demand through its own axes | the transpose family, on values |
| Demand a scatter's values that miss the read entries | the scatter family, on values |
| Demand the wrong columns of a product | the sampled-adjoint case, on values |
| Evaluate a computed part inside a selected concatenation | the concatenation guard |
| Narrow an elementwise value through any table | the concatenation guard |
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
