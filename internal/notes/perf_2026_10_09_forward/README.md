# Issue 19 forward traversal measurements

This compares the pushed `t3code/issue-18-ad-calls` base with the pushed
`t3code/issue-19-one-forward-traversal` branch. Both run in separate, clean
benchmark worktrees on la015. The question is how the leading seed axis changes
generated source, generation time, native evaluation time and workspace.

The protocol is `internal/notes/benchmark_protocol.md`. Both revisions use the
performance governor, disabled boost, glibc vector math, the same samples and
Scaly backend, five fresh processes with empty compilation caches, order seed 0,
a 180-second compile limit and the 50 MiB source limit. Each native cell runs its
correctness gate before timing. No tests or other builds run during measurement.
The original four cells measure sparse Lagrangian Hessians. The two additional
Jacobian cells cover the structure gates affected by this change.

The before worktree is `/tmp/scaly19-benchmark-before`. The after worktree is
`/tmp/scaly19-benchmark-after`. Raw artifacts, logs, provenance, generated C and
native binaries live in their `benchmarks/results/issue19-before` and
`benchmarks/results/issue19-after` directories. This directory retains the raw
and summary comma-separated value files under `data/{before,after}/sweep/`.
`comparison.md` comes from `uv run internal/notes/perf_2026_10_09_forward/compare.py`.
Generation time includes derivative construction and C rendering. The raw
records also give these two times separately. Native time comes from Google
Benchmark. CV is the sample coefficient of variation across five processes.
The retained provenance names each revision by branch. The original sidecars,
including the transient branch commit identifier, remain with the raw artifacts.

All 60 original headline attempts and ten rerun attempts passed their correctness
gates and native timing. There were no compile failures, timeouts or skipped
measurement cells. Before race-car generation had 5.5% CV, and after neural MPC
Jacobian generation had 5.3% CV. Both complete five-process cells were rerun under
the same policy. The comparison uses those reruns, whose generation CVs are 2.1%
and 1.3%. `data/*/original/` retains the first attempts. Selected native CVs range
from 0.2% to 3.5%.

All six selected kernels run slower. The four Hessian increases are 11.5% for
race cars, 18.2% for chain, 20.5% for unbumpercars and 13.1% for neural MPC. The
chain and neural MPC Jacobians increase by 77.2% and 61.1%. These are measurements
of the listed sizes. They do not establish the cause or performance at other
sizes. The runtime regressions are another maintainer open point. Batched matrix
products in #67 and loop fusion in #69 are relevant follow-ups, but no recovery
claim is made for either.

The agent launched these commands in named tmux sessions, after `uv sync`, smoke,
and a one-repetition toolchain check outside the measurement directory:

```sh
PYTHONHASHSEED=0 SCALY_VECTOR_LIBM=glibc SCALY_CC=gcc CC=gcc CXX=clang++ uv run benchmarks/run.py sweep --workloads WORKLOAD --sizes SIZE --backends scaly --repetitions 5 --headline --boost off --order-seed 0 --out benchmarks/results/issue19-REVISION/WORKLOAD.csv
uv run benchmarks/run.py report benchmarks/results/issue19-REVISION
```

Cells are `race_cars:5`, `chain:3`, `unbumpercars:2`, `npmpc:4`,
`chain_jac:33` and `npmpc_jac:100`. No published benchmark page changes.

## Snapshot diff

Only `tests/baseline/c/jac.c` changes. Its header and all other C snapshots stay
identical. The four directions now read the shared primal and partial using a
broadcast index modulo two. The old copies of `vel` and `2 * vel` disappear,
along with their four copy helpers and the unused two-lane type. Buffer packing
shrinks `s0[8]`, `s1[8]`, `s2[16]`, `s3[1]` to `s0[4]`, `s1[16]`, `s2[1]`.
The remaining lane helpers and buffers change numbers. The seed tables, output
order, reduction order and exported application binary interface stay the same.

## Points for the maintainer

The rewritten chain gate and increased neural MPC workspace need maintainer
acceptance. Neither decision is implied by a passing test.

The old chain gate compared the whole Jacobian at five and 33 masses. With the
new traversal those kernels have 1458 and 2473 lines, compared with 5698 and 2754
before. The small case crosses the automatic scalarization cutoff, and the
whole source includes explicit scalarized boundary rows. The new gate measures
the forward mass helper bodies at 33 and 65 masses, preserving the original
requirement that the mass loop stays a loop. Restored bodies have 828 and 827
lines. Forcing those helpers to use scalar expansion produces 10644 and 21428
lines and fails the gate. The perturbation was removed.

Neural MPC fixed workspace changes from zero to 6144 doubles at horizon six.
At horizon 100 total workspace changes from 1600 to 7744 doubles, leaving the
horizon-dependent part at 1600. Both allowed folds for a matrix-vector tangent
were tried: `dY @ A.T` and `(A @ dY.T).T`. They put transposes in different
places but leave the same fixed workspace with the current lowering passes.
The batched intermediates survive as buffers that the previous scalarized
per-seed bodies eliminated. Removing them requires more loop fusion or a
different contraction lowering. This change keeps the decided traversal and
uses tight bounds of 6144, 7744 and 1600 for fixed, total and horizon growth.
It does not preserve the previous workspace limit of 4800 at horizon 100.
