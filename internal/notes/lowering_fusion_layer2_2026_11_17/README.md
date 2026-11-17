# Lowering fusion, layer 2 of #69

Layer 1, PR #452, already chose sums of scatters in lowering, deleted the
`combine_scatter_sums` matcher, deleted `fuse_elementwise` and its `inline_producer`,
and deleted `fold_arith._constant_fills`. Main has no remaining program-level
scatter-sum or constant-fill detection. Uniform constants are scalar reads in
`LowerCtx._step`; a whole-buffer constant fill is chosen in `lowering/movement.py`.
The `zero_fill` attribute read by `widen_ranges` protects an explicit lowering
choice. It does not detect fills.

This change adds consumer agreement and finishes the deferred index divisions in
dense non-elementwise readers. Consumer domains propagate backwards through the
same access maps as `LowerCtx.read`. Agreeing reads share deferred computation.
Independent consumer loops with the same domain merge, with named scalar values
shared across their stores. Coordinates, domain extents, storage dependencies,
invariant hoisting and indexed-write ordering can require storage. Single-reader
movement decisions from layer 1 stay in place, including its alias boundaries for
the mapped-range pass. Reductions, calls and sparse iteration plans retain their
existing boundaries. Sparse traversal belongs to layer 3.

## Measurements against main

Main is `cd2b6c75c9392499398f90c4adc76dd826710dca`. The before process imports its
`src/` tree from a source archive, while the after process imports this checkout.
Both use this checkout's unchanged harness, problems, dependency lockfile and
plugins. Each process builds the five cells below in order, without a JIT cache
lookup, then compiles each generated translation unit to an object. There is one
observation per revision and cell. Generation includes derivative construction,
lowering, rendering and writing the header and source. Compilation excludes a
benchmark wrapper and linking.

These are diagnostics on shared la015, not a quiet-machine study or evidence of
runtime speed. Focused checks overlapped some generation and compile observations.
These diagnostics compare generation and C compilation only. The standard smoke
checks ran as required validation; their built-in timings are excluded here. Python is 3.14.3. Clang is Ubuntu 20.1.8, with
`-O3 -march=native -fno-math-errno -fveclib=libmvec`, matching the harness's C object
compilation flags. Exact commands and observations are in `measurements.json`.

| Kernel | C lines, main / layer 2 | C bytes, main / layer 2 | Generation ms, main / layer 2 | C compile ms, main / layer 2 | Caller workspace doubles, main / layer 2 |
|---|---:|---:|---:|---:|---:|
| chain M=5 | 7,877 / 7,877 | 788,491 / 788,490 | 9,563.3 / 9,785.5 | 3,181.8 / 3,473.5 | 66,744 / 66,744 |
| race_cars 5 | 1,148 / 1,148 | 71,628 / 71,628 | 863.0 / 925.6 | 231.2 / 229.8 | 0 / 0 |
| npmpc_jac 100 | 1,108 / 1,098 | 67,439 / 66,596 | 985.4 / 922.3 | 739.5 / 722.0 | 0 / 0 |
| npmpc 100 | 1,503 / 1,477 | 134,429 / 132,347 | 1,175.4 / 1,154.0 | 1,000.6 / 948.2 | 19,744 / 19,744 |
| npmpc_jac 6 | 1,108 / 1,098 | 67,303 / 66,462 | 840.0 / 851.9 | 346.0 / 311.5 | 0 / 0 |

The four requested compilation cells succeeded, as did the N=6 workspace control.
Raw generated files, logs, objects and the diagnostic driver are in
`benchmarks/results/layer2/`. The commands were:

```sh
PYTHONPATH=benchmarks/results/layer2/main/src:. uv run benchmarks/results/layer2/measure.py benchmarks/results/layer2/before
PYTHONPATH=. uv run benchmarks/results/layer2/measure.py benchmarks/results/layer2/after
```

## Snapshot diff

The pull request file diff is the complete snapshot diff against main. A local copy
is `benchmarks/results/layer2/snapshots.diff`. `snapshot-audit.json`
records all 21 files. Eighteen are byte-identical. Every constant table in every
file has identical type, shape and contents after ignoring its name.

- `spjac.c`, both hunks, changes declaration order and temporary names. Planning
  visits the 36-entry gather table earlier, renaming it from `k5` to `k1`.
  The two 24-entry index tables become `k4` and `k5`, and the constant double
  mask becomes `k7`. The mapped and zero-fill loop names follow the new temporary
  names. Every matching parameter, reference and lane call changes with its
  declaration. Buffer sizes, table contents, stores, loops, arithmetic and the
  header are unchanged.
- `wide.c`, the first hunk, combines the loops computing `a`, `b` and their final
  elementwise consumer into `lanes_1`. It computes `a` once in `v1` and `b` once in
  `v3`, then uses those locals at every agreeing read. The input `y` is loaded
  once. Constants and scalar names move with their first uses. Arithmetic nesting
  and libm operations are preserved. Two removed lane helper bodies and one
  removed 40-double buffer account for the storage change. The tail helper is
  renamed `lanes_2`.
- `wide.c`, the declaration hunk, removes `s2[40]` and renames the second constant
  matrix from `k5` to `k3`. Both matrices' contents are byte-identical to main.
  The two remaining buffers are the first matrix product and the final
  elementwise value consumed by the second matrix product.
- `wide.c`, the loop hunks, replace three 40-entry lane loops and their tail calls
  with one loop and its tail call. The second matrix product reads `s1` and `k3`
  in place of `s2` and `k5`, with the same reduction order. The final scatter calls
  use the renamed tail helper. Its index arithmetic is unchanged.
- `wide.h` changes only `wide_FRAME_BYTES`, from 5,056 to 4,736 bytes. The 320-byte
  reduction is exactly the removed 40-double buffer. `SZ_W` remains zero.

## Tests and perturbations

The agreeing two-consumer diamond and the three independent outputs each lower
to one loop with no private buffer and exactly one sine per input entry. The
three-output controls retain storage when a transpose changes coordinates or a
slice changes the domain. A shared gather computes only its selected entries.
A call between dependent consumers prevents a merge. All compare compiled C
against NumPy in `tests/passes/lowering/test_fusion.py`.

The new reduction, matmul and concat tests first failed on the deferred transpose
divisions. They now contain no integer quotient or remainder and match NumPy.
The existing carried-range and scatter write-order tests pass unchanged.

Each temporary perturbation was restored before validation:

| Perturbation | Result |
|---|---|
| Disable consumer agreement | The diamond and three-output agreement gates fail on retained storage. |
| Ignore coordinate equality | The transpose control fails on duplicated sine work. |
| Ignore domain equality | The shorter-domain control fails on removed storage. |
| Disable delinearization | All three non-elementwise division gates fail. |
| Disable dependency closure through merged groups | The crossed-group regression exhausts its 1 GB limit. |

The raw failures are `benchmarks/results/layer2/perturbation-*.log`. The focused
lowering, mapped-range, lane and snapshot checks passed 540 tests. The test-node
baseline was regenerated and the complete collection passed, with 5,469 tests.


The first full check found that independent groups could introduce a scheduling
cycle through each other. Dependency closure now includes already merged groups.
The existing reverse-over-reverse loop test passes with a 3 GB address-space
limit, and the small crossed-group test pins the cause. Requests through an
already deferred value also collapse agreeing duplicates, and a divergent view
stops new agreement decisions upstream.

The typed-entry sibling-workspace fixture now reads each shared producer at both
forward and reversed coordinates. Agreement removed its old intermediates, so it
no longer exercised spilling. The revised fixture still checks two disjoint
callee workspace windows, with its independent NumPy reference changed to match.
All three affected tests pass, including both strict-aliasing drivers.


The detached final pre-merge hook passed through `~/.scaly-suite/run`, exit 0.
It passed 5,141 tests with 16 skipped, all 13 benchmark smoke checks, types, both
Ruff checks, the documentation build and the private-name check. The final four
workspace/source gates passed in a focused two-worker run. Logs are
`benchmarks/results/layer2/pre-merge-final.log` and
`benchmarks/results/layer2/final-workspace-gates.log`.

Opus 5.5 at high effort approved the code and every changed snapshot line. Its
independent 88-test run and additional compiled NumPy checks passed, including
coordinate remaps, broadcasts, dependent consumers, scatter versions and invariant
hoisting in a carried loop body. The return-value docstring now names all three
results, and the temporary implementation plan is removed before the PR is ready.
Optional planner maintenance is tracked in [#472](https://github.com/PREDICT-EPFL/scaly/issues/472):
avoid planning-only index tables and skip group dependency work for rejected nodes.
Neither note affects generated code or correctness.
