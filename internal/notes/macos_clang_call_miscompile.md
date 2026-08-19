# macOS Apple-clang miscompile of inlined `_raw` callees

> **Frozen note.** Kept for the record, not maintained. For how alloy works now, see
> [`docs/how_it_works/architecture.md`](../../docs/how_it_works/architecture.md).

## Summary

`tests/alloy/test_factory_casadi.py::test_jacobian_through_call_node_matches_casadi_mx`
failed only on the `core tests / macos-latest` CI job. The root cause is a
**deterministic miscompilation by the CI runner's Apple clang** (Xcode 16 /
macOS 15 arm64), not a test-isolation or hash-consing problem. The immediate
workaround was to force generated `_raw` callees to be real (non-inlined) call
frames. The follow-up fix keeps callees inlineable by changing workspace packing
so a `CALL` output cannot reuse a scratch slot that contributed to one of that
same call's inputs.

Mitigated in two layers:

1. `src/alloy/passes/program.py` (`pack_workspace`) tracks the transitive producer
   closure for each private buffer and extends those producer lifetimes through
   any `CALL` that consumes the buffer. This removes the originally identified
   hazardous reuse (`inner_fwd2_y_x_raw(s0, s2, s1)` where `s2` was produced
   from `s1`).
2. The CI compiler still miscompiled the same forward-AD helper after that
   reuse was removed, so `src/alloy/codegen/c.py` now selectively emits
   generated forward-AD helper callees (`*_fwd*`) as
   `static __attribute__((noinline)) void ...`. Normal user `_raw` callees stay
   `static inline`.

## Symptom

The test differentiates `outer(z) = inner(z·z)` with `inner(w) = sin(w) + w²`
(forward-AD through a `CALL` node) and compares against CasADi:

```
[1, 1]: 11.833016900971549 (ACTUAL, alloy)   7.225016900971549 (DESIRED, casadi)
ACTUAL : [[1.045782, 0], [0, 11.833017]]
DESIRED: [[1.045782, 0], [0,  7.225017]]
```

Only the `[1,1]` entry is wrong; `[0,0]` is correct.

## Why the earlier fixes were ineffective

Prior commits chased test isolation / ordering:
`Fix hash-consing id reuse`, `Stabilize macOS core test sharding`,
`Run macOS core tests serially`. None worked, because the failure is not an
ordering effect:

- The **exact same** wrong float `11.833016900971549` appears in both the older
  `-n=auto` (xdist) run and the later serial run. A race or
  ordering-dependent bug would produce varying values or intermittent passes;
  an identical bit pattern across parallelism configurations means the result
  is fully deterministic.

## Investigation

Everything below was reproduced locally on macOS arm64 (Apple clang 21).

1. **Codegen is deterministic.** The rendered C source for the failing function
   is byte-identical across 50 in-process rebuilds, across a `PYTHONHASHSEED`
   sweep, after id-allocation perturbation, and — critically — after running
   the **entire core suite in session order** (diffed: identical). The JIT
   cache key is a SHA-256 of the source text, so a stale/colliding cache entry
   would require identical source, which would compute the same (correct)
   answer. No collision path exists.

2. **The source is mathematically correct.** Hand-tracing the rendered C yields
   exactly `[[1.046, 0], [0, 7.225]]`, matching CasADi.

3. **The source compiles correctly everywhere I can test.** Compiled standalone
   under Apple clang 21, Homebrew clang 18 / 20 / 22, and gcc-15, at
   `-O0/-O1/-O2/-O3/-Ofast`, plus `-fsanitize=address,undefined`. All produce
   the correct result; sanitizers are clean.

Conclusion: the source is correct and robust; the failing artifact is produced
only by the CI runner's compiler. The CI image is `macos-15-arm64`, whose `cc`
is the Xcode-bundled Apple clang (~16/17), older than anything available
locally.

## Decoding the wrong value

The relevant generated code (callee + its call site in the entry):

```c
static inline void inner_fwd2_y_x_raw(const double* x, const double* fwd_x,
                                      double* fwd_y_x, double* w) {
  double s0[2];                  // local: cos(x), computed up-front from x
  const double* t1 = fwd_x;      // seed column 0
  const double* t7 = fwd_x + 2;  // seed column 1
  for (i<2) s0[i] = cos(x[i]);
  for (j<2) fwd_y_x[0*2+j] = s0[j]*t1[j] + 2*(t1[j]*x[j]);
  for (j<2) fwd_y_x[1*2+j] = s0[j]*t7[j] + 2*(t7[j]*x[j]);
}

int J(...) {
  double s0[2], s1[4], s2[4];
  for (i<2) s0[i] = arg[0][i]*arg[0][i];        // s0 = z·z  (inner input x)
  for (j<2) s1[0*2+j] = arg[0][j];              // s1 = [z, z]
  for (j<2) s1[1*2+j] = arg[0][j];
  for (i<4) s2[i] = 2*(k2[i]*s1[i]);            // s2 = seed = 2·I·s1
  inner_fwd2_y_x_raw(s0, s2, s1, NULL);         // x=s0, fwd_x=s2, out=s1
  ... transpose s1 into res[0] ...
}
```

With `z = [0.4, 1.2]`: `x = s0 = [0.16, 1.44]`, seed column 1 = `t7 = [0, 2.4]`.
Correct `fwd_y_x[3] = cos(1.44)·2.4 + 2·(2.4·1.44) = 0.313 + 6.912 = 7.225`.

The observed `11.833 = 0.313 + 2·(2.4·2.4)` requires the **polynomial term to
read `x[1] = 2.4` while the `cos` term still used `x[1] = 1.44`**. `2.4` is the
value of `s2[3]` (the seed). So mid-kernel, a read of the input buffer `x`
(`= s0`, a separate array that is never written after `z·z`) returned a value
belonging to the seed buffer `s2`.

There is no legal aliasing path for this: `s0`, `s1`, `s2` are distinct arrays,
all live across the call, so a correct compiler keeps them in disjoint storage.

## Root cause

Two ingredients combine:

1. **Buffer reuse from the workspace packer** (`pack_workspace` in
   `passes/program.py`): the slot that built the seed (`s1`, holding `[z, z]`) is reused
   as the call's **output** buffer — `inner_fwd2_y_x_raw(s0, s2, s1, ...)`. This
   reuse is liveness-correct (`s1`'s last read is the statement computing `s2`),
   and works on every compiler tested.

2. **`static inline` on the callee**: inlining exposes the caller's packed
   scratch buffers (`s0`, `s1`, `s2`) to the optimizer. The seed `s2` is a
   trivial linear transform of `s1` (`s2 = 2·I·s1`), and `s1` is the clobbered
   output slot. Apple clang's optimizer mis-handles this configuration
   (rematerialization of a still-live input over the reused slot, and/or faulty
   stack-slot coloring), yielding a load of `s0[1]` that returns seed data.

The bug surfaces only when both hold and only on the old Apple clang. Inlining
is the enabling factor: with the callee kept as a real call frame, its pointer
arguments stay opaque and it simply loads `x` from memory — which is correct.

## The fix

The temporary workaround rendered `_raw` callees as non-inlined functions:

```c
static __attribute__((noinline)) void <name>_raw(...) { ... }
```

That keeps caller scratch buffers opaque across the call boundary, defeating the
mis-optimization, but it also adds a real call at every `CALL`-node boundary.
The first retained fix is more surgical: `pack_workspace` now forbids the hazardous
reuse by extending the lifetime of every scratch buffer in a `CALL` input's
producer closure through the `CALL` statement. The failing source changes from:

```c
inner_fwd2_y_x_raw(s0, s2, s1, NULL);  // s2 was produced from s1; output reuses s1
```

to:

```c
inner_fwd2_y_x_raw(s0, s2, s3, NULL);  // output slot is disjoint from input producers
```

However, Apple clang 17 still produced the same wrong value with the output
moved to `s3`, so the second retained fix is selective: only generated forward-AD
helper raw callees are marked `noinline`. A regression guard in
`tests/alloy/test_passes.py` asserts both parts: the failing AD-through-call
pattern gets a distinct output slot and the forward helper is noinline, while a
regular user `inner_raw` callee remains inline.

## Recommendations / follow-up

1. **Validate against the macOS CI job.** The local environment could not
   reproduce the miscompile even with the same Python/package versions; the CI
   runner is the oracle for this compiler bug. If it regresses again, widen the
   selective `noinline` predicate to the affected generated helper family.

2. **Pin / record the CI compiler.** The failure is tied to a specific Apple
   clang. Log `cc --version` in the `core tests` job so future regressions are
   attributable, and be aware that bumping the `macos-*` runner image can change
   the compiler under us (both for better and worse).

3. **Benchmark the change.** Run `benchmarks/scalability_sweep.py` (tracking +
   unbumpercars) before/after the packer guard. The guard may increase stack or
   workspace pressure by keeping some scratch buffers live slightly longer, but
   it should be much cheaper than globally disabling call inlining.

4. **Upstream report (optional).** If a minimal reproducer can be distilled
   (packed buffers + inlined callee + seed-from-output-slot), it is worth filing
   against LLVM / Apple clang. Lower priority than shipping the guard.
