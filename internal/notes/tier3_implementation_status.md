# Tier 3 — implementation status (done, tag `tier3-complete`)

Branch `claude/tier3-ipm` (on `claude/tier2-sparse`), todo items C-123, C-124, C-128, C-127, C-129,
C-132 for the tier's PRs, with C-100, C-107, C-112 and C-125 on the way. Reports:
`tier3_pr0a_report.html` … `tier3_pr5_report.html` and `tier3_review_report.html` (the Tier 3
summary). Timings: `perf_2026_09_26_tier3/`.

## What Scaly can do now (all generated code)
- `sc.while_loop(..., params=...)`: tensors every step reads unchanged, passed where they are,
  with forward and reverse derivatives and sparsity through them (C-112).
- A Function may name an output like an input (C-125); `SparseLDL.solve_with` solves with a
  factor carried out of a loop.
- The Function-call layer passes conforming arrays through as they are and reuses one workspace
  per thread: a trivial call takes 2.9 µs (C-100).
- `scaly.solvers.ipm`: PIQP 0.6.2 over one sparsity structure.
  - `QPStructure` fixes the patterns and which bounds exist; `QPValues.preprocess` applies PIQP's
    preprocessing (free rows, packed box bounds).
  - `ruiz`/`scale`: Ruiz equilibration as a `while_loop`, cost scaling with each backend's quirk.
  - `KKT` and `Kernels`: the dense (condensed Cholesky) and sparse (`SparseLDL` on the whole KKT
    matrix) backends, PIQP's factorization retries and iterative refinement, shared by the
    initial point and the loop.
  - `Solver(structure, backend, Settings()).solve(values, trace=...)`: the initial point, the loop
    (Mehrotra, proximal updates, boundary shift, fine-tune switch, infeasibility detection) and
    the result in PIQP's layout, with PIQP's status codes and a per-iteration trace.
- Test harness: 48 Maros–Mészáros problems (n + m ≤ 1 000) in `tests/data`, vendored PIQP run
  through a trace driver, and a NumPy port of PIQP 0.6.2 as the step-by-step oracle
  (`tests/solvers/ipm/`).

## Gates
| Gate | Result |
|---|---|
| The NumPy reference takes vendored PIQP's decisions on ≥ 90% of the subset | met: 46/48 (T3-0b) |
| The generated IPM matches (per backend, against PIQP's run of the same backend) | met: status 62/62 on both; decision traces 48/48 (sparse) and 47/48 (dense) where PIQP's backends agree |
| T3-6 target: sparse within 1.5× PIQP's solve time | met: 1.26× PIQP's warmed solve where it takes ≥ 50 µs (35 problems), after the review's speed-ups (1.49× before); 1.85× over all 51, where tiny problems pay a ~12 µs Python call; 0.81× PIQP's setup + solve. Dense: 2.24× (its Cholesky kernel, C-117) |

## Deviations from PIQP, documented
- The dense backend also fails a Cholesky pivot below ε times its diagonal entry while refinement is
  off: such a pivot's sign is rounding noise, and without refinement a noisy factor goes uncorrected.
- A bound the structure declares finite that arrives infinite, NaN or at or beyond 1e30 stops the
  solve with `INVALID_BOUNDS` (−11): the solver is specialised to which bounds exist.

## Review round (T3-R, C-133)
Five agents reviewed Tier 3. Everything they confirmed is fixed, each with a regression test that
fails without the fix:
- **Crashes:** three derivative Functions of while loops that collided by name; a cold-cache
  compile race between threads calling a new Function.
- **Wrong or stale numbers:** dense cost scaling used the sparse backend's Ruiz quirk; a bound
  declared finite that arrived infinite gave NaN (now `INVALID_BOUNDS`); a failed first
  factorization returned garbage instead of PIQP's start; infeasibility and NUMERICS exits reported
  the previous pass's state.
- **Resources:** the in-place proof broadcast int64 tables to every step (1.7 GB); a
  Function/handle cycle held workspaces until garbage collection; the trace harness leaked
  510 MB of problem files.
- **Fairness:** `-ftree-vectorize` changed GCC 12+'s cost model; it is now for GCC before 12 only,
  and the CasADi baseline takes the same flag.
- **Tests:** tolerances with 20× headroom, so the IPM tests pass with FMA contraction off and fast;
  the dense exemption read from the run; the IPM tests moved to `tests/solvers/ipm/`.
- **Speed:** max/min reductions in four lanes, sorted segment extrema run by run, Ruiz maxima per
  matrix, the refinement tolerance and the data vector out of each step: −11 to −25% on sparse
  solves, identical iterations.

## Open items handed on
- **C-114:** unpadded factor updates, the lever for sparse speed (the factorization is 40–60% of a step).
- **C-117:** a blocked dense Cholesky, the lever for the dense backend (its kernel is 80% of a step,
  at a quarter of Eigen's speed).
- **C-130:** QRECIPE's sparse path takes 34 iterations against PIQP's 19.
- **C-131, C-134:** the loop-carry copies and the other step overheads the review measured (a further 5–10%).
- **Tier 4 decisions:** parity semantics for the drop-in backend, baselines, and which quirks it keeps.
