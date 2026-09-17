# Fuzzing and agent-based validation

The failure cases a compiler's authors never thought to test are the ones that reach users. The
current strategy is to make scaly work well on a small number of benchmark problems, use it
internally, then hand it to external beta users before asking anyone to switch. This note records
the systems that could find those failures systematically, and how they relate. None of them exists
yet.

Each system tests either the core in isolation (IR, automatic differentiation, codegen, no solver in
the loop) or the core together with a solver on end-to-end optimal-control workloads. The two find
almost disjoint failure classes.

## System 1: differential graph fuzzer (core in isolation)

A property-based fuzzer generates random expression graphs, compiles them through the full JIT
path, and checks values and derivatives against independent oracles. Csmith did this for C
compilers and NNSmith for machine-learning compilers.

The oracles already exist:

- Value oracle. The runtime has no interpreter fallback, but the `OpInfo.numpy` table in
  `src/scaly/ir/expr.py` covers most of the op set. A small graph-walking evaluator over it gives a
  fast in-process reference that stays a test oracle and never becomes a runtime path. CasADi,
  already a dev dependency and already used in `tests/function/test_factory.py`, is a second,
  independent oracle.
- Derivative oracles that need no reference: finite differences (`scaly.ad.finite_difference`),
  forward-vs-reverse agreement, Hessian symmetry, and a sparse Jacobian scattered dense against the
  dense Jacobian. The fuzzer only has to generate graphs and assert these properties.

Design decisions that matter more than the generator itself:

- Use Hypothesis, mainly for its shrinker. Random-graph fuzzing without automatic minimization
  produces 500-node reproductions nobody debugs. Shrinking yields the minimal example, which is the
  input the repository rule that every compiler path needs a small differential test in `tests/`
  asks for. Every fuzzer find becomes a minimized regression test.
- Bias generation toward scaly-shaped graphs. Uniformly random directed acyclic graphs hammer the
  elementwise ops and never build a deep `VMAP`-over-`CALL` chain with `GATHER`-fed stages, where
  the real complexity lives. The highest-yield region is structural-op composition
  (`SLICE`/`GATHER`/`SCATTER`/`RESHAPE`/`CONCAT`) and sparsity propagation through the derivative
  factories. The one production bug of this kind found so far, a contiguous-slice alias that
  mis-detected column-style indexing as contiguous and is now pinned by a regression test, is the
  class a random-shape fuzzer catches immediately. The benchmark problems have fixed structure, so they
  can never find a sparsity pattern that is wrong only under some index permutation.
- Do not fuzz for performance with wall-clock numbers. There is no performance model to act as
  oracle, and timing on random graphs is noise. Assert structural invariants instead: flop count of
  the rendered C proportional to the nonzeros of the reported sparsity, no dense fallback on a
  sparse pattern, intermediate-representation size growing at most linearly in graph size, compile
  time under a generous ceiling.
- Run the fuzz corpus with AddressSanitizer and UndefinedBehaviorSanitizer on the generated shared
  library, and differentially at `-O0` against `-O2`. Both catch undefined behavior in rendered C
  that computes the right answer today.
- Keep the solver out of the loop. Random optimal-control problems are routinely infeasible or
  ill-conditioned, and "IPOPT did not converge" is not a bug signal. Solver testing stays on
  curated problems.

Metamorphic testing extends this: apply semantics-preserving transforms to a graph (unroll a `VMAP`
into concatenated calls, permute stages where valid, re-slice a `CONCAT`) and assert identical
outputs. It needs no oracle and composes with the same generator. The hand-written VMAP-versus-
unrolled tests in `tests/ad/test_vmap.py` are this pattern done manually.

A fuzzing campaign is also the correctness evidence a compiler can cite: N random graphs over the
full op set, checked against CasADi and finite differences, zero mismatches, plus the bugs it found
and fixed. Csmith and NNSmith made this the standard answer to "how do you know the generated code
is correct beyond your benchmarks?".

## System 2: agent-based workload recreation (core and solver)

A separate repository, run on a schedule, in which an agent finds concrete optimal-control workloads
(papers, the acados examples directory, do-mpc, the COPS collection, Betts' book problems) and
recreates them end-to-end with scaly. It is an internal benchmark set nobody reads until it detects
something worth a human's attention. It finds what no graph fuzzer can: "scaly cannot express this
formulation", "the obvious way to write this is 50 times slower than the clever way", "this error
message sends you down the wrong path". That is the class of failure that shows up in adoption.

Triage decides whether the system is useful. Experience building control systems with agents shows
that a large fraction of failures are the agent's, so the agent's raw success rate is too volatile to
measure scaly by, and unfiltered agent reports would erode trust until nobody reads them. The
pipeline that makes the signal reliable:

1. The recreator agent implements each workload twice, once in a reference stack (CasADi or acados)
   and once in scaly, and may only escalate when the twins agree on values and derivatives but
   scaly diverges, errors, or is anomalously slow. One agent's scaly code failing is not a signal.
2. Every escalation carries a minimal reproducible example, not a pointer into the workload
   repository.
3. A second, independent filter agent reproduces the minimal example and tries to attribute the
   failure to the recreation. Only findings that survive become a GitHub issue for a human.

This system's findings are only actionable once the frontend API stops moving. Launched earlier it
mostly measures roadmap churn, and every recreation goes stale with each API change. It belongs at
the core-freeze milestone, where it doubles as the beta program's advance scout. Its main
prerequisite is documentation quality (accurate reference documentation, a small formulation
cookbook, good error messages), since the agent works from the same documentation an external beta
user would.

## System 3: agent-based codegen review (core in isolation, performance-driven)

An agent reads the generated C for a set of graphs, fixed (the benchmark problems) or fuzzed, and
looks for optimization opportunities: redundant recomputation the common-subexpression elimination
missed, loops that should have fused, workspace traffic that could be registers, a sparsity pattern
handled densely, code CasADi generates visibly better for the same graph.

- The output is qualitative suggestions (tickets with the graph, the generated code excerpt, and
  the proposed improvement), so it needs no performance oracle. A human judges whether the
  opportunity is real, and the existing benchmark harness measures any resulting change.
- CasADi-generated code for the same graph, placed side by side, turns "this looks slow" into "the
  other compiler does X here and we do not".
- It composes with System 1. The fuzzer's generator supplies diverse graphs, and interesting ones
  (largest generated source per graph size, worst flop-to-nonzero ratio) form the review queue.
- Unlike System 2 it is useful before the API freezes, because it reads generated code, not the
  frontend.

The same trust discipline applies as in System 2. A second agent or a cheap measurement gate filters
suggestions before they reach a human, or the stream becomes noise.

## Prerequisite worth doing first: a coverage map

A one-off script recording which of the 39 expression ops, which op pairs, and which
derivative-factory paths the current tests and benchmarks exercise. It is small, it tells System 1's
generator where to aim, and it will probably show that the benchmark problems cover a small corner of the
op-composition space.

## What none of these cover

Performance regressions on the canonical problems (the existing benchmark gates own that) and
numerical robustness of the solvers on hard instances (curated problems own that).

## Suggested sequencing

1. Coverage map. Small, and it informs everything after.
2. System 1, the Hypothesis-based differential fuzzer. All oracles exist and it feeds the
   regression-test corpus.
3. System 3, the codegen-review agent. It can start as soon as System 1 supplies graphs and is
   useful before the freeze.
4. System 2, the workload recreator. Specify it now, launch it at core freeze alongside the external
   beta.
