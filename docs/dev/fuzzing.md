# Fuzzing and agent-based validation

The standing fear with building a compiler is the failure cases we never thought to test. The current strategy accepts that: make scaly work extremely well on a small number of concrete benchmark problems, then dogfood it internally and hand it to external beta users (marketed as experimental) to battle-test it before asking anyone to switch. This note records the systems that could accelerate that process systematically, and how they relate.

The useful organizing axis: each system tests either **the core in isolation** (IR, automatic differentiation, codegen — no solver in the loop) or **the core in conjunction with a solver** (end-to-end optimal-control workloads). They find almost disjoint failure classes.

## System 1: differential graph fuzzer (core in isolation)

A property-based fuzzer that generates random expression graphs, compiles them through the full JIT path, and differentially checks values and derivatives against independent oracles. This is a well-trodden path for compilers (Csmith for C, NNSmith for machine-learning compilers), and it is already sketched in `internal/roadmap.md` ("Property/fuzz tests later": random expression trees, random shapes, random sparsity patterns).

Scaly is unusually well set up for it because the oracles already exist:

- **Value oracle.** There is deliberately no interpreter fallback in the runtime, but the `OpInfo.numpy` table in `src/scaly/ir/expr.py` covers most of the op set; a small graph-walking evaluator over it gives a fast in-process reference that is a test oracle, not a runtime path. CasADi (already a dev dependency, already used in `tests/function/test_factory.py`) is a second, independent oracle.
- **Derivative oracles that need no reference at all.** Finite differences (`scaly.ad.finite_difference`), forward-vs-reverse agreement, Hessian symmetry, sparse-Jacobian-scattered-dense vs dense Jacobian. The fuzzer only has to generate graphs and assert these properties.

Design decisions that matter more than the generator itself:

- **Use Hypothesis, primarily for its shrinker.** Random-graph fuzzing without automatic minimization produces 500-node reproductions nobody debugs. Shrinking yields the minimal reproducible example for free, which is exactly what the repository rule ("every path needs a small differential test in `tests/`") wants as input. Every fuzzer find becomes a minimized regression test — the corpus only ratchets up.
- **Bias generation toward scaly-shaped graphs.** Uniformly random directed acyclic graphs hammer the elementwise ops and never build a deep `VMAP`-over-`CALL` chain with `GATHER`-fed stages, which is where the real complexity lives. The highest-yield region is structural-op composition (`SLICE`/`GATHER`/`SCATTER`/`RESHAPE`/`CONCAT`) and sparsity propagation through the derivative factories. The one production bug of this kind found so far — a contiguous-slice alias that mis-detected column-style indexing as contiguous, now pinned by a regression test is precisely the class a random-shape fuzzer catches immediately. The fixed structure of the three benchmarks means they can never find a sparsity pattern that is wrong only under some index permutation; the fuzzer exists to cover that combinatorial space.
- **Do not fuzz for performance with wall-clock numbers.** There is no performance model to serve as oracle, and timing on random graphs is noise. Assert structural invariants that are performance in disguise: flop count of the rendered C proportional to the nonzeros of the reported sparsity, no dense fallback on a sparse pattern, intermediate-representation size not exploding superlinearly in graph size, compile time under a generous ceiling.
- **Cheap high-value additions:** run the fuzz corpus with AddressSanitizer/UndefinedBehaviorSanitizer on the generated shared library, and differentially at `-O0` vs `-O2` — both catch undefined behavior in rendered C that happens to compute the right answer today.
- **Keep the solver out of the loop.** Random optimal-control problems are routinely infeasible or ill-conditioned, and "IPOPT did not converge" is not a bug signal. Solver testing stays on curated problems.

A natural extension is **metamorphic testing**: apply semantics-preserving transforms to a graph (unroll a `VMAP` into concatenated calls, permute stages where valid, re-slice a `CONCAT`) and assert identical outputs. It needs no oracle and composes with the same generator; the hand-written VMAP-versus-unrolled tests in `tests/ad/test_vmap.py` are already this pattern done manually.

**Evidence.** A fuzzing campaign is also the correctness evidence a compiler can point to: "N random graphs over the full op set, differentially checked against CasADi and finite differences, zero mismatches", plus the bugs it found and fixed along the way. Csmith and NNSmith established this as the standard way to answer "how do you know the generated code is correct beyond your benchmarks?".

## System 2: agent-based workload recreation (core + solver)

A separate repository, run on a schedule, in which an agent periodically finds concrete optimal-control workloads (papers, open-source examples: the acados examples directory, do-mpc, the COPS collection, Betts' book problems) and recreates them end-to-end with scaly — its own internal benchmark set that nobody reads until it detects something worth a human's attention. This is dogfooding with the human cost removed: it finds what no graph fuzzer can — "scaly cannot express this formulation", "the obvious way to write this is 50× slower than the clever way", "this error message sends you down the wrong path" — which is the class of failure that shows up in real adoption.

**Triage is the whole game, and agent failures are mostly the agent's fault.** Experience building control systems with agents from scratch shows that a large fraction of failures are attributable to the agent, not the tooling — so the agent's raw success rate is too volatile to serve as a quality metric for scaly, and unfiltered agent reports would erode trust until nobody reads them. The pipeline that makes the signal reliable:

1. The recreator agent must implement each workload **twice** — once in a reference stack (CasADi or acados), once in scaly — and may only escalate when the twins agree with each other on values and derivatives but scaly diverges, errors, or is anomalously slow. Two independent implementations agreeing while scaly disagrees is a real signal; one agent's scaly code failing is not.
2. Every escalation must carry a **minimal reproducible example**, not a pointer into the workload repository.
3. A second, independent **filter agent** reproduces the minimal example and adversarially tries to attribute the failure to the recreation rather than to scaly. Only findings that survive this filter become a GitHub issue presented to a human.

**Timing:** this system's findings are only actionable once the frontend API stops moving — launched earlier it mostly measures roadmap churn, and every recreation goes stale with each API change. It belongs at the core-freeze milestone, where it doubles as the beta program's advance scout. The main prerequisite is documentation quality (accurate reference documentation, a small formulation cookbook, good error messages), since the agent works from the same documentation an external beta user would.

## System 3: agent-based codegen review (core in isolation, performance-driven)

An intermediate system between the two: an agent that reads the generated C for a set of graphs — fixed (the benchmark problems) or fuzzed — and looks for optimization opportunities we are leaving on the table. Not driven by absolute numbers, but by *noticing*: redundant recomputation the common-subexpression elimination missed, loops that should have fused, workspace traffic that could be registers, a sparsity pattern handled densely, code CasADi generates visibly better for the same graph.

Useful properties of this design:

- The output is qualitative suggestions (tickets with the graph, the generated code excerpt, and the proposed improvement), so it needs no performance oracle — the human judges whether the opportunity is real, and the existing benchmark harness measures any resulting change.
- Placing CasADi-generated code for the same graph side by side gives the agent a concrete foil, turning "this looks slow" into "the other compiler does X here and we do not".
- It composes with System 1: the fuzzer's generator supplies diverse graphs; interesting ones (largest generated source per graph size, worst flop-to-nonzero ratio) are the natural queue for review.
- Unlike System 2 it is useful **before** the API freezes, because it reads generated code, not the frontend.

The same trust discipline applies as in System 2: suggestions should be filtered (by a second agent or by a cheap measurement gate) before reaching a human, or the stream becomes noise.

## Prerequisite worth doing first: a coverage map

A one-off script recording which of the 38 expression ops, which op pairs, and which derivative-factory paths the current tests and benchmarks actually exercise. It is small, it tells System 1's generator where to aim, and it will likely show that three benchmarks cover a small corner of the op-composition space — the concrete version of the original fear.

## What none of these cover

Performance regressions on the canonical problems (the existing benchmark gates own that) and numerical robustness of the solvers on hard instances (curated problems own that).

## Suggested sequencing

1. Coverage map (small, informs everything after).
2. System 1, the Hypothesis-based differential fuzzer — all oracles exist, it is already roadmapped, and it feeds the regression-test corpus.
3. System 3, the codegen-review agent — can start as soon as System 1 supplies graphs, useful pre-freeze.
4. System 2, the workload recreator — spec it now, launch it at core freeze alongside the external beta.
