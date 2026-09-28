# What to do next

The single actionable list. Rationale lives elsewhere and is linked, never restated:

- **Why a number is or is not admissible** — [`docs/results/fairness.md`](../docs/results/fairness.md):
  what the comparisons hold constant, the measurement protocol, the reference machine.
- **How the suite got here** — [`internal/notes/benchmark-buildout.md`](notes/benchmark-buildout.md):
  the completed B-track and L-track, formulation history, retired workloads.
- **What shape a refactoring should take** — [`internal/notes/refactorings.md`](notes/refactorings.md):
  one `#` section per refactoring, kept until that refactoring lands.

Reorganized 2026-09-07. Sections are themes that outlive the first release. Inside each section,
**Now** holds what is actively worked on or next in line, and **Deferred** holds what is
intentionally low priority: the reasoning is still good, nothing depends on it yet. A finished item
stays in place with its box checked until it is flushed out by hand; git history and the frozen
notes hold the record after that.

### Identifiers

Every item has an identifier `<PREFIX>-<n>`. The prefix names the section the item sits in; the
number comes from one counter shared by the whole file, which only ever grows.

**Next id: 196**

| Prefix | Section |
|---|---|
| API | API |
| C | Compiler internals |
| S | Solvers |
| CAPI | C API |
| BH | Benchmark harness |
| BP | Benchmark problems |
| L | Licensing |
| D | Documentation |
| R | Release |
| CS | Case studies |

The Track C closeout at the end groups active tasks across sections without changing their identifiers.

Rules:

- A new item takes the next id and bumps the counter. A deleted item never frees its number.
- Moving an item to another section changes its prefix and keeps its number. Grepping the number
  alone finds the item, or proves it is gone.
- Other documents cite the full id and the title, `S-15 Replace METIS 4 with METIS 5`, so the
  reference survives both a move and a retitle.
- A new section adds a row with a prefix that is not in the table and never was.

Why identifiers at all: they give the short stable handle that Linear or GitHub issues give, while
the list stays git-tracked, lives next to the code, and changes per branch, so a worktree can add,
close and reorder its own items and the merge carries them. A global counter rather than one per
section because items move between sections more often than expected, because eight counters are
eight places to get wrong once completed items are deleted, and because the letters then carry
only the theme and nothing else has to stay stable.

Priority order from 2026-09-11, the road to a public repository and a first alpha. Each step is
cheap once and expensive to redo, so the order is the sequencing that matters:

1. R-40 versioning policy.
2. D-32 to D-34 documentation rewrite, D-68 acknowledgements and AI disclosure, D-65 the two
   docs deployments.
3. R-67 make the repository public, after running the suite, ruff and ty locally on the rewritten
   tree.
4. R-66 platform-only wheel tags, R-41 wheels on test PyPI, R-42 `0.1.0a1`.

## API

### Now

- [x] **API-1. Function templates and multi-parameter functions.** Done 2026-09-27 on
      `claude/function-templates` (plan `notes/function_templates_plan_2026_09_27.md` v2, after a
      five-agent review; phases P1a to P4 and P6, P5 deferred as API-77). A body takes one argument per
      parameter; declarations are shapes, names or trees, one per parameter, with `output=` optional;
      a shape left open (`sc.L()`, `(n, None)`, or nothing declared at all) makes `sc.Function` a
      template that traces one `ConcreteFunction` per argument signature under a deterministic C name
      (`f__3x4`), while a fully declared body keeps its exact C symbol. Derivatives and
      `custom_derivative` lift over templates; one name is `wrt` and names default when unique.
      Seeded derivatives take the source's arguments then the seed, and solvers five arguments. The
      whole tree is migrated; C snapshots unchanged; concrete calls faster. Reports:
      `notes/templates_p1a_report.html` to `templates_p6_report.html`, summary
      `notes/templates_summary_report.html`.
- [ ] **API-2. Preserve declared trees through `vmap` and Function-level differentiation.** Settle
      the mapped input convention and retain runtime and static acceptance tests. Rationale:
      refactorings.md "`vmap` and the AD entry points erase the callee's declared trees".
- [ ] **API-3. Decide the zero-input Function contract and reduce the private flat call path.**
      Preserve legitimate parameterless solver oracles. The dispatch half is settled by API-1: `f()`
      evaluates and `f.symbolic_call()` is the symbolic spelling. What remains is whether AD inlines
      zero-input constant callees, and retiring `ad/forward.py`'s `_flat_symbolic_call` seam.
      Rationale: refactorings.md "Zero-input `Function`s, and the flat call seam that survives
      because of them".
- [ ] **API-4. Finish the npmpc template example**, unblocked now that API-1 has landed. The public
      typed decorators, exact `Function` annotations, and shared Scaly/CasADi runtime parameters
      landed first. Replace the remaining decoder-architecture builders with `sc.function`
      templates (a declaration with holes, or bare).
      This benchmark may use the packed parameter length as its specialization key because it does
      not add more MLP layouts; a general template must distinguish individual layer shapes because
      equal parameter counts do not prove equal architectures.
- [x] **API-5. Validation additions**: one public `fwd` and `adj` test on the same nontrivial `VMAP`
      fixture compared against the unrolled form with a forward/reverse duality check, and a
      finite-difference check of the Lagrangian gradient in the pairwise-map sparse-Hessian test.
      Lives in `tests/ad/test_vmap.py` (duality) and `tests/integration/test_vmap.py` (pairwise Hessian).
- [x] **API-142. Explicit Runge-Kutta integrators** (integrators/MPC plan I1,
      `notes/integrators_mpc_plan_2026_09_27.md`). `scaly.integrators`: Butcher tableaus with the
      order conditions checked by B-series over rooted trees, ten named explicit methods (Euler to
      RK4 and the 3/8 rule, the Bogacki-Shampine and Dormand-Prince pairs), and `si.explicit` /
      `si.rk4`, maps with the model's own signature, `dt` folded or an input, substeps unrolled up to
      four and a `scan` beyond. Each stage is one call of the model. The coefficients fold as written
      by hand, so a shooting defect generates byte-identical C to `examples/nmpc_cartpole.py`'s RK4 and
      runs 13 to 23% faster than CasADi's fastest encoding (`perf_2026_09_27_integrators/`). 20/20
      mutants killed. Report: `notes/integrators_i1_report.html`.
- [x] **API-144. Implicit Runge-Kutta integrators** (plan I3). `si.implicit`: Gauss-Legendre, Radau
      IIA and Lobatto IIIA/IIIC tableaus built from their nodes for any number of stages, backward
      Euler, implicit midpoint, trapezoidal, Alexander's SDIRK2/3, each checked against the order
      conditions. Newton from `f(x)`, fixed iterations or to a tolerance in a `while_loop`, simplified
      or full; simplified Newton on a coupled method splits `I - h A (x) J` by the eigenvalues of `A`
      (RADAU5's transformation), a diagonally implicit one solves stage by stage. The derivative is
      the implicit function theorem's at the stages found, two levels of rules, so Hessians and
      forward-over-forward are exact and never reach the iterations. `ad/forward.py`: a call that
      reaches a custom forward rule now takes the joint multi-seed pass, so the rule is mapped over
      the seeds and its shared work hoisted (it was applied once per seed). Radau IIA(3) on the
      cart-pole: step 0.62x and step Jacobians 0.94x CasADi SX's time, with a fifth of the code.
      24/24 mutants killed, three after a test was added. Report: `notes/integrators_i3_report.html`.
- [x] **API-145. Exact, adaptive and symplectic integrators** (plan I4). `si.zoh`, `si.foh` (one
      `expm` each) and `si.linearize` (one Jacobian per input leaf, as generated code).
      `si.adaptive`: DOPRI5 or BS32 with RMS error control in a `while_loop`, NaN when `max_steps`
      runs out, derivatives with the step sequence held (the controller's factor rounded to a power
      of `2^(1/1024)` through an `int64` cast); RK45's accuracy at 20 to 140x less time per Python
      call. `si.symplectic`: Störmer-Verlet and symplectic Euler for `x = [q, v]`, energy bounded over
      1e5 pendulum steps where RK4 loses a third of it. Report: `notes/integrators_i4_report.html`.
- [x] **API-146. Collocation and pseudospectral transcriptions** (plan I5). `si.MultipleShooting`
      (any integrator, the running cost integrated as one more state), `si.Collocation` (Radau or
      Gauss points, equal to Radau IIA and Gauss-Legendre at a fixed control) and
      `si.Pseudospectral` (Radau pseudospectral as GPOPS-II, controls at the nodes, spectral
      convergence): each makes an `Interval` Function `(x, u, [z], xnext, *params, [dt]) ->
      [residuals; cost]` for a horizon to map. Lobatto collocation and the LGL and LG pseudospectral
      schemes were dropped from the plan: LGL collocates one condition more than its unknowns at a
      fixed control, and LG has no node for the control a receding horizon applies. `ad/sparse.py`:
      the structured VMAP Jacobian now accepts contiguous runs of the variable (an NLP's leaves are
      slices of its one variable vector, so every multi-leaf OCP fell back to a whole-horizon
      coloring assembled through dense masks) and differentiates all blocks in one block-seeded
      pass. Horizon Jacobians: collocation 19.2 to 2.1 us, pseudospectral 71.9 to 4.4 us, shooting
      6.9 to 4.8 us; CasADi SX 0.95 us for collocation (a fully expanded form, 70x the code).
      32/32 mutants killed. Report: `notes/integrators_i5_report.html`.
- [x] **API-147. `scaly.mpc` core** (plan M1). `mpc.OCP` over a continuous model and any transcription,
      or a discrete map: `Quadratic` or Function costs at the points or integrated, state and control
      bounds (the transcription's internal variables included), `Path` constraints hard or soft
      (exact l1), `TerminalEquality`, parameters as the union of what the Functions name, some
      varying per stage. `mpc.MPC` for ipopt, sqp or piqp: `solve` (a `Solution`), `__call__` with a
      warm start shifted one interval, and `law`, one Function with the solver nested and the shift in
      its code, for `write_module`; `mpc.simulate`. The cart-pole built by it is the hand-written NLP
      (same solution to 1e-9, same 23 iterations, 0.99x the time). Transcription intervals now give
      residuals and cost as two outputs, so the residual map keeps the stage-wise Jacobian.
      `ad/forward.py`: a matmul's zero tangent is left out, so a linear map through a call proves
      affine and a QP through an OCP reaches piqp. 18/18 mutants killed, one after a test was added.
      Report: `notes/mpc_m1_report.html`.
- [ ] **API-157. Warm starts for a global pseudospectral horizon:** a shift by a sampling time shorter
      than the one interval, by interpolation on the nodes (plan §4.1); segments shift by whole
      intervals today.
- [ ] **API-148. Linear MPC and linear terminal ingredients** (plan M2): LQR, polytopes, maximal
      invariant sets, ellipsoids, the condensed form.
- [ ] **API-149. Nonlinear terminal ingredients** (plan M3): quasi-infinite horizon, certification,
      steady-state targets.
- [ ] **API-150. Real-time iteration, deployment example, docs** (plan M4).
- [ ] **API-151. Integrators/MPC review round** (plan R).
- [ ] **API-152. Indirect methods** (plan X): Pontryagin's boundary value problem by indirect
      multiple shooting.
- [x] **API-171. `scaly.interp`: tables and interpolating splines** (interp plan SP1,
      `notes/interp_plan_2026_09_28.md`). Every interpolant is an `interp.BSpline`, a tensor-product
      B-spline evaluated from existing ops: the `uniform` (floor, clamp, one-step correction),
      `binary` (NaN-padded halvings) and `count` searches, each equal to `searchsorted` on 1e6
      adversarial points; per-cell polynomials (`pp`) or local bases (`basis`), expanded at cell
      midpoints; batches as one `vmap` of the point's Function (a vectorized graph grew with the batch
      under reverse mode); five extrapolation modes, NaN in giving NaN out; `index=` sharing one search;
      `function()` interned by content. `interp.interpolant`: `nearest`, `zoh`, `linear`, `cubic` (four
      boundary conditions) and `spline` k = 1 to 5, per axis, 1-D to 4-D, vector outputs; SciPy to
      1e-13, derivatives to 1e-11 in both modes. CasADi: `linear` and the `bspline` evaluation to
      1e-13; its fitted `bspline` misses its own data by up to 1.6e-10, so the fit agrees within that.
      0.04x to 0.29x CasADi's generated `interpolant` time per point (`perf_2026_09_28_interp/`); the
      count search never won, so `auto` is uniform or binary. 21/22 mutants killed, the survivor dead
      code now removed. Report: `notes/interp_sp1_report.html`.
- [x] **API-172. Shape-preserving kinds, smoothing splines and spline calculus** (interp plan SP2).
      `pchip`, `akima` and `makima` (SciPy's formulas, 1e-13) and `steffen` (the paper's, monotone on
      1e4 random data sets) as C1 cubics on doubled knots, 1-D; `smooth_linear` as CasADi builds it
      (from its source; 9e-16). `interp.smoothing`: P-splines of any degree and dimension, gridded or
      scattered, lambda by GCV (9 ms at 2e4 points, where SciPy's GCV fails), and
      `make_smoothing_spline`. `derivative` and `antiderivative` as splines on the same partition,
      with the extrapolation their operation implies; `integrate` exact in every mode; `inverse()` of
      a strictly monotone curve, Newton bisecting to stay in its cell, `1/f'` by
      `custom_derivative`, a flat end held. 26/26 mutants killed, five after a test was added (the
      bisection needed a searched-for quintic); a dead ulp tolerance removed. Report:
      `notes/interp_sp2_report.html`.
- [x] **API-173. Coefficients and data as expressions** (interp plan SP3). `BSpline` with `Expr`
      coefficients (the basis strategy, the coefficients an input of its Function, broadcast into
      batches; calculus as constant maps). `interpolant` with `Expr` data for every kind: constant
      maps, above 256 sites a C2 cubic's slopes by the Thomas algorithm in two scans (Sherman-Morrison
      for periodic), the shape-preserving formulas as expressions with every unused division made
      safe for reverse mode; `smoothing` with a given lambda. `basis()` and `at()` for points known
      now, with each extrapolation: the Jacobian in 100 coefficients at 2000 points costs 5.7 us
      through `at()`, 351 us at symbolic points, 22.5 ms in CasADi inlined. A broadcast table's fit is
      hoisted out of a map. 16/16 mutants killed, ten after a test was added or fixed (the scan-solve
      test had looked only at the sites). Report: `notes/interp_sp3_report.html`.
- [ ] **API-186. `inverse()` of a spline with `Expr` coefficients:** the bracket from the curve's
      values at run time, the Newton loop carrying the coefficients, and the derivative in them.
- [x] **API-174. Interpolation performance** (interp plan SP4). The `bucket` search (a uniform
      bucket index, a start table, as many compares as the fullest bucket needs): clustered tables
      5.4 to 1.9 ns; `auto` is binary up to 32 cells, then bucket, uniform, binary, within 5% of the
      best fixed choice. The default `linear` extrapolation tabulated as two outer cells (11.7 to
      8.2 ns on a bicubic). Thresholds measured: `DENSE_FIT` 40, `PP_BUDGET` 4 MB, a warning at 2^20
      tabulated values; uniform cells compute their centers. `float32` tables and evaluation (to
      1e-6; derivatives wait on the core AD's float32 tangents), `BSpline.pack`, a table as an input
      of AOT code from a C `main`, the header stating multi-axis buffer layouts. Against CasADi's
      interpolant, bspline node and blazing_spline: 0.03x to 0.30x per point, 25x on batches. The
      dedicated-op gate is not tripped. Hint search not built. 12/12 mutants killed. Report:
      `notes/interp_sp4_report.html`.
- [x] **API-175. Shape-constrained fitting** (interp plan SP5). `interp.constrained`: least squares
      over a B-spline's coefficients with monotone, convex/concave (per axis), bounds, pinned values
      and derivatives and periodic ends as sufficient linear conditions, solved by PIQP when the
      graph is built, one solver per problem size. An active-set polish after the interior point
      makes an inactive-constraint fit equal least squares to rounding (3.5e-10 before). The 2-D
      fits exposed two Python loops in the core sparsity code, now sparse products: the matmul
      pattern and the star-colouring recovery (a 16x16 map's first call 56 s to 6.4 s). 27/27
      mutants killed, three after a test change. Report: `notes/interp_sp5_report.html`.
- [x] **API-176. Interpolation notebooks and CasADi pairs** (interp plan SP6). Six executed
      notebooks in `examples/interp/`: every kind against SciPy, n-D tables with search and layout
      timings and a C `main` passing a table in, calibration and Hammerstein identification with
      the Jacobian pattern at known against symbolic points, shape-constrained fits with KKT
      checks and an OCV inverse, a closed-loop MPCC lap on one periodic arc-length spline, spline
      trajectories with Bezier convex-hull obstacle constraints as a PIQP QP. Five CasADi pairs in
      `examples/interp/pairs/`, `compare.py --dir`: all agree (2e-10 at worst, CasADi's bicubic
      fit), same iteration counts, Scaly 2.5x to 860x faster per run than CasADi's VM, 2x to 420x
      than its JIT. Report: `notes/interp_sp6_report.html`.
- [ ] **C-187. One copy of a constant table per generated module.** Each generated function
      embeds its own copy of every constant it reads, so a spline table read by a stage cost, a path
      constraint and their derivatives is emitted six times (`contouring_control.ipynb`: 941 kB of
      C, most of it the track's 5 440-value table). Emit each distinct constant once at file scope
      and reference it from every function. Measured in the interp review: source 941 to 283 kB,
      and for a 128 x 128 bicubic in an OCP stage cost 34.9 to 5.9 MB and GCC's compile 4.0 to
      1.0 s; clang already merges identical tables in the object, and run time does not change. A
      Program-IR pass collecting `constant` buffers across a PROGRAM's procedures (keyed by dtype,
      size and a digest computed in `_lower_const`), emitted once after `extern "C" {`, is the
      likely shape. `LARGE_TABLE` counts one copy.
- [ ] **API-188. `interp.constrained` as a sparse QP with box bounds.** The fit passes its bounds
      as identity rows of G and every matrix dense, so a 2-D fit of 361 coefficients spends most of
      a solve extracting dense data and colouring dense patterns. With the bounds as PIQP's box
      and P and G on their structural patterns (per-axis bands and the differencing rows), the
      review measured a solve at 9.6 ms instead of 131 ms and a first call without the dense
      colouring. Needs sparse matrix parameters of a fixed pattern in `qp_problem` or a problem of
      its own.
- [ ] **C-189. Dense QP data without AD, and dense colouring without search.** `_qp_data`
      (`solvers/qp.py`) extracts P and G through the coloured Hessian and Jacobian, which with
      dense seeds lowers to products with constant identity matrices, O(n^3) a call for a copy;
      fold `I @ X` in simplification or pass the matrices through. `column_coloring` and
      `star_coloring` search even a full pattern, where n colours are the answer.
- [ ] **C-190. An integer `SUM` in lowering.** The lowering emits a float accumulator for a sum of
      `int64` (a `CONST_FLOAT` of int64 dtype fails verification), so the `count` search sums its
      compares as doubles, which clang stops vectorizing at about 32 cells (8.1 ns against 2.95
      with an integer sum, measured in the interp review).
- [ ] **API-191. Numeric cubic fits in Hermite form.** `interpolant(kind="cubic")` on NumPy data
      fits by `make_interp_spline` collocation, whose B-spline basis is ill-conditioned on clustered
      sites: 4.7e-12 against `CubicSpline`'s 6.7e-16 for sites 1e-6 apart, 2.1e-10 at 1e-9. The
      slopes from `CubicSpline` on the doubled-knot Hermite form (as the `Expr` path already
      stores) keep its accuracy.
- [ ] **C-192. float32 through the core AD.** `zeros_like` and `_ones_like` build float64, Python
      float literals in the AD rules become float64 constants (`Expr._operand` weak-types only
      ints), and the seeds of `jacobian` and `_jvp_many_structural` are float64, so forward and
      reverse mode both fail on float32 graphs. float32 splines evaluate but do not differentiate
      until this lands. C-158 has since given `zeros_like`, `_ones_like` and a Python number beside
      a float expression the value's dtype (`tangent_dtype`); what remains starts with checking a
      float32 spline's derivatives.
- [x] **C-193. Derivative caches keyed on `nonsmooth`.** The call, map, scan and while caches in
      `ad/forward.py` and `ad/reverse.py` ignore `sc.options(nonsmooth=...)`, so a derivative
      built under one setting is reused under another (a gradient at a `maximum` tie, a refusal
      under `"error"`). Key them on the setting, and name non-default derivatives apart. Done by
      C-158's `options_tag`, which keys the caches on every option in force;
      `tests/ad/test_nonsmooth_derivative_caches.py` reproduces both cases through `vmap`, `scan`
      and `while_loop` under `sc.gradient`, `sc.jvp` and `sc.jacobian`. The narrower key is C-195.
- [ ] **C-195. Key a derivative helper only on the options its callee reads.** `options_tag` keys
      every helper cache on all options and tags the names under any non-default one, so under
      `nonsmooth="first"` a smooth callee's derivatives are built again, as new C symbols. Tag a
      callee with `nonsmooth` only when its graph, callees included, holds one of the eight
      nonsmooth ops, and likewise `dense_unroll`/`sparse_unroll` and `max_trajectory` only where
      they are read. `test_a_smooth_derivative_is_shared_across_settings` is its strict xfail.
- [ ] **API-194. A leaner `inverse()` loop.** It costs about 20 forward evaluations per point in a
      batch: the condition recomputes the residual the step just evaluated, and every evaluation
      repeats the clamp and outer-cell shift of a cell already fixed. Carry the residual, evaluate
      the fixed cell's polynomial directly, and consider a fixed-count Newton for batches.
- [x] **API-178. Interpolation review round** (interp plan SPR). Five reviewers: IR and AD,
      numerics, performance, API and docs, tests. About fifteen bugs fixed, each with a test:
      - constant folding of `min`/`max` against C's `fmin`/`fmax`, which left a cast of NaN in C;
      - interning by name alone, and inverse settings missing from the name;
      - an `Index` without its partition;
      - periodic `Expr` fits, and `smoothing`'s dtype;
      - `constrained` at small scales, and its degenerate multipliers;
      - integrals past an end;
      - NaN from `basis` and reverse mode far out;
      - float32 tables far from the origin;
      - the inverse's absolute stop and monotonicity floor;
      - the unbounded periodic wrap.
      Deferred: C-187 and API-188 to API-194. 23/23 mutants undoing the fixes killed; the tests
      reviewer's 26: 21 killed, 3 equivalent, 2 defensive. Report: `notes/interp_spr_report.html`.

### Deferred

- **API-69. Shorter numerical solver calls.** The five input groups make every call spell out
  zero arrays for the initial multipliers, including the empty inequality group; see the README
  example. Consider defaults for the multiplier groups or a keyword form before 1.0.
- **API-70. Rethink the `sc.vmap` mapping tuples.** The `(outer, start, stride)` triples are the
  one construct in the README example a newcomer cannot guess. Consider a named or sliced form
  before 1.0.
- **API-6. A QP-subproblem contract so scaly-sqp can use other QP plugins.** Today `scaly-sqp`
  imports only `include_dir`/`lib_dir` from `scaly_piqp` and its C template calls
  `piqp_setup/update/solve` and reads `qp->result` directly, so a future OSQP, ProxQP or HPIPM
  plugin would be a standalone solver but not an SQP backend. The contract is narrower than
  `render_wrapper`: set up a QP with fixed sparsity, refill values, solve, read the step and
  multipliers in one sign convention, report status and iteration count, clean up. The hard part is
  form reconciliation (two-sided rows and box bounds versus OSQP's single `l <= Ax <= u`, and
  stage-structured solvers) the way CasADi's `conic` layer does it. Documented as a limitation in
  `docs/guide/solver_backends.md`.
- **API-7. Specialized OCP problem/solver tier** in scaly (structured staged OCP lowering to general
  form), then **fatrop** as its consumer plus a casadi-fatrop baseline. osqp / proxqp / acados as
  claims demand.
- **API-77. Sparse pattern holes on template inputs** (the templates plan's deferred P5): `sc.S()`
  with no pattern, bound from a `SparseMatrix` or SciPy argument, with the `p{hex}` instance token
  bare templates already use for `SparseMatrix` arguments. No test, example or benchmark needs one
  yet. Plan: `notes/function_templates_plan_2026_09_27.md` §5 (P5).
- **API-78. Static arguments for templates.** Parameters that take Python values (a horizon, a
  callable, a flag), select an instance and are part of its key, as JAX's `static_argnums`. v1
  refuses body parameters with defaults to keep that syntax free.
- **API-79. Shape inference for `scan` callees** where only the carry is a hole (its shape is
  `init`'s), as `while_loop` already does for its callees. Sliced inputs stay declared: a slice's
  size is not its stride.
- **API-80. More template holes and spellings:** dtype holes bound from numerical arguments
  (`float32` instances), and the keyword declaration form `@sc.function(A=(n, n))`, which could not
  be typed on the numerical side. Also the numerical leaf type: ty refuses a Python float, a list or
  `np.float64` where a leaf takes `np.ndarray`, though the runtime coerces them all, so examples
  still wrap scalars in `np.array(...)`.
- **API-81. Templates for `sc.problem`:** holes in `vars`/`params`, and the decorator's shorthand
  (`vars=3`, names from the body's parameters).
- **API-153. DAEs in implicit integrators and collocation:** semi-explicit index 1, algebraic
  states at the stage points (integrators/MPC plan §7).
- **API-154. Control-invariant and robust (tube) terminal sets** (plan §7).
- **API-155. The generated IPM as an MPC backend,** once Tier 4 #30 (`backend="scaly"`) lands.
- **API-156. Migrate the benchmark problems' hand-written integrators and NumPy plants** onto
  `scaly.integrators`, keeping their `checks.py` gates and CasADi parity.
- **API-179. `ExprOp.SPLINE_EVAL`,** a dedicated interpolation op, only if the gate of the interp
  plan (§10) trips. Evaluated at SP4 (`notes/interp_sp4_report.html`): not tripped, the composite
  evaluation at 0.03x to 0.30x CasADi's time.
- **API-180. Scattered-data interpolants** (radial basis functions, the Gaussian-process posterior
  mean) behind `scaly.interp`'s calling convention.
- **API-181. Piecewise-affine functions on polyhedral or simplicial partitions** (explicit MPC laws,
  PWA models), with point location by a search tree.
- **API-182. B-spline input parametrizations as a transcription option in `scaly.mpc`.**
- **API-183. Shape-constrained fitting online,** through the generated IPM once Tier 4 #30 lands.
- **API-184. Knots as expressions** (CasADi's parametric grid) and free-knot fitting.
- **API-185. Shape-preserving interpolation in n-D,** as a tensor-product Hermite spline with a
  stated definition; SciPy's RGI `"pchip"` is not one (interp plan §3.4).

## Compiler internals

The [completed study](../docs/results/index.md) supplies the current measurements, and §8 of
The 2026-09-07 investigation under
[`notes/perf_2026_09_07/`](notes/perf_2026_09_07/README.md) remains the rationale and validation
record for the completed compiler tasks below.

Start each compiler item by reading how the tools that shaped Scaly solve the same problem, before
designing anything. tinygrad, whose IR and pattern-rewrite infrastructure Scaly's are modelled on,
expands small tensor ops into scalar UOps and simplifies them symbolically (C-44), and has a
scheduler that fuses elementwise producers into their consumers and a symbolic index arithmetic
that turns strided views into closed-form index expressions (C-8 and C-9). MLIR's affine dialect and
its loop-fusion, affine-map and memref-normalization passes are the standard treatment of exactly
the loops we emit, and their design notes state the legality conditions we would otherwise
rediscover. JAX's `vmap` batching rules are the reference for what a mapped derivative rule should
produce without materializing per-trip index tables. The goal is to port the smallest idea that
fits Scaly's two dialects, not to adopt a framework; write down what was read and what was rejected
in `internal/notes/refactorings.md` before the implementation.

### Tier 2+ (PIQP in Scaly)

The living plan is [`notes/piqp_plan.md`](notes/piqp_plan.md) (same text as `claude/piqp-plan.md` in the
claude.ai project; update both together). Tier 1 ends at the tag `tier1-complete`; Tier 2 (general
sparse and dense linear algebra, generated code only) runs on `claude/tier2-sparse`. Tiers 3–4
are the IPM machinery and PIQP proper; Tier 5 (multistage/supernodal) and Tier 6 (platform
selection, optional external kernels) come later. Tier 2 PRs T2-1 … T2-9 and T2-R get ids here when
they start. Review evidence: [`notes/piqp_plan_2026_09_26.html`](notes/piqp_plan_2026_09_26.html).
Reports: `notes/tier2_pr*_report.html`; timings: `notes/perf_2026_09_26_tier2/`.

- [x] **C-101. Step number as a loop-body input (T2-1).** `sc.scan(..., index=True)` and
      `sc.while_loop(..., index=True)` hand the body an `int64` step number. A scan slices it from
      a constant table that lowering never stores: an integer table read one entry per step becomes
      arithmetic on the loop counter (counting down in the backward scan), and a float table whose
      entries along the walk are equal becomes the value (the ones of a summed output's
      cotangent). Procedures holding `int64` values now scalarize, with a store's conversion kept
      as a cast; `where` refuses a Python `bool`, which is what `k == 0` yields
      (`notes/tier2_pr1_report.html`).
- [x] **C-102. `take`, `put_add`, `put` with run-time `int64` indices (T2-2).** On the last axis,
      leading axes kept; an index outside `[0, n)` reads `fill` or drops its value into a scratch
      slot of its own lane, so writes are unconditional and no index reads out of bounds. Forward
      (one and many seeds), reverse and conservative row-local sparsity; `+ - *` with a Python
      integer keep an integer expression integer. Procedures holding them do not scalarize. A
      `put_add` into a scan carry still copies the carry every step, which is T2-3
      (`notes/tier2_pr2_report.html`).
- [x] **C-103. In-place carries through run-time-index updates, proven at the loop (T2-3).**
      `in_place_steps` evaluates every update index and every `take`/`gather`/slice read of a chain
      link for all steps at once from the constant tables the loop slices (and the step number),
      and requires each read to miss the writes of the updates after its link. The carry keeps
      the scratch slots of padded lanes after its entries. `A^T y` by row accumulation at
      n = 2000: 2 283 → 51 µs (`notes/tier2_pr3_report.html`).
- [x] **C-104. `SparseMatrix` (T2-4).** `scaly.linalg.sparse`: a static CSC pattern in NumPy with
      an `Expr` of values. Construction from symbols, patterns, COO, dense expressions, SciPy and
      sparse Jacobians/Hessians; union add, scaling, Hadamard, row/column scaling, transpose,
      products with dense operands and with sparse ones (pattern at build time), blocks, triangles,
      diagonal, `to_dense`. Values cross the `Function` boundary compactly with `sparsity` as the
      output metadata. `Expr` operators defer to operands that set `__array_ufunc__ = None`
      (`notes/tier2_pr4_report.html`).
- [x] **C-105. Symbolic `LDL^T` analysis (T2-5).** `scaly.linalg.symbolic.analyze`: natural, RCM,
      SuperLU MMD and `auto` (least update work) orderings; the permuted lower triangle with a map
      back to the input values; elimination tree, postorder, row and column patterns of `L`, the
      left-looking update lanes, statistics, and consecutive segments chosen by a DP over a padded
      work model. Refuses an ordering whose updates exceed 50 M multiply-adds (500 M since C-109). MMD fill equals
      SuperLU's (`notes/tier2_pr5_report.html`).
- [x] **C-106. Generated dense kernels (T2-6).** `cholesky`, `ldl` (no pivoting, packed) and
      `solve_triangular` as expression ops with loop lowerings (row Crout with four partial sums;
      row sweeps for transposed solves; four rows per pass for several right-hand sides),
      straight-line code up to order 8, derivatives in both modes and to second order, and
      conservative sparsity; `scaly.linalg` adds `cho_solve`, `ldl_solve`, `ldl_unpack`, `solve`.
      Gate met at `-O3` (≤ 1.6× OpenBLAS at n = 32–64, faster below); at gcc 11's `-O2` the
      multi-RHS solve and `matmul` stay 3× (`notes/tier2_pr6_report.html`).
- [x] **C-107. Decide the JIT's default optimization level.** gcc before 12 does not vectorize at
      `-O2`, which costs dense kernels and `matmul` up to 3× on Linux (`notes/tier2_pr6_report.html`);
      clang vectorizes at `-O2`. Decided 2026-09-26: `-O2`, plus `-ftree-vectorize` when the
      compiler is GCC before 12 (T3-R: GCC 12 and later vectorize at `-O2` with the `very-cheap`
      cost model, which the flag would replace, so they get nothing added; the CasADi benchmark
      harness takes the same flag from `jit.vectorize_flags`). `-O3` was measured first and
      rejected: +7% compile time on ordinary functions but 1–5.8× (up to 376 s) on 385 kB
      straight-line ones, and a cold suite of 23 min instead of 7 on the Mac. Still to check on the
      Linux VM that GCC 11 with `-ftree-vectorize` recovers the 3× of `tier2_pr6_report.html`.
- [x] **C-108. Generated sparse `L D L^T` and solves (T2-7).** `scaly.linalg.SparseLDL`: one
      in-place `scan` per column segment over a single carry `[L | D | work | 0 | scratch]`, tables
      padded in range (no bounds checks), factor/solve split, implicit derivatives two levels deep
      (`custom_derivative`), multi-seed forward mode through a custom rule as one `vmap`. Reverse
      mode no longer walks back through a constant-zero cotangent. MPC/grid within 1.3–1.7× of an
      up-looking C LDL, random QPs 2.5× (below the 3× kill line) (`notes/tier2_pr7_report.html`).
- [x] **C-109. Ragged update loops for the sparse factorization (T2-8).** Each update lane loads four index
      tables and three scattered values; the C baseline walks one contiguous column range per `k`.
      Group lanes by `k` into an inner loop of run-time length (T2-8), which also shrinks the tables
      from O(flops) to O(nnz(L)) and generation time with them (10.8 s at 3.1 M lanes). Done:
      `ragged_add`/`ragged_dot` (run-time ranges over fixed maps, closed under AD), the factor
      and sweeps rewritten on them, lazy lane tables, interval-first in-place proof, one constant
      table per content. Factor 1.0–1.4×, solve 0.5–1.6× the C baseline; 2.5 s to generate at
      nnz(L) = 153 k (`notes/tier2_pr8_report.html`).
- [x] **C-110. Schedules, refinement, health and options for the sparse factorization (T2-9).**
      `SparseLDL(schedule="auto"|"scan"|"unroll")` (straight-line code at most
      `sparse_unroll` = 1000 operations; the factorization 5–8× faster than the loops there); `solve(refine=k,
      tol=)` fixed or adaptive (a `while_loop`) refinement inside the implicit rules;
      `inertia()` and `health(signs=, pivot_tol=, x=)`; a per-component sparsity override on the
      solve. Generic: `sc.options(dense_unroll=, sparse_unroll=, max_trajectory=)`, the dense unroll
      decision as a node attribute, `custom_derivative(sparsity=)` honored by calls, `vmap`, `scan`
      and `while_loop` bodies and the structured sparse Jacobian, a reverse-pass memory guard, loop
      patterns that skip loops reading nothing that depends on `wrt`, and a fix for scan patterns
      with a carry of 256 or more entries. Examples: `examples/sqp_newton_sparse.py`,
      `examples/kalman_update.py` (`notes/tier2_pr9_report.html`).
- [ ] **C-111. Generation time of straight-line code.** An unrolled graph costs about 1.2 ms per
      scalar operation to generate, most of it in lowering and the program passes (`match.rewrite`,
      `fuse_elementwise`, `fold_arith`, `pack_workspace`), linearly in the size. It caps
      `sparse_unroll` and `dense_unroll`; halving it would let both double
      (`notes/tier2_pr9_report.html`).
- [x] **C-112. Loop-invariant inputs for `while_loop` (T3-1).** `sc.while_loop(..., params=...)`:
      tensors every step reads unchanged, taken by the body after the carry (and the step number)
      and by the condition after the carry, and passed to both calls by pointer. The node carries
      them as arguments and an explicit `index` flag; forward mode passes their tangents as further
      params, reverse mode sums their cotangents over the steps taken in the backward scan's carry,
      and the sparsity rule brings their pattern in at every step. An `int64` constant param is a
      table for the in-place proof. SparseLDL's adaptive refinement now carries `[x | r]` only:
      its overhead over a plain solve falls 12-45% (`perf_2026_09_26_tier3/t3_1_while_params.py`,
      `notes/tier3_pr1_report.html`). Multi-tensor carries and an int64 carry spill (T2.g) stay
      unneeded: an integer in a carry is exact as a `float64` up to 2^53.
- [x] **C-113. Tier 2 review round (T2-R).** Five review agents (lowering and in-place proofs, AD
      rules, `linalg`, performance, docs and tests). Fixed: generated names shadowing inputs,
      `-0.0` tables and literals, the vmap sparse Jacobian bypassing a forward rule, `put`'s reverse
      rule with repeated indices, stale declared patterns, NaN second derivatives of `x**p` at 0, a
      forward rule's unread tangents formed anyway (jvp of a solve differentiated the
      factorization), solve-variant name collisions, unchecked `symbolic=`, hoisting a while
      loop's count away from its loop. Optimized: `fuse_elementwise` and `pack_workspace` no
      longer quadratic, the JIT call reads array addresses through the buffer protocol (4.2 → 3.1
      µs for a trivial call; part of C-100), in-place proof for loops with constant indices only
      (the adaptive refinement loop). `notes/tier2_review_report.html`.
- [x] **C-114. Unpadded updates in the sparse factorization.** Done by C-135, which goes further.
      Each step runs the padded groups of its segment: 22–65% of the group slots are real work. A group loop over the real entries
      of row `j` (`r_ptr[j] .. r_ptr[j+1]`, each with its column's range and weight read from global
      tables) needs a two-level ragged op whose scale is `-D[k] L[j,k]` read from the carry. A
      hand-written C version runs grid 30×30 in 12.7 µs against 20.6 (and the C baseline's 18.0),
      MPC N = 100 in 20.7 against 35.8; neutral on dense-ish QPs. Needs verify, AD and sparsity
      rules and the in-place proof for the new op (review prototype `~/review-agents/perf/`).
- [ ] **C-115. JIT cache key without lowering.** `render_c_module` (lowering, optimization,
      rendering) runs before the disk-cache lookup, so every new process pays the whole Python
      generation (2.6 s for an unrolled MPC N = 20 factor) even when the library is cached. Key on
      a structural hash of the graph (callees, rules, attributes), the scaly version and the
      compiler flags instead; the risk is a key that misses an input.
- [ ] **C-116. Compile time of straight-line code.** About 1 000 one-element buffers stay live
      until paired stores at the end; storing each result into `res` as it is computed halves
      gcc's time at `-O2` (2.5 → 1.3 s for an unrolled MPC N = 20 factor) with no change at run
      time. Needs an ABI rule on whether `res` may alias `arg`. Related: C-111.
- [ ] **C-117. Dense dot kernels and small solves.** Explicit `double2` accumulators with two
      columns per pass make the order-64 Cholesky 31% faster at `-O2` than the generated code at
      `-O3` (decide with C-107). For small sparse solves: fold `y / D` into the backward sweep and
      write the final permutation straight into the output (5–9% on grid 30×30), and move large
      stack arrays (`double s0[n]`) into the workspace. The dense IPM backend makes this urgent
      (T3-6): its Cholesky is 80% of a step and runs at about 3 G multiply-adds/s at n ≈ 300, a
      quarter of Eigen's blocked `LLT` in PIQP (PRIMALC5 1.26 ms against PIQP's whole 0.39 ms
      iteration). Register blocking of the Crout dot products saves loads but not 4x; a blocked
      factorization with register-tiled update kernels is what closes it.
- [x] **C-118. Tier 2 on the Mac.** The first macOS run (Apple clang 21): 1 575 passed, 2 failed.
      Both failures were one race: `recompile()` removes `~/.cache/scaly/jit/<key>`, and a build
      writing into that directory at the same moment lost its temp file. A build now starts over
      once when its directory vanishes, and never retries a compiler error. `ty` found 30
      diagnostics in the Tier 2 tests (mypy-style `# type: ignore[...]`, `str` where a `Literal`
      is declared), and `SparseMatrix.block` was annotated narrower than it accepts.
- [x] **C-119. Sparse matrices as `Function` arguments and results.** `sc.S(name, pattern)` puts
      the pattern in the signature: the body gets a `SparseMatrix`, a symbolic call must pass one
      with exactly that pattern, an evaluation a SciPy matrix with exactly that pattern (explicit
      zeros count), and a result comes back as a `SparseMatrix` or a `csc_array`. `sc.S(name, ...)`
      infers an output's pattern. Interface only: the C signature carries the values vector, and
      the generated code is identical to a `Function` over it (a test pins this); an output's
      pattern becomes its header sparsity metadata, as `output_sparsities` already did. Trees gained
      `infer` and `sparsities`, and `SymbolicValue` marks a library value that stands for an `Expr`
      leaf. A Python call costs about 10 µs more than the flat seam, mostly SciPy's constructor
      (`perf_2026_09_26_tier2/followup_sparse_args.py`).
- [x] **C-120. Derivatives with respect to an expression that is not an input.** Forward mode and
      the sparsity analysis seeded only `INPUT` nodes, so `jacobian`, `hessian`, `jvp`,
      `sparse_jacobian` and `sparse_hessian` in a slice of an input returned zeros without an
      error (reverse mode was right). A `while_loop` body sees only its carry, so a Newton step there
      differentiates in a slice of it: the hanging-chain example never converged. `independent`
      (in `ir/expr.py`) now puts a stand-in input for such a `wrt`, and `jvp`, `jvp_many`, `vjp`,
      `vjp_many`, `jacobian_sparsity` and the colored, reference and Hessian sparse paths map the
      result back; the stand-in also keeps `simplify` from folding `x[:3][1]` past the `wrt`.
- [x] **C-121. Examples for the Tier 1 and 2 capabilities.** Six control, optimization and solve
      problems in `examples/` (indexed in `examples/README.md`): value iteration for a slippery
      grid (segment ops, `while_loop`, `int64` policy), the LASSO by ADMM (dense Cholesky,
      soft thresholding), LQR weight tuning through the Riccati and closed-loop `scan`s (step
      index, negative stride, reverse mode through loops), truss sizing (`put_add`/`take` with a
      run-time bar table), optimal heating of a plate (`SparseLDL` time stepping, `sc.S` between
      Functions) and a hanging-chain calibration (Newton in a `while_loop`, `isfinite`,
      `custom_derivative`). `tests/integration/test_examples.py` checks each against NumPy/SciPy.
- [x] **C-123. Tier 3 harness (T3-0a).** `tests/data/maros_meszaros/`: the 48 Maros–Mészáros problems
      with n + m ≤ 1000 from `qpsolvers/maros_meszaros_qpbenchmark` (Apache-2.0, NOTICE with the
      commit), stored as `.npz` as distributed (408 kB). `tests/solvers/ipm/problems.py` maps them to
      PIQP's form (unit rows become bounds, `l == u` rows equalities, |value| ≥ 1e19 absent,
      since the set sometimes stores 1e20 as 9.999999999999998e19) and generates infeasible
      problems and linear MPC QPs. `tests/solvers/ipm/piqp_trace.c` runs vendored PIQP with verbose output
      and prints its result at full precision; `exact_trace` reads every iteration at full
      precision from runs truncated at `max_iter = k`, which a test shows reproduce the full run.
      PIQP solves all 51 with both backends; the backends take different iteration counts on 4
      (QCAPRI 84 sparse against 32 dense), so the reference's trace gate must pick a backend. On
      the infeasible set PIQP runs two primal-infeasible problems to the iteration limit and
      calls an unbounded one primal infeasible; the reference has to reproduce both
      (`perf_2026_09_26_tier3/t3_0_piqp_baseline.py`, `notes/tier3_pr0a_report.html`).
- [x] **C-124. The NumPy PIQP reference (T3-0b).** `tests/solvers/ipm/reference.py` ports PIQP 0.6.2's
      solver loop, Ruiz equilibration, preprocessing and KKT system with iterative refinement line
      by line, solving the regularized KKT matrix with SuperLU. Decision traces (status, iterations,
      rho and delta to 1e-3) match PIQP's sparse backend on 46/48 of the stored subset (gate: 90%)
      and on every problem where PIQP's two backends follow the same path; full-precision values
      agree to 1e-14 early and 1e-8 to 1e-5 near convergence; refinement-always matches on 48/48. Found and
      reproduced: signed maxima for box residuals, cost scaling aliasing the Ruiz stopping test,
      IEEE `0/0` in the mu rate and sigma. 23 mutants of the decision rules, all killed
      (`perf_2026_09_26_tier3/t3_0b_reference_gate.py`, `notes/tier3_pr0b_report.html`).
- [x] **C-125. An output named like an input is read back from the output buffer.** Silent wrong
      numbers: lowering keyed buffers by name, so an output's buffer replaced the input of the same
      name and every read of that input went to `res[k]`. Found by the TinyMPC example (C-126),
      which names its output `state_next`, and again by T3-2's tests (`examples/simple.py`:
      `(x, y) -> (y, z)`). Fixed: such an output gets a buffer of its own, output rules reach their
      buffer by position, and the C entry maps parameters to `arg[i]` and `res[i]` by position (the
      typed header already named them `y_in`/`y_out`).
- [x] **C-126. TinyMPC in Scaly (`examples/tinympc`).** TinyMPC's ADMM (the library's `solve`, box
      and second-order-cone constraints, affine dynamics, warm starts) written once in Python and
      generated per problem: a `while_loop` over ADMM iterations around two `scan`s (Riccati backward
      pass, rollout). A NumPy port of the library's loop is the test oracle; the three problem
      families of `mcu-solver-benchmarks` (random QP-MPC, safety filter, rocket landing) are
      closed-loop scripts. `benchmark/run_benchmark.py` replays the same warm-started solves in the
      TinyMPC library (pinned commit, built from source) and in the generated C: identical iteration
      counts on all 52 instances; geometric-mean speed-up 1.38x / 2.79x / 2.66x per family at -O3
      (slower, to 0.80x, for 16-32 inputs, where dense products dominate: C-117); 4-63 KiB of
      object code against 200 KiB. With the bounds as constants (two thirds of the carried loop
      invariants) solves take 2-6 % less, a measure of what C-112 buys. Tests:
      `tests/integration/test_tinympc.py`, 12 mutants, 11 killed (the survivor is equivalent up to
      rounding). Report: `notes/tinympc_benchmark_report.html`.
- [x] **C-128. Ruiz equilibration as generated code (T3-2).** `scaly.solvers.ipm`: `QPStructure`
      (the patterns of `P`'s upper triangle, `A`, `G`, and which bounds are finite; rows free on
      both sides stay as zero rows bounded by [-1, 1]), `QPValues.preprocess`, and `ruiz`/`scale`:
      PIQP's equilibration as a `while_loop` whose params are the matrix values and `c` and whose
      carry holds the scalings, cost scaling and its aliasing included. Scalings, scaled data and
      box scaling match the reference to 1e-13 on all 48 stored problems with and without cost
      scaling. 8-325 us per call, below PIQP's whole setup except on dense `P`
      (`perf_2026_09_26_tier3/t3_2_ruiz.py`, `notes/tier3_pr2_report.html`).
- [x] **C-127. The KKT backends (T3-3).** `scaly.solvers.ipm.KKT`: PIQP's `KKTSystem` (the box and
      inequality barrier regularizations, the reduced right-hand sides, dual and slack recovery
      for one- and two-sided rows) over a dense backend (the condensed matrix assembled from
      sparse products, densified once, Cholesky) and a sparse one (the upper-triangle KKT matrix,
      `SparseLDL`); `Iterate` holds variables in PIQP's layout. Both match the reference's solves to
      1e-7 on 10 stored problems, free rows, MPC and two-sided rows. One factorization and two
      solves take 1.3-1.8x PIQP's whole iteration (sparse) and 1.5-3x (dense); refinement and
      factorization retries were left for Tier 4 (they moved into Tier 3 with C-129) (`perf_2026_09_26_tier3/t3_3_kkt.py`, `notes/tier3_pr3_report.html`).
- [x] **C-129. The generated interior-point solver (T3-4, T3-5).** `scaly.solvers.ipm.Solver`: PIQP
      0.6.2 from Ruiz to `restore_dual`, its loop one `while_loop` whose carry holds the iterate, the
      proximal centres and PIQP's info (residuals and termination, initial point, Mehrotra
      predictor-corrector, fraction to boundary, proximal updates, boundary shift, fine-tune switch,
      infeasibility detection), with a per-iteration trace. Factorization retries and iterative
      refinement (#29) moved in from Tier 4: condensed Cholesky fails on half the stored problems
      along PIQP's path. `Kernels` holds one factorization attempt inside PIQP's retry loop and the
      solve with PIQP's refinement, as Functions the initial point and the loop share;
      `SparseLDL.solve_with` solves with a factor carried out of a loop. Deviation: the dense
      backend also fails a Cholesky pivot below eps times its diagonal entry while refinement is off
      (its sign is noise; LAPACK fails exactly where such pivots come out negative). Gate, each
      backend against PIQP's run of the same backend on 62 problems: status 62/62; decision traces
      48/48 (sparse) and 47/48 (dense, QADLITTL) where PIQP's own backends agree; the reference
      matched step by step to 1e-12 … 1e-5; PIQP's ×100 retries reproduced on nonconvex QPs.
      20 mutants, 19 killed (the survivor is equivalent up to rounding). Time: 1.96× PIQP's solve
      (sparse) and 2.29× (dense), geometric means over 51 problems (C-132 found these mostly
      measurement); 0.7–5.2 s cold first call
      (`perf_2026_09_26_tier3/t3_4_ipm.py`, `notes/tier3_pr4_report.html`).
- [ ] **C-130. QRECIPE's sparse path stalls.** 34 iterations against PIQP's 19 (both backends of
      PIQP agree on 19; the problem is rounding-sensitive). From iteration 7 the generated LDL^T's
      rounding (MMD ordering) moves the path; at iterations 11–13 the steps shrink to 1e-21 until a
      factorization with an exactly zero pivot retries. Candidates: a noise-pivot test for the sparse
      backend (needs the sum of |L|^2 |D| per pivot), or PIQP's AMD ordering for KKT matrices.
- [ ] **C-131. The retry loop copies the factor.** C-138 removed most of it; what remains is the
      init's copy into a loop's store (a dead private init used as the store). The factor travels through the retry loop's
      carry: about five copies of `nnz(L) + n` values (sparse) or `n^2` (dense) per factorization,
      +5–23% on factor + two solves against T3-3's direct factorization (interleaved A/B). T3-6 took
      one copy out (one concatenation into the carry); what is left costs about 1.5 µs of a 32 µs
      sparse step (QSC205). The in-place carry proof does not apply: it counts a value reached
      through a call, here the factorization, as reading every carry entry. The T3-R performance
      review lists the rest, 2-7% sparse and 3-5% dense together: a dead private init used as the
      loop's store (the factorization's three scans each copy the last one's buffer), the final
      while carry returned as a pointer into the store, a call writing straight into the concat
      slice it feeds, and the first attempt peeled out of the retry loop (no zero record).
- [x] **C-132. The generated IPM's speed (T3-6).** A fair protocol first: both sides warmed up
      (the trace driver repeats PIQP's solve in-process, `SCALY_TRACE_REPEAT`; the first calls of
      a burst run up to 1.6x slower on Apple Silicon) and the minimum of equally many samples; the
      1.96x of C-129 was mostly cold starts and medians under a busy indexer. C-side profiling (a
      piece run K times in a loop, with a run-time zero multiplier so it is not folded) put a sparse
      step at 1.1-1.4x PIQP's iteration, the factorization its largest part. Changes: the loop carries
      the residual vectors instead of recomputing them at the top of each step; the Function-call
      layer passes conforming arrays through as they are (C-100) and reuses one workspace per thread
      (the kernel never reads workspace it did not write; a test poisons it with NaN, another runs
      calls from 8 threads); one copy of the factor less per retry loop (C-131). Result, 51 problems:
      sparse 1.49x PIQP's warmed solve where that takes at least 50 µs (35 problems), 2.11x over
      all (tiny problems pay a ~12 µs Python call against 1-5 µs C-timed solves), 0.91x PIQP's
      setup + solve; dense 2.45x / 2.96x / 1.67x (its Cholesky kernel, C-117). 7-12% of it from
      these changes (A/B on 12 problems). Next levers: C-114 (the factorization is 40-60% of a
      sparse step) and C-117 (`perf_2026_09_26_tier3/t3_4_ipm.py`, `notes/tier3_pr5_report.html`).
- [x] **C-133. Tier 3 review round (T3-R).** Five agents (fidelity to PIQP, IR/lowering/JIT, AD
      through while params, performance, tests and docs). Fixed, each with a regression test:
      three derivative Functions of while loops that collided by name (a crash at lowering: one
      body under two active sets, a condition shared by two bodies, the step-number flag); a
      reverse accumulator for params only the condition reads; a cold-cache compile race between
      threads; the in-place proof broadcasting int64 tables to `max_iter` rows (1.7 GB); a
      Function/handle cycle holding workspaces until collection; `-ftree-vectorize` changing GCC
      12+'s cost model (now GCC < 12 only, shared with the CasADi harness); dense cost scaling
      using the sparse Ruiz quirk; bounds declared finite arriving infinite (`INVALID_BOUNDS`); a
      failed first factorization returning garbage; infeasibility and NUMERICS exits reporting
      stale state; `max_iter=0`. Tests: tolerances with 20x headroom (the IPM tests pass with FMA
      contraction off and fast), the dense exemption read from the run, the trace harness's
      file leak (510 MB) and build race, stricter reference tests, tests moved to the mirrored
      `tests/solvers/ipm/`. Speed: extremum reductions in four lanes, sorted segment extrema run
      by run, Ruiz maxima per matrix, the refinement tolerance and the data vector out of the
      step: -11 to -25% sparse, identical iterations; sparse now 1.26x PIQP's warmed solve above the
      call floor (1.85x over all 51, 0.81x its setup + solve), dense 2.24x. 10 mutants on the fixes,
      all killed; the other fixes' tests were each run against the old code and fail there
      (`notes/tier3_review_report.html`).
- [ ] **C-134. IPM step overheads from the T3-R review.** Gate the second `regularized` pass on a
      retry having changed rho and delta (a one-pass `while_loop`; 2-5%); pass the residuals'
      data-only maxima (`c`, `b` and bound terms over their scalings) as loop params (1-3%); and
      stop the scalarizer from unrolling a procedure made of one long reduction
      (`ipm_ruiz_go`: `cc` 0.85 → 0.58 s). See also C-131.
- [x] **C-135. The sparse factorization as one loop nest (supersedes C-114).** `ir.expr.sparse_ldl_factor`
      (`ExprOp.SPARSE_LDL`), `SparseLDL(schedule="loop")`, the default above `sparse_unroll`: a
      left-looking factorization whose column updates come in chunks of up to four columns with the
      same rows from `j` down (`SymbolicLDL.chunks`), one pass over the work column per chunk with the
      sum in a register, no padding, no per-step call, the work column cleared as it is read. The
      update order per entry is the scan's, so the factor is the `scan` schedule's bit for bit but for
      the sign of a zero or a NaN (every test matrix, triangle and ordering; inf and NaN too; the
      review found the zero signs). Chunks of up to four columns, eight since C-139. Factorization 1.5-2.2x faster (C prototype
      on 11 KKT patterns); the generated sparse IPM 0.81x its time (geometric mean over 18 problems,
      identical iterations and solutions). The op has no derivative of its own (the solve's implicit
      rules never need it; `schedule="scan"` stays differentiable): forward mode now forms only the
      tangents a call's derivative reads, the multi-seed call pruning and map rule tolerate a tangent
      they do not read, and the map reverse rule drops constant-zero cotangents. Structural sparsity:
      a factor column depends on its elimination subtree. 16 mutants: 14 killed, 1 equivalent (the
      op's run-time-index membership; scalar expansion already declines run-time loop bounds), 1 dead
      branch removed (`perf_2026_09_27_ipm_speed/`, `notes/ipm_speed_report.html`).
- [x] **C-136. The dense Cholesky by register tiles (the Cholesky half of C-117).** Above
      `dense_unroll`, `cholesky` lowers to Crout by 4x4 tiles of `L`: a tile's dot products over the
      columns left of it run together (16 multiply-adds per 8 loads instead of 2 loads each), then the
      terms inside the tile. 3.3x the old kernel at n = 300 (12 GFMA/s on the M3), 2-2.5x at
      n = 111-230; the generated dense IPM 0.56x its time (geometric mean over 16 problems). Each dot
      product is four partial sums over contiguous quarters, as accurate as the old kernel's four
      interleaved ones: one running sum per entry was 2% faster but let QSHARE1B fail a late
      factorization (pivots there are rounding noise) and take 56 iterations instead of 27. Over the
      55 stored problems two dense paths change: QBEACONF 23 -> 17 iterations, QBORE3D 19 -> 21
      (PIQP's dense backend takes 23 and 20); the decision-trace gate holds. `ldl` keeps the
      entry-at-a-time Crout loops. 7 mutants, all killed (`perf_2026_09_27_ipm_speed/`,
      `notes/ipm_speed_report.html`).
- [x] **C-137. Elementwise producers fuse into max/min reductions (part of C-89).** The four-lane
      extremum read its vector in up to six statements (four lane starts, the blocked loop, the
      tail), and `fuse_elementwise` inlines a producer only into one consumer statement: every
      `norm_inf(x - y)`, residual maximum and step-length minimum stored its vector first. The reads
      now sit in one loop run once, which `unroll_unit_loops` removes after fusion: the same
      arithmetic, no vector temporary. 0.975x on the sparse IPM (geometric mean, 18 problems; up to
      0.89x where the step dominates). 2 mutants, both killed.
- [x] **C-138. Loop carries that are not copied (most of C-131).** Three copies per loop left:
      a while loop's final carry is now its store's first slot read in place (a loop that stopped on
      an odd step moves the second slot there first; one of zero trips copies nothing); the KKT
      solve's refinement sits behind one gate whose carry is the solution alone (with refinement
      off, which is most solves, it moved about 3 600 values per solve through two carries); and the
      first factorization attempt runs before the retry loop, which then usually takes no step
      (no zero record, no copy of the factor out). Sparse 0.957x, dense 0.963x (18 problems,
      identical iterations; solutions identical but for 8e-15 on the one problem that refines).
      7 mutants: all killed, two by new tests (a refinement-off solve is the plain solve bit for bit;
      with no retries allowed a failure still turns refinement on).
- [x] **C-139. Sparse factorization chunks of up to eight columns.** `SPARSE_LDL_MAX_WIDTH` 4 -> 8,
      one loop per width: a supernode wider than four columns updates the work column in half the
      passes. Still the scan's factor bit for bit (the tests now cover every width 1-8). Sparse IPM
      0.977x (23 problems; DUAL3 0.90x, mpc 27x6x30 0.93x, QE226 0.94x). 32-bit index tables were
      tried in the C prototype and gained 1-5%, not enough for a second table type. 2 mutants,
      both killed.
- [x] **C-140. The sparse solve as one op (the solve half of C-117's small-solve items).**
      `ir.expr.sparse_ldl_solve` for `schedule="loop"`: `b` permuted into a work vector, the forward
      sweep by chains of up to eight columns (`SymbolicLDL.solve_chunks`, fundamental supernodes:
      each row below a chain updated once with the sum in a register), then the backward sweep with
      the division by `D` and the output permutation folded in. The solution is the scan's bit for
      bit. Linear in `b` with that derivative (forward, multi-seed and reverse); a derivative in the
      factor is refused. The multi-seed forward mode now also forms a call argument's tangent only
      when the callee's derivative reads it (the implicit rules take the factor as an argument).
      Sparse IPM 0.975x (23 problems, every one faster or equal, solutions identical). Chunking the
      backward sweep too would need four partial sums per column aligned per column to stay bit for
      bit; not done. 10 mutants: all killed, two after a new test (a factor given as an input).
- [x] **C-141. Review round for C-135 ... C-140.** Three agents (IR/lowering, AD, IPM and tests).
      Fixed, each with a test that fails without the fix: `schedule="loop"` failed to compile with
      one entry in `L` or one column (a branch for a chunk width the analysis does not have folded a
      table read past its end: `fold_arith` now leaves a constant index outside its table alone,
      and the loop nests have loops only for the widths present); `SymbolicLDL.chunks` hid the lane
      tables it shares a cache with; the elimination-subtree sparsity rule crashed on a diagonal
      matrix; reverse mode through a plain Function taking the factor asked the factor for a
      cotangent (a call is now differentiated only in the formals that depend on the variables, as
      maps and loops are); forward mode through a call with an argument whose tangent is a known zero
      regressed when the probe of which tangents are read could not be built (it falls back to
      forming them all); the builders now check every table they index. Claims corrected: the loop
      schedules equal the scan's bit for bit but for zero and NaN signs; third derivatives in the
      matrix need `schedule="scan"`. Gaps closed: the dense gate now bounds the iterations of a
      rounding-sensitive problem (within 3 of one of PIQP's runs, which kills a less accurate
      Cholesky: QSHARE1B 56 against 27); refinement takes no step when the residual is already
      small; zero retries with refinement on. 7 mutants on the fixes, 6 killed, 1 equivalent (loops
      for every width again: the `fold_arith` fix alone makes them harmless).
- [ ] **C-122. `Expr` indexing papercuts.** `x[np.int64(2)]` is refused (a Python `int` works), and
      `x[np.array([0, 2])]` fails with NumPy's truth-value error instead of pointing to `sc.gather`.

### Tier 1 primitives

The operations a solver written in Scaly needs, for one fixed sparsity structure per generated
solver. Plan, ordering and test strategy: [`notes/tier1_primitives_plan_2026_09_25.html`](notes/tier1_primitives_plan_2026_09_25.html);
the per-step reports sit beside it as `notes/tier1_pr*_report.html`.

- [x] **C-82. Comparisons, `bool`, `select`, `isfinite`, `copysign`, `cast`.** Elementwise ops in
      both dialects with AD, sparsity, folding and C spellings; `bool` crosses the `double` ABI as
      0.0/1.0.
- [x] **C-83. Options system, non-smooth derivative conventions, `reduce_max`/`reduce_min`.** `sc.options(nonsmooth="split"|"first"|"error")`, read at graph build time; NaN-propagating extremum reductions.
- [ ] **C-89. Keep reduction accumulators in registers and fuse elementwise producers into them.** `sum`, `max` and `min` store the accumulator through the output pointer every trip and run 6-7x slower than NumPy at n = 10^6 (`notes/tier1_pr2_report.html`); `norm_inf` also materializes `abs` first. Scalarize must learn `ASSIGN` or the rule must stay pointer-free.
- [ ] **C-90. Rewrite `select(x > c, c, x)` and friends to `fmin`/`fmax`.** Nested ternaries on random data stay branches under gcc and run 9x slower (`notes/tier1_pr1_report.html`).
- [x] **C-84. Accumulating `scatter`, segment reductions, linear-time `gather` VJP.** The index pattern picks a parallel store (distinct destinations) or a `REDUCE` accumulation; the `gather` adjoint is one scatter.
- [x] **C-85. `scan` with ping-pong carry buffers and trajectory-storing reverse mode.** One SERIAL loop per scan; code size constant in the step count; forward and reverse derivatives are scans.
- [x] **C-91. One reverse scan per forward scan.** Reverse mode differentiates the used outputs of
      one scan, and of one call, together when it reaches the last of them: one backward scan takes
      the final carry's cotangent and every stacked output's (`_group_key` in `ad/reverse.py`).
- [x] **C-86. Differentiable `while_loop` with a static `max_iter`.** A SERIAL loop with `BREAK_IF` and an `exit_var` trip count; forward mode is a while loop over `[c, dc]`, reverse mode a masked `max_iter`-step backward scan.
- [ ] **C-92. Run the backward pass of a `while_loop` for the steps taken only.** It runs all
      `max_iter` masked steps and evaluates the adjoint body in each, so a Newton solve that stops
      after 7 of 50 steps pays 50 backward steps (`notes/tier1_pr5_report.html`). Start the backward
      loop at the step count, which needs a scan whose trip count is read at run time.
- [x] **C-87. In-place carry updates under a conservative donation rule.** `index_add`/`index_set` chains rooted at the carry lower to an `_inplace` body over one aliased slot; the rule is `in_place_chain` in `passes/lowering.py`.
- [x] **C-88. Custom derivatives on a `Function`.** `sc.custom_derivative(fn, jvp=, vjp=)`, honored through `CALL`, `VMAP` and multi-seed forward mode.
- [x] **C-93. Multi-seed forward mode through `scan` and `while_loop`.** `jvp_many` carries every
      seed in one tangent loop (carry `[c, Dc]`, `Dc` seed-major) instead of one loop per seed; a
      `CALL` whose callee holds a loop is differentiated in place, and loop inputs that are a strided
      view of another loop's stored carries are read in place. A dense Hessian through a scan is a
      fixed number of loops for any horizon (`notes/tier1_pr8_report.html`). Also fixed: the JIT
      kept each call's workspace alive until the next garbage collection.
- [ ] **C-94. Drop unused carry components of a scan.** Forward over reverse carries `[λ, λ̇]`
      through the adjoint scan and uses only `λ̇`. Low value now: with every seed in one carry, `λ`
      is 2 of 2 + 2N entries, and for a nonlinear body `λ̇` reads `λ`, so it cannot go at all. What
      is dead is the per-step `adj_u` store and single-slot stacked outputs no one reads.
- [ ] **C-95. Share the primal of a `CALL` with the loops its reverse mode inlines.** The gradient
      through `f(shoot(x))` keeps the call for its outputs and inlines the body's scan for the
      stored carries, so the rollout runs twice (`notes/tier1_pr8_report.html`).
- [ ] **C-96. Fuse adjacent elementwise loops.** Zeroing, copies, scaling, scatter-adds and the
      final transpose each lower to their own loop; a Hessian at N = 10 spends most of its
      non-scan time in them.

### Now

Ordered by measured payoff. The numbers are the 2026-09-07 note's, on the reference machine at the
protocol's compile flags.

- [ ] **C-97. Read and write scan data in place.** The adjoint scan of a Hessian reads a reversed,
      scaled copy of the tangent trajectory (`20 * Ẋ[N-1-k]`); reading `Ẋ` at a negative stride
      with the scale folded into the body removes a pass over N² entries and its buffer, and a
      scan could write its stacked outputs straight into the result. Measured −16 to −26% on the MPC
      Hessian in hand-edited C (`notes/tier1_pr9_report.html`).
- [ ] **C-98. Exploit Hessian symmetry and identity seeds.** With identity seeds on a scanned input,
      seed `s` is zero before step `s` forwards and only rows `t ≥ s` are needed backwards; compute
      one triangle and mirror it. Measured −38 to −43% (RK4) and −13 to −25% (MPC) by hand.
- [ ] **C-99. No ping-pong buffer for small carries.** A carry of a few doubles can stay in
      locals; −40% on the MPC gradient, but 12% slower above about 32 doubles, so gate it.
- [x] **C-100. Trim the Function-call layer.** A trivial JIT call cost 4.6 µs, 2.7 of them in
      flattening and validating inputs before the ctypes call. (C-113 took the address reads from
      `arr.ctypes.data` to the buffer protocol: `run` 4.2 → 3.1 µs.) Done in T3-6 (C-132): an array
      already of the declared shape, dtype and layout passes `L.flatten_numerical` and `run`
      as is, and `DType.numpy()` is cached: a trivial call takes 2.9 µs, a 9-input one 12 instead of
      22 µs.
- [x] **C-43. Lower matmul by layout.** `_lower_matmul` emits every product as
      `for i { out[i] = 0; for k out[i] += A[i,k] v[k] }`, a serial add chain per output that the C
      compiler cannot break without reassociation; `casadi_mtimes_dense` has the same shape, which is
      why both are slow. Emit the reduction loop outermost when the reduction axis is the matrix's
      slow axis (`v @ A`, and `A.T @ v` after the fold below), so the inner loop runs over independent
      outputs and vectorizes; emit a blocked dot with four accumulators as four unrolled statements
      when the reduction axis is contiguous (`A @ v`), because a four-trip inner loop becomes gathers
      under `-march=native`. Add `A.T @ v -> v @ A` and `v @ A.T -> A @ v` to `passes/expr.py` so the
      adjoint products materialize no transpose. Each output keeps its summation order, so results
      are bit-identical. Measured with the throwaway patch: npmpc N=12 83.2 to 45.7 µs (MX 51.1),
      unbumpercars C=8 3485 to 827 µs (MX 9950). Tests: a fixture per shape class against NumPy,
      and the C snapshots updated. The patch is `notes/perf_2026_09_07/experiments.patch`. This is
      a stopgap: the two rules are the special case of a range split with one accumulator per lane
      and a choice of outermost range, which C-8 provides generically; when C-8 lands, C-43's rules
      are deleted, not kept beside it.
- [x] **C-44. Scalarize small stage bodies, driven by the `lowering` hint.** For a callee whose
      body is marked `Expr.scalar()`, or fits a conservative scalar-operation budget under `auto`, and never under
      `block`: unroll to scalar SSA, hash-cons, fold `0`, `1` and constant arithmetic, and render
      expression trees (not one op per statement, which is what stops clang from forming FMAs in
      SX's output). The hint is now active. `notes/perf_2026_09_07/scalarize_stage.py` does the
      transform after the fact on the generated C: race-car eq stage 23.0 to 18.1 µs, chain eq
      stage 1910 to 247 µs, both bit-identical. The only item that moves chain, and it subsumes the
      seed half of C-9 and most of C-10. tinygrad's expand-then-devectorize is the reference
      (`notes/perf_2026_09_07/tinygrad_rangeify.md` §4). Gate: the two stage kernels above within
      10% of the hand-scalarized time. Implemented 2026-09-08 in `passes/program/scalarize.py`; the
      [validation](notes/perf_2026_09_07/README.md#c-44-validation-2026-09-08) separates runtime
      seeds from the constant-seed control. Review follow-up replaces the tensor-width heuristic
      with limits on scalar operations after folding and sharing, expansion work, and aggregate
      generated-code growth. The arithmetic contract and exceptional constant cases are documented
      and tested. The [closeout](notes/perf_2026_09_07/README.md#c-44-closeout) records minimal
      GCC/Clang compilation and runtime checks; the full benchmark rerun follows more Track C work.
      Automatic seed specialization remains C-45.
      Design: [arithmetic policy](notes/algebraic_simplification_2026_09_08.md#proposed-scaly-arithmetic-policy);
      rationale: [measurement protocol](../docs/results/fairness.md#measurement-protocol).
- [x] **C-52. Split program passes into an explicitly ordered package.** Implemented in `scaly.passes.program`, with shared helpers and an explicit pipeline in place of registration side effects; pass order, observer events, and behavior are preserved. [Design](notes/algebraic_simplification_2026_09_08.md#the-architectural-decision).
- [x] **C-55. Preserve intended lowering hints through derivative Function construction.** Implemented 2026-09-08: every derived `Function` built in `ad/` takes the primal callee's effective hint (`block`/`opaque` -> `block`, `scalar` -> `scalar`, `auto` inherits nothing) on its output root, through `Function._effective_lowering`; the chain check `hinted_stage_hessian` and `tests/ad/test_lowering_hints.py` pin selection. The chain benchmark stage now carries `.scalar()` (decided 2026-09-08: the comparison is against each side's best formulation, and this is ours); the M=5 Hessian kernel runs at 835 µs against 1769 µs without. Race-car gets nothing from the hint because the automatic policy already selects its stage ([timing](notes/perf_2026_09_07/README.md#track-c-follow-up-2026-09-08)). [Observed hint loss](notes/perf_2026_09_07/README.md#c-44-closeout).
- [x] **C-53. Share arithmetic simplification across both dialects and program forms.** Implemented 2026-09-08 in `passes/arith.py` (one adapter per dialect, rules for neutral elements, zero annihilation, self-cancellation, negation normalization, bounded constant powers, dtype-checked constant evaluation) and applied through `passes/expr.py`, `scalarize`, and the new `fold_arith` loop-body pass after fusion; `tests/passes/test_arith.py` runs the same cases in all three forms. Left open: `_h{n}` renderer temporaries have no collision guard and deep index expressions are not hoisted, both unobserved in practice. [Design and validation](notes/algebraic_simplification_2026_09_08.md#a-small-common-implementation).
- [x] **C-45. Bake stage-invariant constant tangents into the VMAP forward callee.** Implemented
      2026-09-08 in `ad/forward.py`: a constant `jvp_many` tangent whose per-iteration tiles repeat
      with period `k <= 8` (and at least twice, so short horizons of distinct tiles are not unrolled)
      is baked into one const-seed callee per tile, each mapped over its residue class and assembled
      with stack/transpose/reshape, so no seed table or gather is emitted; other constant patterns
      keep the local-coloring and runtime-seed paths. `tests/ad/test_const_seed_bake.py` pins equal,
      periodic, and fallback tiles. Race-car N=50 measured 26.0 to 24.0 µs together with C-53/C-10,
      static metadata 107 to 90 KB ([timing](notes/perf_2026_09_07/README.md#track-c-follow-up-2026-09-08)).
- [x] **C-47. One accumulation buffer for a sum of scatters.** Chain's entry point zero-filled 43
      buffers of 23,544 doubles and scattered 576 values into each before summing them: 8 MB of memset
      per call and the 1,075,248-double workspace. `passes/program/combine_scatter_sums.py` already
      accumulated such sums into one buffer but required every operand to have the sum's declared
      shape, and chain's scatters are flat `(23544,)` buffers read through a `(24, 981)` reshape.
      Comparing element counts instead (2026-09-09) lets the pass fire: chain M=5 Hessian workspace
      1,075,248 -> 109,944 doubles and 850 -> 230 µs; M=3 266,616 -> 26,028 and 162 -> 46 µs, single
      cells from the harness with `--repetitions 1`. `test_scatter_sum_combines_reshaped_scatters`
      pins the zero-fill count. Arithmetic identities such as `0 / x` belong to C-53.
- [x] **C-50. `-march=native` and `-fno-math-errno` in the JIT.** `codegen/jit.py` compiles with
      `-O2` (or `SCALY_CC_OPT`) and no target flag, so every JIT kernel is SSE2 scalar code without
      fused multiply-adds on a machine that has them; measured on race-car N=50, `-mfma` alone is
      32.9 to 27.8 µs. Add the two flags to the JIT compile line, check that the solver plugins'
      compile paths (scaly-sqp's wrapper, the PIQP and IPOPT hooks) still link, and keep the plugin
      wheels themselves at the portable baseline. One line if it plays well with the solvers.
- [x] **C-51. Coalesce consecutive scalar loads and stores into vector accesses in the C renderer.**
      After C-44 scalarizes a body, adjacent `buf[i], buf[i+1], ...` accesses can be emitted as one
      clang `ext_vector_type` load or store; tinygrad's `memory_coalescing` does this in about 60
      lines (`tinygrad_rangeify.md` §6) and it is the difference between scalar code and visible
      SIMD on clang. After C-44. Landed as store coalescing only (`codegen/c.py`, `_emit_body`),
      width 2 via `vector_size`: race-car N=50 24.4 to 23.2 µs on clang, 30.3 to 30.4 µs on GCC;
      chain M=5 within noise on both. Width 4 slowed GCC on chain by about 5 %, and loads were left
      scalar because scalarized bodies consume them lane by lane.
- [x] **C-9. Affine index maps instead of materialized tables.** Implemented 2026-09-09 in
      `passes/affine.py`: `affine_index_map` factors a concrete index array into ranges whose
      contribution is affine plus a residual table, greedily, outermost first, and
      `LowerCtx.index_at` emits one term per range over the trip index so a fully affine gather or
      scatter declares nothing. No AD rule changed; `ad/forward.py` and `ad/reverse.py` still build
      the arrays, and the structure is recovered at lowering, which also catches every other affine
      gather in the graph. `passes/program/_common.py` gained `_index_values`, the counterpart that
      reads the indices back out of the expression, so `combine_scatter_sums` still sees the
      destinations it needs. `tests/passes/test_affine_index.py` pins the factoring, the forward and
      reverse VMAP derivatives against NumPy and against the unrolled form, and the two gates; all
      three gates fail with the affine path disabled.
      Measured: unbumpercars C=32 source 52.30 MiB to 1.35 MiB and static metadata 53.9 MB to
      345 KB, so the cell compiles and passes its correctness check under the 50 MiB cap (kernel
      compile 49.7 s, 8011 µs). race_cars metadata 90,115 to 39,578 bytes at N=50 and 1,015,708 to
      420,681 at N=500; all eight `int64_t` index tables are gone, and the two 6-entry `unique_j`
      residuals that remain are constant in N.
      Runtime is unchanged: race_cars N=50 24.0 µs and N=500 240 µs against 24.0 and 243 to 247 for
      the tables, three repetitions each. Emitting each level as `(k // stride) % dim` did cost
      about 5% (25.4 µs and 257 µs), because the modulo is a second division. It is gone: since
      `k // stride[i] // dims[i] == k // stride[i - 1]`, the coordinates telescope and the index is
      a combination of the plain quotients `k // stride[i]` with coefficients
      `c[i] - c[i+1] * dims[i+1]`, which is the `(x % c) + (x // c) * c -> x` recombination applied
      once at emission rather than as a folding pass. Exact integer algebra for a non-negative trip
      index, so the indices are unchanged; `tests/passes/test_affine_index.py` gathers through every
      factored case and compares against NumPy to pin that. A single absolute size threshold was
      also tried and rejected: it makes the emitted shape depend on N, which
      `test_vmap_sparse_hessian_c_source_is_constant_in_length` correctly rejects.
      The one part of the gate not met is `static_metadata_bytes` fixed across N, and index
      arithmetic cannot make it so; C-57 owns what still grows.
      Design and what was rejected from tinygrad's `uop/divandmod.py`:
      [refactorings](notes/refactorings.md#affine-index-maps-for-gathers-and-scatters).
- [x] **C-10. Fold the identities the AD rules introduce, at the expression level.** Implemented
      2026-09-08 in `passes/expr.py`: `v @ ones -> sum(v)`, identity-index gathers become reshapes,
      uniform 0/1 masks of the result shape fold, each pinned in `tests/passes/test_expr.py`. The
      matrix forms `A @ ones` and `ones @ A` were tried as stacked row sums and reverted: in loop
      form they lower to one loop per row and lose the fused producer, slower than the matmul. They
      wait for an axis reduction in the IR, which is C-8's accumulator lowering.
- [x] **C-12. One matcher and iterative rewrite driver for both dialects.** Implemented 2026-09-08: `ir/match.py` is generic over both node types with an iterative driver (`fixpoint`, `revisit`, `max_steps`), `rebuild_program` in `passes/program/_common.py` is the program adapter, and `_transform` is gone. No nested patterns or captures: no call site needed them. C-13 closed with it. [Updated design](notes/refactorings.md#shared-compiler-rewrites).
- [x] **C-143. A general dense solve: LU with partial pivoting** (integrators/MPC plan I2).
      `ExprOp.LU` gives `P A = L U` packed with the permutation in an `(n + 1, n)` array, the pivot
      LAPACK's (largest magnitude, first on a tie). Up to `dense_unroll` it is straight-line code with
      every access at a fixed address, the row swap selecting on the run-time pivot; above, loops that
      swap through the pivot row's run-time address. `linalg.lu_solve` (and `trans=True`) and
      `linalg.solve(a, b, assume="gen")`, whose derivative is implicit with two levels of rules, so
      Hessians never reach the factorization, which refuses a derivative in both modes. Against
      Accelerate's `dgesv`: 0.12x at order 4, 0.51x at 8, 0.94 to 1.05x from 12 to 40. 24/24 mutants
      killed, three after a test was added. Report: `notes/integrators_i2_report.html`.
- [x] **C-158. Edge-case tests for the ops added since `main`, and the bugs they found.** 494 tests
      in ten `*_edge_cases.py` files cover the 32 new expression ops: special values, ties, empty and
      size-one shapes, out-of-range indices, the unroll thresholds, loops of zero and one steps,
      folded against compiled, every derivative mode, and `verify_expr` on every op and its
      derivatives. They pinned 52 failures, all fixed. The fixes:
      - reverse mode keeps a `where` mask through elementwise ops;
      - float32 tangents, cotangents and constants take their value's dtype;
      - a Python number beside a float expression takes its dtype;
      - tie weights never divide 0 by 0;
      - derivative-helper caches are keyed by the options, and their names are tagged under non-default ones;
      - non-float loop carries carry no tangent;
      - the in-place proof handles empty step tables and empty groups;
      - `ldl_unpack` works in float32 and `lu_solve` checks the rank first;
      - program-IR constants are keyed by their bits;
      - `maximum`/`minimum` fold as `fmax`/`fmin`;
      - int64 constants are exact;
      - float32 C narrows its constants and libm results.
      26/26 mutants of the fixes were killed, three after a test was added. Report:
      `notes/edge_case_fixes_report.html`.
- [ ] **C-159. float32 through the packed vectors of loop derivatives.** Multi-seed forward mode joins
      the tangents of several inputs in one vector, and a while loop's adjoint packs the carry's
      cotangent with the params'. A float32 carry beside float64 slices or params cannot share one,
      so both raise (strict xfails in `tests/integration/test_loop_edge_cases.py`).
- [ ] **C-160. Multi-seed forward rules for `maximum`, `minimum`, `copysign`, the `max`/`min`
      reductions, `segment_max`/`segment_min` and `index_add`/`index_set`.** They fall back to one pass
      per seed, and `SCALY_STRICT_JVP_MANY=1` refuses them. The notes in `docs/how_it_works/ir.md` list
      only `atan2`, `asin`/`acos`/`atan` and `abs`.
- [ ] **C-161. Truncation and signed zeros left over from C-158.** `_operands` gives a Python float
      beside an integer expression that expression's dtype, so `i64 < 0.5` compares with 0.
      `_attrs_key` in `ir/expr.py` merges `take(..., fill=0.0)` with `fill=-0.0`. int64
      `maximum`/`minimum` render as double `fmax`/`fmin`, which is inexact above 2**53.
- [x] **C-170. A zero derivative for `floor` and `ceil`, and forward mode through integer index
      arithmetic** (interp plan SP0, `notes/interp_plan_2026_09_28.md`). `floor` and `ceil` are
      differentiable, with a zero derivative in forward and reverse mode, refused under
      `nonsmooth="error"`; a sparsity pattern records no dependence through them, except under
      `"error"`, so that differentiating still reaches the refusal. A periodic wrap
      `sin(x - T floor(x / T))` differentiates to 1e-14. Forward mode (`_jvp`, `_jvp_many_structural`)
      gives an integer or bool node no tangent, so an index vector built by broadcast `int64`
      arithmetic no longer fails `sc.jacobian` and `sc.hessian` on a mixed-dtype product. A test pins
      the clamp-before-cast order through the generated C (NaN, ±inf, 1e300), and `cast`'s docstring
      states the undefined behaviour it avoids. Spike for the plan's kill criterion: a spline evaluation
      composed from existing ops runs in 0.06x, 0.03x and 0.39x the time of CasADi's generated
      `interpolant` (1-D cubic, 2-D bicubic, 3-D trilinear) with a tenth of the C lines
      (`perf_2026_09_28_interp/`). 9/9 mutants killed, one after the clamp test was strengthened.
      Report: `notes/interp_sp0_report.html`.

### Deferred

The 2026-09-22 items below come from re-measuring a hand-optimized race-car Hessian on the M4 Max;
[`notes/perf_2026_09_22/`](notes/perf_2026_09_22/README.md) holds the numbers, the generality
analysis of each hand change, the `zig cc` and compiler-extension survey, and the rendering
decision. Like C-8 they are deferred and not required for 0.1.0: the current kernels already beat
CasADi on every benchmark cell, and the release work comes first. They are organized around C-8
rather than beside it. C-77 *is* C-8's step 0 (inline scalarized callees into their mapped loop,
which nothing in C-8 covered) and step 1 (range propagation); doing it lands the first third of
C-8 and the assembly fix at once. C-78 is three independent passes that need none of C-8 and can
go first if cheap wins are wanted. C-79 is one more rewrite over ranges, tinygrad's `shift_to`,
and sits after C-8's step 1 because a widened kernel that still writes 48 outputs to memory gains
little; it must not be built as a separate `VMAP`-loop transformation that C-8 would then delete,
the way C-43 is scheduled for deletion. C-80 is solver work. C-81 was the x86 re-run; it
confirmed the order C-77 then C-79 and settled C-79's lane width and loop shape. When
C-8 is resumed, fold C-77 and C-79 into its step list and close them there.

- [ ] **C-77. Inline scalar callees into their mapped loops and fuse the derivative assembly.**
      C-8 step 0 and step 1. Today `_lower_vmap` emits `FOR { CALL }` and every pass stops at the
      `CALL`, so the sparse-Hessian recovery (`gather(transpose(jvp_many(...)))`) runs as 15
      separate array passes: 35% of the race-car N=200 Hessian, and the reason inlining alone buys
      nothing (43.5 to 43.0 µs) while fusion lets 6 of 16 block entries die (28.4 to 18.0 µs).
      Gates: no assembly loops in the generated C, `workspace` fixed across N in the sweep,
      race-car hess lower N=200 under 22 µs on the M4 without vectorization.
- [ ] **C-78. Constant-tile folding, invariant-divisor reciprocals, terminal-trip peeling.** Three
      small passes. A `static const` table that tiles a period P becomes `k[i % P]`, a scalar at
      P=1; `k0`, `k22`, `k26` from C-57 are exactly these. `x / y` with `y` loop invariant becomes
      `x * inv_y` hoisted, behind a policy flag because it moves the last bit; both race-car
      divisors are invariant and this is the `-ffast-math` gap (18.0 to 15.5 µs). Peeling the last
      trip of a mapped axis whose slice differs makes the recovery gather `k44` affine (period 24,
      residual 13, verified) so C-9's map replaces the table. Gates: no `double` table growing
      with N in the race-car header; no division in the stage body.
- [ ] **C-79. Explicit lanes on mapped ranges.** Port tinygrad's `shift_to`,
      `r -> r_outer * W + r_lane` with `r_lane` of `RangeKind.VECTOR`, applied to the mapped axis
      first (independent trips, no dependence analysis), then a contiguous output axis, then a
      reduction axis with one accumulator per lane. Under a widened range a unit-stride access is
      a vector load or store, a constant stride is a staging transpose at the ABI boundary (buffers
      created under the range are lane major; ABI arrays stay stage major), anything else is per
      lane; libm ops, `minimum` and `maximum` render per lane. W is chosen by the C preprocessor
      from the compiler's target macros, not by Scaly: 8 under `__AVX512F__`, 4 under `__AVX__` or
      256-bit SVE, 2 under `__SSE2__` or `__aarch64__`, otherwise 1 (Cortex-M and ARMv7 have no
      double vectors), overridable with `-DSCALY_LANES=n`; the compiler legalizes any W, so W is a
      performance knob only. The fused stage body is rendered once as an `always_inline` function
      taking the stage index and a count of valid lanes; the main loop runs the `N / SCALY_LANES`
      full trips with the count fixed at W, and one remainder call under
      `#if (N % SCALY_LANES) != 0` handles the last partial vector with clamped loads and guarded
      stores. No scalar tail and no second copy of the body. Staging buffers are sized for W = 8 so
      the workspace size in the generated header does not depend on the macro. Two render modes,
      documented in `docs/api/codegen.md` with their differences and supported compilers when this
      lands: `gnu` (default; `vector_size` types, `v[i]`, `__builtin_shufflevector`,
      `__builtin_convertvector`, `restrict`; gcc ≥ 12, clang, `zig cc`, armclang) and `c` (opt in;
      the widened program as a scalar body inside an inner lane loop over the same staging buffers;
      any C99 compiler). No vendored vector libm, no glibc libmvec, no own polynomials:
      transcendentals are per-lane scalar libm calls on every target, decided for OS and compiler
      portability. C-81 measured the price on the x86 reference machine: 26.6 µs at 8 lanes against
      12.7 with libmvec, the whole gap being trig. Gates: within 1.2× of `variant_w8_lane.c` on the
      x86 reference machine (26.6 µs, gcc 13); no regression on the M4 at W = 2; byte-identical
      output between the two modes.
- [ ] **C-80. Parameter-only oracle prologue.** An oracle whose subgraph depends only on `p`
      runs once per solve and its result is reused across SQP iterations (the race-car cost block).
      Function or solver level, not a program pass; belongs with the S items once C-77 lands.
- [x] **C-81. Re-run the 2026-09-22 variants on the x86 reference machine.** Done 2026-09-22,
      section 5 of `notes/perf_2026_09_22/README.md`, `x86_variants.sh` reproduces it. C-77 stays
      first (108.5 to 59.4 µs on gcc). A native `zig cc` links `-lmvec`. Declared
      glibc `_ZGV*` prototypes reach the hand-written kernel (12.7 µs) and gcc's `simd` attribute
      does so from scalar source, but both are libmvec and stay out for portability; the
      `__builtin_elementwise_*` option is dropped too. Side finding not yet acted on: gcc 13, the JIT's `cc` here, is 40% slower than
      clang 20 on today's generated code at the protocol flags.

- [ ] **C-8. A range-based loop compiler for the program dialect, in the shape of tinygrad's
      rangeify.** Caller workspace still grows with N on race_cars and npmpc. C-47 removed chain's
      43 separate scatter accumulation buffers and reduced M=5 workspace to 109,944 doubles.
      Defer the general loop compiler until after the closeout study and documentation work. The
      2026-09-08 study (`notes/perf_2026_09_07/tinygrad_rangeify.md`) says how the reference design
      gets fusion without a dependence analysis: loop variables (ranges) are first-class values;
      views become index expressions over them; a producer with one consumer inherits the consumer's
      ranges, which is fusion by construction; a producer whose consumers disagree on an axis is
      materialized on that axis only; a reduce becomes `acc init / acc op= x / END(range)`; and one
      substitution `r -> r_outer * amt + r_inner` expresses tile, unroll and upcast, with an
      accumulator per upcast lane. Port that shape, not the framework, in this order, each step with
      a `tests/` fixture: (1) ranges and the three-case propagation rule over the lowered loops, which
      is the fusion pass and the workspace fix; (2) the accumulator lowering of reductions with a
      per-lane split, which supersedes the layout-specific matmul rules C-43 landed in
      `_lower_matmul` (delete them then) and generalizes them to `W @ [v1 v2 v3]`
      and to matrix-matrix products; (3) the reduce-under-broadcast rule so a value is never
      recomputed under an expand. Gates: race-car `workspace` fixed across N in the sweep CSV, chain
      workspace under 100k doubles at M=5, npmpc within 5% of today's kernel with those rules removed.
      The pre-optimization caller shares were 22% of race-car, 3% of npmpc, and 26% of chain.
      Re-measure the remaining costs before resuming this work.

- [ ] **C-57. The static metadata that still grows with N after C-9.** With every affine index
      table gone, race_cars metadata is 39,578 bytes at N=50 and 420,681 at N=500, so it still
      grows roughly linearly. Three things are left, none of them index arithmetic. The generated
      header's sparsity tables are `O(nnz)` by construction (23,677 bytes at N=50: rows, cols, the
      CSR and CSC pointers and both value permutations) and the question is whether a banded or
      per-stage-block encoding can describe them in closed form for a multistage problem instead of
      listing them. The sparse-assembly gather is a genuine `nnz`-length permutation with no affine
      structure (`k44`, 657 entries at N=50), and would need the assembly itself restructured, not
      its index compressed. And three `double` constant tables that C-45's periodic-tile bake did
      not reach (`k0` 1200, `k22` and `k26` 1224 entries at N=50) grow with N; find out which
      tangent or weight each one is and whether the bake's period test is simply too narrow.
      Deferred 2026-09-09: growing metadata is reported separately from executable code and
      artifacts must stay within the compile cap. Revisit if measured artifact size becomes a
      deployment limit.

- [ ] **C-11. Chain: exploit the stage-block structure.** The coloring width grows with M (12, 24,
      42 at M=3,5,9), so the per-stage Hessian pays that many forward-over-reverse sweeps where `SX`
      computes one symbolic Hessian. Investigate a scalar-level second-order pass inside the stage
      body, then a scatter. Deferred beyond this closeout. The
      [C-49 audit](notes/perf_2026_09_07/c49_ad_op_audit.md#what-is-inherent-to-forward-over-reverse-here)
      finds composition accounts for most excess operations, with symmetry the remaining
      second-order opportunity. Reassess only if a current workload justifies the work.

- [ ] **C-58. Inline small pure callees before differentiation.** Revisit a bounded expansion policy
      if measured workloads justify it. Excluded from C-49 closeout to preserve mapped structure
      without introducing a new expansion policy. Diagnosis: [C-49 audit](notes/perf_2026_09_07/c49_ad_op_audit.md#ranked-rule-and-composition-edits-for-an-implementer).

- [ ] **C-54. Add memory-aware program common-subexpression elimination and dead-code cleanup.** Deferred, like all compiler work, unless a measured workload requires it. Build on C-12/C-53 with definition/use tracking and conservative read/write/alias handling; retain required calls and output stores, and test repeated loads across writes. Broader loop motion follows demonstrated workload need; load-node interning alone is not a current stale-value bug. [Design](notes/algebraic_simplification_2026_09_08.md#separate-value-cleanup-from-memory-optimization).
- [x] **C-13. Make the Program IR passes iterative instead of recursive.** Done 2026-09-08 with
      C-12: the program passes, `scalarize`, and the C renderer no longer recurse per expression node,
      and the renderer hoists subtrees deeper than `MAX_SCALAR_DEPTH` into temporaries so clang's
      bracket limit is not hit. Witnesses in `tests/passes/test_program.py`: left folds at 400 and
      3000, the NPMPC-shaped flat per-stage reduction at N=100, and a hinted scalar fold, each
      compiled and checked against NumPy. Recursion proportional to statement nesting (loop and call
      depth) remains and is documented in `docs/how_it_works/lowering.md`.
- **C-14. Confirm or drop the reverse / row-coloured sparse-Jacobian hypothesis.** `sparse_jacobian`
  colours columns only, and npmpc's per-stage block is wider than it is tall, which is consistent
  with running more forward sweeps than a row-coloured or reverse pass would need. Prize is bounded
  and knowable, roughly 0.85 -> 1.1 at the shipped decoder width, and it does not change the
  width-axis result Scaly already wins.

## Solvers

### Now

### Deferred

- **S-16. Separate the IPOPT gap into version against build configuration.** Rebuild 3.14.11 with
  our hook's flags, or 3.14.19 against the wheel's OpenBLAS. "We ship a better-tuned linear algebra
  stack" is defensible; "our IPOPT is newer" is not. Rationale: [fairness](../docs/results/fairness.md).
- **S-17. Make the compiled CasADi IPOPT cache survive worktree removal.** Include the library
  search paths in the cache identity or make cached artifacts independent of them. Rationale:
  refactorings.md "Compiled CasADi artifacts across worktrees".
- **S-18. CasADi `sqpmethod` as a secondary reference column.** Opt-in and record-only; its
  globalization, regularization and QP path differ from `scaly-sqp`. Add only if review asks for it.

## C API

The generated C, C++ and CasADi-compatible interface. [Design](notes/generated_interface_2026_09_18.md).

### Now

- [x] **CAPI-72. Native entry cleanup.** `mem` becomes `int`; the `f_sz_*()` functions and the
      `alloc_mem/init_mem/free_mem` stubs go from `codegen/c.py` and `codegen/aot.py`; the JIT
      reads `module.workspace_size` instead of calling `f_sz_w()`; the ABI doc follows. Touches
      every rendered header, so the whole suite runs and the `benchmarks/results/smoke/**`
      fixtures are regenerated if compared textually. [Design](notes/generated_interface_2026_09_18.md#the-pointer-entry-both-languages-always).
- [x] **CAPI-73. C header with a caller-owned workspace.** After CAPI-72. Buffer structs become
      `f_x_t` (no `_in`/`_out`), 16-byte aligned; `f_workspace_t` is passed to `f_call` instead of
      being stack-allocated inside it. Update the two C++ smoke tests in `tests/codegen/test_c.py`.
      [Design](notes/generated_interface_2026_09_18.md#the-c-header-langc).
- [x] **CAPI-74. C++ header.** After CAPI-73, independent of CAPI-75. `lang="cpp"` renders `f.hpp`
      beside the same `f.c`: a guarded `Buffer<T, Ns...>` with inline aligned storage and a
      `constexpr shape`, a namespace per function with `x_t`/`workspace_t` aliases, `constexpr`
      sparsity tables, `call(..., workspace_t&)`; no enclosing `scaly` namespace. Smoke test: a C++
      caller against a `(N, nx)`-shaped function and a sparse Jacobian, reading a value through
      `csc_val_perm`. [Design](notes/generated_interface_2026_09_18.md#the-c-header-langcpp).
- [x] **CAPI-75. CasADi 3.8 compatible symbols.** After CAPI-73, independent of CAPI-74. `casadi=True`
      adds the guarded `casadi_int`/`casadi_real` typedefs, the query set (`_n_in`, `_n_out`,
      `_name_in`, `_name_out`, `_default_in`, `_sparsity_in`, `_sparsity_out`, `_work`,
      `_work_bytes`, `_checkout`, `_release`, `_incref`, `_decref`), compressed-column sparsity
      tables, and an entry that gathers compact sparse outputs through `csc_val_perm` (adding
      `nnz` to `f_SZ_W`). Dense matrix inputs or outputs are rejected at render time. Tests: load
      the library with `casadi.external` and compare against `numerical_call`; assert the six
      symbols acados needs resolve through `ctypes`. [Design](notes/generated_interface_2026_09_18.md#the-casadi-layer-casaditrue-either-language).
- [x] **CAPI-76. Document the generated interface.** After CAPI-74 and CAPI-75. Rename the ABI page to
      "The generated interface": the pointer ABI, then the C, C++ and CasADi layers; say plainly
      that both header languages compile the same kernel. Update `guide/codegen.md` and the CLI
      help (`--lang`, `--casadi`). [Why ABI and API are both right](notes/generated_interface_2026_09_18.md#what-this-is-called).

## Benchmark harness

### Now

- [x] **BH-48. Adopt `-march=native` in the benchmarks and the AOT guidance; keep distributed
      binaries portable.** Decided 2026-09-08: the sweep and closed-loop harnesses compile both
      providers with `-march=native` (and `-fno-math-errno`), fairness.md states the rule and why;
      AOT users are told in the docs to pass it and it goes in the suggested CFLAGS; the solver
      plugin wheels stay at the portable baseline. The JIT side is C-50. Measured reason: on
      race-car the gain is FMA contraction (`-mfma` alone: Scaly 32.9 to 27.8 µs, SX 21.3 to 20.7,
      because SX's one-op-per-statement code never contracts), not vector width. Record both flag
      sets in fairness.md until BH-20 reruns. Evidence: `notes/perf_2026_09_07/README.md`.
- [ ] **BH-21. Add an immutable publication mode**: clean release candidate, every raw run retained,
      and an archive of source, lockfile, inputs, generated code, logs, statistics and manifest.

### Deferred

- **BH-22. Embedded hardware benchmarks** (Raspberry Pi / Jetson).

## Benchmark problems

### Deferred

- **BP-24. Replace the race-car tracking NMPC with the MPFC distillation**, in the same `race_cars`
  package, and **laopt as an external baseline** for it once laopt is published.
- **BP-25. l4casadi CPU column** on unbumpercars, with batching and exact second-order generation.
- **BP-26. Network-width regime sweep**, and a trained wide decoder from the neural-process-MPC
  authors so the width study becomes a measured closed-loop column instead of an extrapolation.
  Also worth telling them their released episode's reported cost metric cannot be reproduced from
  the trajectory it ships with.
- **BP-27. Diffusion- and GP-based problems; hovercraft MBD/DIAL workload.**

## Case studies

The reproductions of published benchmarks in `examples/case_studies/`, planned in
`notes/case_studies_plan_2026_09_27.html`. Each study's report is `notes/case_study_eN_report.html`.

### Now

- [ ] **CS-1. E1's acados leg.** Port the Fatrop chain to `AcadosOcp` (rockit 0.6.7's acados driver no
      longer compiles against current acados) and swap acados' generated `expl_vde_*`/`expl_ode_hess`
      for Scaly's `--casadi` output. Report: `notes/case_study_e1_report.html`, section 6.
- [ ] **CS-2. IPOPT's oracles without the dense seed contraction.** For a mapped multiple-shooting gap
      the descriptor Jacobian (and Hessian) contract every stage's tangents with constant 0/1 seed
      matrices over the whole horizon, then transpose and gather: 3.3x the stagewise Jacobian at 3
      masses, and the reason Scaly's IPOPT oracles are 1.7 to 1.9x CasADi SX's on the chain. Scatter
      the per-stage values straight into the nonzeros. E1 report, section 4.
- [ ] **CS-3. Star-coloured stage Hessians** for stagewise oracles (the Fatrop drop-in): the dense
      `sc.hessian` over a 42-variable 3D chain stage is 1.14x CasADi SX, which drops the structural zeros.
- [ ] **CS-4. Rerun E1 on the reference machine** (`compare.py`, `sweep.py`); every number so far is from an M3 Max.
- [ ] **CS-5. Local derivatives for repeated patterns** (E6). Colouring itself scaled (case13659 builds),
      but the colour width does not stay small: the OPF Lagrangian Hessian needs 12 directions at
      case118, 28 at case2869, 74 at case9241 (the Jacobian 15 to 48), tracking the largest bus degree.
      Evaluation and generated C both grow as n x width: the Hessian is 5.6x ExaModels' at case9241
      (1.28 against 0.23 s per solve) and the C is 277 MB (12 s to render, 22 s to compile). ExaModels
      forms each constraint pattern's small local Hessian and scatters it, which grows as nnz. Colour
      one `vmap` body (or one gather pattern) and replicate it. E6 report.
- [ ] **CS-6. `sc.solver(sc.qp_problem(n, ...), "piqp")` build-time growth** (E3): 0.23 s at n=50, 14.4 s at n=200.
- [ ] **CS-7. Reverse mode through `SparseLDL.solve` with a matrix right-hand side** (E5), about 100x slower than per-column solves.
- [ ] **CS-8. Second-order cones in the generated IPM** (E3, E8), Nesterov-Todd scaling on the fixed KKT pattern.
- [ ] **CS-9. A `vmap` of matrix-vector products with one constant matrix as a matrix product.** RTN-MPC's
      surrogate at 10 nodes: Scaly runs, per node, a matvec for the value and a two-column product for
      the Jacobian, reading each layer's weights 20 times per call and storing them twice (one copy per
      layout); PyTorch runs one product per layer for all nodes. Scaly is 29 to 38x faster at width 16,
      0.6x at 5 x 128 and 0.05x at 12 x 512 (float32 accounts for 2.2x of that). E2 report.
      The same shape in E5: 64 independent LQ solves per step, each a chain of 8 x 8 or 16 x 16 products,
      run element by element; at 16 states Scaly's lead over XLA's batched kernels falls from 10x to 4x
      (forward) and from 5.5x to 2.1x (gradient). Vectorize a `vmap` across its batch. E5 report.
- [ ] **CS-10. A call node's value and its derivative compute the callee's forward pass twice.** In
      acados' `expl_vde_forw` for a 5 x 128 MLP model written as a `Function`, the Jacobians recompute
      the network beside the call's own value: 60 µs against 31 µs with the network written inline.
      Share the primal between a call and its derivative (or return it from the derivative). E2 report.
      E5: the reverse sweep of an episode calls a `custom_derivative` MPC whose rule needs its output, and
      the output is recomputed for it; with `cost` and `grad` from one Function the whole episode's forward
      also runs twice (200 against 161 ms). E5 report.
- [ ] **CS-11. Dense matrices in the CasADi layer.** `casadi=True` refuses a dense matrix argument
      (Scaly row-major, CasADi column-major), so acados' `Sx` has to travel as a flat column-major
      vector. Transposing in the layer would make any acados signature a drop-in. E2 report.
- [ ] **CS-12. A parameter-dependent QP Hessian is materialized dense before its nonzeros are gathered.**
      The oscillating-masses MPC with `Q`, `R` as parameters: the generated PIQP zero-fills an `n x n`
      array every solve, writes the diagonal and reads it back (40 000 doubles at horizon 16, about
      1.6 n^2 of workspace in all: 29k doubles at n = 104, 243k at n = 392). Extract `P` sparse. E3 report.
- [ ] **CS-13. The rest of E4's problem set.** Parallel park and cartpole are done. Still open:
      escape (Altro.jl's infeasible start, slack controls on the dynamics), the quadrotor maze
      (a quaternion state, so iLQR in the error state as RobotDynamics does), the Kuka arm (RNEA as a
      `scan` over links), and the DIRCOL column: Scaly's IPOPT on a Hermite-Simpson transcription of the
      same problems against Ipopt DIRCOL from Julia. E4 report.
- [ ] **CS-14. Loop-invariant hoisting stops at a custom rule.** E5's episode, written as the benchmark
      writes it, has a Riccati recursion that depends only on broadcast data: the forward `scan` hoists
      it (0.19 ms for 3200 MPC solves), but in the reverse sweep each element's rule recomputes it
      (84 ms). Hoist the invariant part of a rule's body as the forward does. E5 report.
- [ ] **CS-15. `sc.diag` and `sc.stop_gradient`.** A diagonal matrix from a vector is written
      `v.reshape((n, 1)) * I`; DiffMPC's and trajax's truncated gradients (no cotangent through the MPC's
      state) needed a custom rule, where `stop_gradient` would express it directly. E5 report.
- [ ] **CS-16. A JIT cache hit still renders the C.** `jit.py` hashes the rendered source to find the
      compiled library, so a warm start pays the whole rendering: 1.4 s of the SCvx study's 1.8 s warm
      start, against a 2 ms solve. Key the cache on the Function's structure (its IR hash and the
      compile flags) and render only on a miss. E8 report.
- [ ] **CS-17. E8 with the cones in the subproblem.** OpenSCvx's 6-DoF landing keeps every nonconvex and
      conic constraint in its penalty state, so its subproblem is a QP and E8 ran Phase C directly. An
      example whose subproblem keeps second-order cones (the plan's Szmuk and Reynolds formulations) needs
      CS-8 first; the node-count sweep against the ~100 ms class of Reynolds et al. goes with it. E8 report.

## Licensing

Scaly and the three plugins are BSD-2-Clause. The plugin wheels also ship other people's binaries,
so each wheel carries its dependencies' license texts the way CasADi does
(`casadi/include/licenses/<dep>/LICENSE`), except that CasADi's `mumps-external` and
`metis-external` entries are the COIN-OR wrapper's EPL text rather than the real MUMPS and METIS
licenses, which we do not copy. Surveyed 2026-09-07. What we ship and what it asks of us:

| Package | Component | License | Obligation |
|---|---|---|---|
| scaly, scaly-sqp | our code | BSD-2 | none |
| scaly-piqp | PIQP, BLASFEO | BSD-2 | notice |
| | Eigen | MPL-2.0 | notice. PIQP does not define `EIGEN_MPL2_ONLY` itself, so `hatch_build.py` passes it through `CMAKE_CXX_FLAGS`; PIQP 0.6.2 compiles under it, which proves no LGPL Eigen file reaches the library |
| | LDL inside PIQP (`piqp/sparse/LDL_License.txt`) | LGPL-2.1-or-later | notice plus the LGPL-2.1 text. PIQP's `sparse/ldlt` is a modified LDL, instantiated in `ldlt.cpp` and compiled into `libpiqpc`, so the shared library we ship contains LGPL code. That is allowed: the LGPL text travels with it, the modified source is PIQP's public tag, and `libpiqpc` is a separately loaded shared library the user can replace |
| scaly-ipopt | IPOPT | EPL-2.0 | notice, upstream source of the pinned version; a separate dynamically loaded module, so our BSD-2 is unaffected |
| | MUMPS 5.8.2, via COIN-OR `ThirdParty-Mumps` 3.0.12 | CeCILL-C, EPL-2.0 for the wrapper | notice for each |
| | OpenBLAS (static, Linux) | BSD-3 | notice |
| | libgfortran, libquadmath | GPL-3 + GCC runtime exception | notice; the exception covers this use |
| | METIS 5.2.1 | Apache-2.0 | notice |
| | GKlib | Apache-2.0, plus two glibc-derived headers under LGPL-2.1-or-later and one BSD-3-Clause file, per its `LICENSES.md` | notice for each |

### Now

- [x] **L-28. Root `LICENSE.md` (BSD-2-Clause, copyright EPFL, 2026)**, the holder the lab's other
      projects name, with the author only in the pyproject `authors` entry, `license = "BSD-2-Clause"` and
      `license-files = ["LICENSE.md"]` in the root `pyproject.toml`, and the README "License" section
      replaces "TBD".
- [x] **L-29. Copy the same `LICENSE.md` into each of `plugins/scaly-{sqp,piqp,ipopt}/`** with the same
      two pyproject fields. Each plugin is its own sdist and wheel, so each needs the file in its
      own tree; a copy, not a symlink, so sdists stay correct.
- [x] **L-30. Third-party notices generated by the build hooks.** `hatch_build.py` in `scaly-piqp`
      and `scaly-ipopt` copies each dependency's license text from the already-cloned
      `third_party/` sources into `src/scaly_{piqp,ipopt}/licenses/<dep>/` and writes a short
      `THIRD_PARTY_NOTICES.md` listing name, pinned version from `build_config.json`, license and
      upstream URL; the directory joins the wheel `artifacts`. Generating at build time keeps the
      notices from drifting from the pins. libgfortran and libquadmath are not cloned, so their
      GPL-plus-runtime-exception text is vendored once in `plugins/scaly-ipopt/licenses/`, as is
      the LGPL-2.1 text for LDL in `plugins/scaly-piqp/licenses/`. `scaly-piqp` got its own
      `build_config.json`, which also pins BLASFEO to `0.1.4.3`; it built from an unpinned
      `master` before. A missing notices file counts as "not built", so libraries from before this
      change are rebuilt once.
- [x] **L-31. Close the two PIQP questions** in the table: confirm `EIGEN_MPL2_ONLY`, and how the
      LDL code is linked. Both answered in the table.
- [x] **A test that the license directory exists for every dependency named in
      `build_config.json`** in each plugin wheel's file list, shown to fail when an entry is
      removed. Plus one paragraph in `docs/dev/contributing.md`: a new vendored dependency needs a
      `build_config.json` entry and a license copied by the hook. `plugins/*/tests/test_*_notices.py`.

## Documentation

The same category of debt the fairness audit found in the results pages: prose that outran what
the code does.

### Now

- [x] **D-32. Rewrite the user-facing documentation.** Done 2026-09-16: every page under `docs/`
      and the README rewritten for register and accuracy, the wordmark added to both entry points,
      the single status admonition in `docs/index.md` (remove it at 0.1.0), and the plain-speech
      rules recorded in `docs/dev/conventions.md`. `FunctionTemplate` (API-1) is not mentioned in
      the docs; add its page when it lands. Wheel installation instructions belong to R-41.
- [x] **D-33. Rework `docs/how_it_works/comparison.md` (now `influences.md`).** Done 2026-09-16: every Taken/Changed row
      checked against code; `sc.problem` returns a `Problem`, only one upward import is checked. A scoped fix landed on 2026-08-25: the
      CasADi section's "one graph with a per-node hint" paragraph presented an inert mechanism as a
      departure, and it now states the repetition claim that is actually true and measured, with the
      lowering-hint discussion subsequently removed from the published docs. The rest of the page still
      predates a lot. Check every "Taken / Changed" row against what the code does today, and check
      the tinygrad, MLIR and JAX sections the same way.
- [x] **D-34. Audit the whole of `docs/` for claims that outran the implementation.** Done
      2026-09-16; every guide snippet executed, two were broken and are fixed. Originally: the way the
      results pages were audited. The pattern to look for is a stated departure or capability whose
      supporting mechanism is recorded but not wired up, and a number with no machine attached.
      The lowering-hint claims have been removed. Check any surviving timing that predates the
      reference-machine rule in `AGENTS.md`.
- [x] **D-68. Acknowledgements and AI disclosure in the README and `docs/index.md`.** Done 2026-09-16. An
      acknowledgement of NCCR Automation, which funded the research, and a disclosure that AI coding
      agents (Claude, Codex and others) were used to write parts of the code and documentation.
      Before R-67, so the first public snapshot carries both.
- [ ] **D-65. Two documentation deployments.** GitHub Pages publishes the user-facing docs from
      `main` on each version tag, through a `docs.yml` job with `pages: write` and `id-token: write`
      permissions on a tag trigger. Cloudflare Pages publishes the latest docs from `main` and a
      preview per branch, which the existing every-branch build already produces. Each site carries
      a banner or version switcher saying which one it is.
- [x] **D-177. Interpolation docs** (interp plan SP7). `docs/guide/interp.md` (the kinds, the
      extrapolation modes, evaluation, derivatives and the Jacobian's pattern at known against
      symbolic points, where coefficients come from, generated code, a section for CasADi users),
      `docs/api/interp.md`, the nav, the API index row, `interp/` in the codebase map, a section in
      `examples/README.md`. `test_import_boundaries.py` now checks that every interp name, method,
      property and constant is documented, which found eight undocumented properties. The stale
      `BSpline` search docstring is fixed. `zensical build`: no issues, and no griffe warning from
      interp. Report: `notes/interp_sp7_report.html`.
- [x] **D-35. Reconcile the problem READMEs with the audit.** Completed 2026-09-11. The problem
      READMEs now describe only the current formulations and link measured comparisons to the
      canonical result pages.

### Deferred

- **D-36. GPU backend milestone definition**: what a first accelerator target must demonstrate before
  any backend work starts.

## Release

These steps make the tree public and permanent, and each is cheap to do once and expensive to redo.

### Now

- [x] **R-37. Move the paper design note out of this repository.** Done 2026-09-16; it goes to
      the paper's own repository. The note was never on main but is in dev's history, which R-62
      rewrites.
- [ ] **R-38. Windows support.** Decide the toolchain (MSVC or clang) and the target: the core JIT
      plus `scaly-sqp` and `scaly-piqp` first; `scaly-ipopt` on Windows is a separate later item
      because it drags in Fortran and its own licensing survey. Candidate toolchain: make `ziglang`
      (R-71) a required dependency on Windows through a `sys_platform == 'win32'` marker, and build
      the Windows `scaly-piqp` wheel with `zig cc` as the CMake C and C++ compiler, so the JIT and
      the solver library share one toolchain and no MSVC-versus-MinGW runtime mismatch can arise.
- [x] **R-39. Finalize the name.** Decided 2026-09-14: `scaly`. `scali` was the first choice, but
      PyPI refused it as too similar to `scaii`, an abandoned 2019 project: PyPI treats `l`, `i`
      and `1` as one character when comparing names, and nobody can override that check. `scaly`
      is pronounced the same, is a real word people spell right after hearing it, and matches the
      scale-filled S of the logo.
- [x] **R-60. Reserve the PyPI names** `scaly`, `scaly-sqp`, `scaly-piqp` and `scaly-ipopt`: a
      placeholder package per name at version `0.0.0a0` whose description says what it will become,
      built with `uv build` and uploaded with `uv publish` from the gitignored
      `package-placeholders/`. A pre-release version so `0.1.0a1` stays free; PyPI never lets a
      version be reused. `scaly` is reserved as of 2026-09-14; the three plugins wait on PyPI's
      new-project rate limit. The `scali-sqp`, `scali-piqp` and `scali-ipopt` placeholders published
      before the name changed stay up as tombstones, since deleting a project frees its name for
      anyone; point their descriptions at the `scaly` packages once those exist.
- [x] **R-59. Rename everything to scaly.** Done 2026-09-16: the package, the three plugin
      distributions and directories, the `scaly_*` modules, `SCALY_*` environment variables, the two
      console scripts, the JIT cache directory, CI cache paths, `.config/wt.toml`, `zensical.toml`,
      `docs/` and `internal/`, and the `sc` import alias. The Foxglove layouts never contained the
      old string. The GitHub repository was renamed in place and this
      clone's `origin` now points at `PREDICT-EPFL/scaly`; any other clone needs
      `git remote set-url origin git@github.com:PREDICT-EPFL/scaly.git` and, if its directory was
      renamed too, a recreated `.venv` since uv's console scripts hardcode the venv path.
- [x] **R-61. Remove the pre-extraction history from file contents.** Done 2026-09-16; the
      library roadmap went with it, since everything it planned is either implemented or too
      vague to keep.
- [x] **R-63. Audit `internal/` for publication.** Done 2026-09-16 with an independent second
      pass. It stays in the public repository and off the documentation site, which Zensical
      guarantees because it builds `docs/` only. The `no-private-names` `pre-merge` hook in
      `.config/wt.toml` keeps the gate alive after the audit.
- [x] **R-62. Rewrite history and force-push the renamed repository.** After R-37, R-59, R-61 and
      R-63 (all done): drop the local `refs/t3/checkpoints/*` refs, `git filter-repo --invert-paths` on
      the paths R-37 removed, re-add `origin`, force-push every branch and
      tag, delete stale remote branches, and hard-reset or re-clone every other clone and worktree
      rather than pulling. Do this while the repository is still private, since GitHub keeps
      unreachable commits fetchable by SHA until its garbage collection.
- [x] **R-64. Publication metadata and hygiene.** A secrets scan over the rewritten history,
      `CITATION.cff`, a real pyproject description, and ruff's `target-version` aligned with `requires-python`.
- [ ] **R-67. Make the repository public.** After R-62, R-64 and the merge into main, with the suite,
      ruff and ty green locally. The first CI run happens here because the month's Actions minutes
      are spent, and both workflows will consume them once they refill.
- [x] **R-40. Versioning policy.** What a minor bump promises about the generated C symbols, the
      sparsity-table prefixes and the plugin ABI; written into `docs/dev/versioning.md`, with the
      plugins pinning `scaly>=0.1.0a1,<0.2`.
- [ ] **R-66. Platform-only wheel tags for the plugins.** Nothing in the plugins touches the Python
      C API, the solvers load through ctypes, yet `hatch_build.py` in `scaly-piqp` and `scaly-ipopt`
      sets `infer_tag = True`, which stamps the running interpreter's `cpXY-cpXY-<platform>` tag.
      Set the tag to `py3-none-<platform>` explicitly instead, so one wheel per OS and architecture
      serves every Python version and no per-interpreter build matrix or stable ABI is needed.
- [ ] **R-41. Wheel building and publishing.** cibuildwheel with one matrix entry per OS and
      architecture, solvers built natively on each runner as CI already does. `scaly`, `scaly-sqp`
      and `scaly-piqp` first; `scaly-ipopt` follows once its Fortran runtime licensing (L-30) is
      settled. Test PyPI first. After L-28 to L-31 and R-66. Decide the compiler the wheels are
      built with at that point; the system compiler on each runner is the default. `cmake` comes
      from PyPI as a build requirement of both plugins; a C++ compiler and gfortran stay system-wide
      prerequisites for anyone building the wheels themselves.
- [ ] **R-71. `zig cc` as the last fallback compiler.** After `SCALY_CC`, `CC` and `cc` on `PATH`,
      the JIT tries `python -m ziglang cc` when `ziglang` is importable. It wraps clang, so the flag
      dialect the JIT already emits applies. The compiler becomes a command list rather than a
      binary path, which `scaly_toolchain` must print. Expose it as the `scaly[toolchain]` extra.
- [ ] **R-42. Freeze measurements on `0.1.0a1`, tag `v0.1.0a1`** and publish the wheels and a durable
      archive. After R-41.

## Track C closeout before merging to dev

- [x] **C-59. Complete the approved optimization cleanup.** Follow the
      [review and plan](notes/optimization_cleanup_2026_09_10.md), preserving the arithmetic and
      measurement contracts in [fairness](../docs/results/fairness.md).
  - [x] Pin regressions and capture the baseline.
  - [x] Centralize Program analyses, names, and procedure reachability.
  - [x] Normalize compilation expressions and consolidate derivative helpers.
  - [x] Close fusion, arithmetic, and scalarization interactions.
  - [x] Move store coalescing and scalar scheduling into Program passes.
  - [x] Validate the combined compiler and refresh documentation.
  - [x] Address Fable's review and repeat the controlled comparison. The remaining generation overhead is measured and accepted for this cleanup.

Complete the implementation and independent reviews, then run BH-20 on the combined tree.
C-56 and BP-23 can run independently. C-46 and C-49 share derivative construction and run together.
After integration, refresh the test baselines and run the full suite, formatting, lint, and strict
type checks. Run the study without competing tests or compilation. Remove the temporary
implementation worktrees after integrating and reviewing their changes.
Keep metadata and caller-workspace growth as measured limitations. After updating the results,
merge into dev and prioritize documentation.

- [x] **C-56. Prevent workspace slot names from colliding with user buffers.** `pack_workspace`
      allocates scratch slots around every existing parameter, output, and private buffer name.
      The regression in `tests/passes/test_program.py` reproduces silent wrong results with outputs
      named `s1` and `s2`. Implemented and independently reviewed 2026-09-09.

- [x] **C-46. Share stage-invariant and cross-formal derivative work.** The program pass hoists
      invariant buffers before mapped loops. Forward differentiation now combines active formals
      and packs compatible specialized results into one mapped callee output, sharing primal and
      adjoint expressions. This does not coalesce arbitrary original Function outputs or batch
      every runtime seed into a matrix product. Implemented and independently reviewed 2026-09-09.
      [Controlled closeout probes](notes/perf_2026_09_07/README.md#c-46-and-c-49-closeout-2026-09-09).

- [x] **C-49. Reduce derivative operation count without pre-differentiation inlining.** Joint
      propagation across active formals, negation identities, lean division rules, vector self-dot
      and elementwise-square rules, and shared square-root reciprocals are implemented. Tests cover
      seed layouts, caches, overlapping slices, logical input names, and finite-scale derivatives.
      Independently reviewed 2026-09-09. The [audit](notes/perf_2026_09_07/c49_ad_op_audit.md)
      records the diagnosis; [closeout probes](notes/perf_2026_09_07/README.md#c-46-and-c-49-closeout-2026-09-09)
      record the combined C-46/C-49 results. Pre-differentiation inlining is deferred as C-58.

- [x] **BP-23. Vmap the unbumpercars wall rows.** The four wall barriers per car now use one
      mapped Function, preserving constraint order and all existing numerical gates. The growth
      check covers Jacobian source and retained pair/wall call families through the full exact
      Lagrangian Hessian. The old inline-wall formulation fails the growth check. Outer Hessian assembly
      still grows with car count and remains part of deferred C-8. Independently reviewed 2026-09-09.

- [x] **BH-19. Harness gaps.** `dispatch_trip_count`, `dispatch_workspace` and `dispatch_arithmetic`
      were empty for the race_cars and unbumpercars Scaly cells because `_dispatch_metrics` gave up
      on any kernel mapped over two axes (`N` and `N+1` stages; cars and pairs). It now reports the
      dispatch-loop family carrying the most arithmetic per call and the workspace over every
      dispatch, so all four problems fill the columns. The unrolled pair rows no longer exist in
      `filters.py` (removed by the pair-row port); we chose not to recover them from history for a
      same-protocol control, so the mapped-pair result is reported descriptively.
      Closeout follow-up implemented and independently reviewed 2026-09-09: hoisted procedures
      retain their original callee identity for dispatch metrics. Unit-trip and longer maps exclude
      the one-time prologue. Disabling the identity lookup makes the regression fail.

- [x] **BH-20. Complete the closeout study.** Completed 2026-09-11
      in `benchmarks/results/study-2026-09-10`. The [results overview](../docs/results/index.md)
      and [scalability tables](../docs/results/scalability.md) contain the completed study.
      The interrupted 2026-09-09 attempt
      remains preserved in its original result directory and frozen investigation note.
