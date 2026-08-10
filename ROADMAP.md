# Alloy benchmarks, solver plugins, and paper roadmap

Last updated: 2026-07-15. This document supersedes everything that lived in
`fast_benchmarks/` (FastBench prototype, `BENCHMARK_SUITE_PLAN.md`,
`REAL_BENCHMARK_CANDIDATES.md`, `STRATEGY_NOTES.md`), now removed. It covers the
benchmark suite, the solver-plugin packaging, and the path to the first paper.
The library-internal phased plan (IR, AD, codegen, solvers) remains
[`docs/roadmap.md`](docs/roadmap.md); this file is about everything around it.

## 1. Paper thesis

The first paper is deliberately modest. Alloy is presented as an experimental
tool for our own research that others are encouraged to try and give feedback
on — not a silver bullet, not a CasADi replacement. Legitimacy is built
gradually through internal use (bumpercars safety filter, hovercraft MBD/DIAL,
racing MPFC) rather than claimed upfront.

Main claims:

1. **Alloy is an alternative way of building a codegen tool, focused on
   deployment** (JIT-as-default through the universal C ABI, single-`.so`
   embedded solvers, no Python in the hot path).
2. **Alloy fixes concrete CasADi issues in code-size scalability and
   performance** (constant-source loop-preserving codegen; mixed dense-NN +
   sparse-scalar workloads).
3. **Alloy's architecture allows initial GPU support** — enough to write new
   things easily, with no pretense of matching torch efficiency.

Explicit non-claims: we do not present the ambition of writing solvers *in*
alloy, and we do not claim GPU code competitive with torch.

Note: claim 3 depends on an initial GPU backend that is currently deferred in
`docs/roadmap.md` ("GPU codegen deferred until a workload justifies it"). The
paper scope makes it a workload requirement now — a minimal GPU lowering
milestone must be planned in the library roadmap before the paper freeze
(candidate driving workload: batched neural dynamics evaluation in the
bumpercars filter). Open until scoped.

Benchmarks are simple and focused on the claims: they demo that alloy is
useful, on the problem types we actually work on. The most complex applications
are *not* distilled into benchmarks — alloy gets integrated into their own
repos directly, and that integration is the real validation.

## 2. Benchmark suite

### 2.1 Location: in the alloy repo, under `benchmarks/`

The earlier separate-repo decision is reversed. Rationale: the suite's scope
shrank to CasADi + laopt baselines (no acados/BOPTEST/OpenFAST-class
dependencies), so the original reasons (heavy deps, credibility of a standalone
suite) no longer outweigh the friction of a second repo. The suite tests alloy
through its installed API surface regardless.

Plan:

- Problem definitions move from `tests/` into `benchmarks/`
  (`test_tracking_workload.py`, `test_unbumpercars_workload.py`,
  `test_safety_filter_workload.py` fixtures become benchmark problem modules;
  thin unit tests may remain in `tests/`).
- Benchmarks are excluded from the default `uv run pytest -n=auto tests/` run.
  A dedicated runner script executes them; it is wired into CI and the
  worktrunk checks.
- The existing gbench harness generators (`benchmarks/alloy_*_benchmark.py`,
  `benchmarks/scalability_sweep.py`) are absorbed into the new layout — their
  mechanics (per-cell compile, fail-fast dense-reference correctness check,
  monotonic short-circuit on timeout/size caps) become the sweep runner.

Sketch (to be refined during implementation):

```text
benchmarks/
  README.md
  run.py                  # entry point: smoke / sweep / closed-loop
  problems/
    chain_of_masses/
    race_cars/            # tracking NMPC now; MPFC lands in the same package
    bumpercars_filter/
  harness/                # sim loop, gbench wrapper gen, mcap dump, metrics
  results/                # gitignored raw outputs
```

### 2.2 Problems (start small)

| problem | scaling axis | notes |
|---|---|---|
| chain of masses | number of masses | classic hanging-chain NMPC (Wirsching/Bock/Diehl form); match the laopt paper's instance parameters where possible for an external reference point |
| race cars: tracking NMPC, kinematic bicycle | horizon N | full-size Formula Student car on vendored FSDS tracks, with the minimum-curvature spline reference generator and lateral corridor constraint from `minimal_tracking_nmpc`; **later replaced by Johannes' MPFC with dynamic bicycle** (backlog), which lands in the same `race_cars/` package |
| bumpercars safety filter, CT neural dynamics + discrete-time CBF | number of cars | **based on `examples/ct_dt_cbf_filter/`** (CT neural model + RK4 + one-step DT position CBF, centralized, `al.map_` over the car axis; CasADi + alloy implementations, closed loop, and per-step instrumentation already exist there). Fold the desired control into the **simulator's** dynamics only (not the OCP's), so the closed loop is plant + filter with no third controller entity. The older input-affine safety-filter variants in `benchmarks/` are **removed** — the CT DTCBF filter is the one that is preserved. |

Future problem candidates (not now): diffusion-based stuff, GP stuff,
hovercraft MBD/DIAL.

Retained doctrine from the previous strategy round:

- **Claims-first admission**: a problem/backend enters only if it supports a
  paper claim or serves as a smoke/cross-validation case.
- **Two measurement protocols over the same parameterized problems**: sweeps
  along the claim-bearing axis for kernel/codegen metrics; 1–2 application-
  justified canonical operating points where closed-loop runs and absolute
  comparisons happen. No fixed-size absolute-only problems.
- **Correctness gates before speed**: every backend's outputs cross-checked
  against a reference (NumPy/CasADi) on shared episodes before any timing is
  recorded; per-cell artifacts (generated source, compile logs, raw traces)
  archived with provenance (alloy commit/version, mode).
- **Steelman the baselines**: recurring passes tuning CasADi/laopt options per
  problem, tuning logs archived.

### 2.3 Solvers and baselines

Eventual plugin coverage: QP — piqp, osqp, proxqp; NLP — ipopt, fatrop, laopt,
acados. Start small and grow problems and solvers together.

**Now** (no structure beyond what `al.nlp` already provides):

| toolchain | solver path |
|---|---|
| casadi | IPOPT via Opti, JIT enabled |
| alloy | IPOPT (existing `al.nlp` backend, moved to plugin) |
| alloy-sqp | **one solver, two oracle providers**: the custom SQP (§3.3) run once with alloy C-ABI oracles and once with CasADi-codegen oracles (same ABI) — a controlled FE comparison |

The alloy-sqp design gives the cleanest fairness story available: the solver
binary, QP subsolver, globalization, and tolerances are *identical* across the
two columns; only the oracle provider changes — which is exactly where alloy's
claims live. CasADi's builtin `sqpmethod` is at most a secondary reference
column: it differs in globalization details (line search, regularization,
no laopt-style constraint relaxation), so a cross-solver comparison against it
would conflate algorithm and oracle differences.

**Later**: laopt as an *external baseline* where its problem implementations
already exist (MPFC in the racing repo), once it is published;
casadi+FATROP via Opti (JIT); alloy+FATROP (needs the structured OCP tier,
backlog); acados.

### 2.4 Metrics and tests

Two complementary benchmark types:

1. **Artificial FE scalability benchmarks** — focus on alloy's actual novelty,
   the function evaluation (oracles: dynamics, constraints, sparse
   Jacobians/Hessians). Measured with the Google Benchmark C++ harness (small
   generated wrapper, as done until now). Inputs are **representative problem
   instances harvested from closed-loop runs** (dumped states/params deemed
   representative), not random points. Swept along each problem's scaling
   axis; also record generated source size/LOC, compile time, workspace size.
2. **Closed-loop benchmarks** — show usefulness in practical scenarios at the
   canonical operating points. Use the solvers' internal timings (solver vs FE
   split where available; for laopt the alloy adapter times FE itself, since
   laopt exposes no timing — see §3.3). Dump everything needed for
   visualization into **MCAP files** (written with the `foxglove-sdk` Python
   package; every message definition is a JSON schema generated from a
   Pydantic model) and build **Foxglove layouts**: curve
   plots for key quantities plus 3D viz for chain of masses, 2D viz for racing
   and bumpercars.

Closed-loop simulator: **non-real-time model-in-the-loop only** (plant = same
or perturbed model integrated with a fixed-step integrator; no real-time
constraints, no hardware). Embedded targets (Raspberry Pi / Jetson) are
explicitly out of scope for now (backlog).

## 3. Solver plugin system

### 3.1 Architecture (unchanged from previous round, restated)

Solver integrations are separate Python packages developed in the alloy repo as
a **uv workspace** (per <https://docs.astral.sh/uv/concepts/projects/workspaces/>),
so users install only what they need while alloy stays batteries-included:

- Core owns the small versioned **oracle protocol** — what `al.qp`/`al.nlp`
  normalize to today (`x, p → f, h_eq, g_ineq` + factory-built `spjac`/
  `sphess`, PIQP-style explicit constraint categories). Registry via
  `importlib.metadata` entry points behind the existing `solver="..."` API.
- A **structured staged-OCP tier** (per-stage dims, MAP-based stage functions;
  canonical form that *lowers* to the general tier) is deferred to the backlog
  together with the specialized OCP problem/solver — none of ipopt/laopt needs
  it.
- **Vendoring rules**: vendor per-package only when the AOT/C-ABI story needs a
  linkable `.so` + headers; duplicate linear-algebra deps statically with
  hidden symbol visibility, never a shared linalg package; licensing isolated
  per package (EPL-2.0 IPOPT, etc.).
- Consequence: PIQP and IPOPT eventually move out of `hatch_build.py` into
  `alloy-piqp` / `alloy-ipopt`, making core a pure-Python wheel (NumPy only).

Initial workspace members: `alloy` (core), `alloy-piqp`, `alloy-ipopt`,
`alloy-sqp`. Layout sketch (solver plugins live under `plugins/`):

```text
alloy/                    # repo root = uv workspace root
  pyproject.toml          # core package + workspace table
  src/alloy/
  plugins/
    alloy-piqp/
    alloy-ipopt/
    alloy-sqp/
  benchmarks/
  examples/
  tests/
  docs/
```

**Wheel building for the plugin packages is deferred to the backlog.**
Development proceeds on workspace path dependencies (editable installs);
wheels/cibuildwheel/release engineering become relevant only when distributing
to outside users, and nothing in the benchmark or paper path needs them.

### 3.2 laopt: investigation findings (2026-07-13) — adapter path dropped

**Decision: no laopt adapter for now.** Instead we build our own C++ SQP
(`alloy-sqp`, §3.3), using laopt's textbook-but-comprehensive SQP as the
algorithmic reference — the same approach previously taken in anvil. The
findings below motivated the decision and are kept for the record; laopt
returns later as an external baseline for MPFC (§2.3, backlog).

laopt (Waibel, Schwan, Jones — EPFL LA; header-only C++17; **not yet public**,
"available upon publication" — coordinate with the authors before depending on
it) solves NLPs of exactly the form `min φ(ξ) s.t. c(ξ)=0, ξ_lb≤ξ≤ξ_ub,
h_lb≤h(ξ)≤h_ub`. Verified against the code (`~/dev/laopt`):

- **Problem form matches `al.nlp`** (two-sided general inequalities + separate
  box; equalities are rows with `lb==ub`, reclassified at the QP layer with
  `EQ_TOL=1e-10`). No runtime parameter vector through the oracle — fixed data
  live as C++ members; re-solves re-linearize. An adapter holds alloy's `p`
  and updates it between solves.
- **No external-oracle interface today.** Users write templated C++ functions
  (CRTP `Differentiable`); derivatives come from laopt's Eigen forward-AD or
  from a CasADi AD backend it drives itself (`differentiable_casadi.hpp`,
  optionally JIT). It does **not** load pre-generated CasADi-C-ABI functions
  in the main path (the raw C ABI appears only in a benchmark comparison).
  The integration point for alloy is laopt's internal problem dispatch:
  `eval_objective`, `eval_objective_gradient`, `eval_objective_hessian` (GN),
  `eval_constraints(cons, lb, ub)`, `eval_constraints_jacobian`,
  `eval_lagrangian_hessian(obj_factor, dual, hess)` — all Eigen sparse **CSC**,
  which alloy's `SparsityType` already converts to. Hessian: Gauss-Newton is
  the default, exact optional — the alloy oracle should serve both.
- **PIQP is the default and only enabled-by-default QP backend**, consumed as
  the *header-only C++ template* `piqp::SparseSolver` via `find_package` —
  compiled into the binary, no shared library. Therefore **the `alloy-piqp`
  dylib (`libpiqpc`, the C interface) cannot be reused by laopt**; instead
  `alloy-laopt` builds PIQP headers into its own extension. This is consistent
  with the duplicate-don't-share vendoring rule.
- **Dimensions are compile-time template parameters.** An `alloy-laopt`
  backend therefore likely generates and JIT-compiles a small C++ translation
  unit per problem (instantiating `SQPSolver<Problem, PIQPSolver<double>>`
  with the problem dims, linking alloy's generated oracle `.so`) — the same
  render/compile/cache model as `src/alloy/jit.py`. Whether laopt tolerates
  dynamic sizes instead is an open question to check with the authors.
- **laopt reports no timings** (only iteration counts and convergence
  metrics). Since the alloy adapter *is* the oracle, it times FE itself; total
  solve time is wrapped externally. For the casadi-laopt baseline, instrument
  the same way.

Paper reference points (mini race car, N=25, tf=0.9 s): laopt SQP/PIQP 8.8 ms
vs IPOPT 15.6 ms, FATROP 9.9 ms, acados 10.0 ms; laopt RTI 1.5 ms.

### 3.3 `alloy-sqp`: a custom C++ SQP plugin

Rather than adapting laopt (private, pre-publication, no external-oracle path,
compile-time dims), we write our own C++ SQP shipped as the `alloy-sqp`
plugin, following the same textbook-but-comprehensive algorithm set as laopt
(Gauss-Newton default / exact Hessian option, L1-merit or filter line search,
constraint relaxation, Gershgorin regularization) — the approach already
prototyped in anvil's SQP solver.

Design points:

- **Oracle interface = the CasADi C ABI convention**
  (`int f(const double** arg, double** res, int* iw, double* w, void* mem)`)
  with CSC sparsity metadata. Alloy emits this natively; CasADi's codegen
  emits the same ABI. One solver binary therefore runs with either oracle
  provider — the controlled-comparison design of §2.3, and the reason this
  solver is a measurement instrument first, product feature second.
- **QP subsolver: PIQP through its C interface** (`piqp_c`) — i.e. it links
  the same shared library `alloy-piqp` vendors, so the dylib is genuinely
  reused (unlike laopt, which inlines PIQP's C++ templates). Runtime dims,
  no per-problem C++ compilation.
- **Timing built in from day one**: FE time per oracle function vs QP time vs
  line-search overhead, exposed in the stats — the split laopt doesn't offer
  and the paper needs.
- Exposes solve + stats through the same `SolverFunction` surface as the
  piqp/ipopt backends (entry-point registered, `solver="sqp"`).
- Paper positioning caution: this stays consistent with the §1 non-claim —
  it is a solver *for* alloy (plugin infrastructure and measurement
  instrument), not a solver written *in* alloy's IR, and it is not a headline
  contribution. Report it as such.

Open questions: warm-start strategy across SQP solves in closed loop; whether
the anvil implementation is ported or rewritten against the oracle protocol;
how much of laopt's globalization detail to replicate before diminishing
returns.

### 3.4 Binding doctrine: generated C glue, one artifact (decided 2026-07-14)

Hand-written Python solver bindings are **transitional, not the model**. The
target is the anvil pattern throughout: every function callable from Python is
JIT-compiled, and JIT thinly wraps AOT with a ctypes interface generated
specifically for that function. Solver integrations therefore become
**generated C wrappers emitted alongside the oracle code** — calling the
solver's C API (`piqp_c`, `IpStdCInterface.h`) directly and linking the
vendored dylibs — not per-solver Python extension modules. One artifact gets
built, and it is the same artifact used at deployment.

Consequences:

- The Python-interleaved solve path (`al.qp`/`al.nlp` calling oracle and
  solver from Python) is **removed** (initially planned as a demoted
  reference/debug mode; dropped outright 2026-07-15 — correctness is gated by
  analytic/KKT/CasADi references instead, and IPOPT/PIQP update rarely enough
  that binding drift is not a live risk).
- The nanobind `_piqp_ext` (c09d11f) and the low-level IPOPT Python-callback
  binding were transitional and are **deleted (2026-07-15)** — the generated
  path covers all their consumers. **No further investment in Python
  bindings** — in particular, no Python sparse-PIQP binding: sparse support
  lands in the generated wrapper, where the CSC pattern is known at codegen
  time and baked in as static tables (values-only `piqp_update_sparse` at
  runtime).
- **Stats/error resurfacing** is the one real design task: alloy owns a small,
  stable C stats struct (status code, iterations, objective, and the
  FE / solver / glue timing split the paper needs) that every generated
  wrapper fills from the solver's native info struct. The mapping lives in the
  per-solver codegen template and is compiled against the vendored headers, so
  upstream struct/enum drift breaks loudly at vendor bumps instead of
  silently — the drift problem hand-written bindings have is dissolved
  structurally. Solver-native structs never cross into Python; Python-side
  status names/dataclasses are driven by the alloy enum.
- Solver settings are baked into the generated wrapper as constants; the JIT
  cache keys on them, so option-tuning sweeps recompile per point — acceptable
  given per-cell artifact archiving (§2.4) already assumes per-cell builds.
- `alloy-sqp` (§3.3) is already a native citizen of this model (C-ABI oracles
  in, `piqp_c` inside, stats struct out) and needs no adaptation.

This reframes L2 (§5) from a one-off filter deliverable into the universal
solver-integration mechanism, and raises its priority accordingly.

### 3.5 Plugin-owned codegen templates (decided + landed 2026-07-15)

L2 landed with the per-solver C wrapper templates living in core
(`src/alloy/codegen/solver_c.py`) — an abstraction leak: everything else about
a solver was already plugin-local (vendored lib, headers, entry point), but
adding a new solver still meant editing the alloy codebase. That is exactly
CasADi's plugin model (`Conic`/`Nlpsol` plugins are written in-tree against
internal headers), and its weakness: third parties cannot ship a solver
integration as their own package.

Decision: **the wrapper template is part of the plugin.** The
`SolverBackend` entry-point protocol (`src/alloy/solvers/registry.py`) gains a
`render_wrapper(fun, ctx)` codegen hook next to the packaging metadata;
protocol version bumped to 2. The PIQP/IPOPT templates moved to
`alloy_piqp/codegen.py` / `alloy_ipopt/codegen.py` unchanged (the generated C
is byte-identical for single-backend units, verified by JIT-cache hits across
the move; a unit reaching *both* backends orders includes/link flags
alphabetically now instead of piqp-first — a one-time cache miss, accepted). Core keeps
everything that makes the contract stable — problem normalization and oracle
assembly, the `_raw` calling convention and Program-IR kernel rendering, the
`alloy_solver_stats` struct + status enum + clock, JIT/cache/link-flag
plumbing, and the framing of every wrapper (stats storage + accessor) — and
`toolchain.py` / `codegen/solver_c.py` lost their hardcoded piqp/ipopt
knowledge (backend discovery, includes, link flags, and the
`ALLOY_<NAME>_LIB` override env vars are all registry-driven now).

The full contract a plugin must respect is documented in
[`docs/solver_plugins.md`](docs/solver_plugins.md): required `_raw` signature,
oracle output orderings per descriptor family, stats-filling obligations,
status mapping through vendored enum constants, workspace pass-through, and
the protocol-versioning rules. A structural test suite
(`tests/alloy/test_solver_registry.py`) pins the registry gates and the
core/plugin codegen handoff with a fake in-test backend.

Why now (and not with B4): `alloy-sqp` (§3.3) is the first new backend; had it
been written against a core-owned template file it would have grown core the
same way CasADi grows. Writing it as an external plugin against this protocol
is the validation that the interface is complete — see B4.

Paper relevance: this is a claim-1-adjacent architecture point — solver
integrations are pip-installable packages (vendored lib + headers + a Python
render hook), not in-tree plugins. Worth a paragraph in the deployment story.

## 4. MPFC benchmark (backlog item, scouted 2026-07-13)

Johannes' MPFC is the racing formulation from the laopt paper, living in
`~/dev/racing-project-ros2/mpc-control-racing-ros2/lib/mpc-control-racing/`
(`SplineFollowingTire_LAMP`: laopt + multiple shooting + PIQP). Scouting
summary for the future distillation:

- **Formulation**: dynamic bicycle (6 states, simplified Pacejka front/rear,
  exponential drivetrain map; identified 1:10 RC car, m=144 g) stacked with a
  double-integrator progress parameter → 8 states, 3 inputs (+1 slack).
  Contouring cost `w_P‖spline(γ) − pos‖²`, progress reward `−w_γ(γ−γ₀)`,
  body-velocity/yaw regularization, input + input-rate costs; linearized track
  half-width constraints with slack. MS N=25 segments, IRK2, tf=0.9 s.
- **Track data**: YAML; minimal artifact is the spline file
  (`config/la_track_spline2.yaml`, 36 cubic pieces + left/right offsets),
  optionally the sampled centerline (`la_track_sampled.yaml`) for the progress
  locator/width.
- **Closed-loop simulator: use `simulation-driving`** —
  `~/dev/racing-project-ros2/simulation-driving-ros2/lib/simulation-driving/`
  (`DrivingSimulator`, `BotSimulator`) is exactly the simple model-in-the-loop
  simulator this benchmark needs; distill/port its plant loop rather than
  writing one from scratch. Lap counting via γ-wrap. The MPC numeric core has
  zero ROS2 coupling — extraction is low effort.
- The repo's CasADi `MO*` variants implement the same formulation via CasADi
  Opti + IPOPT — the natural reference when writing the casadi baseline impl.
- Distillation goal: only the model, one track, and a simple closed loop over
  1–2 laps. Reimplement model + cost in alloy (and CasADi), copy the track
  YAML, write the small runner. Do **not** port the C++ controller stack.

## 5. Execution plan

Phases are ordered; each is independently useful. Problems and solvers grow
together across iterations, **interleaved with library improvements** (L-track)
that the benchmarks expose as prerequisites.

### Library prerequisites (L-track)

Verified gaps, documented first-hand in `examples/ct_dt_cbf_filter/README.md`
("Missing Alloy features exposed by this prototype"):

- **L1 — sparse Lagrangian Hessian through `Ops.MAP`**
  (`sphess:lagrangian:z:z` over mapped neural RK4). Required for exact-Hessian
  IPOPT/SQP columns on the bumpercars problem. **Decision (2026-07-13):
  proper fix only** — second-order AD rules + sparse second-order lowering for
  `Ops.MAP`. No unrolled-map or per-car manual-assembly fallback (that is
  exactly the code-size blowup claim 2 argues against); Gauss-Newton columns
  fill the gap until L1 lands. **Completed (2026-07-14)**: MAP reverse AD
  (cached concat-adjoint mapped once, three stride-class assemblies),
  tensor-form matmul VJP + structural `jvp_many` SCATTER/GATHER/TRANSPOSE
  rules, sphess-through-MAP end-to-end with the permanent
  `ALLOY_STRICT_JVP_MANY` tripwire, and the bumpercars `--exact-hessian`
  column cross-validated against CasADi's `ctdt_hess_lag` at rtol 1e-8. The
  cross-check also flushed out a repo-lifetime CALL-VJP bug (formal
  substitution rewrote symbols inside the incoming cotangent; fixed). Known
  caveat documented in `docs/spec.md`: shared stride-0 `diff=True` formals
  degrade Hessian coloring to O(length).
- **L2 — generated-C solve path as the universal solver integration**
  (reframed and priority raised 2026-07-14, §3.4): every solve callable from
  Python goes through a generated C wrapper — the nested `SOLVER_CALL`
  single-`.so` path, extended — that calls the solver's C API directly with
  the generated oracle kernels; no Python/ctypes callbacks in the loop.
  Scope: (a) the alloy-owned stats struct with per-layer timing (FE vs solver
  vs glue: callback transition, buffer copies, kernel); (b) sparse PIQP via
  `piqp_c` with the CSC pattern baked at codegen time; (c) IPOPT through
  `IpStdCInterface.h` with callbacks pointing at generated kernels; (d) the
  Python-interleaved path demoted to reference/debug mode, and the nanobind
  `_piqp_ext` + `_piqp_ctypes.py` deleted after the generated path soaks.
  The bumpercars standalone filter is the first consumer, not the scope.
  **Status: COMPLETE (2026-07-15). (a) L2-1 (2026-07-14): stats ABI +
  standalone dense QP through the generated wrapper. (b)+(c) L2-2:
  `al.qp(..., sparse=True)` bakes structural CSC patterns of P/A_eq/G_ineq as
  static tables with compact CSC-ordered oracle values and values-only
  `piqp_update_sparse`; the IPOPT wrapper passes the caller's workspace into
  the eval callbacks (root cause of the chain-scale crash), fills the stats
  struct (status via vendored enum constants, iterations via the intermediate
  callback, eval counts, FE/solver/glue split), consumes full warm starts
  (`x0`, `lam_eq0`/`lam_ineq0`, and the signed box multiplier `lam_box0`,
  sign-split into IPOPT's z_L/z_U), and clamps bounds to ±2e19. (d) done
  without a soak period: the Python-interleaved path, the nanobind
  `_piqp_ext` + `_piqp_ctypes.py`, and the ctypes IPOPT callback binding are
  deleted; plugins ship vendored libs + headers + entry-point metadata only.
  The bumpercars CT-DTCBF filter (first consumer) runs through `al.nlp`'s
  generated wrapper with cross-step warm starts and stats-derived
  instrumentation.**
- **L3 — IPOPT low-level binding parity**: accept/return `lam_x`/`lam_g`
  warm starts and final multipliers, expose iteration count and callback
  counts. **Completed (2026-07-13).**
- **L4 — parameterized model constants**: physical constants and `dt` as
  symbolic parameters in the decorated ODE functions instead of baked-in
  values (blocks tuning sweeps). **Status: COMPLETE (2026-07-13).** Chain,
  race-car, and CT-DTCBF dynamics receive their physical constants and step
  size through symbolic parameter vectors in both Alloy and CasADi paths
  (`1326b75`).
- **L5 — plugin-owned solver codegen templates** (§3.5): move the per-solver
  C wrapper templates out of core into the plugins via the `render_wrapper`
  protocol hook, de-hardcode piqp/ipopt from `toolchain.py` /
  `codegen/solver_c.py`, document the plugin contract
  (`docs/solver_plugins.md`), bump the plugin protocol to v2.
  **Status: COMPLETE (2026-07-15).** Generated C verified byte-identical
  across the move; structural protocol tests added.

Scheduling: L3 and L4 are small and land early (with B2); L1 and L2 are the
substantial ones and gate the exact-Hessian and callback-free benchmark
columns respectively — interleave them between B2 and B4.

### Benchmark phases (B-track)

- **B0 — benchmark infra in-repo**: create `benchmarks/` layout; move the
  workload problem definitions out of `tests/`; dedicated runner script;
  exclude from default pytest; wire into CI + worktrunk checks (smoke tier —
  correctness gates + LOC/workspace invariants at minimal sizes — runs
  pre-merge; sweeps and closed-loop runs are manual-only); port the
  regression guards (correctness + generated-LOC/workspace invariants as
  assertions; runtime record-only). `examples/ct_dt_cbf_filter/` moves
  wholesale to `benchmarks/problems/bumpercars_filter/` and `examples/` is
  deleted. All legacy `benchmarks/` scripts are removed once their mechanics
  are absorbed into the new harness (`alloy_safety_filter_benchmark.py`,
  `measure_safety_filter.py`, `alloy_solver_aot_demo.py`,
  `viz_tracking_eq_jac_probe.py`, `alloy_tracking_eq_jac_benchmark.py`,
  `alloy_unbumpercars_ineq_jac_benchmark.py`, `scalability_sweep.py`,
  `scalability_results.csv`, `gen/`). **Status: COMPLETE (2026-07-13,
  `8b1dd3f`).** The suite lives under `benchmarks/`, smoke runs in CI, sweeps
  remain manual, and generated artifacts carry provenance. The obsolete
  input-affine unbumpercars workload was removed from the benchmark runner
  during B3 cleanup; its focused Program-IR regression fixture remains for
  compiler coverage.
- **B1 — workspace + existing plugins**: convert the repo to a uv workspace
  with solver plugins under `plugins/`; extract `alloy-piqp` / `alloy-ipopt`
  from `hatch_build.py` (core goes pure-Python); formalize the oracle
  protocol + entry-point registry behind the existing `solver=` API.
  Wheels: backlog. **Status: COMPLETE (2026-07-13, `2c51692`).** Core is a
  NumPy-only package and PIQP/IPOPT are independently registered workspace
  plugins.
- **B2 — problems v1** (with L3/L4): chain of masses, race-car tracking NMPC
  (kinematic bicycle, from the existing fixture), bumpercars CT-DTCBF filter promoted
  from `examples/ct_dt_cbf_filter/` (desired control folded into dynamics).
  Alloy + CasADi implementations, NumPy/CasADi reference gates, sweep axes
  wired, `expand=True` added to the CasADi columns. **Status: COMPLETE
  (2026-07-13, `3111b52` + `1326b75`).** The three selected formulations,
  symbolic model parameters, CasADi expansion, reference checks, and
  generated-solver bumpercars path are present.
- **B3 — closed loop + viz**: model-in-the-loop simulator, episode/lap logic,
  MCAP dumping, Foxglove layouts (curves + 3D chain / 2D cars), canonical
  operating points chosen and documented, FE-instance harvesting for the
  gbench benchmarks. **Status: COMPLETE (2026-07-15).** A common Pydantic →
  JSON-schema MCAP recorder emits solver/control telemetry, `/tf`, and native
  `SceneUpdate` geometry, with centered scene frames, trajectory trails, and
  arena bounds. Layouts are not generated: each problem keeps one hand-authored
  `foxglove-layout.json`, exported from Foxglove Desktop, next to its runner. The
  canonical points are chain M=5/N=12 (20 steps; harvested state rolled into
  the N=40 FE transcription), race cars N=40 (one lap of the 340 m FSDS
  `fsds_competition_1`, 1367 steps at 0.05 s), and
  bumpercars C=4 (80 steps, seed 42). Their midpoint successful oracle inputs
  feed the canonical gbench cells; CT-DTCBF C=2/4/8 Alloy + CasADi sweep cells
  replace the legacy input-affine unbumpercars axis. The
  `benchmarks/run.py closed-loop` command owns smoke/canonical execution and
  reproducibility artifacts. The race-car problem carries both closed-loop
  columns of §2.3's **Now** table: `--backend alloy` and `--backend casadi`
  build a deliberately identical NLP (same `z`/`p` layout, same rows in the same
  order, same IPOPT options, `expand=True`) and are gated against each other by
  a cross-backend trajectory comparison, so only the oracle provider differs.
- **B4 — `alloy-sqp`** (after L1/L2): the custom SQP per §3.3 over `piqp_c`;
  the one-solver-two-oracles columns (alloy oracles vs CasADi-codegen oracles)
  added to all B2 problems; FE/QP/line-search timing split in stats.
  **Must be built as an external plugin against the L5 protocol (§3.5,
  `docs/solver_plugins.md`)**: `BACKEND` + `render_wrapper` in
  `plugins/alloy-sqp`, zero edits to `src/alloy/` — if a core edit turns out
  to be needed, that is a gap in the plugin protocol to fix explicitly (with
  a protocol-version bump if breaking), not a reason to special-case core.
  B4 doubles as the acceptance test that L5's interface is complete.
- **B5 — paper assembly**: full sweeps + closed-loop runs at canonical points,
  figures, GPU-claim experiment (gated on the GPU backend milestone, §1).

## 6. Backlog

- Replace the race-car tracking NMPC with the **MPFC** distillation (§4), in
  the same `benchmarks/problems/race_cars/` package.
- **Make the Program IR passes iterative instead of recursive.**
  `passes._transform` / `_expand_inlinables` recurse per node, ~5 Python frames
  per expression level, so an expression deeper than ~200 chained elementwise
  ops dies with a bare `RecursionError` during lowering. Found while building
  the race-car objective at N=40 as a left fold (`cost = cost + ...`);
  worked around there by expressing the cost as one flat weighted-square
  reduction. Explicit-stack rewrite of the tree walks, plus a diagnosable error
  if a depth cap is ever kept. See `docs/spec.md` "Known limitation: pass
  recursion depth" and the `xfail` in `tests/alloy/test_passes.py`.
- **laopt as an external baseline** for MPFC (its implementation already
  exists in the racing repo), once laopt is published.
- **Specialized OCP problem/solver in alloy** (structured staged-OCP tier that
  lowers to general-form problems for solvers that don't exploit structure).
- **fatrop** plugin (consumer of the structured tier) + casadi-fatrop baseline.
- osqp / proxqp / acados plugins as claims or users demand them.
- **Wheel building** for workspace packages (cibuildwheel, per-package native
  builds, release engineering).
- Embedded hardware benchmarks (Raspberry Pi / Jetson).
- Diffusion- and GP-based problems; hovercraft MBD/DIAL workload.
- GPU backend milestone definition in `docs/roadmap.md` (prerequisite for
  paper claim 3).
