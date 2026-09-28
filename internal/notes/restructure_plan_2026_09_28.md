# Scaly restructure plan: minimal core, one method interface, five distributions

> Planned work, written 2026-09-28 for execution in Claude Code. It supersedes the package list in
> `internal/notes/scaly_core_split_2026_09_28.html` (17 packages, an `mpc` package); the analysis
> in that report (inventory, couplings, op footprints) still holds and is referenced below.
>
> Everything here is settled unless it appears under **Open items** at the end.

## How to execute this plan

- Read `AGENTS.md`, `docs/dev/codebase.md`, `docs/dev/contributing.md`, `docs/dev/versioning.md`
  and `docs/dev/solver_plugins.md` first. Their rules bind every step: imports go down the
  `IMPORT_LAYERS` table, every new module gets an `IMPORT_LAYERS` entry and a one-line ownership
  docstring, no silent fallbacks, never cite a commit hash made on the current branch, always
  `uv run`.
- Work one **step** at a time (§4). A step is one branch and one merge. Do not start a step until
  the previous step's gate passes.
- After each step: tick it in the status table below, add a one-line entry to the step's
  **Log** line, and regenerate `tests/baseline/pytest_nodeids.txt` if tests moved (procedure in
  `docs/dev/contributing.md`).
- The gate for every step includes: `uv run ruff format`, `uv run ruff check`, `uv run ty check`,
  `uv run pytest -n=auto` (full suite, including solver-marked tests with the plugins built), and
  `tests/test_c_snapshot.py` byte-identical unless the step says otherwise. Steps touching IR, AD or
  lowering also run `uv run benchmarks/run.py smoke` and compare medians against the previous step.
- Pre-1.0 breaks are unshimmed (`docs/dev/versioning.md`). When a public name moves, update every
  caller, example, notebook and doc page in the same step; do not leave aliases.
- When a step reveals that this plan is wrong, stop, write the finding under **Open items**, and ask.

### Status

| Step | Title | Status |
|---|---|---|
| 0.1 | Housekeeping | ☑ |
| 0.2 | Extern calls raise on derivatives | ☑ |
| 1.1 | `EXTERN_CALL` and the extern-callee protocol | ☑ |
| 1.2 | Output-adapter registry (C++, CasADi) | ☐ |
| 1.3 | Core stops importing solvers | ☐ |
| 2.1 | Op registry, builtins registered through it | ☐ |
| 2.2 | Table-driven AD, sparsity, folding, verification | ☐ |
| 2.3 | Public lowering context, op traits, option namespaces, pass slots | ☐ |
| 2.4 | `scaly.ext`: library-author Function API, extension versions in the JIT cache key | ☐ |
| 3.1 | Linear-algebra ops move into `scaly.linalg` | ☐ |
| 3.2 | `linalg.banded` and `linalg.stagewise` | ☐ |
| 4.1 | Method registry and the common `Info`/`Status` | ☐ |
| 4.2 | `scaly.opt`: problems, `solver()`, external methods | ☐ |
| 4.3 | IPM as the method `opt.ipm` | ☐ |
| 4.4 | `scaly.roots` | ☐ |
| 4.5 | `integrators` and `interp` as method registries | ☐ |
| 5.1 | `scaly.sets` | ☐ |
| 5.2 | `scaly.ocp`: continuous and discrete OCPs, transcription, formulation | ☐ |
| 5.3 | OCP methods, warm start, terminal ingredients; `mpc` removed | ☐ |
| 6.1 | Library code promoted from examples | ☐ |
| 6.2 | `nn`, `geometry`, `export` namespaces | ☐ |
| 7.1 | Conformance suites and the `method` marker | ☐ |
| 7.2 | Examples: per-namespace folders, PEP 723 headers, public API only | ☐ |
| 8.1 | `distributions.toml`, manifests, metapackage | ☐ |
| 8.2 | Namespace import mechanics (`extend_path`, lazy attributes) | ☐ |
| 8.3 | CI: isolation jobs, plugin jobs, release job, release script | ☐ |
| 9.1 | Optional: indexing consolidation | ☐ |

## 1. Settled decisions

1. **One repository.** Everything stays in this repo as one uv workspace.
2. **One import namespace.** Users write `import scaly as sc`, `from scaly import linalg as la`,
   `sc.linalg.ldl`, `from scaly.ocp import DiscreteOCP`. Every module has exactly one name
   (`scaly.linalg.sparse`); there are no `scaly_linalg` aliases.
3. **One source tree, sliced into distributions.** All code lives under `src/scaly/`. Each
   distribution's wheel contains a declared subset of that tree. In the repo everything is always
   importable; isolation is enforced by tests and CI (§3.6), not by the file system.
4. **Few distributions** (§2.3): `scaly-core`, `scaly-numerics`, `scaly-control`, `scaly-tools`,
   `scaly-experimental`, `scaly-testing`, the `scaly` metapackage, and one distribution per
   external solver.
5. **Versioning:** all first-party distributions release in lockstep from one tag; external-solver
   plugins version independently (§3.7).
6. **One pattern for every numerical domain:** a problem class, any number of methods, one
   `solver(problem, method)` entry point that returns a `ConcreteFunction`. Generated methods and
   external solvers are indistinguishable to the caller (§3.1).
7. **The IPM is a method of `scaly.opt`** (`opt.ipm`), like PIQP and IPOPT.
8. **OCPs are continuous or discrete.** Transcription converts a continuous OCP into a discrete one
   and is an OCP-level operation. Terminal cost and terminal set are part of the OCP.
9. **No `mpc` namespace.** MPC is an OCP, a solver method and a warm-start shift. The shift lives
   in `scaly.ocp`; closed-loop simulation lives in examples.
10. **The core keeps no linear-algebra kernels beyond `MATMUL`.** Cholesky, LDL, LU, triangular
    solve, sparse LDL and the ragged ops become registered extension ops of `scaly.linalg`.
11. **`sets` ships with `scaly-control`** (the OCP terminal ingredients need it). **`nn`, `geometry`
    and `estimation` start in `scaly-experimental`.**

## 2. Target architecture

### 2.1 The core (`scaly-core`)

Owns: `ir`, `function`, `ad`, `passes`, `codegen` (`abi`, `c`, `aot`, `jit`, `toolchain`,
`__main__`), `utils` (`env`, `names`, `options` with AD policy only), and the new `scaly.ext`.

Contains no solver, adapter, linear-algebra or domain vocabulary. Target size ≈ 15k lines
(today's core directories are 16.8k on `devrush`; about 2.2k leave, about 0.5k of extension
machinery arrives).

**Core ExprOps after step 3.1 (61):**

| Group | Ops |
|---|---|
| leaves | `INPUT CONST` |
| unary | `NEG SIN COS TAN ASIN ACOS ATAN SINH COSH TANH ERF EXP LOG SQRT ABS FLOOR CEIL` |
| binary | `ADD SUB MUL DIV POW ATAN2 MINIMUM MAXIMUM COPYSIGN` |
| predicates | `LT LE EQ NE AND OR NOT ISFINITE SELECT CAST` |
| reductions | `SUM MAX MIN` |
| layout | `RESHAPE TRANSPOSE SLICE STACK CONCAT MATMUL` |
| indexing | `GATHER SCATTER SEGMENT_MAX SEGMENT_MIN INDEX_ADD INDEX_SET TAKE PUT PUT_ADD` (9 → 5 in optional step 9.1) |
| structure | `CALL VMAP SCAN WHILE EXTERN_CALL` |

Registered by `scaly.linalg`: `CHOLESKY LDL LU TRISOLVE SPARSE_LDL SPARSE_LDL_SOLVE RAGGED_ADD RAGGED_DOT`.

### 2.2 User-visible namespaces

| Namespace | Owns | Distribution |
|---|---|---|
| `scaly` | compiler; `function`, `vmap`, `scan`, `while_loop`, `custom_derivative`, derivatives, codegen | core |
| `scaly.ext` | op registry, method registry, `EXTERN_CALL` protocol, adapter registry, pass slots, option namespaces, library-author Function API | core |
| `scaly.linalg` | `SparseMatrix`/`S`, dense and sparse factorizations, `banded`, `stagewise` (Riccati), `custom_linear_solve`; linear-system methods | numerics |
| `scaly.roots` | `Root`, `LeastSquares`; Newton, safeguarded Newton–bisection, Gauss–Newton, LM; `custom_root` | numerics |
| `scaly.opt` | `problem`, `QP`/`NLP` forms, `solver()`; methods `ipm`, and the external-solver base used by plugins | numerics |
| `scaly.integrators` | tableaus, polynomial bases and quadrature, explicit/adaptive/symplectic/implicit schemes, ZOH/FOH, linearization, variational discretization | numerics |
| `scaly.interp` | axes, `BSpline`, fitting, shape-constrained fitting (through `scaly.opt`) | numerics |
| `scaly.ocp` | `ContinuousOCP`, `DiscreteOCP`, transcription, formulation, terminal ingredients, warm start, OCP methods | control |
| `scaly.sets` | `Polytope`, `Ellipsoid`, invariant sets | control |
| `scaly.viz` | graph views, render recording, browser | tools |
| `scaly.export` | C++ header, CasADi external functions, acados drop-in | tools |
| `scaly.nn` | MLP/activation builders, PyTorch state-dict loading | experimental |
| `scaly.geometry` | quaternions, SO(3)/SE(3), retraction and local coordinates | experimental |
| `scaly.estimation` | KF/EKF/UKF, MHE (incubating; starts when two examples share code) | experimental |
| `scaly.testing` | pytest plugin, conformance suites, hyper-dual numbers, reference helpers | testing |

Experimental **methods** of standard namespaces (e.g. `ocp.altro`, `ocp.scvx`) live at their natural
path (`src/scaly/ocp/altro.py`) and are assigned to `scaly-experimental` file by file in
`distributions.toml`. Promotion is a one-line change there; paths and names do not move.

### 2.3 Distributions

| Distribution | Content (from `distributions.toml`) | Depends on | Versioning |
|---|---|---|---|
| `scaly-core` | `scaly/__init__.py`, `ext.py`, `ir/`, `function/`, `ad/`, `passes/`, `codegen/`, `utils/` | numpy, scipy | lockstep |
| `scaly-numerics` | `linalg/`, `roots/`, `opt/` (minus experimental files), `integrators/`, `interp/` | core | lockstep |
| `scaly-control` | `ocp/` (minus experimental files), `sets/` | numerics | lockstep |
| `scaly-tools` | `viz/`, `export/` | core | lockstep |
| `scaly-experimental` | `nn/`, `geometry/`, `estimation/`, experimental method files | control, numerics | lockstep |
| `scaly-testing` | `testing/` | core, pytest | lockstep |
| `scaly` | metapackage: core + numerics + control + tools; extras `experimental`, `piqp`, `ipopt`, `sqp`, `solvers`, `all` | — | lockstep |
| `scaly-piqp`, `scaly-ipopt`, `scaly-sqp` (later `scaly-fatrop`, …) | `plugins/<name>/` | numerics (and control for OCP solvers) | independent |

### 2.4 Repository layout

```
alloy/
├── distributions.toml              single source of truth: distribution → paths, deps, versioning
├── pyproject.toml                  scaly-core; [tool.uv.workspace] members = ["packages/*", "plugins/*", "meta/*"]
├── src/scaly/
│   ├── __init__.py  ext.py  ir/  function/  ad/  passes/  codegen/  utils/          core
│   ├── linalg/  roots/  opt/  integrators/  interp/                                  numerics
│   ├── ocp/  sets/                                                                   control
│   ├── viz/  export/                                                                 tools
│   ├── nn/  geometry/  estimation/                                                   experimental
│   └── testing/                                                                      testing
├── packages/<dist>/pyproject.toml  thin manifests generated from distributions.toml (step 8.1)
├── meta/scaly/pyproject.toml       the metapackage
├── plugins/scaly-piqp|ipopt|sqp/   external solvers, own src/ and tests/ (unchanged location)
├── tests/
│   ├── core/  linalg/  roots/  opt/  integrators/  interp/  ocp/  sets/  viz/  export/  nn/ …
│   ├── conformance/                one suite per problem class, parametrized over the method registry
│   ├── integration/                cross-namespace tests; the example and notebook runner
│   ├── data/opt/maros_meszaros/  data/ocp/ …
│   └── baseline/<dist>_nodeids.txt  c/
├── examples/
│   ├── linalg/ roots/ opt/ integrators/ interp/ ocp/ sets/ nn/ …    per namespace; the docs gallery
│   └── case_studies/<name>/       drivers in git; results/ regenerated or in Git LFS
├── bench/                          scaly-bench (today's benchmarks/), never published
├── docs/  internal/
```

## 3. Interfaces

### 3.1 Problems, methods, solvers

Every numerical domain follows one pattern.

```python
# scaly/ext.py (sketch)
class Method(Protocol[P]):
  name: ClassVar[str]                    # registry key, "opt.ipm", "ocp.ilqr"
  problem: ClassVar[type]                # the problem class it solves
  api: ClassVar[int]                     # method-API version of that problem class it implements
  def supports(self, problem: P) -> Support: ...        # ok, or reasons (nonconvex, too large, needs bounds …)
  def build(self, problem: P, *, name: str) -> ConcreteFunction: ...

class MethodRegistry:                    # one instance per problem family, e.g. registry("opt")
  def get(self, name: str) -> type[Method]: ...          # lazy, through entry points
  def installed(self) -> dict[str, EntryPoint]: ...
  def auto(self, problem) -> Method: ...                  # first installed method whose supports() is ok, in a declared preference order
```

- **Entry points.** Every method, built-in or plugin, is declared in the entry-point group
  `scaly.methods` under `<domain>.<name>`. Built-ins declare theirs in their distribution's
  manifest, so the registry never special-cases them.
- **Domain entry point.** `sc.<domain>.solver(problem, method, *, name=None)`; `method` is a method
  instance, a registry name (`"piqp"`, resolved within the domain), or `"auto"`.
- **Method classes on the namespace.** `scaly.<domain>.__getattr__` resolves method classes lazily
  from the registry, so `sc.opt.PIQP` works when `scaly-piqp` is installed and gives an install hint
  otherwise. Methods are frozen dataclasses of their options, validated at construction.
- **Uniform built Function.** Per problem class the signature is fixed: inputs are parameters and a
  warm start; outputs are the solution (primal, dual where applicable) and an `Info` pytree. `Info`
  always has `status: Status` (a common enum in `scaly.ext`, grown from today's `ScalySolveStatus`),
  `iter`, and domain residuals. External methods write these as outputs too; the stats-struct
  accessor remains for timing only.
- **Derivatives are owned by the problem class**, attached with `custom_derivative` after `build`:
  the KKT system at the solution for `opt` (equality constraints and inequalities with strict
  complementarity; the active set is identified by tolerance), `F_z^{-1}` for `roots`, the
  stage-wise KKT system for `ocp`, the implicit stage equations for implicit integrators. A method
  may override (e.g. differentiate its unrolled iterations). Until a problem class has its rule,
  differentiating through it **raises**.
- **Composition.** Methods take methods: `sc.opt.SQP(qp=sc.opt.IPM(kkt=sc.linalg.SparseLDL()))`,
  `sc.ocp.Direct(sc.opt.IPOPT(), form="sparse")`,
  `sc.integrators.RadauIIA(3, newton=sc.roots.Newton(linear=sc.linalg.LU()))`.
- **Conformance.** A method is supported when it passes its problem class's suite in
  `scaly.testing` (§3.6).
- **Provenance.** Every built Function records method name, options, distribution version and, for
  external methods, the vendored upstream version, in `ConcreteFunction` metadata, in the header
  comment of generated C, and in the JIT cache key.

**Problem classes and methods at the end of this plan:**

| Domain | Problem classes | Generated methods | External methods |
|---|---|---|---|
| `linalg` | `LinearSystem` (dense, sparse, banded, stage-structured) | `LU`, `Cholesky`, `LDL`, `SparseLDL`, `Thomas`, `Riccati` | later: QDLDL, MA57 |
| `roots` | `Root`, `LeastSquares` | `Newton`, `NewtonBisection`, `GaussNewton`, `LevenbergMarquardt` | — |
| `opt` | `QP`, `NLP` | `IPM` | `PIQP`, `IPOPT`, `SQP` (plugins) |
| `integrators` | `ODE` | explicit, adaptive, symplectic, implicit RK, collocation (today's `TABLEAUS`) | — |
| `interp` | `Fit` | `BSpline`, `Linear`, `Smoothing`, `Constrained(qp=…)` (today's `KINDS`) | — |
| `ocp` | `DiscreteOCP` | `Direct(opt_method, form)`, `ILQR`, `TinyADMM`; experimental `ALTRO`, `SCvx` | later: Fatrop |

### 3.2 Extension API (`scaly.ext`)

Detail and footprints are in the 2026-09-28 report, §4–5. Summary of what step 2 delivers:

1. **Op registry.** `ExprOp` becomes a string-keyed registry of `OpDef` (arity, type inference,
   verify rule, NumPy fold, `jvp`, `jvp_many` defaulting to stacked single-seed JVPs, `vjp`,
   sparsity defaulting to dense, lowering rule, traits). Builtins register through it.
   Registering a name twice raises.
2. **Public lowering context**: buffer allocation, statement emission, fresh names (replacing
   `ctx._tmp`), output aliasing, and the loop helpers the linalg rules use (`_blocked_sum`,
   `_copy_loop`, `_lane_loops`).
3. **Op traits** replacing hard-coded sets: `elementwise`, `runtime_index`, `update` (with a
   positions callback, replacing `_Ragged`), `exact_reads`, `expensive`.
4. **Pass slots**: named anchors in the explicit pipeline (`insert_after("fuse_elementwise", p)`).
5. **Option namespaces**: `sc.options(linalg=dict(dense_unroll=8))`; each namespace declares
   whether it affects derivative-helper naming (`options_tag()` hashes only those).
6. **`EXTERN_CALL` protocol**: `dependencies()`, `extern_sources()`, `render(ctx)`,
   `build_requirements()`, optional state blob with a ctypes view (§4, step 1.1).
7. **Output adapters**: `register_adapter(name, header=, extra_source=, entry_prologue=, extra_workspace=)`.
8. **Library-author Function API**: public `from_exprs` (today `ConcreteFunction._from_exprs`, about
   40 uses), `lift`, template tokens, the `SymbolicValue` leaf protocol, a public library loader
   (today `codegen.jit._load_library`, used by the benchmarks).
9. **`EXT_API_VERSION`**, checked by every distribution and plugin at import.

### 3.3 Optimal control (`scaly.ocp`)

```python
cocp = sc.ocp.ContinuousOCP(ode=f, T=2.0, stage_cost=l, terminal_cost=Vf,
                            x_bounds=..., u_bounds=..., constraints=[...], terminal=Xf)
docp = sc.ocp.transcribe(cocp, sc.ocp.Collocation(degree=3, points="radau"), N=40)
# or directly: docp = sc.ocp.DiscreteOCP(step=F, N=40, stage_cost=..., terminal=Xf)

solve = sc.ocp.solver(docp, sc.ocp.Direct(sc.opt.PIQP(), form="sparse"))
solve = sc.ocp.solver(docp, sc.ocp.ILQR())               # same signature
traj, info = solve(x0, params, warm)
warm = sc.ocp.shift(docp)(traj)                          # receding horizon: a Function
```

- **`DiscreteOCP`** is the general multistage form: stage variables `x_k`, `u_k`, internal `w_k`;
  dynamics `x_{k+1} = F_k(x_k, u_k, w_k; p)`; stage equalities and inequalities; stage cost;
  terminal cost; terminal set; bounds; per-stage parameters (`varying`). Collocation internals are
  `w_k`. It carries its stage structure explicitly, which is what `linalg.stagewise`, structured
  IPM variants and external OCP solvers read.
- **`ContinuousOCP`** holds `ode`, horizon length, costs, path constraints, terminal ingredients.
  It is not solvable: `sc.ocp.solver` on it raises `TypeError` naming `transcribe`.
- **Transcription** (`MultipleShooting(integrator)`, `Collocation`, `Pseudospectral`) maps
  `ContinuousOCP → DiscreteOCP`, including the cost quadrature (`cost="points"|"integral"`).
- **Formulation** `sc.ocp.to_problem(docp, form="sparse"|"condensed") → (opt problem, layout)`.
  `Direct(method, form)` is formulation followed by any `opt` method.
- **Terminal ingredients** are OCP arguments; `sc.ocp.terminal` computes them: `lqr(A, B, Q, R)`,
  `max_invariant_set(...)`, `largest_ellipsoid(...)` returning `scaly.sets` objects;
  `TerminalEquality`.
- **Warm start**: `sc.ocp.shift(docp)` returns a Function mapping a trajectory (primal and dual)
  to the shifted warm start; its logic is today's `MPC._shifted`/`_shift_function`.
- **Removed**: `MPC`, `ClosedLoop`, `simulate`, `Solution`. Examples write the closed loop.
  `bench/` keeps its own closed-loop drivers.
- `mpc.ocp.linear(A, B)` becomes `scaly.integrators.linear.affine(A, B)` beside `zoh`/`foh`.

### 3.4 Solvers across the core boundary

- `SOLVER_CALL` becomes `EXTERN_CALL`. Core knows nothing about oracles, QP/NLP fields, stats or
  solver library paths.
- `scaly.opt.external` holds today's solver plumbing (`registry`, `paths`, `stats`, `graph`,
  `model`, `codegen/solver.py`) and a base class for external QP/NLP methods that implements the
  extern-callee protocol. The plugin protocol (`render_wrapper`, `lib_dir`, `include_dir`, …) is
  unchanged in spirit; `SOLVER_PLUGIN_PROTOCOL_VERSION` becomes the method-API version of
  `opt.QP`/`opt.NLP`, plus the extern-callee protocol version.
- The entry-point group `scaly.solvers` becomes `scaly.methods` with `opt.<name>` keys.

### 3.5 Where today's code goes

| Today (`devrush`) | Target | Step |
|---|---|---|
| `ir/expr.py` linalg builders, `SPARSE_LDL_TABLES`, validation | `linalg/ops/*.py` | 3.1 |
| linalg rules in `ir/expr_spec.py`, `ad/forward.py`, `ad/reverse.py`, `ad/sparsity.py`, `passes/lowering.py` | `linalg/ops/*.py` | 3.1 |
| `utils/options.py` `dense_unroll`, `sparse_unroll` | `linalg` option namespace | 2.3, 3.1 |
| `codegen/cpp.py`, `codegen/casadi.py` | `export/cpp.py`, `export/casadi.py` | 1.2, 6.2 |
| `codegen/solver.py`, `solvers/{registry,paths,stats,graph,model}.py` | `opt/external/` | 1.3, 4.2 |
| `solvers/{problem,_oracle,qp,nlp,solver}.py` | `opt/` | 4.2 |
| `solvers/ipm/*`, `examples/qp_solvers/generated_piqp.py` | `opt/ipm/` (method `opt.ipm`) | 4.3 |
| `integrators/implicit._Newton`, `interp/spline.Inverse` Newton loop | `roots/` (used by both) | 4.4 |
| `interp/fit.py` Thomas and cyclic sweeps | `linalg/banded.py` | 3.2 |
| Riccati copies (`examples/lqr_tuning.py`, TinyMPC cache, npmpc, diffmpc) | `linalg/stagewise.py` | 3.2, 6.1 |
| `integrators/transcription.py` | `ocp/transcription.py` | 5.2 |
| `mpc/ocp.py` | `ocp/problem.py`, `ocp/formulate.py` | 5.2 |
| `mpc/terminal.py` (`lqr`, `max_invariant_set`, `largest_ellipsoid`) | `ocp/terminal.py` | 5.3 |
| `mpc/polytope.py`, `mpc/terminal.Ellipsoid` | `sets/` | 5.1 |
| `mpc/controller.py` shift logic | `ocp/warmstart.py` | 5.3 |
| `mpc/controller.py` `MPC`, `ClosedLoop`, `simulate`, `Solution` | deleted; examples | 5.3 |
| `examples/tinympc/{solver,problem}.py` | `ocp/tinyadmm.py` | 6.1 |
| `examples/ilqr.py` | `ocp/ilqr.py` | 6.1 |
| `examples/case_studies/altro/al_ilqr.py` | `ocp/altro.py` (experimental) | 6.1 |
| `case_studies/scvx/scaly_impl.py` PTR loop; `segment_variational` | `ocp/scvx.py` (experimental); `integrators/variational.py` | 6.1 |
| `case_studies/diffmpc` implicit Riccati rule | derivative of `linalg.stagewise` | 6.1 |
| `case_studies/symforce` `gauss_newton`, `lm_function` | `roots/` | 6.1 |
| `case_studies/symforce` quaternions; `scvx/model.cross` | `geometry/` | 6.2 |
| `utils/torch_state_dict.py`; MLPs in `neural_mpc`, `benchmarks/problems/npmpc`, `unbumpercars/filters` | `nn/` | 6.2 |
| `neural_mpc` `acados_functions`, `install_dropin` | `export/acados.py` | 6.2 |
| `case_studies/fatrop_chain/fatrop_dropin.py` | later plugin `scaly-fatrop` (OCP method) | Open items |
| `scvx/model.tsit5`; `examples/casadi/_lgl.py` | `integrators/tableau.py`; `integrators/polynomial.py` | 6.1 |
| `benchmarks/harness/hyperdual.py`; root `conftest.py` marker logic | `testing/` | 7.1 |
| `benchmarks/` | `bench/` | 8.3 |
| duplicated `text_bytes`, `time_c`, `compile_*`, `machine()`, `parse_ipopt`, `compare.py` runners | `bench/` | 8.3 |

### 3.6 Tests and examples

- `tests/` mirrors `src/scaly/` by namespace; core tests move under `tests/core/`.
- **Conformance suites** live in `scaly.testing.conformance`, one per problem class (`QP` on
  Maros–Meszaros, `NLP` on a CUTEst subset, `DiscreteOCP` on the benchmark OCPs, `Root`, `Fit`,
  `LinearSystem`). `tests/conformance/` parametrizes each over every installed method. Plugins run
  the same suite in their own tests.
- `@pytest.mark.method("opt.piqp")` replaces `@pytest.mark.solver(...)`: skip when not installed,
  fail under `SCALY_REQUIRE_METHODS=1` (replacing `SCALY_REQUIRE_SOLVERS`).
- One node-ID baseline per distribution: `tests/baseline/<dist>_nodeids.txt`. C snapshots stay
  with the core.
- **Examples** live in `examples/<namespace>/` and are the docs gallery. Each script opens with a
  PEP 723 header listing distributions (`scaly`, `scaly-piqp>=0.6`, …); notebooks carry the same
  list in notebook metadata. Examples import only public names (a lint test rejects `sys.path`
  edits and underscore imports from `scaly`). The integration runner executes every example and
  notebook, skipping those whose requirements are not installed. An example is owned by the
  highest-tier namespace it imports, which selects its CI job.
- **Case studies** keep drivers in git; large `results/` files are regenerated or moved to Git LFS
  (Open items).
- **Layering, two levels.** `tests/test_import_layering.py` keeps its per-module table inside the
  core and gains a distribution table read from `distributions.toml`: core imports no other
  distribution's paths; each distribution imports only what it declares; the graph is acyclic.

### 3.7 Versioning and release

- **Lockstep** for `scaly-core`, `scaly-numerics`, `scaly-control`, `scaly-tools`,
  `scaly-experimental`, `scaly-testing`, `scaly`: one tag `vX.Y.Z` gives each distribution version
  `X.Y.Z`; first-party inter-dependencies are pinned `==X.Y.Z`. `scaly-experimental` makes no
  stability promise and its modules warn on import.
- **Independent** for external-solver plugins. Each declares its supported `scaly-numerics` range
  (`>=X.Y,<X.(Y+1)` pre-1.0), the method-API version it implements (checked at registration), and
  its vendored upstream version (metadata and changelog). The metapackage offers plugins as extras
  with ranges, not pins.
- **Method-API versions** are per problem class (`opt.QP` API 1, `opt.NLP` API 1, `ocp.DiscreteOCP`
  API 1, …), declared by the owning namespace as the range it accepts.
- `scripts/release.py X.Y.Z` rewrites versions and pins in all lockstep manifests from
  `distributions.toml`, tags, builds every wheel, and runs the release job (§4, step 8.3) before
  upload.
- `docs/dev/versioning.md` is rewritten accordingly in step 8.3 (it currently says core and plugins
  version independently, which stays true for plugins).

## 4. Steps

Each step lists **Goal**, **Changes**, **Gate**. Gates add to the common gate in "How to execute".

### Phase 0: safety and housekeeping

**0.1 Housekeeping.**
Changes: delete untracked `src/scaly/ir/.ipynb_checkpoints/`, `plugins/alloy-*`, `mymodule.py`,
`pla/`, `generated/` (ask before deleting anything tracked); add `integrators/` and `mpc/` to the
package map in `docs/dev/codebase.md`; drop `typing_playground` from `testpaths` and move it to
`internal/typing_playground/`; resolve the uncommitted `tests/integration/test_examples_gallery.py`
change, which depends on the untracked `examples/tiny_qp.py` (commit both or neither).
Gate: common gate; `git status` clean apart from intended changes.
Log: done 2026-09-28. Scratch files moved to the Trash; `tiny_qp.py` rewritten as a gallery example
(`main()`, closed form and KKT check) and committed with its test; `typing_playground` moved, its
checks documented in its README; `ty` ratchet recorded (Open items).

**0.2 Extern calls raise on derivatives.**
Changes: in `ad/forward.py`, `ad/reverse.py`, a seeded `SOLVER_CALL` raises
`NotImplementedError` naming `custom_derivative`; `ad/sparsity.py` returns a dense mask instead of
an empty one. Tests: differentiating through a solver raises; `sparse_jacobian` of a graph
containing a solver has the dense pattern in the solver's outputs.
Gate: common gate; `tests/ad/test_zero_tangent_products.py` updated deliberately, not weakened.
Log: done 2026-09-28. `jvp`, `jvp_many` and `vjp` raise at a seeded `SOLVER_CALL`; its mask is dense in the
columns its arguments read. The QP builder now refuses a nested solver before building oracles.
`test_zero_tangent_products.py` needed no change. Smoke medians within noise.

### Phase 1: invert the solver and adapter edges (still one distribution)

**1.1 `EXTERN_CALL` and the extern-callee protocol.**
Changes: rename the op; its attribute implements `dependencies()`, `extern_sources()`,
`render(ctx)`, `build_requirements()`, optional state blob. `passes/lowering.py` (today lines
134–170 and 1404) uses the protocol instead of `solvers.graph`/`ExternalOracle`;
`passes/program/_common.py` and `pack_workspace.py` read `extern_deps`/`extern_workspace` instead
of `solver_oracles`/`solver_external_workspace`; `function/sugar.py` copies an `extern` attribute
generically instead of `.descriptor`; `function/model.py` drops the `SolverStats` import. The
solver package implements the protocol for its descriptors.
Gate: C snapshots byte-identical; solver-marked tests pass; a test with a fake extern callee in
`tests/core/` compiles, links an extra C source and runs.
Log: done 2026-09-28. Protocol in `function/extern.py` (`ExternCallee`, `extern_function`,
`extern_functions`, `BuildRequirements` with a shared link resolver so flags stay identical);
`SolverDescriptor` implements it via `solvers/wrapper.py`, which replaces `codegen/solver.py`.
`fun.descriptor` is now `fun.extern` (read with `solver_descriptor`); plugin protocol v8. aot and
the JIT's stats read went through the protocol here already, leaving 1.3 the toolchain, env and
import-guard parts. The fake-callee test is `tests/function/test_extern.py` (tests mirror `src/`
until 7.1).

**1.2 Output-adapter registry.**
Changes: `register_adapter` in `codegen/aot.py`; `codegen/cpp.py` and `codegen/casadi.py` register
themselves; remove `lang`/`casadi` fields and branches from `CModule`, `_render_header`,
`_render_source`, `_render_solver_entry`; `codegen/c.py` no longer imports `casadi` (entry-prologue
hook); CLI `--lang`/`--casadi` become `--adapter NAME` (update docs and examples).
Gate: C, C++ and CasADi snapshots byte-identical; `tests/interp/test_casadi_parity.py` passes.

**1.3 Core stops importing solvers.**
Changes: `codegen/aot.py`, `jit.py`, `toolchain.py` aggregate build requirements and state blobs
through the extern-callee protocol; `CompiledFunction.solver_stats` becomes
`callee_state(name)` (with a `solver_stats` helper in the solver package); toolchain diagnostics
take report sections from extensions; `SCALY_SOLVER_*` environment handling moves out of
`utils/env.py`.
Gate: a new test asserts no module in `ir`, `ad`, `function`, `passes`, `codegen`, `utils` imports
`scaly.solvers`, `scaly.codegen.cpp` or `scaly.codegen.casadi`, including function-local imports.

### Phase 2: open the vocabulary

**2.1 Op registry.** `OpDef`, `register_op`, `ExprOp` names resolved through the registry; all
builtins registered through it; `Expr` interning keyed by the registered name.
Gate: C snapshots byte-identical; `tests/test_import_boundaries.py` updated; interning identity
tests pass.

**2.2 Table-driven rules.** Convert the if-chains in `_jvp`, `_jvp_many_structural`, `_local_vjp`,
`_jac_mask_uncached`, `passes/expr._evaluate` and the verify table into lookups on `OpDef`; keep
`CALL`, `VMAP`, `SCAN`, `WHILE` as core special cases.
Gate: byte-identical snapshots; benchmark smoke medians within noise; an out-of-tree toy op defined
in a test differentiates (forward, reverse, multi-seed), reports sparsity, lowers and compiles.

**2.3 Lowering context, traits, option namespaces, pass slots.** As §3.2 items 2–5. Replace
`_UPDATE_OPS`, `_EXACT_READS`, `RUNTIME_INDEX_OPS`, `_EXPENSIVE_OPS`, `_Ragged` with traits;
`viz/graph.py` colours by trait.
Gate: byte-identical snapshots; changing a `linalg` option no longer renames AD helpers (test).

**2.4 `scaly.ext`.** Public module collecting §3.2; `from_exprs`, `lift`, tokens, public loader;
`EXT_API_VERSION`; extension versions in the JIT cache key (bump `_JIT_CACHE_VERSION`). Replace
the private uses in `linalg`, `interp`, `integrators`, `mpc`, `solvers`, examples and benchmarks.
Gate: grep finds no `_from_exprs`, `_lift`, `_tokens`, `_load_library` outside the core.

### Phase 3: linear algebra out of the core

**3.1 Linear-algebra ops move into `scaly.linalg`.**
Changes: `linalg/ops/{dense,trisolve,sparse_ldl,ragged}.py` register the eight ops with all rules;
remove them from `ir/expr.py`, `ir/expr_spec.py`, `ad/*`, `passes/lowering.py`; the AD messages
naming `linalg.solve`/`SparseLDL` move with them; `tests/linalg` absorbs
`tests/integration/test_ragged*.py` and the linalg parts of `test_lowering`,
`test_verifier_edge_cases`, `test_sparsity`.
Gate: byte-identical snapshots with `scaly.linalg` imported; the core test directory passes in a
process where importing `scaly.linalg` is blocked (a conftest fixture); IPM speed benchmark
unchanged.

**3.2 `linalg.banded` and `linalg.stagewise`.**
Changes: move the Thomas and cyclic sweeps out of `interp/fit.py`; add a stage-wise (Riccati)
factorization for block-tridiagonal KKT systems with its implicit derivative; unify
`ipm.QPStructure`'s COO normalizer with `SparseMatrix`.
Gate: `interp` results unchanged; `stagewise` differential tests against dense solves and against
the TinyMPC Riccati cache.

### Phase 4: the method interface

**4.1 Method registry, `Info`, `Status`.** `scaly.ext.Method`, `MethodRegistry`, `Support`,
`Status`, `Info`; entry-point group `scaly.methods`; lazy method classes on namespaces.
Gate: registry unit tests with fake methods; missing-method errors name the distribution to install.

**4.2 `scaly.opt`.**
Changes: `solvers/` becomes `opt/` (§3.5); `sc.problem`/`sc.solver`/`qp_problem` become
`sc.opt.problem`/`sc.opt.solver`/`sc.opt.QP`; public normal forms `extract_qp`, `nlp_oracles`
(today private `_prove_quadratic`, `_qp_data`, `_lowered`, …); `opt.external` base class; plugins
move to `scaly.methods` entry points and emit `status`/`iter` as outputs; `interp/constrained.py`
and `mpc` updated. Rewrite `docs/dev/solver_plugins.md` and `docs/guide/solvers.md`.
Gate: plugin tests pass; no import of an underscore name from `scaly.opt` outside it.

**4.3 IPM as `opt.ipm`.**
Changes: `solvers/ipm/` becomes `opt/ipm/`; `examples/qp_solvers/generated_piqp.py` becomes the
method's `build`; `sc.opt.solver(qp, sc.opt.IPM())` returns the same signature as PIQP; PIQP status
codes mapped onto `Status`; case studies (`scvx`, `embedded_qp`) and
`tests/integration/test_case_study_*` switch to it and stop editing `sys.path`.
Gate: `tests/opt/ipm` (moved from `tests/solvers/ipm`) including PIQP decision traces; QP
conformance passes for both `opt.ipm` and `opt.piqp`.

**4.4 `scaly.roots`.** Problem classes `Root`, `LeastSquares`; methods `Newton`,
`NewtonBisection`, `GaussNewton`, `LevenbergMarquardt`; `custom_root`. `integrators/implicit.py`
and `interp/spline.Inverse` use them; four gallery examples switch from hand-written Newton loops.
Gate: integrator and interp results unchanged to recorded tolerances.

**4.5 `integrators` and `interp` as registries.** `TABLEAUS` and `KINDS` become method registries;
`tsit5` and LGL added.
Gate: integrator order-condition tests; interp suite.

### Phase 5: control

**5.1 `scaly.sets`.** Move `Polytope` and `Ellipsoid`; constraints returned as `(expr, lo, hi)`.

**5.2 `scaly.ocp` problems, transcription, formulation.** `ContinuousOCP`, `DiscreteOCP`,
`transcribe`, `to_problem` as in §3.3; today's `OCP` arguments map onto these (the `ode=`/`step=`
split becomes the two classes, `condensed=` becomes `form=`).
Gate: every `tests/mpc` OCP case reproduced through `transcribe` + `to_problem` with identical
optimal values; stage-structure metadata tested.

**5.3 OCP methods, warm start, terminal ingredients; `mpc` removed.** `sc.ocp.solver`,
`Direct`, `shift`, `ocp.terminal`; delete `src/scaly/mpc/`; rewrite `examples/mpc/*` notebooks
into `examples/ocp/` with explicit closed loops; rewrite `docs/guide/mpc.md` as `ocp.md`.
Gate: notebooks run; closed-loop results match the recorded ones.

### Phase 6: code from the examples

**6.1** Promote per §3.5: `ILQR`, `TinyADMM` (standard); `ALTRO`, `SCvx` (experimental);
`integrators/variational.py`; `tsit5`/LGL; Gauss–Newton/LM into `roots`; diffmpc rule into
`stagewise`. Replace hand-written RK4s with `integrators.rk4`.
Gate: DiscreteOCP conformance for each OCP method; case-study numbers reproduce within recorded
tolerances.

**6.2** `scaly.nn`, `scaly.geometry`, `scaly.export` (C++, CasADi, acados) populated per §3.5;
duplicated MLP code removed from examples, benchmarks and tests.

### Phase 7: tests and examples

**7.1** `scaly.testing` with the pytest plugin, `method` marker, conformance suites, hyper-dual
numbers; tests reorganized per §3.6.
**7.2** Examples per namespace with PEP 723 headers; public-API lint test; runner skips on missing
requirements.

### Phase 8: distributions

**8.1** `distributions.toml`; generate `packages/*/pyproject.toml` (Hatch `force-include` of the
listed paths into the wheel; a small build hook so an sdist rebuild finds the same paths);
`meta/scaly/pyproject.toml`; rename the root project to `scaly-core`.
Gate: every file under `src/scaly/` lands in exactly one wheel (test); `uv sync --all-packages`
works.

**8.2** Core `scaly/__init__.py` gets `pkgutil.extend_path` and a lazy `__getattr__` over the
known namespaces with install hints; each domain `__init__` resolves method classes lazily.
Gate: in a clean venv from built wheels, `import scaly as sc; from scaly import linalg as la;
sc.linalg is la`; `sc.ocp` without `scaly-control` raises an `AttributeError` naming it; `ty` and
`pyright` resolve `scaly.linalg` both in the workspace and in the wheel venv.

**8.3** CI jobs: workspace at HEAD; one isolation job per distribution (install its wheel with only
declared dependencies, run its tests); plugins against HEAD and against their oldest and newest
supported `scaly`; release job (all wheels, `scaly[experimental,solvers]` in a clean env,
conformance and examples). `scripts/release.py`. `benchmarks/` renamed `bench/`. Rewrite
`docs/dev/versioning.md`, `docs/dev/codebase.md`, `AGENTS.md` commands.

### Phase 9: optional

**9.1 Indexing consolidation.** `SCATTER`, `SEGMENT_MAX`, `SEGMENT_MIN` become one
`SEGMENT_REDUCE{add,max,min}`; `INDEX_ADD`/`INDEX_SET` become `PUT_ADD`/`PUT` with constant
indices. Snapshots may change only where these ops appear; differential tests against NumPy.

## 5. Traps

- **Interning.** Two registrations of one op name, or an IR class reachable under two module paths,
  silently break structural identity (AD, CSE, passes). The registry raises on duplicates; nothing
  aliases modules.
- **Registration order.** An op must be registered before an `Expr` using it is built; this holds
  because building needs the builder, which lives in the registering module. Never register ops
  from `scaly/__init__.py`.
- **Benchmarks as the only coverage.** Before moving or deleting example or benchmark code that
  exercises an IR, AD or codegen path, copy a small reproduction into `tests/` (`AGENTS.md`).
- **Vendored solver hooks.** Do not touch `plugins/*/hatch_build.py` logic beyond entry-point and
  import renames; read the comments there first. A cold rebuild takes 5–8 minutes.
- **Foxglove layouts.** Never edit `benchmarks/problems/*/foxglove-layout.json` while moving
  `benchmarks/` to `bench/`; move the files unchanged.
- **Generated C stability.** Exported symbols and `{kind}_{of}_{wrt}` names are a published promise
  (`docs/dev/versioning.md`). Phases 1–3 must not change them; bump `_JIT_CACHE_VERSION` whenever
  generated output changes.

## 6. Open items

- Where `internal/` reports and perf data live long term (571k lines in history): separate notes
  repository or Git LFS.
- Case-study `results/` (about 152k lines of JSON): regenerate on demand, or Git LFS.
- Fatrop as an external OCP method (`scaly-fatrop` plugin): after step 5.3, when `DiscreteOCP`
  API 1 is fixed.
- Derivatives for `opt` problems with degenerate complementarity: behaviour to define (raise, or a
  generalized derivative).
- Whether `roots`, `opt` and `ocp` share one `Info` base with domain subclasses, or one `Info` per
  domain (decide in step 4.1).
- **`ty check` was red on `devrush` before step 0.1** (258 diagnostics, none in `src/` or
  `plugins/`). Step 0.1 fixed the five in `tests/` and the eight in plain example scripts, and
  excluded `examples/case_studies/*/baseline/**` (they run in the baselines' own environments). The
  remaining 198 are in notebooks and case-study drivers, almost all sibling imports through
  `sys.path` (`plotstyle`, `scaly_impl`, ...), which step 7.2 removes. Until then the ty part of the
  common gate is a ratchet: no diagnostic outside the list recorded at step 0.1. Decide whether to
  fix them earlier.
