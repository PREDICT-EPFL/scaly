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
  lowering also run `uv run bench/run.py smoke` and compare medians against the previous step.
- Pre-1.0 breaks are unshimmed (`docs/dev/versioning.md`). When a public name moves, update every
  caller, example, notebook and doc page in the same step; do not leave aliases.
- When a step reveals that this plan is wrong, stop, write the finding under **Open items**, and ask.

### Status

| Step | Title | Status |
|---|---|---|
| 0.1 | Housekeeping | ☑ |
| 0.2 | Extern calls raise on derivatives | ☑ |
| 1.1 | `EXTERN_CALL` and the extern-callee protocol | ☑ |
| 1.2 | Output-adapter registry (C++, CasADi) | ☑ |
| 1.3 | Core stops importing solvers | ☑ |
| 2.1 | Op registry, builtins registered through it | ☑ |
| 2.2 | Table-driven AD, sparsity, folding, verification | ☑ |
| 2.3 | Public lowering context, op traits, option namespaces, pass slots | ☑ |
| 2.4 | `scaly.ext`: library-author Function API, extension versions in the JIT cache key | ☑ |
| 3.1 | Linear-algebra ops move into `scaly.linalg` | ☑ |
| 3.2 | `linalg.banded` and `linalg.stagewise` | ☑ |
| 4.1 | Method registry and the common `Info`/`Status` | ☑ |
| 4.2 | `scaly.opt`: problems, `solver()`, external methods | ☑ |
| 4.3 | IPM as the method `opt.ipm` | ☑ |
| 4.4 | `scaly.roots` | ☑ |
| 4.5 | `integrators` and `interp` as method registries | ☑ |
| 5.1 | `scaly.sets` | ☑ |
| 5.2 | `scaly.ocp`: continuous and discrete OCPs, transcription, formulation | ☑ |
| 5.3 | OCP methods, warm start, terminal ingredients; `mpc` removed | ☑ |
| 6.1 | Library code promoted from examples | ☑ |
| 6.2 | `nn`, `geometry`, `export` namespaces | ☑ |
| 7.1 | Conformance suites and the `method` marker | ☑ |
| 7.2 | Examples: per-namespace folders, PEP 723 headers, public API only | ☑ |
| 8.1 | `distributions.toml`, manifests, metapackage | ☑ |
| 8.2 | Namespace import mechanics (`extend_path`, lazy attributes) | ☑ |
| 8.3 | CI: isolation jobs, plugin jobs, release job, release script | ☑ |
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
- **Uniform built Function.** Per problem class the signature is fixed. For the domains that solve
  (`opt`, `roots`, `ocp`) inputs are parameters and a warm start; outputs are the solution (primal,
  dual where applicable) and an `Info` pytree. `Info` always has `status: Status` (a common enum in
  `scaly.ext`, grown from today's `ScalySolveStatus`), `iter`, and domain residuals. External methods
  write these as outputs too; the stats-struct accessor remains for timing only. The other domains
  fix their own result (decided at step 4.5): an integrator method builds the discrete map
  `F(x, ...) -> xnext`, an interp method a `BSpline`, neither with a warm start or an `Info`; one
  method class per named method, today's functions (`rk4`, `implicit`, `interpolant`, ...) kept as
  shorthand.
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
| `integrators` | `ODE` | one per named method: `RK4`, `Tsit5`, `DOPRI5`, ..., `RadauIIA(s, newton=…)`, `SDIRK3`, ..., `Adaptive(pair)`, `StormerVerlet` (today's `TABLEAUS`) | — |
| `interp` | `Fit` | one per kind: `Linear`, `Cubic`, `PCHIP`, ..., `PerAxis`, `Smoothing`, `Constrained(qp=…)` (today's `KINDS`) | — |
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
Log: done 2026-09-28. Registry and hooks in `codegen/adapter.py`, names resolved through a
`scaly.adapters` entry-point group so `aot` and `c` import neither adapter. API `adapters=("cpp",
"casadi")`, CLI `--adapter NAME` (repeatable: a C++ header with the CasADi layer is a real
combination). There were no C++ or CasADi snapshots; four adapter variants joined the corpus,
rendered from the pre-1.2 code, and match. 1.1's smoke medians were within noise.

**1.3 Core stops importing solvers.**
Changes: `codegen/aot.py`, `jit.py`, `toolchain.py` aggregate build requirements and state blobs
through the extern-callee protocol; `CompiledFunction.solver_stats` becomes
`callee_state(name)` (with a `solver_stats` helper in the solver package); toolchain diagnostics
take report sections from extensions; `SCALY_SOLVER_*` environment handling moves out of
`utils/env.py`.
Gate: a new test asserts no module in `ir`, `ad`, `function`, `passes`, `codegen`, `utils` imports
`scaly.solvers`, `scaly.codegen.cpp` or `scaly.codegen.casadi`, including function-local imports.
Log: done 2026-09-28. `Function.solver_stats()` became the generic `callee_state()`, with
`sc.solver_stats(fn, name=None)` in the solver package (126 call sites moved). The toolchain report
and the solver env vars come from `scaly.toolchain_report` and `scaly.env_vars` entry points. The
guard (`test_core_names_no_solver_or_adapter`) counts `TYPE_CHECKING` imports too, and was shown to
fail on a function-local import.

### Phase 2: open the vocabulary

**2.1 Op registry.** `OpDef`, `register_op`, `ExprOp` names resolved through the registry; all
builtins registered through it; `Expr` interning keyed by the registered name.
Gate: C snapshots byte-identical; `tests/test_import_boundaries.py` updated; interning identity
tests pass.
Log: done 2026-09-28. `OpDef` records in a name-keyed registry (`register_op`, `op_def`,
`registered_ops`) in `ir/expr.py`; builtins register through it from a table, and `ExprOp` stays
as the `StrEnum` of builtin names (a member hashes and compares as its string, so an `Expr.op` is
always a plain registered name). `OP_INFO`/`OpInfo` retired. Registry rules (`jvp`, `lower`, ...)
arrive with 2.2.

**2.2 Table-driven rules.** Convert the if-chains in `_jvp`, `_jvp_many_structural`, `_local_vjp`,
`_jac_mask_uncached`, `passes/expr._evaluate` and the verify table into lookups on `OpDef`; keep
`CALL`, `VMAP`, `SCAN`, `WHILE` as core special cases.
Gate: byte-identical snapshots; benchmark smoke medians within noise; an out-of-tree toy op defined
in a test differentiates (forward, reverse, multi-seed), reports sparsity, lowers and compiles.
Log: done 2026-09-28. `OpDef` carries `jvp`, `jvp_many`, `vjp`, `sparsity`, `fold`, `verify` and
`lower`, set by `register_op(..., **rules)` or `define_rules` (a second definition raises). Each
chain became a table of rule functions in the module that owns the kind (generated from the
branches, then reviewed); `CALL`, `VMAP`, `SCAN`, `WHILE`, `INPUT`, `CONST` and predicates stay
special. Defaults: `jvp_many` stacks per-seed `jvp` (builtins without one keep the whole-graph
fallback, marked explicitly so output is unchanged), sparsity is dense in what the arguments read,
`fold` applies `numpy`. `spec_expr` reads per-op verify rules from the registry. The toy op test is
`tests/ir/test_op_registry.py`; AD build time unchanged.

**2.3 Lowering context, traits, option namespaces, pass slots.** As §3.2 items 2–5. Replace
`_UPDATE_OPS`, `_EXACT_READS`, `RUNTIME_INDEX_OPS`, `_EXPENSIVE_OPS`, `_Ragged` with traits;
`viz/graph.py` colours by trait.
Gate: byte-identical snapshots; changing a `linalg` option no longer renames AD helpers (test).
Log: done 2026-09-28. `LowerCtx` has a documented public part (`emit`, `fresh_id`, `fresh_name`,
`bind`, `copy_loop`, `blocked_sum`, `lane_loops` beside the buffer methods), so a rule outside
`passes/` needs only `ctx` and `ir.program`. Traits on `OpDef` (`elementwise` naming its program
op, which replaces `_UNARY`/`_BINARY`; `expensive`; `runtime_index`; `exact_reads`; `update` and
`reads` position callbacks, `_Ragged` behind a `PositionRanges` protocol). Option namespaces with
`affects_derivatives`; `linalg` holds `dense_unroll`/`sparse_unroll` (declared in core until 3.1)
and derivatives now keep each factorization's unroll choice, so a `linalg` option cannot change a
derivative. Pass slots `insert_after`/`insert_before`. 2.2's smoke ran on a loaded machine: every
runtime, CasADi's too, +45%, the ratios unchanged.

**2.4 `scaly.ext`.** Public module collecting §3.2; `from_exprs`, `lift`, tokens, public loader;
`EXT_API_VERSION`; extension versions in the JIT cache key (bump `_JIT_CACHE_VERSION`). Replace
the private uses in `linalg`, `interp`, `integrators`, `mpc`, `solvers`, examples and benchmarks.
Gate: grep finds no `_from_exprs`, `_lift`, `_tokens`, `_load_library` outside the core.
Log: done 2026-09-28. `scaly/ext.py` collects the registries, protocols and the public
`Function.from_exprs`, `Function.lift`, `ConcreteFunction.tokens`, `codegen.jit.load_library`
(renamed everywhere, 737 call sites). `EXT_API_VERSION` lives in `utils/ext_api.py` so the JIT key
can include it; each plugin calls `require_ext_api` at import; `BuildRequirements.versions` puts the
linked plugin distributions in the key (`_JIT_CACHE_VERSION` 5). The grep gate is a test that scans
`src`, tests, plugins, examples and benchmarks. 2.3's smoke, rerun isolated with the worktree's own
plugins, matched 2.2's.

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
Log: done 2026-09-28. `linalg/ops/{dense,trisolve,sparse_ldl,ragged}.py` register the eight ops
under their old names with every rule, trait and verify rule; `ExprOp`, the rule tables and
`passes/lowering.py` no longer name them, and `linalg/options.py` declares the `linalg` namespace,
which `sc.options(linalg=...)` now loads on first use. The op modules take only public compiler
names (test), so `scaly.ext` gained what their rules share: `is_zero_const`, `zeros_many`,
`JVPManyUnsupported`, the mask helpers (`incidence`, `mask_compose`, `mask_or`, `empty_mask`) and
`NoAdjoint`, a `vjp` rule's answer for an argument it cannot differentiate, which replaces the
reverse sweep's `sparse_ldl_solve` special case. `import scaly` no longer loads `linalg`,
`integrators`, `interp` or `mpc` (`sc.<name>` loads one; test), `sc.S`/`sc.SparseMatrix` became
`sc.linalg.S`/`sc.linalg.SparseMatrix`, and the core-naming guard (renamed
`test_core_names_no_solver_adapter_or_package_built_on_it`) now covers those packages. The root
conftest blocks the packages in `SCALY_BLOCK_IMPORTS`; the core directories pass with
`scaly.linalg` blocked (command in `contributing.md`) once their few linalg uses moved to
`tests/linalg` (verifier corpus, options, sparse leaves, ragged) or became core-only (a Newton step,
a refusing producer, a generated-name clash). `test_sparsity` had no linalg part. Generated C for 192
linalg functions and derivatives and for the IPM harness's quick set (36 solvers) is byte-identical
to 2.4's, so the IPM's speed is unchanged by construction; the harness's `gen.py` had missed 2.4's
rename and was fixed. Smoke matched 2.4's medians and
line counts.

**3.2 `linalg.banded` and `linalg.stagewise`.**
Changes: move the Thomas and cyclic sweeps out of `interp/fit.py`; add a stage-wise (Riccati)
factorization for block-tridiagonal KKT systems with its implicit derivative; unify
`ipm.QPStructure`'s COO normalizer with `SparseMatrix`.
Gate: `interp` results unchanged; `stagewise` differential tests against dense solves and against
the TinyMPC Riccati cache.
Log: done 2026-09-28. `linalg/banded.py` holds the Thomas and cyclic sweeps, public as
`solve_tridiagonal` and `solve_cyclic_tridiagonal` (scan bodies renamed `linalg_thomas_*`); Expr-data
cubic fits past `DENSE_FIT`, every boundary, give bit-identical values and gradients to 3.1's.
`linalg/stagewise.py`: `Riccati(A, B, Q, R, QN, S=, N=)` factors a stage-structured LQ problem's KKT
system by a backward `scan` (each matrix shared or per stage), `solve(x0, q, r, c, qN)` returns
states, controls and multipliers, and the implicit rules, two levels deep as `SparseLDL`'s, make a
tangent or cotangent one more solve with the same factorization plus per-stage outer products
(summed for a shared matrix). Tested against the dense KKT solve, complex-step Jacobians of it in
both modes, Hessians against differences, a NumPy recursion, and the TinyMPC cache in three
scenarios; the examples' Riccati copies move onto it in 6.1. `QPStructure.from_patterns` and
`SparseMatrix.symbol` share `linalg.sparse.csc_coordinates`, so a QP pattern may take any form a
`SparseMatrix` reads, and `generated_piqp.py` lost its own converter; the IPM harness's 36 quick
solvers are byte-identical.

### Phase 4: the method interface

**4.1 Method registry, `Info`, `Status`.** `scaly.ext.Method`, `MethodRegistry`, `Support`,
`Status`, `Info`; entry-point group `scaly.methods`; lazy method classes on namespaces.
Gate: registry unit tests with fake methods; missing-method errors name the distribution to install.
Log: done 2026-09-28. `function/method.py` (collected in `scaly.ext`): the `Method` protocol,
`Support`, `MethodRegistry` over the `scaly.methods` group (entries `<domain>.<name>`, loaded and
checked on first use: the entry name, and `api` against the problem class's `method_api`; `auto`
tries `preference` first and skips a broken install with a warning; `resolve` takes an instance, a
name or `"auto"`), `registry(domain, hints=, preference=)`, and `attribute(module)`, the lazy
`__getattr__` a domain installs for `sc.<domain>.<Class>`. The core names no domain or method: a
domain passes `MethodHint(name, cls, distribution)`s, which is how a missing method names what to
install. `Status` replaces `ScalySolveStatus` (same codes, the stats ABI; `sc.Status`), and
`solvers.stats` moved to layer 5, since nothing below imports it since 1.3. **Decided (Open items):
one `Info` base, `status` and `iter`, with a subclass per domain for its residuals**, so conformance
suites and tooling read `info.status` whatever the domain; it is a Function output through the new
`Record` tree (a dataclass, rebuilt on both sides of a call). A numerical call returns `Info`'s
`int64` fields as `float64`, as the C entry returns every output.

**4.2 `scaly.opt`.**
Changes: `solvers/` becomes `opt/` (§3.5); `sc.problem`/`sc.solver`/`qp_problem` become
`sc.opt.problem`/`sc.opt.solver`/`sc.opt.QP`; public normal forms `extract_qp`, `nlp_oracles`
(today private `_prove_quadratic`, `_qp_data`, `_lowered`, …); `opt.external` base class; plugins
move to `scaly.methods` entry points and emit `status`/`iter` as outputs; `interp/constrained.py`
and `mpc` updated. Rewrite `docs/dev/solver_plugins.md` and `docs/guide/solvers.md`.
Gate: plugin tests pass; no import of an underscore name from `scaly.opt` outside it.
Log: done 2026-09-29. `solvers/` is `opt/` (problems, the normal forms, `solver`) and `opt/external/`
(descriptor, wrapper frame, stats, paths, graph queries, and `External`, the base of every plugin's
method class); `solvers/` keeps only `ipm` for 4.3. `sc.problem`, `sc.solver`, `sc.qp_problem`,
`sc.Problem` and the rest of the solver names left the top level for `sc.opt` (`sc.opt.NLP`, and
`sc.opt.QP(n, n_eq, n_ineq)`, a subclass, for the matrix form); about 700 callers, 97 files by
codemod. The plugins are method classes (`PIQP(sparse=, options=)`, `IPOPT(options=)`,
`SQP(options=)`, options checked when the method is made) in `scaly.methods` under `opt.<name>`;
`scaly.solvers` and the backend objects are gone, and `METHOD_API` 9 continues the plugin protocol.
`sc.opt.solver(problem, method="auto")` takes an instance, a name or `"auto"` (PIQP, IPOPT, SQP in
that order, a QP method's `supports` proving the problem quadratic). Every solver returns an
`opt.Info` (status, iter, objective, primal residual) after its solution: core's frame defines the
exported raw function around the plugin's `<raw>_solve` and copies them from the stats, so no
plugin wrapper changed; the solver C snapshot moved by exactly that frame. `extract_qp` (`QPForm`,
with `patterns`/`in_pattern`, and `matrix_pattern`) and `nlp_oracles` (`NLPOracles`) are the public
normal forms; `generated_piqp.py` uses them and renders byte-identical C, and a test refuses an
underscore import from `scaly.opt` outside it. Four notebooks with uncommitted edits by the user were
rewritten in both the working copy and the index, their edits left unstaged.

**4.3 IPM as `opt.ipm`.**
Changes: `solvers/ipm/` becomes `opt/ipm/`; `examples/qp_solvers/generated_piqp.py` becomes the
method's `build`; `sc.opt.solver(qp, sc.opt.IPM())` returns the same signature as PIQP; PIQP status
codes mapped onto `Status`; case studies (`scvx`, `embedded_qp`) and
`tests/integration/test_case_study_*` switch to it and stop editing `sys.path`.
Gate: `tests/opt/ipm` (moved from `tests/solvers/ipm`) including PIQP decision traces; QP
conformance passes for both `opt.ipm` and `opt.piqp`.
Log: done 2026-09-29. `solvers/ipm/` is `opt/ipm/` and `scaly.solvers` is gone. `opt/ipm/method.py`
holds `IPM(sparse=True, options=)`, PIQP's settings by name checked when it is made, registered by
core itself as `opt.ipm`; `"auto"` tries PIQP, IPM, IPOPT, SQP. Its `build` is what
`generated_piqp.py` did (deleted; its `structure_summary` and `qp_data` moved into
`examples/qp_solvers/compare.py`), now with the signature every opt solver has: a warm start (ignored,
as PIQP ignores it) and the parameters in, the solution in the variable tree, signed multipliers and
an `Info` out, PIQP's status codes mapped onto `Status` (`INVALID_BOUNDS` to `ERROR`). Finding: 4.2's
frame copied PIQP's `primal_obj` into `Info.objective`, which leaves out the objective's constant.
Fixed in core: `build_qp` gives the descriptor an `objective_constant` Function of the parameters when
the extracted `f0` is not zero, and the frame adds it to the stats and the `Info`; no plugin wrapper
changed, and the solver C snapshot, whose constant is zero, did not move. `IPM`'s objective is the
problem's own through `nlp_oracles`. The SCvx and embedded-QP drivers, the embedded-QP and
`qp_solvers` notebooks, and the three integration tests use `sc.opt.IPM`; the tests edit no
`sys.path`, and the drivers keep only their sibling imports for 7.2. SCvx reproduces bit for bit;
the embedded-QP notebook's horizon-20 solve takes the same 7 iterations to the same objective with the
same workspace, its C 3 kB larger (the warm-start inputs and `Info`); the recorded `results/` were not
rerun. `tests/opt/test_qp_conformance.py`, over `opt.ipm` and `opt.piqp` at default options, solves 22
Maros–Mészáros problems and two random QPs twice (the parameter changed, on one compiled solver) and
checks feasibility, stationarity and complementarity with the signed multipliers and the `Info`
fields, and that the five problems with no solution are not reported solved; dropping the objective
constant, flipping IPM's box-multiplier sign and reporting every status `OK` each fail it. The 36
solvers of the IPM quick set render byte-identical C to the tree before the move. `docs/guide/solvers.md`,
`solver_backends.md` (an IPM column and section), `how_it_works/solvers.md`, `dev/codebase.md`,
`dev/solver_plugins.md` and the API page describe it; API-155 and API-183 no longer wait on it.

**4.4 `scaly.roots`.** Problem classes `Root`, `LeastSquares`; methods `Newton`,
`NewtonBisection`, `GaussNewton`, `LevenbergMarquardt`; `custom_root`. `integrators/implicit.py`
and `interp/spline.Inverse` use them; four gallery examples switch from hand-written Newton loops.
Gate: integrator and interp results unchanged to recorded tolerances.
Log: done 2026-09-29. `scaly.roots` (lazy, layer 5): `root`/`least_squares` trace `Root` and
`LeastSquares` (`RootSpec` bounds the unknowns); `Newton` (`tol` absolute plus `rtol`, fixed
iterations with `tol=None`, `linear` of `"lu"`, `"cholesky"`, `"sparse_ldl"`, `simplified`,
`max_step`), `NewtonBisection`, `GaussNewton`, `LevenbergMarquardt` (Nielsen's damping), declared by
core under `roots.<name>`; `sc.roots.solver` returns the warm start and parameters to the solution
and `roots.Info` (status, iter, residual). The derivative is the problem class's: `custom_root`, the
integrator's two-level implicit-function identity made generic, attached by `Root.function`
(through the stationarity condition for least squares), solving with Newton's `linear` when it is
symmetric so a sparse problem's derivative stays sparse. Each method's `iterate` works on
expressions, and an `Info` the caller ignores costs nothing: `integrators/implicit.py` now runs
`Newton.iterate` with its stage-matrix `Linear` (its `_Linear` and `_EigenSplit` stay) and
`custom_root`, and renders byte-identical C for every Newton variant and its first and second
derivatives (48 files). `BSpline.inverse` runs `NewtonBisection.iterate`; that loop bisected when a
converged Newton step rounded to x itself (x then the bracket's end), throwing the root away, and
now stops there instead, which moves about a third of inverse values by up to 9 ulps and the
median error from 0.75 to 0.4 ulp; the interp suite passes. Newton's residual test relative to |z|
(the integrator's, kept there as `rtol = tol`) passed on a diverging iterate, hence `rtol=0` by
default; every method now reports OK only at a finite solution (Gauss-Newton had called x = inf
converged). The four examples are `bratu_newton`, `periodic_orbit`, `option_greeks` and
`hanging_chain`, not `sqp_newton_sparse`, whose loop is an SQP with inertia correction in Python:
Bratu reproduces bit for bit, the periodic orbit to the last digit, the chain fit to 1e-12 (with
LU steps: its Hessian is indefinite at the initial guess, where the old loop fell back to a gradient
step), implied vols to 1e-13 instead of 5e-10. `tests/roots` checks every method against SciPy
(`root`, `brentq`, `least_squares`) and the derivatives against finite differences or the NumPy
implicit formula; ten mutations each fail it, three after a test was added (LM's rejection, `rtol`, the
stuck step).
Guide and API pages, the codebase map; API-186 notes its route.

**4.5 `integrators` and `interp` as registries.** `TABLEAUS` and `KINDS` become method registries;
`tsit5` and LGL added.
Gate: integrator order-condition tests; interp suite.
Log: done 2026-09-29. Shape decided with the user before starting (the question was an open item):
§3.1's warm-start-and-`Info` Function is for the solving domains, and each other domain fixes its
own result; one method class per named method; today's functions stay as shorthand. `integrators`:
`ODE(f, dt, name)`, `si.solver(ode, method)` returning the discrete map, and 23 methods registered as
`integrators.<name>` (`Euler` ... `RK4`, `RK38`, `BS32`, `DOPRI5`, `Tsit5` with `steps`;
`BackwardEuler` ... `SDIRK3` and the families `GaussLegendre(s)`, `RadauIIA(s)`, `LobattoIIIA(s)`,
`LobattoIIIC(s)` with `steps` and `newton=sc.roots.Newton(...)`; `Adaptive(pair)`; `StormerVerlet`,
`SymplecticEuler`), public bases `ExplicitRK`/`ImplicitRK` for a package's own. Each builds the same
map as its shorthand: every one renders byte-identical C to it, and the 48 integrator files of the
4.4 harness are unchanged (`implicit` now goes through `implicit_map`). `MultipleShooting` takes a
method (or its name) instead of a function and options, its callers in tests, `mpc` and three
notebooks moved; an implicit interval's name is now `{model}_radau_iia2_shooting`. `tsit5` joins
`TABLEAUS` with its embedded weights (order 5, embedded 4; its rate on the linear test reads 5.1 to
5.3, a small leading constant, tested like DOPRI5's), and `adaptive` takes it; `si.lgl(n)` is the
Legendre-Gauss-Lobatto rule (nodes, weights, differentiation matrix), matching
`examples/casadi/_lgl.py` to rounding, the examples left for 6.1. `interp`: `Fit(x, y, extrap, fill,
search, strategy, dtype, name)`, `interp.solver(fit, method)` returning the `BSpline`, and 12 methods
registered as `interp.<kind>` (the ten kinds with their options, `Smoothing`, `Constrained`), plus
`PerAxis` to mix kinds; each fits its shorthand's spline (same digest). `constrained` takes `qp=`, any
opt QP method, its status read from the `Info`; the generated IPM cannot serve there, since its code
keeps only the bounds finite when built and a monotone row's upper side is infinite at run time
(documented, with IPM's own docs). The shared `Method` protocol's `build` now returns what the
domain says. `BUILT_ON_THE_CORE` in the layering test gains `scaly.roots`, missed in 4.4. `ty` caught
the condensed OCP's rollout still reading the old `MultipleShooting` fields, a path no test reached:
fixed, with a test that a continuous model's condensed and sparse forms agree. The
`explicit_methods` notebook checks every explicit tableau and now counts Tsit5 (re-executed). Four
mutations (dropped `steps`, an ignored pair, a collapsed `PerAxis`, shooting ignoring its method)
each fail the new tests.

### Phase 5: control

**5.1 `scaly.sets`.** Move `Polytope` and `Ellipsoid`; constraints returned as `(expr, lo, hi)`.
Log: done 2026-09-29. `sets/polytope.py` (moved from `mpc/`) and `sets/ellipsoid.py` (from
`mpc/terminal.py`); `sc.sets` is lazy like the other domains. `constraints(x)` returns a tuple of
`sets.Constraint` triples `(expr, lo, hi)`, so `sets` imports nothing from `opt`; the OCP turns each
into its own group, named `terminal_set` (the ellipsoid's was `terminal_ellipsoid`, which the
`terminal_sets` notebook's refusal check now reads). `mpc.Polytope`/`mpc.Ellipsoid` are gone;
`lqr`, `max_invariant_set` and `largest_ellipsoid` stay in `mpc.terminal` until 5.3 and return
`sets` objects. Tests, the two notebooks (the user's saved `linear_mpc` staged from HEAD), the API
page `sets.md` and the codebase map moved with it.

**5.2 `scaly.ocp` problems, transcription, formulation.** `ContinuousOCP`, `DiscreteOCP`,
`transcribe`, `to_problem` as in §3.3; today's `OCP` arguments map onto these (the `ode=`/`step=`
split becomes the two classes, `condensed=` becomes `form=`).
Gate: every `tests/mpc` OCP case reproduced through `transcribe` + `to_problem` with identical
optimal values; stage-structure metadata tested.
Log: done 2026-09-29. `ocp/problem.py`: `ContinuousOCP(ode, T, ...)` (a description, not solvable),
`DiscreteOCP(step=F, N, ...)` (the multistage form, its `stage` a `StageStructure` of `nx`, `nu`,
`nw`, `n_dynamics`, its `params`), `transcribe(cocp, transcription, N=)` with `dt = T / N`, and
`Quadratic`, `Path`, `TerminalEquality`; `ocp/formulate.py`: `to_problem(docp, form="sparse" |
"condensed")` returning the `opt` problem and its `Layout` (variable leaves and blocks, the
per-stage multiplier runs, the condensed `states`), built once per form. The transcriptions moved
from `integrators` to `ocp/transcription.py`. The code is `mpc.OCP`'s, and `mpc.OCP` is for this
step a thin constructor over it (its `Quadratic`, `Path`, `TerminalEquality` now `ocp`'s), so
every `tests/mpc` case runs through `transcribe` + `to_problem`; the control laws of seven OCPs
(three transcriptions, soft paths, varying references, terminal equality, polytope and ellipsoid,
sparse and condensed) render byte-identical C with identical optimal costs. `tests/ocp/test_problem.py`
solves a discrete OCP sparse and condensed and a continuous one three ways straight from `opt`,
and pins the stage structure and the layout's blocks and runs.

**5.3 OCP methods, warm start, terminal ingredients; `mpc` removed.** `sc.ocp.solver`,
`Direct`, `shift`, `ocp.terminal`; delete `src/scaly/mpc/`; rewrite `examples/mpc/*` notebooks
into `examples/ocp/` with explicit closed loops; rewrite `docs/guide/mpc.md` as `ocp.md`.
Gate: notebooks run; closed-loop results match the recorded ones.
Log: done 2026-09-29. `ocp/method.py`: the `ocp` registry (entry point `ocp.direct`), `Info`
(`status`, `iter`, `objective`, `primal_residual`) and `solver(problem, method)`, whose Function is
`xs, us, point, info = solve(x0, *params, warm)` for every method; a `ContinuousOCP` is refused with
the pointer to `transcribe`. `ocp/direct.py`: `Direct(method, form)`, `to_problem` solved by any
`sc.opt` method, with its `layout`, `warm_size`, `shift` and `initial_guess`. `ocp/warmstart.py`:
`shift(problem, method)`, a Function from the point a solve reached to the next warm start (the old
controller's stage shift), and `initial_guess`. `ocp/terminal.py` moved from `mpc/`; `mpc.linear`
is `si.affine`. Two refinements of §3.3: the solver returns `xs, us` and the flat primal-dual
`point` rather than one trajectory tree, and `shift` takes the method as well as the problem, since
the point's layout is the method's (the form, and later ILQR's own). PIQP's sparse backend is no
longer chosen for the user: `Direct(sc.opt.PIQP(sparse=True))` names it. `src/scaly/mpc/` is gone
(`MPC`, `ClosedLoop`, `simulate`, `Solution`); a receding horizon is a loop over the solver and
`shift`, a control law an `@sc.function` of the two. Before the package went, the new solver
reproduced `MPC.solve` exactly (states, controls, point and shift) on seven OCPs. `tests/mpc` became
`tests/ocp/test_direct.py` and `test_terminal.py` over a `support.Controller` of solver and shift,
with the refusals in `test_problem.py`; `tests/integration/test_interp_ocp.py` moved too. The five
notebooks are in `examples/ocp/` with their loops written out, re-executed: every recorded result
reproduces (iterations, costs, slacks and where they give way, regions of attraction, QP sizes,
agreement with NumPy and `solve_ivp`). Solve times ran 20 to 45% above the recorded ones on a loaded
machine, and HEAD's code timed alongside ran the same (the contouring pair, 1.16 s against 1.17 s);
the timing sentences now say what the outputs show. `interp/contouring_control` (the user's saved
copy staged from HEAD) and the `contouring_mpc` pair moved to `sc.ocp`; the pair's results are
bit-identical to HEAD's, iteration counts included. The composed law is larger C than the old
`MPC.law` (3 176 lines against 2 718 for the lap), the solver and the shift being Functions of their
own. `docs/guide/mpc.md` is `ocp.md`, `api/mpc.md` merged into `api/ocp.md`. Suite: 3808 passed,
42 skipped; C snapshots unchanged; ty within the ratchet (the moved notebooks' diagnostics renamed).

### Phase 6: code from the examples

**6.1** Promote per §3.5: `ILQR`, `TinyADMM` (standard); `ALTRO`, `SCvx` (experimental);
`integrators/variational.py`; `tsit5`/LGL; Gauss–Newton/LM into `roots`; diffmpc rule into
`stagewise`. Replace hand-written RK4s with `integrators.rk4`.
Gate: DiscreteOCP conformance for each OCP method; case-study numbers reproduce within recorded
tolerances.
Log: done 2026-09-29. Four OCP methods beside `Direct`, each a frozen dataclass of its options with
the common signature, warm start, `shift` and `initial_guess`, and entry points `ocp.<name>`:
`ILQR` (`ocp/ilqr.py`, from `examples/ilqr.py`, which it reproduces bit for bit: 39 iterations, the
same cost; the example is now a `ContinuousOCP` with Euler shooting solved by it), `TinyADMM`
(`ocp/tinyadmm.py`: TinyMPC's generated ADMM core `admm_solver`, the library's infinite-horizon
cache `tinympc_cache` and a finite-horizon `finite_cache`; `examples/tinympc` is an adapter over the
core and `test_tinympc` still sees the library's iteration counts), `ALTRO` (experimental,
`ocp/altro.py`, from the case study's `al_ilqr.py`, deleted: the study's four runs take Altro.jl's
iteration counts, 18, 18, 112 and 93, with the recorded agreement, 3.8e-10 against 3.9e-10), and
`SCvx` (experimental, `ocp/scvx.py`: a generic penalized-trust-region SCP over a QP method, `IPM` by
default). `integrators/variational.py` is the SCvx study's `segment_variational` generalized (any
explicit tableau, a zero- or first-order hold per control component, stages without weight
skipped); the study now uses it and the library's Tsit5 (its `model.tsit5` is gone) and its results
are bit-identical. `step_map(ocp)` (`ocp/formulate.py`) is the shared discrete map.
`tests/ocp/test_conformance.py` runs every installed method on four reference problems against the
direct method on IPOPT (a new method fails it until listed); each method has its own tests,
mutation-checked. Deviations and findings: TinyMPC's library weighs references by `Q + rho I`,
which `admm_solver` keeps for the example and `TinyADMM` replaces by the problem's `Q`; the
finite-horizon cache makes a converged `TinyADMM` solve the stated problem. On a problem whose
speed bound binds at consecutive knots PIQP stopped 4e-6 short of the optimum IPOPT and TinyADMM
agree on. iLQR as written in the example stops by an accepted step's improvement only, so on an LQ
problem at its optimum rounding makes it reject steps until `mu` overflows; the method also stops
when the backward pass predicts a negligible decrease. The ALTRO port lacked Altro.jl's
`dJ_zero_counter`, so a problem whose iLQR converges exactly ran every inner solve to 300
iterations; the method has it (`dj_counter_limit`), the case study's counts unchanged. The
pseudospectral CasADi pair takes `si.lgl` (same iterations, states to 3e-13). Not moved, recorded
under Open items: DiffMPC's first-control rule (the stagewise rule gives its gradients to 1e-14, a
test pins it, but loses the hoist), SymForce's LM (needs manifold variables), the benchmarks'
Runge-Kutta steps (8.3). Eleven examples' hand-written Runge-Kutta and Euler steps are `si.rk4` and
`si.explicit` maps, every result the same to 7e-15 and every iteration count equal (`nmpc_cartpole`'s
C byte-identical); the notebooks in `examples/notebooks` keep their written-out steps, which they
teach. Suite: 3876 passed, 44 skipped; C snapshots unchanged; ty within the ratchet.

**6.2** `scaly.nn`, `scaly.geometry`, `scaly.export` (C++, CasADi, acados) populated per §3.5;
duplicated MLP code removed from examples, benchmarks and tests.
Log: done 2026-09-29. `scaly.export`: the `cpp` and `casadi` adapters moved from `codegen/` (entry
points `scaly.export.*`, so `codegen` names no adapter and the layering test lost its exemption), and
`export/acados.py`, the neural MPC study's `acados_functions` and `install_dropin` generalized to any
model (`xdot(x, u, p)` built into each body, as the study found a call node costs twice) and to
several controls, whose `Sp` the study's version treated as a vector; a new test checks two controls
and a parameter against NumPy and compiles the drop-in. `scaly.nn`: `load_torch_state_dict` moved
from `utils` (`nn/torch.py`, with `layers_from_state_dict`), and `nn/layers.py` with `mlp`, `dense`,
the activations the code uses (tanh, sigmoid, SiLU, ReLU, smooth ReLU, spelled as it spells them),
and `unpack`/`pack`/`size` for weights in one vector. The neural MPC study, both benchmarks'
networks (unbumpercars' SiLU and smooth-ReLU models, npmpc's bias-free sigmoid decoder) and the
surrogate notebook use them, the C of every changed function byte-identical, so the benchmarks'
recorded timings stand; the surrogate notebook, re-executed, already differed from its recorded
outputs at HEAD (1304 L-BFGS iterations, not 1435, from earlier numerics), unchanged by the move.
`scaly.geometry`: unit quaternions (Hamilton, scalar last, SymForce's and SciPy's order),
`cross`/`skew`, and manifolds with a retraction and local coordinates (`Euclidean`, `SO3`, `Pose3`
as SymForce's SO(3) x R^3), tested against SciPy's rotations; the SymForce study (bit-identical, its
C byte-identical) and the SCvx model's `cross` use them, and the study's replica test checks the
library's SO(3). Left: `deploy_in_c.py`'s scalar-first quaternion kinematics and the SE(2) of
`pose_graph_slam`, other conventions, and the test fixtures that copy networks on purpose
(`test_vmap_mlp`). Suite: 3902 passed, 44 skipped; C snapshots unchanged; ty within the ratchet.

### Phase 7: tests and examples

**7.1** `scaly.testing` with the pytest plugin, `method` marker, conformance suites, hyper-dual
numbers; tests reorganized per §3.6.
Log: done 2026-09-29. `scaly.testing` (import layer 10, outside `sc.__all__`): `plugin` (the
`method` marker through a `pytest11` entry point; `method_available` checks the domain registry and,
for an external solver, that its library loads; `SCALY_REQUIRE_METHODS=1` turns a missing method
into a usage error, a subprocess test pins both), `hyperdual` (from `benchmarks/harness`),
`helpers` (from `tests/opt/problem_helpers.py`), `qp` (the Maros–Meszaros data moved from
`tests/data`, random and MPC QPs) and `conformance.{qp,ocp,roots}`. `tests/conformance/` lists every
installed method per class and fails on an unlisted one: QP over `ipm`, `piqp`, `ipopt` (88 cases);
DiscreteOCP over the five OCP methods; Root/LeastSquares over the four roots methods, each refusal
checked to be refused by `solver()` too. The piqp and ipopt plugins run the QP suite in their own
tests. 142 `solver("x")` marks became `method("opt.x")`, plus two hand-rolled `REGISTRY.installed()`
skips; CI and `benchmarks/run.py` read the new variable. Tests: the compiler's under `tests/core/`
(with `integration/` for its workload fixtures and `baseline/c/` for the snapshots); the node-ID
baseline stays in `tests/baseline/`; `tests/typing/` stays at the root, as it spans namespaces.
`tests/core` now passes with every other namespace blocked (`docs/dev/contributing.md` gives the
command); getting there moved the CasADi and C++ adapter tests to `tests/export/`, the solver-path,
solver-derivative, nested-PIQP and unnamed-leaf tests to `tests/opt/`, a spline read to
`tests/interp/`, the op-colour check to `tests/viz/`, and made the snapshot's `table` and adapted
entries skip without `scaly.interp`/`scaly.export`. The adapter-registry tests now use throwaway
adapters and a fake entry point. Deviations: the DiscreteOCP suite uses four small reference
problems (LQ, boxed, tracking, pendulum by multiple shooting) solved tight by `Direct` on IPOPT
rather than the benchmark OCPs, which `src/` cannot import; `sqp` is held out of the QP suite
(todo S-19); the NLP suite (no CUTEst data, S-20) and the LinearSystem suite (no linalg registry,
S-21) are not written; "Fit" is the least-squares half of the roots suite; per-distribution node-ID
baselines wait for 8.1's `distributions.toml`. Suite: 4023 passed, 54 skipped (the 12 new are
conformance refusals); C snapshots unchanged; ty within the ratchet (188).
**7.2** Examples per namespace with PEP 723 headers; public-API lint test; runner skips on missing
requirements.
Log: done 2026-09-29. The gallery scripts and `examples/notebooks/` went into `examples/core/`,
`linalg/`, `roots/`, `opt/`, `integrators/`, `ocp/` and `nn/` beside the existing `integrators/`,
`interp/` and `ocp/`; `qp_solvers/` is `opt/qp_solvers/`, `tinympc/` is `ocp/tinympc/` (its
`run_benchmark.py` one folder up, next to the modules it imports); `casadi/` and `case_studies/` stay.
Generated C still lands in `examples/generated/<name>/`. Every script outside `case_studies/` opens
with a PEP 723 header (78), every notebook carries the same list under `scaly` in its metadata (37):
`scaly` (`scaly[experimental]` for `nn`), the solver plugins it names, `matplotlib`/`casadi`. Finding:
ty checks a PEP 723 script as a standalone file, so its sibling imports stop resolving and
`[tool.ty.overrides]` do not reach it; the 50 scripts that import modules beside them set
`[tool.ty.environment] extra-paths = ["."]` in the header, which ty reads. Notebooks keep the project's
config, so `pyproject.toml` puts the notebook folders on ty's `extra-paths`, and each folder has the
shared `plotstyle.py` beside its notebooks (nine identical copies, a test keeps them one file); no
notebook edits `sys.path` any more. `examples/casadi/compare.py` starts its workers with `python -P`
and `PYTHONPATH` (a pair directory's own `_common` first) instead of editing `sys.path`, checked on
both pair directories. `scaly.testing.examples` reads the declarations (`requirements`, `unmet`,
`methods` from the `scaly.methods` entry points' distributions). `tests/integration/test_example_runner.py`
(was `test_notebooks.py`) discovers every script and notebook, skips on unmet requirements, marks
plugin methods, runs scripts in their own process (46, the CasADi and interp pairs among them, which
no test ran before) and notebooks in-process with example-local modules evicted between them; 28
scripts are left to their reference tests (`ELSEWHERE`, checked to be there) and five measuring
harnesses are listed with the reason in `NOT_RUN`. `tests/integration/test_examples_lint.py` rejects
`sys.path` edits and underscore names from `scaly`, and checks each declaration against the plugins
and third-party packages the code uses (dropping `scaly-piqp` from `tiny_qp.py` fails it).
Deviations: `case_studies/` keeps its `sys.path` edits and has no headers (their baselines run in
other environments and they share helpers across studies; todo CS-18); `interp`'s constrained fit
calls PIQP inside the library, so `shape_constrained.ipynb` declares it by hand; the "owned by the
highest-tier namespace" CI selection waits for 8.3's jobs. In this repository an example runs with
`uv run python examples/...` (`AGENTS.md`, `docs/dev/contributing.md`), since `uv run` on a headered
script installs from the index. ty: 188 to 142 diagnostics, the rest pre-existing (the ratchet's
paths follow the moves). Suite: 4328 passed, 54 skipped; `lookup_tables_nd.ipynb`'s timing assertion
failed once under the load of three concurrent agent suites and passes on rerun; C snapshots
unchanged.

### Phase 8: distributions

**8.1** `distributions.toml`; generate `packages/*/pyproject.toml` (Hatch `force-include` of the
listed paths into the wheel; a small build hook so an sdist rebuild finds the same paths);
`meta/scaly/pyproject.toml`; rename the root project to `scaly-core`.
Gate: every file under `src/scaly/` lands in exactly one wheel (test); `uv sync --all-packages`
works.
Log: done 2026-09-29, written in a worktree by a parallel agent and gated in the main checkout,
after the bench part of 8.3 (the repository's tests are `tests/bench`). `distributions.toml` holds each distribution's paths (the most specific listing wins, so
`ocp/altro.py` and `ocp/scvx.py` ship with `scaly-experimental` while `scaly-control` lists `ocp/`),
dependencies, extras and tests, the plugins by manifest, the repository's own tests, and every entry
point and script once: each lands in the manifest of the distribution shipping its module.
`scripts/distributions.py` reads it (owners of a file, module or node ID; what a distribution may
import) and writes `packages/<name>/`, `meta/scaly/` and the marked block of the root
`pyproject.toml`, now `scaly-core`'s (`--check` lists stale files). Deviation: no static
`force-include`, which cannot say "`ocp/` less two files" and names `../../src`, absent from an
sdist; one hook, `hatch_build.py` (copied beside each manifest, as an sdist carries its own), ships
the slice at its import path in a wheel and under `src/` in an sdist, and nothing in an editable
build, where `dev-mode-dirs` puts the one `src/` on the path. The workspace has `packages/*`,
`plugins/*` and `meta/*`; the root's dev group names every distribution, so `uv sync` is `uv sync
--all-packages`. The plugins depend on `scaly-numerics` (`versioning.md`, `solver_plugins.md`).
Node IDs: one baseline per owner, `tests/baseline/<owner>_nodeids.txt` (conformance files by problem
class, `tests/integration` and `tests/typing` the metapackage's, the structural tests and
`tests/benchmarks` the `repository`'s); a run checks each owner whose tests it collects in full, and
`--write-nodeid-baselines` writes them from the collected items, replacing the grep recipe and its
stderr hazard. `test_import_layering.py` gains the distribution table (core imports no other, each
imports only what it declares, third-party libraries included and `sc.<ns>` counted as an import,
acyclic), with three edges in `DIST_TOLERATED` for 8.2; `BUILT_ON_THE_CORE` is derived from the
table and gains `scaly.viz`. `tests/test_distributions.py` builds every wheel and sdist with
hatchling: each file in exactly one wheel, a wheel rebuilt from its sdist identical, entry points
where their modules are, manifests current, every distribution installed. Findings: `scaly-testing`
needs more than the plan's core and pytest (its helpers and suites import `opt`, `integrators`,
`ocp`), so it has extras `numerics` and `control` and the plugin keeps loading with the core alone;
`scaly_sqp.casadi` imported CasADi undeclared (extra `casadi` in its manifest); 7.2's
`scaly.testing.examples.methods()` left out only `scaly`, now every lockstep distribution
(`FIRST_PARTY`, tested against the table); `tests/core` skips whole modules without CasADi, which an
isolation job (8.3) has to install or tolerate; §3.7's warning on importing experimental modules is
left for 8.3. Run in this worktree before the change of plan: `uv sync` installed all ten
distributions; `uv build` of core, control and the metapackage; the new tests, the layering test,
`tests/testing`, the example runner and lint: 543 passed; ten mutations (a manifest shipping another's
file, an unlisted path, an sdist without its slice, core, numerics, a library and `sc.ocp` crossing
undeclared, stale baselines, `methods()` reverted, a misplaced entry point) each fail their test; the
baselines' union is 7.2's plus the 26 new tests. Gated in the main checkout: `uv sync` installed the six
first-party distributions and the metapackage as workspace members with the lock unchanged;
`scripts/distributions.py --check` clean; the common gate with the per-distribution baselines
(`pytest --collect-only -q --write-nodeid-baselines`), ty within the ratchet (142) and the core-only
run with every other namespace blocked (1508 passed, 7 skipped).

**8.2** Core `scaly/__init__.py` gets `pkgutil.extend_path` and a lazy `__getattr__` over the
known namespaces with install hints; each domain `__init__` resolves method classes lazily.
Gate: in a clean venv from built wheels, `import scaly as sc; from scaly import linalg as la;
sc.linalg is la`; `sc.ocp` without `scaly-control` raises an `AttributeError` naming it; `ty` and
`pyright` resolve `scaly.linalg` both in the workspace and in the wheel venv.
Log: done 2026-09-29, written in a worktree by a parallel agent and gated in the main checkout. `scaly/__init__.py` extends its
`__path__` with `pkgutil.extend_path` after the core's own imports, and `_NAMESPACES` maps each
namespace to its distribution: `sc.<name>` imports it on first read and, when it is missing, raises
`AttributeError("scaly.ocp needs scaly-control, which is not installed (uv add scaly-control)")`;
`sc.expr_graph` names scaly-tools the same way, and `sc.viz` joins the table, while `scaly.testing`
stays out of the namespace. `roots`, `integrators` and `interp` get the lazy method-class hook `opt`
has; `ocp` gets it too and stops importing `ALTRO` and `SCvx`, which leave its `__all__` and resolve
through the registry, whose hints name scaly-experimental. `DIST_TOLERATED` is empty. Tests:
`tests/core/test_namespaces.py` (a missing namespace and the graph views name their distribution,
`sc.linalg` is the imported module, a `scaly/` portion elsewhere on the path merges);
`tests/test_distributions.py` holds `_NAMESPACES` to the table and runs, in an interpreter with no
site-packages, the unpacked wheels of core and numerics (`sc.linalg is la`, `sc.ocp` names
scaly-control) and of core, numerics and control (`sc.ocp.ALTRO` and `SCvx` name scaly-experimental).
Gate: `uv build` of scaly-core and scaly-numerics into a fresh venv with only those two wheels
(and NumPy, SciPy): `sc.linalg is la`, and `sc.ocp` raises the `AttributeError` naming scaly-control;
ty and pyright resolve `sc.linalg.SparseLDL` to its class in the workspace and in that venv, and
pyright flags a name `linalg` does not have. Suite (8.1 and 8.2 together): 4365 passed, 54 skipped;
C snapshots unchanged; ty within the ratchet (142).

**8.3** CI jobs: workspace at HEAD; one isolation job per distribution (install its wheel with only
declared dependencies, run its tests); plugins against HEAD and against their oldest and newest
supported `scaly`; release job (all wheels, `scaly[experimental,solvers]` in a clean env,
conformance and examples). `scripts/release.py`. `benchmarks/` renamed `bench/`. Rewrite
`docs/dev/versioning.md`, `docs/dev/codebase.md`, `AGENTS.md` commands.
Log: Bench part done 2026-09-29. `benchmarks/` is `bench/` and `tests/benchmarks/` is `tests/bench/`
(`git mv`, the four Foxglove layouts byte-unchanged); the package imports as `bench`, every command
is `uv run bench/run.py ...` (CI, the `wt` pre-merge hook, `AGENTS.md`, which gains the smoke
command, the dev and results pages, `bench`'s READMEs, two example references), and ruff includes
and ty excludes `bench/**` as they did `benchmarks/**`. Inside `bench/` the one duplicated helper
was the provenance call, five callers each passing the repository root and the harness compiler;
`provenance.collect(cli_args)` defaults both. The helpers §3.5 names (`text_bytes`, `time_c`,
`compile_*`, `machine()`, `parse_ipopt`) live in the case studies and `examples/{opt/qp_solvers,casadi}`'s
`compare.py`, not in `benchmarks/`, and an example can import `bench` only through the `sys.path`
edit 7.2 removes, so they stay duplicated until the comparison runners themselves move into `bench/`
(after 7.2, Open items). The chain, race-car and unbumpercars problems step with
`si.rk4(model, dt=None)`: `chain_step_fn`, `race_car_rk4` over `race_car_ode` (the model a Function,
where it was inlined) and the unbumpercars continuous-time map and discrete model's pose step, each
model taking one parameter per slot as `si` requires; the NumPy plants and the CasADi mirrors keep
their written steps, being the references the gates compare against, and npmpc has no Runge-Kutta
step (a learned discrete map, a NumPy midpoint plant). The step multiplies by `h/6` where the
written one divided, so the generated C of those three problems changed; measured against the tree
before the move at seeded inputs, the values agree to 1e-16 (chain's step) or exactly (chain's and
the race car's equality constraints, both unbumpercars steps), and the C moved by one division
turned multiplication per step, except the unbumpercars discrete model, whose pose step is now a
called map (158 lines against 169, eleven more multiplications and additions). The recorded results
predate it (`fairness.md`, todo BH-49), to be re-recorded on a quiet reference machine. `tests/integrators/test_explicit.py`
gains the two shapes the problems now use, the map in a vmapped shooting transcription with its step
read from a broadcast tail and the map vmapped directly with a scalar step formal, against the
written-out step (values, sparse Jacobian, sparse Lagrangian Hessian); the written-out shapes stay
covered in `tests/core` (`test_stage_transcription`, `test_derivatives`, `test_vmap`). A stage
weight scaled by 1e-6 in `increment` fails both new tests, and reading `dt` from the head of the
arguments fails their module. Written in a worktree by a parallel agent that could not sync an
environment there, and gated in the main checkout after 7.2 (committed before 8.1, which it does not
touch): 4332 passed, 54 skipped; C snapshots unchanged; ty within the ratchet (142);
`uv run bench/run.py smoke` passed. The untracked `benchmarks/results/` and `third_party/` moved to
`bench/`.
Rest done 2026-09-29. `scripts/isolation.py <distribution>` builds the wheels of the distribution,
of what it depends on and of scaly-testing, installs them into a fresh environment beside only their
declared third-party dependencies, the extras the distribution declares and pytest, and runs the
tests `distributions.toml` gives it there, leaving out paths inside them that another distribution
owns more specifically (`tests/ocp/test_altro.py` from scaly-control) and turning the conftest's
node-ID check off (`SCALY_NODEID_BASELINES=off`: an isolated collection differs by design); a
plugin runs `--against head` or against the `lowest`/`highest` release of its `scaly-numerics`
range from the index. First runs found what the workspace hides: modules building a plugin's method
at import time (`test_solver_derivatives`, `test_direct`, `test_terminal`, `test_tinyadmm`,
`test_scvx`, OCP conformance; now cached functions or factories), 16 tests building a plugin's
method without its `method` mark, listing checks demanding every method installed (now: every
installed method listed), linalg tests reaching `scaly.export` and the TinyMPC example (`scaly.ocp`)
unguarded, a bench recorder test under `tests/viz` (now `tests/bench`), a CasADi test undeclared by
scaly-ipopt. Every lockstep distribution and all three plugins now pass alone (core 1324, numerics
1572, control 49, tools 18, experimental 22, testing 14, piqp 51, ipopt 47, sqp 43 passed). CI gains
`isolation` (a matrix over the six lockstep distributions), `plugin-isolation` (each plugin against
the workspace, its methods required), `plugin-range` and `release`, the last two on a `v*` tag or by
hand. `scripts/release.py X.Y.Z` sets the table's version, regenerates the manifests (pins move with
it), relocks, builds every wheel and sdist into `dist/` and runs the release check (a clean env with
`scaly[experimental,solvers]` from those wheels alone and the examples' third-party packages;
conformance, the example runner and its lint, methods required); `--tag` tags `vX.Y.Z` on a clean
tree; nothing is uploaded. §3.7's warning: `sc.ExperimentalWarning` (`scaly.utils.experimental`)
on importing `scaly.nn`, `scaly.geometry`, ALTRO and SCvx, filtered in the suite, checked in a
subprocess. `docs/dev/versioning.md` is rewritten (lockstep and independent, method APIs, tags,
the release process), `codebase.md` and `contributing.md` describe the isolation check, `AGENTS.md`
lists both scripts. Deviations: the method API is checked for equality when a method is looked up,
not as a range at registration, and the doc says so; `plugin-range` cannot pass before the lockstep
distributions are on the index; the release job checks and archives, publishing stays a manual
`uv publish`. The release check ran end to end here: every wheel built, and in the clean
environment the conformance suites, every example and notebook and their lint gave 455 passed, 12
skipped (the conformance refusals).

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
- **`ty check` was red on `devrush` before step 0.1** (258 diagnostics, none in `src/` or
  `plugins/`). Step 0.1 fixed the five in `tests/` and the eight in plain example scripts, and
  excluded `examples/case_studies/*/baseline/**` (they run in the baselines' own environments). The
  remaining 198 are in notebooks and case-study drivers, almost all sibling imports through
  `sys.path` (`plotstyle`, `scaly_impl`, ...), which step 7.2 removes. Until then the ty part of the
  common gate is a ratchet: no diagnostic outside the list recorded at step 0.1. Decide whether to
  fix them earlier. After 7.2: 142. The gallery's sibling imports resolve; what remains is the case
  studies' (todo CS-18) and type errors in notebook cells (NumPy unions, Matplotlib argument types),
  which need edits to executed notebooks.
- **DiffMPC's first-control rule stays in its case study (found at 6.1).** `linalg.stagewise`'s
  implicit derivative of `Riccati.solve` gives the study's gradients to 1e-14 (a test pins it), but
  `solve` runs the affine backward pass together with the rollout from `x0`, so a batch episode can
  no longer hoist the `x0`-free part of the recursion: the study's hoisted forward pass went from
  0.14 to 2.4 ms. Wanted in `stagewise`: an `x0`-free policy `(K_k, k_k)` with its implicit rule,
  after which the study's rule can go.
- **SymForce's Levenberg-Marquardt stays in its case study (found at 6.1).** `roots` has generic
  Gauss-Newton and LM (4.4); the study's is another algorithm (SE(3) retraction, factor-wise sparse
  normal equations on `SparseLDL`, SymForce's damping schedule). Moving it needs manifold variables
  (a retraction and a tangent size) and a sparse normal-equation solve in `roots.LeastSquares`; the
  manifold comes with `geometry` (6.2).
- **The recorded benchmark results predate the problems' `scaly.integrators` steps (found at 6.1,
  code done at 8.3).** The chain, race-car and unbumpercars problems step with `si.rk4` since 8.3,
  which changed their generated C; the published tables were recorded before, and re-recording
  them needs the reference machine quiet (todo BH-49), so it was not done with the code.
- **The case studies' comparison runners stay in `examples/` (found at 8.3).** `text_bytes`,
  `time_c`, the `compile_*` helpers, `machine()` and `parse_ipopt` are copied across
  `examples/case_studies/*` and `examples/{opt/qp_solvers,casadi}/compare.py`. An example cannot import
  `bench` without a `sys.path` edit, so sharing them means moving the runners into `bench/` (after
  7.2, which is reorganizing `examples/`), or `bench` becoming an installed workspace member.
