# Benchmarks

This package owns Scaly's reproducible benchmark workloads and correctness-gated harness. Generated sources, binaries, logs, CSV files, and provenance live under `benchmarks/results/` and are not committed.

## Layout

- `problems/` contains importable chain-of-masses, race-car NMPC, unbumpercars HCBF filter, and neural-process-MPC formulations.
- `harness/` contains Google Benchmark wrapper generation, dense-reference checks, sweep mechanics, and provenance capture.
- `run.py` is the entry point for CI smoke gates and scalability sweeps.

## Prerequisites

A C++ compiler, `git`, and `cmake`. Google Benchmark does not have to be installed:
the harness clones the pinned tag in `harness/gbench.py` and builds it into
`benchmarks/third_party/gbench/<tag>` (gitignored) the first time a cell is compiled,
then links that static library by absolute path. The build takes a few seconds and is
skipped once the header and library are in place. It stays inside the checkout, so
parallel worktrees never read a prefix another one is still installing, and the
scaly JIT cache under `~/.cache/scaly` keeps holding only what scaly itself compiles.
Delete the directory to force a rebuild, and read `build.log` inside it if the build
fails.

## Smoke gates

```bash
uv run python benchmarks/run.py smoke
uv run python benchmarks/run.py smoke --skip solver_call
uv run python benchmarks/run.py smoke --select problems
```

Smoke runs three groups, all on by default:

- `problems` — each problem's own formulation gates, owned by the problem
  (`problems/*/checks.py`). For `race_cars` that is the vendored track data, the
  spline reference generator, the parameter-tail order, a pin on the physical
  constants, the episode's shapes and bounds, the recorded scene, and the
  Scaly-vs-CasADi trajectory agreement; for the chain of masses the dimensions and
  RK4 plant, the equality Jacobian against CasADi and a dense reference, the NLP
  objective against CasADi, the single end-mass reference behind both cost terms,
  the episode's shapes, and the recorded scene; for `unbumpercars` the
  Scaly-vs-CasADi oracle, the parameter-tail order, the pair barrier's relative
  degree, the discrete-MLP plant's three pieces and its reversed control order,
  both symbolic discrete-MLP prediction paths against a NumPy reference, per-step
  Scaly-vs-CasADi solution agreement on a binding state, and the default
  exact-Hessian rollout; for `npmpc` the vendored checkpoint and reference episode,
  every weight and bound read out of the reference implementation's own config file, the
  parameter-tail order, the terminal Riccati weight, the transcribed constraint rows, the
  closed-loop swing-up, the recorded scene, and a per-step cross-implementation comparison
  against their released episode — [`problems/npmpc/README.md`](problems/npmpc/README.md)
  records what that gate does and does not establish, and which perturbations it survives.
  All four problems also gate the SQP column against the
  IPOPT column (`sqp_matches_ipopt`): the same smoke episode through both solvers,
  compared per step in applied control, planned trajectory, objective, and
  constraint violation, with tolerances chosen from measured healthy-step
  differences (documented next to each gate). On divergence the failure names the
  first diverging step with both solvers' status and violation — the input the
  SQP robustness work consumes. Gates needing IPOPT or CasADi report `skipped: ...`
  rather than passing silently.
- `benchmarks` — Python and compiled-C derivative kernels against a dense reference,
  plus sparsity, workspace, and loop-preservation invariants. Every Scaly cell evaluates the exact
  sparse Jacobian or Hessian and sparsity supplied by its solver descriptor. For `npmpc` the
  loop-preservation gate runs on two axes: the generated source must not grow with the
  horizon (the decoder uses VMAP, not per-stage unrolling) and must not grow with the
  decoder width either, since the weights are read out of the parameter tail rather than
  baked in as literals. Independent dense NumPy references cover every equality and inequality row
  in the race-car and `npmpc` solver descriptors.
- `solver_call` — the QP/IPOPT solver-call ABI, when vendored solver libraries
  are present.

Any failure produces a nonzero exit status.

### Adding a problem gate

A problem's `checks.py` holds one function per gate and a `CHECKS` table mapping a
short name to `(check, needs_ipopt, needs_casadi)`. `run_checks()` walks the table
and yields `(name, outcome)`, where `outcome` is `"ok"` or `"skipped: ..."` when a
required dependency is missing — a gate never reports success without having run.
Checks raise (bare `assert` or `numpy.testing`) instead of returning a bool, so the
failure message carries the offending values.

Build a per-cell dense reference from the *stage* function, not by differentiating
an unrolled twin of the whole formulation. The neural-process-MPC reference was
written the second way first: at N = 100 it consumed 900 MB and never finished,
where scattering per-stage blocks takes 2.2 ms at N = 100 against 25 s at N = 25
and agrees with the unrolled version to 0.0 wherever that is affordable. It is
also *more* independent of the kernel under test, not less, because it never
builds the VMAP graph at all.

Confirm a new gate can actually fail, by perturbing the thing it checks and
watching it fire. A gate that cannot fail is worse than no gate, because it reads
as coverage. Beware perturbations that are secretly no-ops: scaling an objective
does not move its argmin, so multiplying a cost by a constant will not disturb a
trajectory-agreement gate, while biasing one control term will.

## Sweeps

```bash
uv run benchmarks/run.py sweep
uv run benchmarks/run.py sweep --workloads race_cars --sizes 1,5,10,50 --backends scaly,casadi_sx,casadi_mx,casadi_call_mx,casadi_map_sx
uv run benchmarks/run.py sweep --out benchmarks/results/sweep/my-sweep.csv
```

Sweeps run CasADi 3.8's `Function.transform()` simplification flow on every CasADi kernel before
generating code. `--casadi-transform` is enabled by default; use `--no-casadi-transform` to disable it. The unsuffixed workload on each axis measures the exact sparse Lagrangian Hessian from the solver
descriptor. Add `_jac` to `chain`, `race_cars`, `npmpc`, or `npmpc_decoder` to run the constraint
Jacobian row retained for the long paper. The default sweep runs only the Hessian workloads.

The sweep defaults to three repetitions. Each cell starts in a fresh Python process, with backend
order varied using `--order-seed`. `--repetitions 1` is useful for a toolchain check. The raw CSV
records repetition and backend order. A sibling `.summary.csv` reports mean, median, sample
standard deviation, coefficient of variation, minimum, and maximum across successful processes.
Failed attempts remain in the raw CSV and the summary counts. A single sample has no dispersion
estimate. `kernel_compile_ms`, `wrapper_compile_ms`, and `link_ms` separate generated code from the
measurement wrapper. `compile_ms` retains their total elapsed time.

The regular commands work on Linux and macOS. On hosts without Linux CPU frequency controls,
provenance records empty policies and an unknown boost state. CPU affinity is unknown when the
operating system does not expose it. Fresh processes, varied order, and dispersion reporting remain
available. The reference-machine `--headline` frequency check is Linux-specific and rejects
unsupported hosts explicitly. It does not try to change macOS power settings.

For headline runs on Linux, set every CPU policy to `performance` and choose the boost state first. Then use
`--headline --boost on` or `--headline --boost off`. The runner checks those settings before and
after each cell and requires at least three repetitions. Provenance records the frequency policies,
boost state, and CPU affinity. Run boost-on and boost-off comparisons into separate output
directories with the same grid and seed. Keep compilation and other benchmark jobs out of timed
runs. The runner validates machine settings but does not change them.

Each `(workload, size, backend)` cell retains its generated C/header, raw float64 samples, wrapper, binary, and compile log next to the CSV, under `<csv-parent>/repeat_<n>/<workload>/<backend>_<axis><size>/`. With the default CSV this is `benchmarks/results/sweep/repeat_<n>/<workload>/`. Rows stream to CSV as cells finish; a sibling `.provenance.json` records the exact CLI, git state, package and compiler versions, platform, Python, and timestamp. It also records the resolved IPOPT library and its MUMPS, METIS, and OpenBLAS build. After canonical closed-loop runs, the chain-of-masses M=5, race_cars N=40, unbumpercars C=8, and neural-process-MPC N=12 cells automatically consume their harvested `representative_fe_inputs.npz` rather than synthetic samples.

Synthetic unbumpercars kernel inputs use seeded states inside the arena without rejecting
collisions. Large car counts therefore remain measurable even when collision-free closed-loop
placement cannot fit them. The closed-loop runner still requires collision-free initial states.
The provenance records this input policy for every affected cell.

The stage-based workloads default to Scaly and four CasADi encodings of the repeated dynamics
stage: unrolled `SX`, unrolled `MX`, repeated calls to an elemental `MX` `Function`, and a serial map
of an elemental `SX` `Function` in an `MX` outer graph. The objective and inequality expressions
remain in the problem's canonical outer graph. The chain smoke tier uses literal `casadi_sx` at M=3 and mapped SX at M=5;
full sweeps record compilation outcomes at each requested mass count. Unbumpercars defaults to Scaly, `SX`, and `MX` because its pairwise
formulation has no single repeated stage. If an explicit backend does not apply to a workload, the
sweep records `not_applicable` and continues.

The CSV separates generated artifact bytes into `executable_bytes` and `static_metadata_bytes`.
The executable count is the C translation unit without `static const` declarations. The metadata
count includes those declarations and the generated header. It therefore includes sparse index
tables, automatic-differentiation seed tables, and numeric constants that the generator emits as
arrays. A generator that writes a constant inline counts it as executable source. Their sum is
`artifact_bytes`; `source_bytes` remains the complete C translation unit for compatibility with
earlier runs.

The CSV records floating-point and integer ABI workspace and the required lengths of the argument
and result pointer arrays. `coloring_width` is specific to each backend and must not be compared across
backends. For Scaly, it is the number of compressed tangent directions the generated
derivative actually executes. A structured Jacobian sums the independently executed directions from
each formal or local batch; a sparse Hessian reports the global star-color count. CasADi leaves the
field blank because its generated code does not expose its internal derivative count. Scaly rows
group the mapped dispatch loops by trip count and report the family whose trip count times
per-iteration arithmetic is largest: `dispatch_trip_count` is that family's trip count and
`dispatch_arithmetic` is the sum of its callees' floating-point Program IR operations per iteration.
Other families are not reported (the unbumpercars pair rows, for example, lose to the per-car
network step). `dispatch_workspace` is the maximum floating-point scratch across every mapped
callee, stack slots and `w[]` slots included. The dispatch fields are empty for kernels without a
map. CasADi does not expose the derivative function inside its generated map as a stable inspection
boundary, so its dispatch fields are empty.

All benchmark artifacts follow the same command-first layout:

```text
benchmarks/results/
  closed-loop/<problem>/<solver>+<oracle>/  # episodes, plots, and harvested inputs
  sweep/<problem>/<cell>/           # generated code, binaries, samples, and logs
  sweep/scalability.csv             # default sweep table and provenance sidecar
  smoke/<problem>/<cell>/           # smoke-generated code, binaries, samples, and logs
  smoke/closed-loop/<problem>/<solver>+<oracle>/  # short episodes, isolated from canonical artifacts
  smoke/solver_call/                 # generated solver-call smoke artifacts
```

**Only harness numbers count.** Timing a backend from Python includes dispatch overhead.
Treat a Python-level reading as a smoke test for whether a cell builds, never as a result.

The doctrine is claims-first: broad sweeps establish scaling and canonical points support comparisons; correctness gates always run before speed is measured; every result carries enough provenance to reproduce it. The governing claim matrix lives in the paper repository.

The gates here guard the *measurements*, not the compiler. Op and composition coverage lives in `tests/` as small artificial cases checked against unrolled or NumPy references; a benchmark problem must never be the only thing exercising an IR, AD, or codegen path. That separation is what lets the problem set follow the workload roadmap without silently dropping compiler coverage.

The split runs both ways: a check that is about *a problem* rather than about Scaly belongs in that problem's `checks.py`, not in `tests/`, so the pytest suite never imports a benchmark problem. `race_cars` is the worked example — `problems/race_cars/checks.py` owns its formulation gates, and `tests/integration/test_stage_transcription.py` carries a self-contained copy of the RK4 stage-transcription shape that problem surfaced, so retiring the problem cannot drop the compiler coverage. The chain-of-masses and unbumpercars problems follow the same shape; the IR behaviours they lean on are reproduced self-contained in `tests/ad/test_sparsity.py` and `tests/function/test_factory.py`.

## Studies: every headline run in one directory

```bash
uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name>
uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name> --only sweep --problems race_cars,unbumpercars
uv run benchmarks/run.py study --out-dir benchmarks/results/<study-name> --only closed-loop --no-headline --repetitions 1
uv run benchmarks/run.py report benchmarks/results/<study-name>
```

`study` runs the frozen headline grids in `harness/study.py`: the exact-Hessian sweep of every
problem over its published sizes, and each supported canonical IPOPT and SQP closed loop,
each as its own `run.py` invocation with five fresh processes, seed 0, and the headline CPU checks
with boost off unless told otherwise. It writes `sweep/<problem>/<problem>.csv` and
`closed-loop/<problem>/` under the output directory, refuses a directory that already has content,
and records every command and its exit status with the usual provenance in `study.json`. It ends
by running `report`, which renders `sweep/<problem>/table.md`, `closed-loop/closed_loop.summary.json`,
`closed-loop/closed_loop.summary.csv` and a combined `report.md`. The figures on the benchmark pages
read `docs/assets/benchmarks/closed_loop.summary.csv` and `sweep.summary.csv`, the four
`sweep/<problem>/<problem>.summary.csv` files concatenated. `report` can
be rerun on any study directory, including one whose runs were launched by hand. Every episode
writes a per-step `telemetry.csv` beside its MCAP, which is what `report` reads.

## Closed-loop episodes and Foxglove

Run a short end-to-end episode, including the generated Scaly/IPOPT path and MCAP recording:

```bash
uv run python benchmarks/run.py closed-loop --problem race_cars --smoke
```

Canonical runs omit `--smoke`:

```bash
uv run python benchmarks/run.py closed-loop --problem chain
uv run python benchmarks/run.py closed-loop --problem chain --solver sqp --oracle scaly
uv run python benchmarks/run.py closed-loop --problem chain --solver sqp --oracle casadi
uv run python benchmarks/run.py closed-loop --problem race_cars --solver ipopt --oracle casadi
uv run python benchmarks/run.py closed-loop --problem race_cars --solver sqp --oracle scaly
uv run python benchmarks/run.py closed-loop --problem race_cars --solver sqp --oracle casadi
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver ipopt --oracle scaly
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver sqp --oracle scaly
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver sqp --oracle casadi
uv run python benchmarks/run.py closed-loop --problem unbumpercars --solver none
uv run python benchmarks/run.py closed-loop --problem npmpc --solver ipopt --oracle scaly
uv run python benchmarks/run.py closed-loop --problem npmpc --solver ipopt --oracle casadi
uv run python benchmarks/run.py closed-loop --problem npmpc --solver sqp --oracle scaly
uv run python benchmarks/run.py closed-loop --problem npmpc --solver sqp --oracle casadi
```

Each solver episode also writes `modes.csv`. The `jit` row adds observed construction time to the
first solve. The `prebuilt` row reports the first solve with construction and loading excluded.
Both use the mean wall time of subsequent solve calls for per-step cost. These wall timings include
Python dispatch and are separate from the native solver statistics. Construction includes the
standalone oracles used to check solver results, so the `jit` startup cost includes that benchmark
validation setup.

To combine recorded runs into the five-row deployment table, run
`uv run benchmarks/run.py modes <compiled-results> <interpreted-results> --out <table.csv>`.
Use runs with the same workload configuration and machine settings. The command reads the saved
mode files and does not rerun any benchmark.

For CasADi's third mode, add `--casadi-interpreted` to an `ipopt+casadi` run. Its artifacts live under
an `interpreted/` directory, and its mode row names the CasADi wheel's IPOPT provider. That row is a
deployment comparison and does not isolate oracle cost against the compiled columns' shared IPOPT.

`closed-loop --repetitions 3` runs sequential fresh processes with separate empty compilation caches
under `repeat_<n>/cache/<oracle>/`. Reusing one of those output directories fails. The parent writes
`modes.summary.csv` with startup and per-step dispersion. A single episode uses the configured cache,
so its `jit` row may measure a cache load. Add `--headline --boost on` or `--headline --boost off` to
check frequency settings around every repeated episode. `--oracle both` compares the compiled
providers, rotating their order between repetitions from an initial `--order-seed` shuffle. The
summary provenance retains the actual run order. Problem smoke runs each problem in its own
process, so large symbolic graphs from earlier problems do not remain resident.

The closed-loop interface selects the optimizer with `--solver` and the generated
function provider with `--oracle`. Unsupported pairs are rejected per problem;
for example, chain does not currently provide an IPOPT/CasADi runner. Canonical
runs write to `benchmarks/results/closed-loop/<problem>/<solver>+<oracle>/`, while
`--smoke` writes under `benchmarks/results/smoke/closed-loop/`, so a CI smoke
cannot replace a harvested canonical input. The two SQP columns run the same
`scaly-sqp` implementation and PIQP
subsolver with identical settings; only the generated C-ABI oracle provider
changes. The CasADi SQP adapter applies `transform({})` to every generated oracle and
supplies only the upper Hessian triangle, matching the Scaly SQP descriptor. For `race_cars`, the IPOPT/Scaly and IPOPT/CasADi columns solve a
deliberately identical problem — same decision-variable and
parameter layout, same cost and constraint rows in the same order, same IPOPT
with the same options — so the only difference is who differentiates and
evaluates the oracles. The `race_cars/oracles_agree` smoke gate holds them to
that with a cross-provider trajectory comparison, and the default
`ipopt+scaly` run is the column the FE sweep harvests from. The open-loop
unbumpercars controller is represented by `--solver none` and writes under
`none/` because it has no oracle.

All SQP columns use exact Lagrangian Hessians by default, assembled into PIQP's
sparse interface from the oracle sparsity patterns and convexified by a modified
sparse LDL^T over the same pattern, after constraint-normal `A.T @ A` damping
where the problem has equalities. Canonical runs use `scaly-sqp`'s sparse QP path. The dense
path remains a diagnostic option rather than a published comparison.

The canonical unbumpercars SQP column uses the default filter first, at the
solver's default tolerances, and gives rare active-barrier solves up to 1000
iterations. If the filter strictly rejects all trials, the controller retries
the same warm start through l1/watchdog-five;
both oracle providers use the identical two-solver policy. This completes all
200 canonical solves without braking fallback or collisions.

The canonical N=40 race episode starts both SQP oracle columns from the same
problem-owned nominal primal in `problems/race_cars/data/nominal_N40.npz.b64`.
It is the deterministic IPOPT solution of the canonical initial NLP and is used
only as the first NMPC guess; every online call, including that first call, is
solved and timed by `scaly-sqp`, and subsequent calls use the shifted SQP
primal/dual warm start. The independent SQP-versus-IPOPT smoke gate does not use
this N=40 nominal.

Each run prints the exact artifact directory. It contains `episode.mcap`,
`rollout.npz`, configuration/summary/provenance JSON, and
`representative_fe_inputs.npz`. The MCAP includes `/tf`, the scene channels, and
control, state, horizon, and solver telemetry channels. `/tf` carries a static
`world → scene` transform plus one `scene → car/<vehicle>` transform per step, and
every primitive drawn on a car is expressed in that car frame rather than in track
coordinates, so setting the 3D panel's display frame to `car/vehicle` makes the
camera ride along. Planar scenes are centered through the transform and retain
trajectory trails, which stay in the `scene` frame; the
unbumpercars scene also contains a persistent arena boundary, and the race-cars scene
contains the track (center line plus one cube per cone, coloured as on a real
track), with the reference and predicted horizons redrawn every step.

Race-car recording happens inside the plant loop. Every completed step is
flushed to the open MCAP before the next solve; if a later solve raises, the
recorder's context still closes a valid file. The runner then writes a partial
`rollout.npz` and a summary containing the failing step plus Scaly and native
statuses before re-raising the failure. The complete failing warm start is also
written to `failing_solver_inputs.npz` for one-call replay. Successful episodes
keep the same artifact shapes and channels as before.

The scene is split across `/scene` (vehicles and trails), `/scene/horizon`, and
`/scene/static` (track and arena) because of how Foxglove seeks: jumping in the
timeline hands each panel only the newest message on each subscribed topic. Geometry
that has to survive a jump therefore has to be the last message on a topic nothing
else writes to. With that split, the track and arena are logged once at time zero
and stay visible however far ahead you scrub — a single 22 kB message per episode
rather than one per step. A scene topic is invisible in the 3D panel until its layout
turns it on, so a recorder that starts writing a new one shows nothing until the layout
is re-exported from Desktop with it enabled — worth remembering when new geometry does
not appear. Unbumpercars writes no horizon, so it enables only the other two.

Recording is one class per problem. `Recorder` owns what every episode has — the
writer, `/run/metadata`, `/telemetry`, `/control`, `/scene`, `/scene/static` and
`/tf` — and `ChainRecorder`, `RaceCarRecorder` and `UnbumpercarsRecorder` add their
own channels in `_open_channels` (`RaceCarRecorder` and `UnbumpercarsRecorder` share
the planar-vehicle machinery through `PlanarRecorder`). Channels register as they
are opened, so an episode offers Foxglove only the topics its problem actually
writes rather than a padded list with the other two problems' topics sitting empty.

The neural-process MPC scene is the suite's only rotating linkage, so `NpmpcRecorder`
subclasses `Recorder` directly rather than going through `PlanarRecorder`: the pose is
two angles rather than a position and a yaw, and the arm and pendulum rods hang off a
`scene → arm → pendulum` transform chain and are logged in body coordinates. The base,
the shaft and the arm-angle bound the single slack softens go on `/scene/static`; the tip
trail and the predicted tip path are in the `scene` frame, so `furuta_tip` and the
transform chain have to agree, which `tests/viz/test_recording.py` pins.

In the chain-of-masses scene the fixed wall anchor and the actuated end mass are picked out by
colour, the applied control is an arrow on the end mass (the control *is* that mass's
velocity), and the end mass keeps a trail. Its open-loop plan goes on
`/scene/horizon` as one faint strip per horizon node plus the path the plan takes the
end mass along, and its end-mass reference is a labelled marker on `/scene/static`,
logged once for the same seeking reason as the track. What that reference is, and how
it departs from laopt, is documented in
[`problems/chain/README.md`](problems/chain/README.md).

Layouts are **not** generated. Each problem keeps one hand-authored layout,
exported from Foxglove Desktop, next to its runner:

```text
benchmarks/problems/chain/foxglove-layout.json
benchmarks/problems/race_cars/foxglove-layout.json
benchmarks/problems/unbumpercars/foxglove-layout.json
```

In Foxglove Desktop:

1. Open the Layouts menu and import the problem's `foxglove-layout.json`.
2. Select that layout.
3. Open `episode.mcap` with **Cmd/Ctrl+O**, or run
   `foxglove-studio /absolute/path/to/episode.mcap`.

To change a layout, edit it in Desktop, export it, and overwrite the checked-in
file. The checked-in JSON is a backup of that export, never a file to edit in
place: only Desktop's own export is trusted here — the SDK's `foxglove.layouts`
builder emits a different `version`/`content` envelope, and an earlier
hand-rolled `configById` generator produced files Desktop refused to import — and
a hand-written edit is lost the moment the layout is re-exported. `AGENTS.md`
makes these files off limits to agents for the same reason.

Canonical operating points are deterministic and intentionally modest:

| problem | canonical point | scene |
|---|---|---|
| chain of masses | `M=5`, controller `N=12`, 90 plant steps at 0.2 s (long enough to settle) | 3D chain, end-mass control arrow and trail, open-loop plan, end-mass reference marker |
| race cars | controller `N=40`, one lap of `fsds_competition_1` (340 m at 0.05 s), IPOPT and SQP with two oracle providers | track, cones, planar vehicle, reference and predicted horizons |
| unbumpercars HCBF | 8 cars, 200 plant steps at 0.1 s, seed 42, IPOPT and SQP with two oracle providers | planar cars with body and keep-out rings |
| neural process MPC | controller `N=12`, 100 plant steps at 0.02 s from hanging, IPOPT and SQP with two oracle providers | 3D Furuta pendulum on a `scene → arm → pendulum` transform chain, tip trail, predicted tip path, arm-angle bound |

The midpoint successful closed-loop oracle input is harvested for future
Google Benchmark cells. Closed-loop runs are manual-only; CI keeps using the
faster correctness and code-size smoke gates.

## Problems

| problem | scaling axis | reference |
|---|---|---|
| chain of masses | number of masses | laopt/acados chain-mass formulation; `M=5` is the canonical point |
| race cars | horizon | CasADi SX/MX; reference stages followed by symbolic vehicle parameters in `p` |
| unbumpercars HCBF filter | number of cars | CasADi MX; neural weights followed by symbolic vehicle parameters and `dt` |
| neural process MPC | horizon (`npmpc`) and decoder width (`npmpc_decoder`); append `_jac` for the long-paper Jacobian rows | the Furuta-pendulum controller of *Neural Process Model Predictive Control*; a conditional-neural-process decoder evaluated at every horizon node, weights and latent code in the parameter tail. Formulation, vendored data, and departures from the reference implementation are in [`problems/npmpc/README.md`](problems/npmpc/README.md) |

### Race-car track data

`problems/race_cars/data/tracks/` holds four Formula Student Driverless Simulator
tracks, vendored as CSV from the `minimal_tracking_nmpc` reference implementation:
`center_line.csv` (`x,y,right_width,left_width`) and `cones.csv`
(`cone_type,X,Y,Z,std_*,right,left`). Each is a closed circuit 340–460 m long with
a corridor about 1.7 m wide per side.

### Race-car solvers and oracles

`problems/race_cars/closed_loop.py` owns the plant, the planner, warm starts and
lap logic; `build_solver(config, solver, oracle)` selects the optimizer and
generated function provider independently.
`problems/race_cars/casadi_nlp.py` is the CasADi mirror of the same NLP. The
harness code-generates the complete `nlpsol` with `expand=True` in a fresh
process, compiles it against Scaly's IPOPT, and calls it through the same C-level
boundary used for timing. Generated-code instrumentation fills the same
`SolverStats` fields as the Scaly column. The
`sqp+scaly` and `sqp+casadi` columns provide the controlled
one-solver, two-oracle comparison; their telemetry separates FE, QP, and
globalization time.

### Neural-process-MPC solvers and oracles

`problems/npmpc/closed_loop.py` owns the plant, the warm starts and the episode;
`problems/npmpc/casadi_nlp.py` is the CasADi mirror, built from the same
`ca_npmpc_pieces` the sweep kernels use and reading its bounds from the same two
functions as the Scaly formulation. The harness code-generates the complete
`nlpsol` with `expand=False` and links it to the same IPOPT library as the
Scaly column. The [current results](../docs/benchmarks/index.md) report the controlled comparison.

`problems/race_cars/reference.py` fits a minimum-curvature closed cubic spline to
the center line, samples it uniformly in arc length, and reads a constant-speed
reference horizon off it at every step, re-anchored to the car's projected arc
length. The reference implementation solves the spline-fitting QP with OSQP through
`qpsolvers`/`scipy.sparse`; because the constraints are all equalities, the port
solves the KKT system with `numpy.linalg.solve` instead and needs no new
dependency. It agrees with the OSQP fit to that solver's tolerance and satisfies
the continuity constraints several orders of magnitude more tightly.
