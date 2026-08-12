# Benchmarks

This package owns Alloy's reproducible benchmark workloads and correctness-gated harness. Generated sources, binaries, logs, CSV files, and provenance live under `benchmarks/results/` and are not committed.

## Layout

- `problems/` contains importable chain-of-masses, race-car NMPC, and unbumpercars HCBF filter formulations.
- `harness/` contains Google Benchmark wrapper generation, dense-reference checks, sweep mechanics, and provenance capture.
- `run.py` is the entry point for CI smoke gates and scalability sweeps.

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
  Alloy-vs-CasADi trajectory agreement; for the chain of masses the dimensions and
  RK4 plant, the equality Jacobian against CasADi and a dense reference, the NLP
  objective against CasADi, the single end-mass reference behind both cost terms,
  the episode's shapes, and the recorded scene; for `unbumpercars` the
  Alloy-vs-CasADi oracle, the parameter-tail order, the pair barrier's relative
  degree, the discrete-MLP plant's three pieces and its reversed control order,
  both symbolic discrete-MLP prediction paths against a NumPy reference, per-step
  Alloy-vs-CasADi solution agreement on a binding state, and the opt-in
  exact-Hessian rollout. Gates needing IPOPT or CasADi report `skipped: ...`
  rather than passing silently.
- `benchmarks` — Python and compiled-C Jacobians against a dense reference,
  plus sparsity, workspace, and loop-preservation invariants.
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

Confirm a new gate can actually fail, by perturbing the thing it checks and
watching it fire. A gate that cannot fail is worse than no gate, because it reads
as coverage. Beware perturbations that are secretly no-ops: scaling an objective
does not move its argmin, so multiplying a cost by a constant will not disturb a
trajectory-agreement gate, while biasing one control term will.

## Sweeps

```bash
uv run python benchmarks/run.py sweep
uv run python benchmarks/run.py sweep --workloads race_cars --sizes 1,5,10,50 --backends alloy,casadi_sx
uv run python benchmarks/run.py sweep --out benchmarks/results/sweep/my-sweep.csv
```

Each `(workload, size, backend)` cell retains its generated C/header, raw float64 samples, wrapper, binary, and compile log next to the CSV, under `<csv-parent>/<workload>/<backend>_<axis><size>/`. With the default CSV this is `benchmarks/results/sweep/<workload>/`. Rows stream to CSV as cells finish; a sibling `.provenance.json` records the exact CLI, git state, package/compiler versions, platform, Python, and timestamp. After canonical closed-loop runs, the chain-of-masses M=5, race_cars N=40, and unbumpercars C=8 cells automatically consume their harvested `representative_fe_inputs.npz` rather than synthetic samples.

All benchmark artifacts follow the same command-first layout:

```text
benchmarks/results/
  closed-loop/<problem>/<backend>/  # episodes, plots, and harvested inputs
  sweep/<problem>/<cell>/           # generated code, binaries, samples, and logs
  sweep/scalability.csv             # default sweep table and provenance sidecar
  smoke/<problem>/<cell>/           # smoke-generated code, binaries, samples, and logs
  smoke/solver_call/                 # generated solver-call smoke artifacts
```

The doctrine is claims-first: broad sweeps establish scaling and canonical points support comparisons; correctness gates always run before speed is measured; every result carries enough provenance to reproduce it. See [ROADMAP.md §2](../ROADMAP.md#2-benchmark-suite) for the governing claim matrix.

The gates here guard the *measurements*, not the compiler. Op and composition coverage lives in `tests/` as small artificial cases checked against unrolled or NumPy references; a benchmark problem must never be the only thing exercising an IR, AD, or codegen path. That separation is what lets the problem set follow the workload roadmap without silently dropping compiler coverage.

The split runs both ways: a check that is about *a problem* rather than about Alloy belongs in that problem's `checks.py`, not in `tests/`, so the pytest suite never imports a benchmark problem. `race_cars` is the worked example — `problems/race_cars/checks.py` owns its formulation gates, and `tests/alloy/test_stage_transcription.py` carries a self-contained copy of the RK4 stage-transcription shape that problem surfaced, so retiring the problem cannot drop the compiler coverage. The chain-of-masses and unbumpercars problems follow the same shape; the IR behaviours they lean on are reproduced self-contained in `tests/alloy/test_alloy_sparsity.py` and `tests/alloy/test_factory_casadi.py`.

## Closed-loop episodes and Foxglove

Run a short end-to-end episode, including the generated Alloy/IPOPT path and MCAP recording:

```bash
uv run python benchmarks/run.py closed-loop --problem race_cars --smoke
```

Canonical runs omit `--smoke`:

```bash
uv run python benchmarks/run.py closed-loop --problem chain
uv run python benchmarks/run.py closed-loop --problem race_cars --backend alloy
uv run python benchmarks/run.py closed-loop --problem race_cars --backend casadi
uv run python benchmarks/run.py closed-loop --problem unbumpercars --backend alloy
```

`race_cars` and `unbumpercars` accept `--backend`, and write to
`benchmarks/results/closed-loop/<problem>/<backend>/` so the columns can coexist. For `race_cars` the two
backends solve a deliberately identical problem — same decision-variable and
parameter layout, same cost and constraint rows in the same order, same IPOPT
with the same options — so the only difference is who differentiates and
evaluates the oracles. The `race_cars/backends_agree` smoke gate holds them to
that with a cross-backend trajectory comparison, and `--backend alloy` is the
column the FE sweep harvests from.

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
| race cars | controller `N=40`, one lap of `fsds_competition_1` (340 m, 1367 plant steps at 0.05 s), alloy and casadi backends | track, cones, planar vehicle, reference and predicted horizons |
| unbumpercars HCBF | 8 cars, 200 plant steps at 0.1 s, seed 42, alloy and casadi backends | planar cars with body and keep-out rings |

The midpoint successful closed-loop oracle input is harvested for future
Google Benchmark cells. Closed-loop runs are manual-only; CI keeps using the
faster correctness and code-size smoke gates.

## Problems

| problem | scaling axis | reference |
|---|---|---|
| chain of masses | number of masses | laopt/acados chain-mass formulation; `M=5` is the canonical point |
| race cars | horizon | CasADi SX/MX; reference stages followed by symbolic vehicle parameters in `p` |
| unbumpercars HCBF filter | number of cars | CasADi MX; neural weights followed by symbolic vehicle parameters and `dt` |

### Race-car track data

`problems/race_cars/data/tracks/` holds four Formula Student Driverless Simulator
tracks, vendored as CSV from the `minimal_tracking_nmpc` reference implementation:
`center_line.csv` (`x,y,right_width,left_width`) and `cones.csv`
(`cone_type,X,Y,Z,std_*,right,left`). Each is a closed circuit 340–460 m long with
a corridor about 1.7 m wide per side.

### Race-car backends

`problems/race_cars/closed_loop.py` owns the plant, the planner, warm starts and
lap logic; `build_solver(config, backend)` swaps only the controller.
`problems/race_cars/casadi_nlp.py` is the CasADi mirror of the same NLP, wrapped
to present the same call signature and the same `SolverStats` as an Alloy
`SolverFunction`, with `expand=True` and per-oracle timings read out of
`nlpsol.stats()`. One canonical lap on this machine (Apple clock, IPOPT 3.14,
identical iteration counts and an identical trajectory to 1e-12):

| backend | mean total | mean FE | mean iters | RMS lateral error |
|---|---|---|---|---|
| alloy | 2.73 ms | 0.62 ms | 10.4 | 0.04403 m |
| casadi | 6.57 ms | 1.51 ms | 10.4 | 0.04403 m |

Treat these as an indicative single-machine reading, not a published claim: the
sweep harness is where the steelmanned per-cell numbers belong, and the
one-solver-two-oracles columns land properly with `alloy-sqp` in B4.

`problems/race_cars/reference.py` fits a minimum-curvature closed cubic spline to
the center line, samples it uniformly in arc length, and reads a constant-speed
reference horizon off it at every step, re-anchored to the car's projected arc
length. The reference implementation solves the spline-fitting QP with OSQP through
`qpsolvers`/`scipy.sparse`; because the constraints are all equalities, the port
solves the KKT system with `numpy.linalg.solve` instead and needs no new
dependency. It agrees with the OSQP fit to that solver's tolerance and satisfies
the continuity constraints several orders of magnitude more tightly.
