# Closed-loop control

This page compares complete controllers. Each [benchmark problem](index.md#the-problems) runs a
full simulated episode, solving one optimization problem per control step, once with Scaly's
generated functions and once with CasADi's. Everything else stays the same:

- **SQP** runs Scaly's sequential quadratic programming solver, with PIQP for the quadratic
  subproblems.
- **IPOPT** runs the same IPOPT 3.14.19 library on both sides.

Each provider runs one episode in a fresh process with an empty compilation cache. The step timers
measure native solver time and leave out Python, the plant simulation and building the controller.
CasADi uses SX for race cars and chain and MX for the two neural-network problems, the fastest
validated Hessian encoding at each problem's size in the [sweep](scalability.md). The benchmark has
no CasADi version of the chain controller with IPOPT.

## Where the time goes

Each pair of bars divides the mean time per control step into function evaluation, every
objective, constraint and derivative value the solver asks for, and the rest of the solver's work.
Each controller has its own time axis, and the number next to Scaly's bar says how many times
faster its step is on average.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_share.vl.json"></div>

The same solver does the same work on both sides, so the gray part stays about the same and the
step gains roughly what function evaluation loses. Function evaluation takes 97 to 98% of
CasADi's step on unbumpercars, which evaluates a neural network for every pair of cars, and the
step is 7.3 to 8.2× faster there. It takes 39 to 56% on neural-process MPC, where the step is 1.25
to 1.46× faster, and 5 to 8% on the race car, where the step gains 4 to 8% although function
evaluation is 2.6 to 3.1× faster. On the chain, CasADi's function evaluation takes 14% less time
than Scaly's, but it is under 3% of the step and the two steps take the same time to within 1%.

## Step times against the sampling time

A controller has to finish each step within its sampling time, the interval between two control
updates, given in each panel's title. The figure shows how the time per step is spread over the
episode, for function evaluation and for the whole solve, each on its own time axis. Each box
spans the middle half of the steps, with the median marked inside, and the whiskers reach the
fastest and the slowest step. A gray mark shows the sampling time where it is close enough to
the measured times to fit on the axis. The arrow joins the two medians, and its label is the median
of the per-step speedups, CasADi's time divided by Scaly's at the same step.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_times.vl.json"></div>

Every Scaly controller finishes all of its steps within the sampling time. CasADi's unbumpercars
controllers do not. With SQP, 48 of the 200 steps take longer than the 100 ms sampling time, and
with IPOPT, 186 of them do. On the other problems, no step on either side takes more than 55% of
the sampling time.

Both sides solve the same problem at every step, so the per-step speedup hardly changes along an
episode. For every controller, 80% of the steps lie within 7% of the median speedup, for function
evaluation and for the whole solve. This spread describes the episode, not measurement noise, as
the [fairness page](fairness.md#what-the-results-do-not-show) explains.

| Problem | Solver | Median step, ms | Slowest step, ms | Steps over the sampling time | Iterations per step |
| --- | --- | ---: | ---: | ---: | ---: |
| Race-car MPC | SQP | 2.23 / 2.32 | 6.87 / 7.03 | 0 / 0 | 2.7 |
| Race-car MPC | IPOPT | 4.51 / 4.88 | 9.84 / 14.12 | 0 / 0 | 10.4 |
| Neural-process MPC | SQP | 1.02 / 1.48 | 7.61 / 11.06 | 0 / 0 | 4.7 |
| Neural-process MPC | IPOPT | 3.48 / 4.34 | 6.75 / 10.53 | 0 / 0 | 10.1 |
| Chain of masses | SQP | 6.24 / 6.25 | 8.11 / 8.11 | 0 / 0 | 3.0 |
| Chain of masses | IPOPT | 6.56 | 10.90 | 0 | 7.4 |
| Unbumpercars | SQP | 6.73 / 55.33 | 79.0 / 669.1 | 0 / 48 | 7.9 |
| Unbumpercars | IPOPT | 19.04 / 139.46 | 41.4 / 297.5 | 0 / 186 | 13.7 |

Each cell gives Scaly's value, then CasADi's, except for the chain with IPOPT, which has only
Scaly. Iterations are the mean over the episode's steps.

## Do both sides do the same work?

Both sides should take the same solver iterations and ask for the same function evaluations at
every step, since only the source of the generated functions changes. They do for every
controller except unbumpercars with SQP, where three of the 200 steps take a different number of
iterations. The two trajectories still agree to within 7×10⁻⁹, but that controller's speedup
compares slightly different amounts of work.

## Building a controller

Before its first step, each controller constructs its problem, generates C for its functions and
compiles it, starting from an empty cache.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_build.vl.json"></div>

Scaly builds the race-car controllers in under 3 seconds, 3.3 and 6.2 times sooner than CasADi,
and the unbumpercars controllers 4.9 and 8.6 times sooner. The controllers compile with GCC at
`-O2`, so these times differ from the compile times on the
[scalability page](scalability.md#from-model-to-object-file), which use Clang at `-O3`. On the
chain, CasADi builds its SQP controller 2.6 seconds sooner than Scaly.
