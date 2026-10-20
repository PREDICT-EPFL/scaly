# Closed-loop control

This page compares complete controllers. Each [benchmark problem](index.md#the-problems) runs a
full simulated episode, solving one optimization problem per control step, once with Scaly's
generated functions and once with CasADi's. Everything else stays the same:

- **SQP** runs Scaly's sequential quadratic programming solver, with PIQP for the quadratic
  subproblems.
- **IPOPT** runs the same IPOPT 3.14.19 library on both sides.

Each provider runs one episode in a fresh process with an empty compilation cache. The step timers
measure native solver time and leave out Python, the plant simulation and building the controller.
CasADi uses SX for race cars at a 50-step horizon and for chain, and MX for the two neural-network
problems. These are the fastest validated Hessian implementations at each problem's size in the
[sweep](scalability.md). The benchmark has no CasADi version of the chain controller with IPOPT.

## Where the time goes

Each bar divides the mean time per control step into function evaluation, every objective,
constraint and derivative value the solver asks for, and the rest of the solver's work. Each
problem has its own time axis.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_share.vl.json"></div>

The same solver does the same work on both sides, so the gray part stays about the same and the
step gains roughly what function evaluation loses. Function evaluation takes 97 to 98% of
CasADi's step on unbumpercars, which evaluates a neural network for every pair of cars, and 39 to
56% on neural-process MPC. On the race car it takes only 5 to 8%, so Scaly's 2.7 to 2.8× faster
function evaluation barely changes the step. On the chain, CasADi's function evaluation takes 11%
less time than Scaly's, but it is under 3% of the step.

## Step times against the sampling time

A controller has to finish each step within its sampling time, the interval between two control
updates, given in each panel's title. The figure shows how the time per step is spread over the
episode, for function evaluation and for the whole solve, each on its own time axis. Each box
spans the middle half of the steps, with the median marked inside, and the whiskers reach the
fastest and the slowest step. A gray line shows the sampling time where it is close enough to
the measured times to fit on the axis.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_times.vl.json"></div>

Every Scaly controller finishes all of its steps within the sampling time. CasADi's unbumpercars
controllers do not. With SQP, 48 of the 200 steps take longer than the 100 ms sampling time, and
with IPOPT, 186 of them do. On the other problems, no step on either side takes more than 56% of
the sampling time.

Both sides solve the same problem at every step, so the per-step speedup, CasADi's time divided by
Scaly's at the same step, hardly changes along an episode. For every controller, 80% of the steps
lie within 7% of the median speedup, for function evaluation and for the whole solve. This spread
describes the episode, not measurement noise, as the
[fairness page](fairness.md#what-the-results-do-not-show) explains.

## Do both sides do the same work?

Both sides should take the same solver iterations and ask for the same function evaluations at
every step, since only the source of the generated functions changes. They do for every
controller except unbumpercars with SQP, where two of the 200 steps take a different number of
iterations. The two trajectories still agree to within 6×10⁻⁹, but that controller's speedup
compares slightly different amounts of work. On average, the SQP controllers take 2.7 to 7.9
iterations per step and the IPOPT ones 10.1 to 13.7.

## Building a controller

Before its first step, each controller constructs its problem, generates C for its functions and
compiles it, starting from an empty cache.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_build.vl.json"></div>

The controllers compile with GCC at `-O2`, so these times differ from the compile times on the
[scalability page](scalability.md#from-model-to-object-file), which use Clang at `-O3`.
