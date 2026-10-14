# Closed-loop control

This page compares complete controllers. Each [benchmark problem](index.md#the-problems) runs a
full simulated episode, solving one optimization problem per control step, once with Scaly's
generated functions and once with CasADi's. Everything else stays the same:

- **SQP** runs Scaly's sequential quadratic programming solver, with PIQP for the quadratic
  subproblems.
- **IPOPT** runs the same IPOPT 3.14.19 library on both sides.

Each provider runs one episode in a fresh process with an empty compilation cache. The timers
measure native solver time and leave out Python, the plant simulation and building the controller.
CasADi uses SX for race cars and chain and MX for the two neural-network problems, the fastest
validated Hessian encoding at each canonical size in the [sweep](scalability.md).

## Where the time goes

Each bar splits the mean step into function evaluation, the objective, constraint and derivative
values the solver asks for, and the rest of the solver's work. Each problem has its own time axis.
Hover over a bar to see the median and standard deviation of the step time and the median of the
paired per-step speedups.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_breakdown.vl.json"></div>

Function evaluation dominates unbumpercars, which evaluates a neural network for every pair of
cars, so there the mean total solve time is 7.2 to 8.2× faster with Scaly. On the race car,
function evaluation is a small part of the step, and the mean total speedup is 1.03× with SQP and
1.08× with IPOPT. Chain's function evaluation is slightly faster with CasADi, and the mean total
solve times differ by less than 1%.

## Measured times

| Problem | Solver | Scaly, ms | CasADi, ms | Speedup | Function-evaluation speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Race-car MPC | SQP | 2.49 | 2.57 | 1.03× | 2.95× |
| Race-car MPC | IPOPT | 4.54 | 4.89 | 1.08× | 2.59× |
| Neural-process MPC | SQP | 1.27 | 1.84 | 1.45× | 2.19× |
| Neural-process MPC | IPOPT | 3.55 | 4.45 | 1.25× | 1.79× |
| Chain of masses | SQP | 6.41 | 6.39 | 1.00× | 0.88× |
| Chain of masses | IPOPT | 6.82 | | | |
| Unbumpercars | SQP | 10.72 | 87.36 | 8.15× | 9.11× |
| Unbumpercars | IPOPT | 20.24 | 146.51 | 7.24× | 8.71× |

Times are means over the control steps of one episode, and speedups are ratios of those means.
The [overview](index.md#closed-loop-control) reports the median of the speedups between matching
steps instead, so its figures differ slightly. The benchmark has no CasADi version of the chain
controller with IPOPT.

Step times vary along an episode because the solver's work changes from one step to the next.
Relative to their means, unbumpercars and neural-process MPC with SQP vary most, since their
iteration count changes along the episode. With one process per controller, neither this spread
nor the overview's percentile ranges estimate the variation between runs, as the
[fairness page](fairness.md#what-the-results-do-not-show) explains.

## Do both sides do the same work?

Both sides should take the same solver iterations and ask for the same function evaluations at
every step, since only the source of the generated functions changes. They do for every
controller except unbumpercars with SQP, where three of the 200 steps take a different number of
iterations. The two trajectories still agree to within 7×10⁻⁹, but that controller's speedup
compares slightly different amounts of work.
