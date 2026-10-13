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

Each bar splits the step into function evaluation, the objective, constraint and derivative values
the solver asks for, and the rest of the solver's work. Each problem has its own time axis.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_breakdown.vl.json"></div>

Function evaluation dominates unbumpercars, which evaluates a neural network for every pair of
cars, so there the mean total solve time is 7.2 to 8.2× faster with Scaly. On the race car,
function evaluation is a small part of the step, and the mean total speedup is 1.03× with SQP and
1.08× with IPOPT. Chain's function evaluation is slightly faster with CasADi, and the mean total
solve times differ by less than 1%.

## Measured times

| Problem | Solver | Scaly, ms | CasADi, ms | Speedup | Function-evaluation speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Race-car MPC | SQP | 2.49 | 2.57 | 1.03× | 2.98× |
| Race-car MPC | IPOPT | 4.54 | 4.89 | 1.08× | 2.59× |
| Neural-process MPC | SQP | 1.27 | 1.84 | 1.45× | 2.18× |
| Neural-process MPC | IPOPT | 3.55 | 4.45 | 1.25× | 1.79× |
| Chain of masses | SQP | 6.41 | 6.39 | 1.00× | 0.88× |
| Chain of masses | IPOPT | 6.82 | | | |
| Unbumpercars | SQP | 10.72 | 87.36 | 8.15× | 9.11× |
| Unbumpercars | IPOPT | 20.24 | 146.51 | 7.24× | 8.71× |

Times are means over control steps. Speedups are ratios of those means, so they can differ from
the medians of the per-step ratios below. The benchmark has no CasADi version of the chain
controller with IPOPT.

## Variation along an episode

The median describes a typical step. The sample standard deviation describes how much step
times vary as the controller's workload changes along the episode. These are separate quantities,
shown in separate panels, and neither estimates variation between fresh processes. Unbumpercars
and neural-process MPC have larger spreads with SQP, whose iteration count changes along the
episode. Hover over a point to see its mean, median, standard deviation and number of steps.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_step_statistics.vl.json"></div>

The next figure pairs matching control steps and divides CasADi's time by Scaly's time at each
step. Points show the median ratio and lines span its 10th to 90th percentiles. They are not
confidence intervals. Ratios above one favor Scaly and ratios below one favor CasADi. Chain's
median total ratio is 0.99×. Unbumpercars SQP's is 8.10×, with a range of 7.55 to 8.39×.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_speedup.vl.json"></div>

## Do both sides do the same work?

Both sides should take the same solver iterations and ask for the same function evaluations at
every step, since only the source of the generated functions changes. They do for every
controller except unbumpercars with SQP, where three of the 200 steps take a different number of
iterations. The two trajectories still agree to within 7×10⁻⁹, but that controller's speedup
compares slightly different amounts of work.
