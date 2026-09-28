# Closed-loop control

This page compares complete controllers. Each [benchmark problem](index.md#the-problems) runs a
full simulated episode, solving one optimization problem per control step, once with Scaly's
generated functions and once with CasADi's. Everything else stays the same:

- **SQP** runs Scaly's sequential quadratic programming solver, with PIQP for the quadratic
  subproblems.
- **IPOPT** runs the same IPOPT 3.14.19 library on both sides.

The time per step is the solver's own time, averaged over every step of the episode and over five
fresh processes. It leaves out Python, the plant simulation and building the controller.

## Where the time goes

Each bar splits the step into function evaluation, the objective, constraint and derivative values
the solver asks for, and the rest of the solver's work. Each problem has its own time axis.

<div class="vega-chart" data-spec="../../assets/benchmarks/closed_loop_breakdown.vl.json"></div>

The rest of the solver takes about the same time on both sides, as it should, since it is the same
code. What changes is function evaluation. It dominates unbumpercars, which evaluates a neural
network for every pair of cars, so there Scaly makes the whole step 7 to 9× faster. On the race
car, function evaluation was already a small part of the step, and the whole step gets only 5 to
7% faster.

## Measured times

| Problem | Solver | Scaly, ms | CasADi, ms | Speedup | Function-evaluation speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Race-car MPC | SQP | 2.49 | 2.67 | 1.07× | 4.2× |
| Race-car MPC | IPOPT | 4.71 | 4.96 | 1.05× | 2.1× |
| Neural-process MPC | SQP | 1.27 | 1.84 | 1.45× | 2.2× |
| Neural-process MPC | IPOPT | 3.61 | 4.48 | 1.24× | 1.8× |
| Chain of masses | SQP | 6.27 | 14.43 | 2.30× | 53× |
| Chain of masses | IPOPT | 6.78 | | | |
| Unbumpercars | SQP | 11.33 | 106.58 | 9.41× | 10.5× |
| Unbumpercars | IPOPT | 21.11 | 146.56 | 6.94× | 8.3× |

Times are per control step. The benchmark has no CasADi version of the chain controller with
IPOPT.

## Do both sides do the same work?

Both sides should take the same solver iterations and ask for the same function evaluations at
every step, since only the source of the generated functions changes. They do for every
controller except unbumpercars with SQP, where 5 of the 1,000 steps take a different number of
iterations. The two trajectories still agree to within 5×10⁻⁹, but that controller's speedup
compares slightly different amounts of work.
