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
network for every pair of cars, so there Scaly makes the whole step 7 to 10× faster. On the race
car, function evaluation was already a small part of the step, and the whole step gets only 7 to
8% faster.

## Measured times

| Problem | Solver | Scaly, ms | CasADi, ms | Speedup | Function-evaluation speedup |
| --- | --- | ---: | ---: | ---: | ---: |
| Race-car MPC | SQP | 2.47 | 2.67 | 1.08× | 5.4× |
| Race-car MPC | IPOPT | 4.60 | 4.92 | 1.07× | 2.6× |
| Neural-process MPC | SQP | 1.27 | 1.85 | 1.45× | 2.2× |
| Neural-process MPC | IPOPT | 3.53 | 4.46 | 1.26× | 1.8× |
| Chain of masses | SQP | 6.31 | 14.42 | 2.29× | 45× |
| Chain of masses | IPOPT | 6.85 | | | |
| Unbumpercars | SQP | 10.79 | 107.01 | 9.92× | 11.1× |
| Unbumpercars | IPOPT | 20.35 | 146.73 | 7.21× | 8.7× |

Times are per control step. The benchmark has no CasADi version of the chain controller with
IPOPT.

## Do both sides do the same work?

Both sides should take the same solver iterations and ask for the same function evaluations at
every step, since only the source of the generated functions changes. They do for every
controller except unbumpercars with SQP, where 15 of the 1,000 steps take a different number of
iterations. The two trajectories still agree to within 4×10⁻⁸, but that controller's speedup
compares slightly different amounts of work.
