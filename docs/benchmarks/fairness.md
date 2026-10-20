# Are the comparisons fair?

A speedup only means something if the two sides differ in the one thing being compared. Here
that is the generated code for the model functions. This page lists what the benchmarks hold
fixed, the machine they ran on, and what the results do not show.

## What both sides share

- **The same model.** Scaly and CasADi build the same formulation of each problem, and each
  timed Hessian is checked against an independent reference before it counts.
- **The same compiler and flags.** Both use the same optimization level and `-march=native`,
  because Scaly compiles on the machine it runs on. Both use glibc's vector math library, with
  no fast-math options that could reorder floating-point arithmetic.
- **The same solver.** Closed-loop pairs share the solver build, its settings, bounds, warm starts
  and tolerances. Only the provider of the model functions changes.
- **The same timing boundary.** The timers cover native code only. Python, plant simulation and
  controller construction are never inside them.
- **CasADi's best implementation.** CasADi is measured with several implementations, and each
  comparison uses the fastest one that compiled at that size, including where CasADi wins. Closed
  loops use the fastest validated Hessian implementation at their canonical size, SX for race cars
  and chain and MX for neural-process MPC and unbumpercars. The same implementation supplies every
  oracle.
- **Fresh processes and fixed CPU policy.** Each cell or provider runs once with an empty cache,
  one benchmark at a time. Order seed 0 fixes the backend order. The runner checks the
  `performance` governor and disabled boost state before and after each headline run.

## The machine

| | |
| --- | --- |
| Machine | Minisforum F7BSC desktop |
| CPU | AMD Ryzen 9 7940HS, 8 cores, boost disabled, `performance` governor |
| System | Ubuntu 24.04, glibc 2.39 |
| Compilers | GCC 13.3.0 for controllers, Clang 20.1.8 for the generated Hessians |
| Versions | Python 3.14.3, Scaly measured October 20, 2026, CasADi 3.8.0, PIQP 0.6.4 |

Absolute times depend on the machine. The results come from this one machine only and say
nothing about others, such as Apple silicon.

## What the results do not show

- **A missing CasADi time is not a speedup.** When no tested implementation compiles within 180
  seconds, the result is that Scaly compiled and those implementations did not. Another formulation
  of the same model might still compile.
- **Function-evaluation timers differ slightly in scope.** Scaly's includes a bounds function with
  no CasADi counterpart, and Scaly evaluates the objective and constraints together where CasADi
  evaluates them separately. These differences are part of what is measured.
- **One controller does not do identical work.** Unbumpercars with SQP takes a different number of
  iterations on two of 200 steps, as described on the [closed-loop page](closed_loop.md).
- **Step spread is not measurement noise.** One fresh process per cell shortens the study but
  provides no estimate of variation between runs. Closed-loop standard deviations and percentile
  ranges describe the changing workload along an episode. The paired per-step speedup compares
  matching steps, with the iteration mismatch above still part of the unbumpercars SQP result.
  `--repetitions N` retains the option to measure variation between processes.
- **Binary size is not normalized.** The two generators store constants and sparsity patterns in
  different forms, so binary size compares the files each one ships, not the information in them.
