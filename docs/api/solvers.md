# Solvers

Declare an optimization problem with `problem` and `ProblemSpec`, then select an installed backend
with `solver`. The result is a `Solver` that you call with the problem parameters.
Use `x0=` for an initial guess or `warm=` for a previous result, and `.stats()`
to inspect the latest numerical solve. Its `.function` exposes the full input signature
for integrations that need a `Function`.
See the [solver guide](../guide/solvers.md) for a complete example and
[solver backends](../guide/solver_backends.md) for backend-specific options.

Problem construction and statistics are useful in application code. Plugin descriptors and graph
queries support solver integrations and compiler extensions.

## Problem construction

::: scaly.solvers.problem.Bounded

::: scaly.solvers.problem.bounded

::: scaly.solvers.problem.NO_LB

::: scaly.solvers.problem.NO_UB

::: scaly.solvers.problem.ProblemSpec

::: scaly.solvers.problem.Problem

::: scaly.solvers.problem.problem

## Solver selection

::: scaly.solvers.solver.solver

::: scaly.solvers.solver.Solver

::: scaly.solvers.paths.solver_loadable

::: scaly.solvers.paths.SolverLibraryError

::: scaly.solvers.qp.qp_problem

::: scaly.solvers.qp.QPData

::: scaly.solvers.qp.NotQuadratic

## Plugin interfaces

These are the names a solver plugin builds on. [Solver plugins](../dev/solver_plugins.md) explains
how they fit together.

::: scaly.solvers.registry.SolverBackend

::: scaly.solvers.registry.NlpSolverBackend

::: scaly.codegen.solver.SolverWrapperCtx

::: scaly.solvers.model.SolverDescriptor

::: scaly.solvers.model.ExternalOracle

::: scaly.solvers.model.descriptor_function

## Statistics

::: scaly.solvers.stats.SolverStats

::: scaly.solvers.stats.SolverStatus

::: scaly.solvers.stats.ScalySolveStatus

## Graph queries

::: scaly.solvers.graph.is_solver_function

::: scaly.solvers.graph.solver_callees

::: scaly.solvers.graph.solver_compile_flags
