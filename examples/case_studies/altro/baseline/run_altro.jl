# The Altro.jl side of the ALTRO case study: the IROS 2019 paper's problems on current Altro.jl (0.5).
#
#   julia --project=<env> run_altro.jl <problem> <projected_newton> <constraint_tolerance> <out.json>
#
# The problems are the paper-era definitions from TrajectoryOptimization.jl at 320dbaca (2019-07-18,
# `problems/parallel_park.jl`, `problems/cartpole.jl`), ported to the current API: the same models
# (RobotZoo's DubinsCar and Cartpole are the 2019 `car_model` and `cartpole_model`), the same explicit
# RK3, horizons, weights, bounds, goal constraints and initial controls. The 2019 cost multiplied every
# stage term by dt; the current one does not, so Q and R are scaled by dt here. Timing is BenchmarkTools
# over cold solves, each restarted from the initial trajectory (as Altro's `benchmark_solve!` does when
# called before the first solve). The first solve, which compiles, and the package load are timed apart.

const T0 = time()
using Altro, TrajectoryOptimization, RobotDynamics, RobotZoo, StaticArrays, LinearAlgebra, BenchmarkTools, JSON
const TO = TrajectoryOptimization
const RD = RobotDynamics
const T_LOAD = time() - T0

function parallel_park()
    model = RobotZoo.DubinsCar()
    dmodel = RD.DiscretizedDynamics{RD.RK3}(model)
    n, m = RD.dims(model)
    N, dt = 51, 0.06
    x0, xf = SA[0.0, 0.0, 0.0], SA[0.0, 1.0, 0.0]
    Q, R, Qf = 1e-2 * dt * Diagonal(@SVector ones(n)), 1e-2 * dt * Diagonal(@SVector ones(m)), 100.0 * Diagonal(@SVector ones(n))
    obj = LQRObjective(Q, R, Qf, xf, N)
    cons = ConstraintList(n, m, N)
    add_constraint!(cons, BoundConstraint(n, m, u_min=-2.0, u_max=2.0), 1:1)
    add_constraint!(cons, BoundConstraint(n, m, x_min=[-0.25, -0.001, -Inf], x_max=[0.25, 1.001, Inf], u_min=-2.0, u_max=2.0), 2:N-1)
    add_constraint!(cons, GoalConstraint(xf), N)
    prob = Problem(dmodel, obj, x0, (N - 1) * dt; xf=xf, constraints=cons)
    initial_controls!(prob, [@SVector ones(m) for k = 1:N-1])
    rollout!(prob)
    prob, Dict(:penalty_scaling => 10.0)
end

function cartpole()
    model = RobotZoo.Cartpole(1.0, 0.2, 0.5, 9.81)
    dmodel = RD.DiscretizedDynamics{RD.RK3}(model)
    n, m = RD.dims(model)
    N, tf = 101, 5.0
    dt = tf / (N - 1)
    x0, xf = SA[0.0, 0.0, 0.0, 0.0], SA[0.0, pi, 0.0, 0.0]
    Q, R, Qf = 1e-2 * dt * Diagonal(@SVector ones(n)), 1e-1 * dt * Diagonal(@SVector ones(m)), 100.0 * Diagonal(@SVector ones(n))
    obj = LQRObjective(Q, R, Qf, xf, N)
    cons = ConstraintList(n, m, N)
    add_constraint!(cons, BoundConstraint(n, m, u_min=-3.0, u_max=3.0), 1:N-1)
    add_constraint!(cons, GoalConstraint(xf), N)
    prob = Problem(dmodel, obj, x0, tf; xf=xf, constraints=cons)
    initial_controls!(prob, [@SVector fill(0.01, m) for k = 1:N-1])
    rollout!(prob)
    prob, Dict{Symbol,Float64}()
end

const PROBLEMS = Dict("parallel_park" => parallel_park, "cartpole" => cartpole)

function main(args)
    name, pn, tol, out = args[1], parse(Bool, args[2]), parse(Float64, args[3]), args[4]
    t = time()
    prob, extra = PROBLEMS[name]()
    opts = SolverOptions(; projected_newton=pn, constraint_tolerance=tol, projected_newton_tolerance=1e-4, show_summary=false, verbose=0, extra...)
    solver = ALTROSolver(prob, opts)
    t_setup = time() - t
    Z0 = deepcopy(TO.get_trajectory(solver))  # the initial trajectory: every timed solve starts from it
    t_first = @elapsed solve!(solver)  # compiles, and is the solve whose result is recorded
    X, U = deepcopy(states(solver)), deepcopy(controls(solver))
    # max_violation(::ALTROSolver) also reads the projected-Newton solver, whose trajectory is never
    # updated when that phase is off; the augmented-Lagrangian solver's own violation is the one to report.
    st, it, J = status(solver), iterations(solver), cost(solver)
    viol = pn ? max_violation(solver) : max_violation(solver.solver_al)
    # Altro's benchmark_solve! restarts from the trajectory it finds, which after a solve is the
    # solution; this restarts every sample from the initial trajectory, as a cold solve.
    b = @benchmark begin
        TO.initial_trajectory!($solver, $Z0)
        solve!($solver)
    end samples = 50 evals = 1
    iterations(solver) == it || error("a benchmarked solve took $(iterations(solver)) iterations, the recorded one $it")
    res = Dict(
        "problem" => name,
        "projected_newton" => pn,
        "constraint_tolerance" => tol,
        "status" => string(st),
        "iterations" => it,
        "cost" => J,
        "max_violation" => viol,
        "time_min" => minimum(b).time * 1e-9,
        "time_median" => median(b).time * 1e-9,
        "allocs" => minimum(b).allocs,
        "t_load" => T_LOAD,
        "t_setup" => t_setup,
        "t_first_solve" => t_first,
        "X" => [Vector(x) for x in X],
        "U" => [Vector(u) for u in U],
        "julia" => string(VERSION),
    )
    open(out, "w") do f
        JSON.print(f, res)
    end
    println(JSON.json(Dict(k => v for (k, v) in res if !(k in ("X", "U")))))
end

main(ARGS)
