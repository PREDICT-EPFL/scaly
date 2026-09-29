// Replays a fixed sequence of OCP instances (replay_io.hpp) through the upstream laOPT Furuta MPC:
// FurutaNPOCP<12> (neural) or FurutaEqOCP<12> (equation), transcribed by MultipleShootingXDiff and
// solved with the upstream's configureSolver/solveOnce (laopt_solver.hpp): IPOPT (tol 1e-6,
// max_iter 50) or SQP with PIQP (max_iter 15, laOPT defaults otherwise).
//
//   replay_laopt --solver ipopt|sqp --method neural|equation --instances f.json --out o.json
//                [--no-prepare] [--carry-duals]
//
// As in run_furuta_*_laopt_client.cpp, the problem is taped once and reused. Per instance: the
// initial-state band from x0 (set_initial_state), the primal guess from x_guess/u_guess with zero
// slack, one solve() call, no retries. Before the instances, one solve of instance 0 outside the
// recorded steps (prepare_ms), like the upstream CasADi clients' warm-up; --no-prepare skips it. The SQP
// solver keeps its multipliers between solve() calls; they are zeroed before each instance so that
// every instance starts from its own guess alone (as a freshly constructed solver, as in
// eval_furuta_*_laopt), unless --carry-duals. IPOPT starts from the primal guess alone either way.
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "laopt/laopt.hpp"
#include "npmpc/mpc/laopt/multiple_shooting_xdiff.hpp"

#include "npmpc/mpc/laopt/FurutaEqOcpEigen.hpp"
#include "npmpc/mpc/laopt/FurutaNPOcpEigen.hpp"
#include "npmpc/mpc/laopt/laopt_solver.hpp"

#include "IpoptConfig.h"
#include "replay_io.hpp"

#ifndef NPMPC_IPOPT_BUILD
#define NPMPC_IPOPT_BUILD "unknown"
#endif

namespace {

using namespace npmpc::mpc::furuta_laopt;

constexpr int N = 12; // must match horizon_steps in the config

// The two OCPs with their transcriptions, as in eval_furuta_np_laopt.cpp / eval_furuta_eq_laopt.cpp.
struct Neural {
    using Ocp = FurutaNPOCP<N>;
    using Transcription = laopt_tools::MultipleShootingXDiff<Ocp, N, laopt::ERK4>; // integrator unused (DiscreteDynamics)
    static constexpr double kLagrangeWeight = 1.0; // discrete dynamics: Lagrange terms summed unweighted
    static std::shared_ptr<Ocp> make(const replay::InstanceFile& f) { return std::make_shared<Ocp>(f.weights, f.mpcConfig); }
};
struct Equation {
    using Ocp = FurutaEqOCP<N>;
    using Transcription = laopt_tools::MultipleShootingXDiff<Ocp, N, laopt::IRK2>; // implicit midpoint, as in Python
    static constexpr double kLagrangeWeight = 1.0 / N; // continuous dynamics: h * Lagrange, h = 1/N
    static std::shared_ptr<Ocp> make(const replay::InstanceFile& f) { return std::make_shared<Ocp>(f.mpcConfig); }
};

// ocpObjective of eval_furuta_{np,eq}_laopt.cpp: the laOPT objective as MultipleShootingXDiff assembles
// it, sum_k w * lagrange(x_k, x_{k+1}, u_k) + mayer (terminal and slack cost).
template<typename M>
double ocpObjective(typename M::Ocp& ocp, const typename M::Transcription::StateTrajectory& x,
                    const typename M::Transcription::InputTrajectory& u, const typename M::Ocp::Param& p)
{
    using Ocp = typename M::Ocp;
    const Eigen::Vector<double, 1> t0 = Eigen::Vector<double, 1>::Constant(ocp.t0);
    const Eigen::Vector<double, 1> tf = Eigen::Vector<double, 1>::Constant(ocp.tf_lb);
    double obj = 0.0;
    for (int k = 0; k < N; ++k) {
        obj += M::kLagrangeWeight * ocp.lagrange_term_impl(typename Ocp::State(x.col(k)), typename Ocp::State(x.col(k + 1)),
                                                           typename Ocp::Input(u.col(k)), p, t0, tf, double(k) / N);
    }
    return obj + ocp.mayer_term_impl(typename Ocp::State(x.col(N)), p, t0, tf);
}

template<typename M, SolverType S>
replay::Run replayInstances(const replay::InstanceFile& f, bool prepare, bool carryDuals)
{
    using Ocp = typename M::Ocp;
    using Transcription = typename M::Transcription;
    using OptProblem = laopt::Problem<Transcription>;
    using Solver = SolverFor<S, OptProblem>;
    using StateTrajectory = typename Transcription::StateTrajectory; // (NX, N+1)
    using InputTrajectory = typename Transcription::InputTrajectory; // (NU, N)
    using namespace std::chrono;

    /* Build OCP, transcription, problem (tape) and solver once, as the upstream clients do */
    const steady_clock::time_point tSetup0 = steady_clock::now();
    std::shared_ptr<Ocp> ocp = M::make(f);
    std::shared_ptr<Transcription> transcription = std::make_shared<Transcription>(ocp);
    std::shared_ptr<OptProblem> optProblem = std::make_shared<OptProblem>(transcription); // generates the tape
    Solver solver(optProblem);
    configureSolver(solver);
    const double setupMs = duration<double, std::milli>(steady_clock::now() - tSetup0).count();
    if (std::abs(ocp->settings.dt - f.dt) > 1e-12) {
        throw std::runtime_error("dt in the instance file differs from the MPC config's");
    }

    auto solveInstance = [&](const replay::Instance& inst) {
        typename Ocp::State x0;
        StateTrajectory xGuess;
        InputTrajectory uGuess;
        for (int j = 0; j < kNX; ++j) { x0(j) = inst.x0[j]; }
        for (int k = 0; k <= N; ++k) {
            for (int j = 0; j < kNX; ++j) { xGuess(j, k) = inst.xGuess[k][j]; }
        }
        for (int k = 0; k < N; ++k) { uGuess(0, k) = inst.uGuess[k]; }

        ocp->set_initial_state(x0);
        // Guesses must be set after the solver is constructed (see MultipleShooting::set_X_guess).
        transcription->set_X_guess(xGuess);
        transcription->set_U_guess(uGuess);
        transcription->set_p_guess(Ocp::Param::Zero());
        if constexpr (!kIsIpopt<Solver>) {
            if (!carryDuals) {
                solver.set_initial_dual(Eigen::VectorXd::Zero(solver.dual().size()));
                solver.set_initial_dual_bounds(Eigen::VectorXd::Zero(solver.dual_bounds().size()));
            }
        }

        replay::Step step;
        const steady_clock::time_point t0 = steady_clock::now();
        const SolveResult result = solveOnce(solver);
        step.solveMs = duration<double, std::milli>(steady_clock::now() - t0).count();
        step.converged = result.converged;
        step.status = result.status;
        if constexpr (kIsIpopt<Solver>) {
            const Ipopt::SmartPtr<Ipopt::SolveStatistics> stats = solver.ipopt_application->Statistics();
            step.iter = Ipopt::IsValid(stats) ? stats->IterationCount() : -1;
        } else {
            step.iter = solver.info().iter;
            step.qpIter = solver.info().qp_iter;
        }

        const StateTrajectory x = transcription->get_X_opt();
        const InputTrajectory u = transcription->get_U_opt();
        const typename Ocp::Param p = transcription->get_p_opt();
        for (int k = 0; k <= N; ++k) { step.x.emplace_back(x.col(k).data(), x.col(k).data() + kNX); }
        step.u.assign(u.data(), u.data() + N);
        step.slack.assign(p.data(), p.data() + kNP);
        step.objective = ocpObjective<M>(*ocp, x, u, p);
        return step;
    };

    replay::Run run;
    double prepareMs = std::nan(""); // written as null when skipped
    if (prepare) {
        const steady_clock::time_point t0 = steady_clock::now();
        solveInstance(f.instances.front());
        prepareMs = duration<double, std::milli>(steady_clock::now() - t0).count();
    }
    for (const replay::Instance& inst : f.instances) { run.steps.push_back(solveInstance(inst)); }

    const replay::LoadedIpopt ipopt = replay::loadedIpopt();
    const bool ipoptSolver = kIsIpopt<Solver>;
    run.fields = {
        {"implementation", replay::jsonString(ipoptSolver ? "laopt-ipopt" : "laopt-sqp-piqp")},
        {"ipopt", replay::jsonString(std::string(NPMPC_IPOPT_BUILD) + " IPOPT " + ipopt.version +
                                     (ipoptSolver ? "" : " (linked, unused by SQP)"))},
        {"ipopt_library", replay::jsonString(ipopt.path)},
        {"ipopt_headers", replay::jsonString(IPOPT_VERSION)},
        {"solver", replay::jsonString(solverName<Solver>())},
        {"method", replay::jsonString(f.method)},
        {"sqp_duals", replay::jsonString(ipoptSolver ? "n/a" : (carryDuals ? "carried" : "zeroed"))},
        {"setup_ms", replay::jsonNumber(setupMs)},
        {"prepare_ms", replay::jsonNumber(prepareMs)},
    };
    return run;
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const replay::Args args(argc, argv);
        const std::string solver = args.value("--solver");
        const std::string method = args.value("--method");
        const replay::InstanceFile f = replay::loadInstances(args.value("--instances"), kNX, N);
        if (f.method != method) {
            throw std::invalid_argument("--method " + method + " but the instance file says " + f.method);
        }
        const bool prepare = !args.flag("--no-prepare");
        const bool carryDuals = args.flag("--carry-duals");

        replay::Run run;
        if (method == "neural" && solver == "ipopt") { run = replayInstances<Neural, SolverType::IPOPT>(f, prepare, carryDuals); }
        else if (method == "neural" && solver == "sqp") { run = replayInstances<Neural, SolverType::SQP_PIQP>(f, prepare, carryDuals); }
        else if (method == "equation" && solver == "ipopt") { run = replayInstances<Equation, SolverType::IPOPT>(f, prepare, carryDuals); }
        else if (method == "equation" && solver == "sqp") { run = replayInstances<Equation, SolverType::SQP_PIQP>(f, prepare, carryDuals); }
        else { throw std::invalid_argument("--solver ipopt|sqp, --method neural|equation"); }

        replay::writeRun(args.value("--out"), run);
        int converged = 0;
        double ms = 0.0;
        for (const auto& s : run.steps) { converged += s.converged; ms += s.solveMs; }
        std::cout << "replay_laopt " << solver << " " << method << ": " << run.steps.size() << " instances, "
                  << converged << " converged, mean solve " << ms / run.steps.size() << " ms -> "
                  << args.value("--out") << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "replay_laopt: " << e.what() << std::endl;
        return 1;
    }
}
