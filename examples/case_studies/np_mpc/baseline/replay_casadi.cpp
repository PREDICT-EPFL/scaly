// Replays a fixed sequence of OCP instances (replay_io.hpp) through the upstream C++ CasADi MPC:
// the casadi::Opti problem of MPCController::buildOptimization (controller.cpp), over FurutaNPMPC
// (neural) or FurutaMPC (equation), solved by IPOPT through Opti.
//
//   replay_casadi --method neural|equation --instances f.json --out o.json [--expand] [--jit] [--no-prepare]
//
// MPCController keeps its Opti and variables private, so the ~25 lines of buildOptimization are
// replicated here from the public MPCBase methods. The solver options are the upstream's
// configureSolver(problem, "ipopt") (solver.cpp) as published; --expand and --jit rebuild the same
// option dict with expand=true and/or jit=true (compiler "shell", jit_options flags -O3, the options
// solver.cpp carries switched off). Before the timed instances, one solve of instance 0 (prepare_ms)
// builds the NLP solver, as MPCController::prepare does before the upstream clients' closed loop;
// with --jit that is where the generated code is compiled. --no-prepare skips it, and the first
// step then carries that cost. Per instance: set_value(x0), set_initial(x, u, slack = 0), one
// solve() in try/catch (on failure, the last iterate from debug(), converged = false).
#include <chrono>
#include <cmath>
#include <iostream>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <casadi/casadi.hpp>

#include "npmpc/mpc/mpc_config_io.hpp"
#include "npmpc/mpc/problem/FurutaMPC.hpp"
#include "npmpc/mpc/problem/FurutaNPMPC.hpp"
#include "npmpc/mpc/solver.hpp"
#include "npmpc/nps/weights_io.hpp"

#include "replay_io.hpp"

namespace {

constexpr int N = 12; // must match horizon_steps in the config
constexpr int kNX = 4;

// configureSolver(problem, "ipopt") of solver.cpp with expand/jit as given.
void configureIpopt(casadi::Opti& problem, bool expand, bool jit)
{
    casadi::Dict jit_options = {{"flags",   "-O3"},
                                {"verbose", true}};
    problem.solver("ipopt", {
        {"ipopt.max_iter", 50},
        {"ipopt.tol", 1e-6},
        {"ipopt.print_level", 0},
        {"ipopt.sb", "yes"},
        {"print_time", 0},

        {"expand", expand},
        {"jit", jit},
        {"compiler", "shell"},
        {"jit_options", jit_options}
    });
}

std::vector<double> row(const casadi::DM& m, casadi_int r)
{
    std::vector<double> out;
    for (casadi_int c = 0; c < m.size2(); ++c) { out.push_back(static_cast<double>(m(r, c))); }
    return out;
}

} // namespace

int main(int argc, char** argv)
{
    using namespace std::chrono;
    try {
        const replay::Args args(argc, argv);
        const std::string method = args.value("--method");
        const bool expand = args.flag("--expand");
        const bool jit = args.flag("--jit");
        const bool prepare = !args.flag("--no-prepare");
        const replay::InstanceFile f = replay::loadInstances(args.value("--instances"), kNX, N);
        if (f.method != method) {
            throw std::invalid_argument("--method " + method + " but the instance file says " + f.method);
        }

        /* Model and Opti, as the upstream clients and MPCController build them */
        const steady_clock::time_point tSetup0 = steady_clock::now();
        npmpc::mpc::problem::MPCParams params = npmpc::mpc::loadMPCParamsFromYaml(f.mpcConfig);
        if (params.horizonSteps != N || params.xSize != kNX || params.uSize != 1) {
            throw std::runtime_error("the MPC config's dimensions differ from the compiled N = 12, NX = 4, NU = 1");
        }
        if (std::abs(params.dt - f.dt) > 1e-12) {
            throw std::runtime_error("dt in the instance file differs from the MPC config's");
        }
        std::optional<npmpc::nps::NeuralProcess> np;
        std::unique_ptr<npmpc::mpc::problem::MPCBase> mpc;
        if (method == "neural") {
            if (params.z.is_empty()) { throw std::runtime_error("no latent `z` in " + f.mpcConfig); }
            np.emplace(npmpc::nps::loadNeuralProcessFromYaml(f.weights));
            mpc = std::make_unique<npmpc::mpc::problem::FurutaNPMPC>(&*np, params.z, params.cost);
        } else if (method == "equation") {
            if (params.p.is_empty()) { throw std::runtime_error("no plant parameters `p` in " + f.mpcConfig); }
            mpc = std::make_unique<npmpc::mpc::problem::FurutaMPC>(params.p, params.cost);
        } else {
            throw std::invalid_argument("--method neural|equation");
        }

        // MPCController::buildOptimization.
        casadi::Opti problem;
        const int xSize = params.xSize;
        const int uSize = params.uSize;
        casadi::MX x = problem.variable(N + 1, xSize);
        casadi::MX u = problem.variable(N, uSize);
        casadi::MX slack = problem.variable(1, xSize);
        casadi::MX x0Param = problem.parameter(1, xSize);

        // Initial state.
        problem.subject_to(x(0, casadi::Slice()) <= x0Param + 1e-3);
        problem.subject_to(x(0, casadi::Slice()) >= x0Param - 1e-3);

        // Constraints.
        for (const auto& c : mpc->integrationConstraints(x, u, params.dt)) {
            problem.subject_to(c);
        }
        for (const auto& c : mpc->stateConstraints(x, slack, params.hardBound, params.slackBoundX)) {
            problem.subject_to(c);
        }
        for (const auto& c : mpc->inputConstraints(u, params.hardBound)) {
            problem.subject_to(c);
        }
        for (const auto& c : mpc->slackConstraints(slack, params.slackBoundX)) {
            problem.subject_to(c);
        }

        // Cost.
        casadi::MX cost = mpc->costFunction(x, u) + mpc->slackCost(slack, params.slackBoundX);
        problem.minimize(cost);

        if (!expand && !jit) {
            npmpc::mpc::configureSolver(problem, params.solver); // as published
        } else {
            if (params.solver != "ipopt") { throw std::runtime_error("--expand/--jit need solver: ipopt"); }
            configureIpopt(problem, expand, jit);
        }
        const double setupMs = duration<double, std::milli>(steady_clock::now() - tSetup0).count();

        auto solveInstance = [&](const replay::Instance& inst) {
            casadi::DM xGuess = casadi::DM::zeros(N + 1, xSize);
            casadi::DM uGuess = casadi::DM::zeros(N, uSize);
            for (int k = 0; k <= N; ++k) {
                for (int j = 0; j < xSize; ++j) { xGuess(k, j) = inst.xGuess[k][j]; }
            }
            for (int k = 0; k < N; ++k) { uGuess(k, 0) = inst.uGuess[k]; }
            problem.set_value(x0Param, casadi::DM(inst.x0).T());
            problem.set_initial(x, xGuess);
            problem.set_initial(u, uGuess);
            problem.set_initial(slack, casadi::DM::zeros(1, xSize));

            replay::Step step;
            casadi::DM xOpt, uOpt, sOpt;
            double obj = 0.0;
            const steady_clock::time_point t0 = steady_clock::now();
            try {
                casadi::OptiSol sol = problem.solve();
                step.solveMs = duration<double, std::milli>(steady_clock::now() - t0).count();
                step.converged = true;
                xOpt = sol.value(x);
                uOpt = sol.value(u);
                sOpt = sol.value(slack);
                obj = static_cast<double>(sol.value(problem.f()));
            } catch (const std::exception&) {
                step.solveMs = duration<double, std::milli>(steady_clock::now() - t0).count();
                step.converged = false;
                casadi::OptiAdvanced debug = problem.debug();
                xOpt = debug.value(x);
                uOpt = debug.value(u);
                sOpt = debug.value(slack);
                obj = static_cast<double>(debug.value(problem.f()));
            }
            try {
                const casadi::Dict stats = problem.stats();
                if (auto it = stats.find("iter_count"); it != stats.end()) { step.iter = static_cast<int>(it->second.to_int()); }
                if (auto it = stats.find("return_status"); it != stats.end()) { step.status = it->second.to_string(); }
                if (auto it = stats.find("t_wall_total"); it != stats.end()) { step.solverMs = 1e3 * it->second.to_double(); }
            } catch (const std::exception&) {
                // no stats available
            }
            for (int k = 0; k <= N; ++k) { step.x.push_back(row(xOpt, k)); }
            for (int k = 0; k < N; ++k) { step.u.push_back(static_cast<double>(uOpt(k, 0))); }
            step.slack = row(sOpt, 0);
            step.objective = obj;
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
        std::string implementation = "casadi-opti-ipopt";
        if (expand) { implementation += "-expand"; }
        if (jit) { implementation += "-jit"; }
        run.fields = {
            {"implementation", replay::jsonString(implementation)},
            {"ipopt", replay::jsonString("casadi-wheel IPOPT " + ipopt.version)},
            {"ipopt_library", replay::jsonString(ipopt.path)},
            {"casadi", replay::jsonString(casadi::CasadiMeta::version())},
            {"solver", replay::jsonString("IPOPT")},
            {"method", replay::jsonString(method)},
            {"expand", expand ? "true" : "false"},
            {"jit", jit ? "true" : "false"},
            {"setup_ms", replay::jsonNumber(setupMs)},
            {"prepare_ms", replay::jsonNumber(prepareMs)},
        };
        replay::writeRun(args.value("--out"), run);

        int converged = 0;
        double ms = 0.0;
        for (const auto& s : run.steps) { converged += s.converged; ms += s.solveMs; }
        std::cout << "replay_casadi " << implementation << " " << method << ": " << run.steps.size() << " instances, "
                  << converged << " converged, mean solve " << ms / run.steps.size() << " ms -> "
                  << args.value("--out") << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "replay_casadi: " << e.what() << std::endl;
        return 1;
    }
}
