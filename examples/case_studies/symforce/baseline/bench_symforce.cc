/* The SymForce side of the SymForce study: robot 3D localization, 5 poses and 20 landmarks.
 *
 *   bench_symforce <out.json>
 *
 * Compiled with -DNO_FIXED for sizes whose fixed-size code SymForce was not given the time to generate.
 * Built against SymForce's own C++ optimizer and the example's checked-in generated code
 * (symforce/examples/robot_3d_localization) by baseline/setup.sh. For each of SymForce's two variants
 * (dynamic: one generated function per factor type; fixed: one generated linearization of the whole
 * problem), it times what symforce/benchmarks/robot_3d_localization times, with the same calls:
 *   linearize: Linearizer::Relinearize at the initial values;
 *   iterate:   Optimizer::Optimize(values, 1 iteration) on values that each call leaves updated;
 * and, beside them, the whole solve: Optimize from the initial values to SymForce's stopping test.
 * The benchmark's loop of 1000 calls is kept; each figure is the best of 20 such loops, in
 * microseconds per call. The whole solve's per-iteration trace is written for comparison.
 */
#include <chrono>
#include <cstdio>
#include <fstream>
#include <vector>

#include <symforce/examples/robot_3d_localization/common.h>
#include <symforce/examples/robot_3d_localization/run_dynamic_size.h>
#ifndef NO_FIXED
#include <symforce/examples/robot_3d_localization/run_fixed_size.h>
#endif
#include <symforce/opt/factor.h>
#include <symforce/opt/optimizer.h>

using namespace robot_3d_localization;
using Clock = std::chrono::steady_clock;

template <typename F>
double best_us(F&& f, int calls = 1000, int loops = 20) {
  double best = 1e300;
  for (int l = 0; l < loops; ++l) {
    auto t0 = Clock::now();
    for (int i = 0; i < calls; ++i) f();
    double us = std::chrono::duration<double, std::micro>(Clock::now() - t0).count() / calls;
    best = std::min(best, us);
  }
  return best;
}

struct Result {
  double linearize, iterate, solve;
  std::vector<sym::optimization_iteration_t> trace;
  double final_error;
  int iterations;
  std::vector<double> poses;
};

Result run(const std::vector<sym::Factor<double>>& factors, const char* name) {
  Result r{};
  {
    sym::Values<double> values = BuildValues<double>(kNumPoses, kNumLandmarks);
    sym::Optimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors, name);
    sym::Linearizer<double>& linearizer = optimizer.Linearizer();
    sym::SparseLinearization<double> linearization;
    r.linearize = best_us([&] { linearizer.Relinearize(values, linearization); });
  }
  {
    sym::Values<double> values = BuildValues<double>(kNumPoses, kNumLandmarks);
    sym::Optimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors, name);
    typename sym::Optimizer<double>::Stats stats;
    r.iterate = best_us([&] { optimizer.Optimize(values, 1, false, stats); });
  }
  {
    const sym::Values<double> initial = BuildValues<double>(kNumPoses, kNumLandmarks);
    sym::Optimizer<double> optimizer(RobotLocalizationOptimizerParams(), factors, name);
    typename sym::Optimizer<double>::Stats stats;
    sym::Values<double> values;
    r.solve = best_us([&] { values = initial; optimizer.Optimize(values, -1, false, stats); }, 100, 20);
    values = initial;
    optimizer.Optimize(values, -1, false, stats);
    r.trace = stats.iterations;
    r.iterations = static_cast<int>(stats.iterations.size()) - 1;
    r.final_error = stats.iterations.back().new_error;
    for (int i = 0; i < kNumPoses; ++i) {
      const auto pose = values.At<sym::Pose3<double>>(sym::Keys::WORLD_T_BODY.WithSuper(i));
      for (int k = 0; k < 7; ++k) r.poses.push_back(pose.Data()[k]);
    }
  }
  return r;
}

void write(std::ofstream& out, const char* name, const Result& r, bool last) {
  out << "\"" << name << "\": {\"linearize_us\": " << r.linearize << ", \"iterate_us\": " << r.iterate
      << ", \"solve_us\": " << r.solve << ", \"iterations\": " << r.iterations << ", \"final_error\": ";
  char buf[64];
  std::snprintf(buf, sizeof buf, "%.17g", r.final_error);
  out << buf << ", \"trace\": [";
  for (size_t i = 0; i < r.trace.size(); ++i) {
    const auto& it = r.trace[i];
    std::snprintf(buf, sizeof buf, "%.17g", it.new_error);
    char lam[64];
    std::snprintf(lam, sizeof lam, "%.17g", it.current_lambda);
    out << (i ? ", " : "") << "{\"iteration\": " << it.iteration << ", \"lambda\": " << lam
        << ", \"new_error\": " << buf << ", \"accepted\": " << (it.update_accepted ? "true" : "false") << "}";
  }
  out << "], \"poses\": [";
  for (size_t i = 0; i < r.poses.size(); ++i) {
    std::snprintf(buf, sizeof buf, "%.17g", r.poses[i]);
    out << (i ? ", " : "") << buf;
  }
  out << "]}" << (last ? "\n" : ",\n");
}

int main(int argc, char** argv) {
  const Result dynamic = run(BuildDynamicFactors<double>(kNumPoses, kNumLandmarks), "dynamic");
  std::ofstream out(argc > 1 ? argv[1] : "symforce.json");
  out << "{\n";
#ifndef NO_FIXED
  write(out, "dynamic", dynamic, false);
  const Result fixed = run({BuildFixedFactor<double>()}, "fixed");
  write(out, "fixed", fixed, true);
  std::printf("fixed:   linearize %.2f us, iterate %.2f us, solve %.1f us in %d iterations\n", fixed.linearize, fixed.iterate, fixed.solve, fixed.iterations);
#else
  write(out, "dynamic", dynamic, true);
#endif
  out << "}\n";
  std::printf("dynamic: linearize %.2f us, iterate %.2f us, solve %.1f us in %d iterations\n", dynamic.linearize, dynamic.iterate, dynamic.solve, dynamic.iterations);
  return 0;
}
