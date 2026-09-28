#pragma once

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>

#ifndef SCALY_SOLVER_STATS_DEFINED
#define SCALY_SOLVER_STATS_DEFINED
#define SCALY_SOLVER_STATS_VERSION 3
#define SCALY_SOLVE_OK 0
#define SCALY_SOLVE_ACCEPTABLE 1
#define SCALY_SOLVE_MAX_ITER 2
#define SCALY_SOLVE_PRIMAL_INFEASIBLE 3
#define SCALY_SOLVE_DUAL_INFEASIBLE 4
#define SCALY_SOLVE_NUMERICS 5
#define SCALY_SOLVE_USER_STOP 6
#define SCALY_SOLVE_ERROR 7
typedef struct {
  int32_t version;
  int32_t status;
  int32_t native_status;
  int32_t iter;
  double obj;
  double t_total;
  double t_fe;
  double t_solver;
  double t_qp;
  double t_globalization;
  double t_glue;
  int32_t n_eval_f;
  int32_t n_eval_grad_f;
  int32_t n_eval_g;
  int32_t n_eval_jac_g;
  int32_t n_eval_h;
  int32_t _pad0;
  double primal_viol;
  double step_inf;
  double alpha;
  double merit_penalty;
  int32_t backtracks;
  int32_t qp_iter;
} scaly_solver_stats;
#endif

#ifndef SCALY_SUCCESS
#define SCALY_SUCCESS 0
#endif
#ifndef SCALY_ERR_NULL_ABI
#define SCALY_ERR_NULL_ABI 1
#endif
#ifndef SCALY_ERR_NULL_WORK
#define SCALY_ERR_NULL_WORK 2
#endif
#ifndef SCALY_ERR_NULL_RESULT
#define SCALY_ERR_NULL_RESULT 3
#endif
#ifndef SCALY_ERR_NULL_INPUT
#define SCALY_ERR_NULL_INPUT 4
#endif

#ifndef casadi_real
#define casadi_real double
#endif
#ifndef casadi_int
#define casadi_int long long int
#endif

#define qp_host_SZ_ARG 1
#define qp_host_SZ_RES 1
#define qp_host_SZ_IW 0
#define qp_host_SZ_W 0

#ifndef SCALY_BUFFER_HPP
#define SCALY_BUFFER_HPP
// A fixed-shape, row-major, 16-byte aligned buffer with inline storage: an aggregate to place on
// the stack, in a static, or behind make_unique. shape is the Expr shape; data is its flattening.
template <typename T, std::size_t... Ns>
struct alignas(16) Buffer {
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * std::size_t{1});
  T data[size > 0 ? size : 1];
  T* ptr() { return data; }
  const T* ptr() const { return data; }
  template <typename... I> T& operator()(I... idx) { return data[offset(idx...)]; }
  template <typename... I> const T& operator()(I... idx) const { return data[offset(idx...)]; }
  template <typename... I> static constexpr std::size_t offset(I... idx) {
    static_assert(sizeof...(I) == ndim, "Buffer: one index per dimension");
    const std::size_t index[] = {static_cast<std::size_t>(idx)..., 0};
    std::size_t off = 0;
    for (std::size_t k = 0; k < ndim; ++k) {
      assert(index[k] < shape[k]);
      off = off * shape[k] + index[k];
    }
    return off;
  }
};
#endif

namespace qp_host {
// The pointer ABI for qp_host; the same C symbols the C header declares.
extern "C" int qp_host(const double** arg, double** res, int* iw, double* w, int mem);
extern "C" int corpus_qp_stats(scaly_solver_stats* out);
extern "C" casadi_int qp_host_n_in(void);
extern "C" casadi_int qp_host_n_out(void);
extern "C" const char* qp_host_name_in(casadi_int i);
extern "C" const char* qp_host_name_out(casadi_int i);
extern "C" casadi_real qp_host_default_in(casadi_int i);
extern "C" const casadi_int* qp_host_sparsity_in(casadi_int i);
extern "C" const casadi_int* qp_host_sparsity_out(casadi_int i);
extern "C" int qp_host_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
extern "C" int qp_host_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
extern "C" int qp_host_checkout(void);
extern "C" void qp_host_release(int mem);
extern "C" void qp_host_incref(void);
extern "C" void qp_host_decref(void);

using mu_t = Buffer<double, 2>;
using cost_t = Buffer<double>;
using workspace_t = Buffer<double, qp_host_SZ_W>;
constexpr int sz_arg = qp_host_SZ_ARG;
constexpr int sz_res = qp_host_SZ_RES;
constexpr int sz_iw = qp_host_SZ_IW;
constexpr int sz_w = qp_host_SZ_W;

inline int call(const mu_t& mu, cost_t& cost, workspace_t& workspace) {
  const double* arg[sz_arg > 0 ? sz_arg : 1] = {mu.ptr()};
  double* res[sz_res > 0 ? sz_res : 1] = {cost.ptr()};
  return qp_host(arg, res, nullptr, sz_w ? workspace.ptr() : nullptr, 0);
}
}  // namespace qp_host
