#pragma once

#include <stdint.h>

#ifndef ALLOY_SOLVER_STATS_DEFINED
#define ALLOY_SOLVER_STATS_DEFINED
#define ALLOY_SOLVER_STATS_VERSION 3
#define ALLOY_SOLVE_OK 0
#define ALLOY_SOLVE_ACCEPTABLE 1
#define ALLOY_SOLVE_MAX_ITER 2
#define ALLOY_SOLVE_PRIMAL_INFEASIBLE 3
#define ALLOY_SOLVE_DUAL_INFEASIBLE 4
#define ALLOY_SOLVE_NUMERICS 5
#define ALLOY_SOLVE_USER_STOP 6
#define ALLOY_SOLVE_ERROR 7
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
} alloy_solver_stats;
#endif

#ifndef ALLOY_SUCCESS
#define ALLOY_SUCCESS 0
#endif
#ifndef ALLOY_ERR_NULL_ABI
#define ALLOY_ERR_NULL_ABI 1
#endif
#ifndef ALLOY_ERR_NULL_WORK
#define ALLOY_ERR_NULL_WORK 2
#endif
#ifndef ALLOY_ERR_NULL_RESULT
#define ALLOY_ERR_NULL_RESULT 3
#endif
#ifndef ALLOY_ERR_NULL_INPUT
#define ALLOY_ERR_NULL_INPUT 4
#endif

#define qp_host_SZ_ARG 1
#define qp_host_SZ_RES 1
#define qp_host_SZ_IW 0
#define qp_host_SZ_W 0

// Universal CasADi-style ABI for qp_host.
#ifdef __cplusplus
extern "C" {
#endif
int qp_host(const double** arg, double** res, int* iw, double* w, void* mem);
int qp_host_sz_arg(void);
int qp_host_sz_res(void);
int qp_host_sz_iw(void);
int qp_host_sz_w(void);
void* qp_host_alloc_mem(void);
int qp_host_init_mem(void* mem);
void qp_host_free_mem(void* mem);
int corpus_qp_stats(alloy_solver_stats* out);
#ifdef __cplusplus
}
#endif

// Optional typed buffer wrappers for statically known shapes.
typedef struct { double data[2]; } qp_host_mu_in;
typedef struct { double data[1]; } qp_host_cost_out;
#ifdef __cplusplus
static_assert(sizeof(qp_host_mu_in) == sizeof(double) * 2, "qp_host_mu_in size mismatch");
static_assert(sizeof(qp_host_cost_out) == sizeof(double) * 1, "qp_host_cost_out size mismatch");
static inline int qp_host_call(const qp_host_mu_in& in_mu, qp_host_cost_out& out_cost) {
  double w[qp_host_SZ_W > 0 ? qp_host_SZ_W : 1];
  const double* arg[qp_host_SZ_ARG > 0 ? qp_host_SZ_ARG : 1] = {in_mu.data};
  double* res[qp_host_SZ_RES > 0 ? qp_host_SZ_RES : 1] = {out_cost.data};
  return qp_host(arg, res, nullptr, qp_host_SZ_W ? w : nullptr, nullptr);
}
#endif
