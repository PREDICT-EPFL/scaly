#pragma once

#include <stddef.h>
#include <stdint.h>

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

#ifndef SCALY_ALIGNAS
#ifdef __cplusplus
#define SCALY_ALIGNAS(n) alignas(n)
#else
#define SCALY_ALIGNAS(n) _Alignas(n)
#endif
#endif

#define qp_host_SZ_ARG 1
#define qp_host_SZ_RES 1
#define qp_host_SZ_IW 0
#define qp_host_SZ_W 0

// The pointer ABI for qp_host.
#ifdef __cplusplus
extern "C" {
#endif
int qp_host(const double** arg, double** res, int* iw, double* w, int mem);
int corpus_qp_stats(scaly_solver_stats* out);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[2]; } qp_host_mu_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } qp_host_cost_t;
typedef struct { SCALY_ALIGNAS(16) double data[qp_host_SZ_W > 0 ? qp_host_SZ_W : 1]; } qp_host_workspace_t;
static inline int qp_host_call(const qp_host_mu_t* mu, qp_host_cost_t* cost, qp_host_workspace_t* workspace) {
  const double* arg[qp_host_SZ_ARG > 0 ? qp_host_SZ_ARG : 1] = {mu->data};
  double* res[qp_host_SZ_RES > 0 ? qp_host_SZ_RES : 1] = {cost->data};
  return qp_host(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
