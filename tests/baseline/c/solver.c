#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include "piqp/piqp.h"
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

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

#ifndef SCALY_SOLVER_TIMING_DEFINED
#define SCALY_SOLVER_TIMING_DEFINED
static double scaly_clock_s(void) {
#ifdef __APPLE__
  return 1e-9 * (double)clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
#else
  struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
  return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
#endif
}
#endif

#ifdef __cplusplus
extern "C" {
#endif

static inline void corpus_qp_oracle_raw(const double* mu, double* qp_P, double* qp_c, double* qp_x_lb, double* qp_x_ub, double* w) {
  (void)w;
  *(double2*)(qp_P) = (double2){1.0, 0.0};
  *(double2*)(qp_P + 2) = (double2){0.0, 1.0};
  *(double2*)(qp_c) = (double2){(-mu[0]), (-mu[1])};
  *(double2*)(qp_x_lb) = (double2){((double)(-INFINITY)), ((double)(-INFINITY))};
  *(double2*)(qp_x_ub) = (double2){((double)INFINITY), ((double)INFINITY)};
}

static scaly_solver_stats corpus_qp_stats_data;
// PIQP dense solver wrapper for corpus_qp (n=2, p=0, m=0).
static void corpus_qp_raw(const double* in0, const double* in1, const double* in2, const double* in3, const double* in4, double* out0, double* out1, double* out2, double* out3, double* w) {
  (void)in0;
  (void)in1;
  (void)in2;
  (void)in3;
  double stats_t0 = scaly_clock_s();
  static double P_buf[4];
  static double c_buf[2];
  static double xlb_buf[2];
  static double xub_buf[2];
  static double Pcol[4];
  double fe_t0 = scaly_clock_s();
  corpus_qp_oracle_raw(in4, P_buf, c_buf, xlb_buf, xub_buf, w);
  for (int i = 0; i < 2; ++i) { if (isinf(xlb_buf[i]) && xlb_buf[i] < 0.0) xlb_buf[i] = -PIQP_INF; if (isinf(xub_buf[i]) && xub_buf[i] > 0.0) xub_buf[i] = PIQP_INF; }
  double stats_t_fe = scaly_clock_s() - fe_t0;
  for (int j = 0; j < 2; ++j) for (int i = 0; i < 2; ++i) Pcol[i + j * 2] = P_buf[i * 2 + j];
  static piqp_workspace* corpus_qp_ws = NULL;
  static piqp_settings corpus_qp_settings;
  double solver_t0 = scaly_clock_s();
  if (corpus_qp_ws == NULL) {
    piqp_set_default_settings_dense(&corpus_qp_settings);
    corpus_qp_settings.verbose = 0;
    piqp_data_dense setup_data;
    setup_data.n = 2;
    setup_data.p = 0;
    setup_data.m = 0;
    setup_data.P = Pcol;
    setup_data.c = c_buf;
    setup_data.A = NULL;
    setup_data.b = NULL;
    setup_data.G = NULL;
    setup_data.h_l = NULL;
    setup_data.h_u = NULL;
    setup_data.x_l = xlb_buf;
    setup_data.x_u = xub_buf;
    piqp_setup_dense(&corpus_qp_ws, &setup_data, &corpus_qp_settings);
  } else {
    piqp_update_dense(corpus_qp_ws, Pcol, c_buf, NULL, NULL, NULL, NULL, NULL, xlb_buf, xub_buf);
  }
  piqp_solve(corpus_qp_ws);
  double stats_t_solver = scaly_clock_s() - solver_t0;
  piqp_result* res = corpus_qp_ws->result;
  for (int i = 0; i < 2; ++i) out0[i] = res->x[0 + i];
  for (int i = 0; i < 2; ++i) out1[i] = res->z_bu[0 + i] - res->z_bl[0 + i];
  int32_t stats_status;
  switch (res->info.status) {
    case PIQP_SOLVED: stats_status = SCALY_SOLVE_OK; break;
    case PIQP_MAX_ITER_REACHED: stats_status = SCALY_SOLVE_MAX_ITER; break;
    case PIQP_PRIMAL_INFEASIBLE: stats_status = SCALY_SOLVE_PRIMAL_INFEASIBLE; break;
    case PIQP_DUAL_INFEASIBLE: stats_status = SCALY_SOLVE_DUAL_INFEASIBLE; break;
    case PIQP_NUMERICS: stats_status = SCALY_SOLVE_NUMERICS; break;
    default: stats_status = SCALY_SOLVE_ERROR; break;
  }
  corpus_qp_stats_data.version = SCALY_SOLVER_STATS_VERSION;
  corpus_qp_stats_data.status = stats_status;
  corpus_qp_stats_data.native_status = (int32_t)res->info.status;
  corpus_qp_stats_data.iter = (int32_t)res->info.iter;
  corpus_qp_stats_data.obj = res->info.primal_obj;
  corpus_qp_stats_data.t_fe = stats_t_fe;
  corpus_qp_stats_data.t_solver = 0.0;
  corpus_qp_stats_data.t_qp = stats_t_solver;
  corpus_qp_stats_data.t_globalization = 0.0;
  corpus_qp_stats_data.n_eval_f = 1;
  corpus_qp_stats_data.n_eval_grad_f = 0;
  corpus_qp_stats_data.n_eval_g = 0;
  corpus_qp_stats_data.n_eval_jac_g = 0;
  corpus_qp_stats_data.n_eval_h = 0;
  corpus_qp_stats_data._pad0 = 0;
  corpus_qp_stats_data.primal_viol = res->info.primal_res;
  corpus_qp_stats_data.step_inf = 0.0;
  corpus_qp_stats_data.alpha = 0.0;
  corpus_qp_stats_data.merit_penalty = 0.0;
  corpus_qp_stats_data.backtracks = 0;
  corpus_qp_stats_data.qp_iter = (int32_t)res->info.iter;
  double stats_t_total = scaly_clock_s() - stats_t0;
  corpus_qp_stats_data.t_total = stats_t_total;
  corpus_qp_stats_data.t_glue = stats_t_total - stats_t_fe - stats_t_solver;
}

int corpus_qp_stats(scaly_solver_stats* out) {
  if (!out) return 1;
  *out = corpus_qp_stats_data;
  return 0;
}

int qp_host_sz_arg(void) { return 1; }
int qp_host_sz_res(void) { return 1; }
int qp_host_sz_iw(void) { return 0; }
int qp_host_sz_w(void) { return 0; }
void* qp_host_alloc_mem(void) { return NULL; }
int qp_host_init_mem(void* mem) { (void)mem; return SCALY_SUCCESS; }
void qp_host_free_mem(void* mem) { (void)mem; }

int qp_host(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  static const double k0[2] = {0, 0};
  static const double k1[1] = {};
  double s0[2];
  double s1[2];
  double s2[1];
  double s3[1];
  corpus_qp_raw(k0, k0, k1, k1, arg[0], s0, s1, s2, s3, NULL);
  res[0][0] = 0.0;
  for (long long i_cost = 0; i_cost < 2; ++i_cost) {
    double v0 = s0[i_cost];
    res[0][0] = (res[0][0] + (v0 * v0));
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
