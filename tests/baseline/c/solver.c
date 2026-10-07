/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c qp_host.c
 * clang -O3 -fno-math-errno -c qp_host.c
 * Link with: -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <time.h>
#include <string.h>
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
#ifndef SCALY_SOLVER_OPTION_DEFINED
#define SCALY_SOLVER_OPTION_DEFINED
typedef struct { const char* name; int kind; int64_t integer; double number; const char* text; } scaly_solver_option;
#endif
static double scaly_option_number(const scaly_solver_option* options, const char* name) {
  while (strcmp(options->name, name)) ++options;
  return options->kind == 0 ? (double)options->integer : options->number;
}

#ifdef __cplusplus
extern "C" {
#endif

static inline void corpus_qp_oracle_raw(const double* mu, double* qp_P, double* qp_c, double* qp_x_lb, double* qp_x_ub, double* w, const scaly_solver_option* const* solver_options) {
  (void)w;
  *(double2*)(qp_P) = (double2){1.0, 0.0};
  *(double2*)(qp_P + 2) = (double2){0.0, 1.0};
  *(double2*)(qp_c) = (double2){(-mu[0]), (-mu[1])};
  *(double2*)(qp_x_lb) = (double2){((double)(-INFINITY)), ((double)(-INFINITY))};
  *(double2*)(qp_x_ub) = (double2){((double)INFINITY), ((double)INFINITY)};
}

static scaly_solver_stats corpus_qp_stats_data;
// PIQP solver wrapper for corpus_qp (n=2, p=0, m=0, nnz P/A/G = 4/0/0).
static void corpus_qp_raw(const double* in0, const double* in1, const double* in2, const double* in3, const double* in4, double* out0, double* out1, double* out2, double* out3, double* w, const scaly_solver_option* const* solver_options) {
  (void)in0;
  (void)in1;
  (void)in2;
  (void)in3;
  const scaly_solver_option* options = solver_options[0];
  double stats_t0 = scaly_clock_s();
  static double P_buf[4];
  static double c_buf[2];
  static double xlb_buf[2];
  static double xub_buf[2];
  double fe_t0 = scaly_clock_s();
  corpus_qp_oracle_raw(in4, P_buf, c_buf, xlb_buf, xub_buf, w, solver_options);
  for (int i = 0; i < 2; ++i) { if (isinf(xlb_buf[i]) && xlb_buf[i] < 0.0) xlb_buf[i] = -PIQP_INF; if (isinf(xub_buf[i]) && xub_buf[i] > 0.0) xub_buf[i] = PIQP_INF; }
  double stats_t_fe = scaly_clock_s() - fe_t0;
  static piqp_workspace* corpus_qp_ws = NULL;
  static piqp_settings previous_settings;
  piqp_settings settings = {0};
  double solver_t0 = scaly_clock_s();
  piqp_set_default_settings_dense(&settings);
  for (const scaly_solver_option* option = options; option->name; ++option) {
    if (!strcmp(option->name, "check_duality_gap")) settings.check_duality_gap = option->integer;
    if (!strcmp(option->name, "compute_timings")) settings.compute_timings = option->integer;
    if (!strcmp(option->name, "delta_init")) settings.delta_init = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "eps_abs")) settings.eps_abs = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "eps_duality_gap_abs")) settings.eps_duality_gap_abs = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "eps_duality_gap_rel")) settings.eps_duality_gap_rel = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "eps_rel")) settings.eps_rel = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "infeasibility_threshold")) settings.infeasibility_threshold = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "iterative_refinement_always_enabled")) settings.iterative_refinement_always_enabled = option->integer;
    if (!strcmp(option->name, "iterative_refinement_eps_abs")) settings.iterative_refinement_eps_abs = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "iterative_refinement_eps_rel")) settings.iterative_refinement_eps_rel = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "iterative_refinement_max_iter")) settings.iterative_refinement_max_iter = option->integer;
    if (!strcmp(option->name, "iterative_refinement_min_improvement_rate")) settings.iterative_refinement_min_improvement_rate = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "iterative_refinement_static_regularization_eps")) settings.iterative_refinement_static_regularization_eps = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "iterative_refinement_static_regularization_rel")) settings.iterative_refinement_static_regularization_rel = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "kkt_solver")) settings.kkt_solver = (piqp_kkt_solver)option->integer;
    if (!strcmp(option->name, "max_factor_retires")) settings.max_factor_retires = option->integer;
    if (!strcmp(option->name, "max_iter")) settings.max_iter = option->integer;
    if (!strcmp(option->name, "preconditioner_iter")) settings.preconditioner_iter = option->integer;
    if (!strcmp(option->name, "preconditioner_reuse_on_update")) settings.preconditioner_reuse_on_update = option->integer;
    if (!strcmp(option->name, "preconditioner_scale_cost")) settings.preconditioner_scale_cost = option->integer;
    if (!strcmp(option->name, "reg_finetune_dual_update_threshold")) settings.reg_finetune_dual_update_threshold = option->integer;
    if (!strcmp(option->name, "reg_finetune_lower_limit")) settings.reg_finetune_lower_limit = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "reg_finetune_primal_update_threshold")) settings.reg_finetune_primal_update_threshold = option->integer;
    if (!strcmp(option->name, "reg_lower_limit")) settings.reg_lower_limit = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "rho_init")) settings.rho_init = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "tau")) settings.tau = (option->kind == 0 ? (double)option->integer : option->number);
    if (!strcmp(option->name, "verbose")) settings.verbose = option->integer;
  }
  if (corpus_qp_ws && (previous_settings.check_duality_gap != settings.check_duality_gap || previous_settings.compute_timings != settings.compute_timings || previous_settings.delta_init != settings.delta_init || previous_settings.eps_abs != settings.eps_abs || previous_settings.eps_duality_gap_abs != settings.eps_duality_gap_abs || previous_settings.eps_duality_gap_rel != settings.eps_duality_gap_rel || previous_settings.eps_rel != settings.eps_rel || previous_settings.infeasibility_threshold != settings.infeasibility_threshold || previous_settings.iterative_refinement_always_enabled != settings.iterative_refinement_always_enabled || previous_settings.iterative_refinement_eps_abs != settings.iterative_refinement_eps_abs || previous_settings.iterative_refinement_eps_rel != settings.iterative_refinement_eps_rel || previous_settings.iterative_refinement_max_iter != settings.iterative_refinement_max_iter || previous_settings.iterative_refinement_min_improvement_rate != settings.iterative_refinement_min_improvement_rate || previous_settings.iterative_refinement_static_regularization_eps != settings.iterative_refinement_static_regularization_eps || previous_settings.iterative_refinement_static_regularization_rel != settings.iterative_refinement_static_regularization_rel || previous_settings.kkt_solver != settings.kkt_solver || previous_settings.max_factor_retires != settings.max_factor_retires || previous_settings.max_iter != settings.max_iter || previous_settings.preconditioner_iter != settings.preconditioner_iter || previous_settings.preconditioner_reuse_on_update != settings.preconditioner_reuse_on_update || previous_settings.preconditioner_scale_cost != settings.preconditioner_scale_cost || previous_settings.reg_finetune_dual_update_threshold != settings.reg_finetune_dual_update_threshold || previous_settings.reg_finetune_lower_limit != settings.reg_finetune_lower_limit || previous_settings.reg_finetune_primal_update_threshold != settings.reg_finetune_primal_update_threshold || previous_settings.reg_lower_limit != settings.reg_lower_limit || previous_settings.rho_init != settings.rho_init || previous_settings.tau != settings.tau || previous_settings.verbose != settings.verbose)) { piqp_cleanup(corpus_qp_ws); corpus_qp_ws = NULL; }
  previous_settings = settings;
  piqp_data_dense data = { 2, 0, 0, P_buf, c_buf, NULL, NULL, NULL, NULL, NULL, xlb_buf, xub_buf };
  if (!corpus_qp_ws) piqp_setup_dense(&corpus_qp_ws, &data, &settings);
  else piqp_update_dense(corpus_qp_ws, P_buf, c_buf, NULL, NULL, NULL, NULL, NULL, xlb_buf, xub_buf);
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

int qp_host_with_options(const double** arg, double** res, int* iw, double* w, int mem, const scaly_solver_option* const* solver_options) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!solver_options) return SCALY_ERR_NULL_INPUT;
  if (!solver_options[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  static const double k0[2] = {0, 0};
  static const double k1[1] = {};
  double s0[2];
  double s1[2];
  double s2[1];
  double s3[1];
  corpus_qp_raw(k0, k0, k1, k1, arg[0], s0, s1, s2, s3, NULL, solver_options);
  res[0][0] = 0.0;
  for (long long i_cost = 0; i_cost < 2; ++i_cost) {
    double v0 = s0[i_cost];
    res[0][0] = (res[0][0] + (v0 * v0));
  }
  return SCALY_SUCCESS;
}

const scaly_solver_option* corpus_qp_default_options(void) {
  static const scaly_solver_option options[] = {
    { "verbose", 0, 0, 0.0, NULL },
    { NULL, 0, 0, 0.0, NULL }
  };
  return options;
}

int qp_host(const double** arg, double** res, int* iw, double* w, int mem) {
  const scaly_solver_option* options[] = { corpus_qp_default_options() };
  return qp_host_with_options(arg, res, iw, w, mem, options);
}

#ifdef __cplusplus
}
#endif
