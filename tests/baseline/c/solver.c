/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c qp_host.c
 * clang -O3 -fno-math-errno -c qp_host.c
 * Link with: -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
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
#ifndef SCALY_SOLVER_OPTION_DEFINED
#define SCALY_SOLVER_OPTION_DEFINED
typedef struct { const char* name; int kind; int64_t integer; double number; const char* text; } scaly_solver_option;
#endif

#ifdef __cplusplus
extern "C" {
#endif

void corpus_qp_solve(const double*, const double*, const double*, const double*, const double*, double*, double*, double*, double*, double*);

void corpus_qp_set_options(const scaly_solver_option* options);
int corpus_qp_stats(scaly_solver_stats* out);
static inline void corpus_qp_support_raw(const double* decision, const double* lam_decision, const double* lam_eq, const double* lam_ineq, const double* mu, double* decision_2, double* lam_decision_2, double* lam_eq_2, double* lam_ineq_2, double* w) {
  (void)w;
  corpus_qp_solve(decision, lam_decision, lam_eq, lam_ineq, mu, decision_2, lam_decision_2, lam_eq_2, lam_ineq_2, NULL);
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
  corpus_qp_set_options(solver_options[0]);
  static const double k0[2] = {0, 0};
  static const double k1[1] = {};
  double s0[2];
  double s1[2];
  double s2[1];
  double s3[1];
  corpus_qp_support_raw(k0, k0, k1, k1, arg[0], s0, s1, s2, s3, NULL);
  double acc_cost = 0.0;
  for (long long i_cost = 0; i_cost < 2; ++i_cost) {
    double v0 = s0[i_cost];
    acc_cost = (acc_cost + (v0 * v0));
  }
  res[0][0] = acc_cost;
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
