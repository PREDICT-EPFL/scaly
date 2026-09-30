#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));
#if defined(__has_builtin)
#if __has_builtin(__builtin_elementwise_maximum) && __has_builtin(__builtin_elementwise_minimum)
#define SCALY_FMAX_NAN(a, b) __builtin_elementwise_maximum(a, b)
#define SCALY_FMIN_NAN(a, b) __builtin_elementwise_minimum(a, b)
#endif
#endif
#ifndef SCALY_FMAX_NAN
static inline double scaly_fmax_nan(double a, double b) { return ((a < b) || (b != b)) ? b : a; }
static inline double scaly_fmin_nan(double a, double b) { return ((b < a) || (b != b)) ? b : a; }
#define SCALY_FMAX_NAN(a, b) scaly_fmax_nan(a, b)
#define SCALY_FMIN_NAN(a, b) scaly_fmin_nan(a, b)
#endif

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline void control_step_raw(const double* c, const double* u, double* n, double* m, double* w) {
  (void)w;
  double v0 = c[0];
  double v1 = u[0];
  double v2 = (((1.0 < v0) ? 1.0 : v0) + v1);
  double v3 = c[1];
  double v4 = ((1.0 < v3) ? 1.0 : v3);
  double v5 = u[1];
  double v6 = c[2];
  double v7 = (((1.0 < v6) ? 1.0 : v6) + v5);
  double v8 = SCALY_FMAX_NAN(v2, v4);
  *(double2*)(n) = (double2){v2, (v4 + ((v1 * v1) + (v5 * v5)))};
  n[2] = v7;
  m[0] = SCALY_FMAX_NAN(v8, v7);
}

int control(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  if (!res[1]) return SCALY_ERR_NULL_RESULT;
  double s0[6];
  const double* t1 = s0 + 3;
  for (long long c_t0 = 0; c_t0 < 3; ++c_t0) {
    s0[c_t0] = arg[0][c_t0];
  }
  for (long long k_t0 = 0; k_t0 < 3; ++k_t0) {
    int64_t v0 = (k_t0 % 2);
    control_step_raw((s0 + (v0 * 3)), (arg[1] + (2 * k_t0)), (s0 + ((1 - v0) * 3)), (res[1] + k_t0), NULL);
  }
  for (long long c_final = 0; c_final < 3; ++c_final) {
    res[0][c_final] = t1[c_final];
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
