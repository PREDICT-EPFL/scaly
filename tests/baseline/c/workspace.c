/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c workspace.c
 * clang -O3 -fno-math-errno -c workspace.c
 * Link with: -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));
#ifndef SCALY_LANES
#if defined(__AVX512F__)
#define SCALY_LANES 8
#elif defined(__AVX__) || (defined(__ARM_FEATURE_SVE_BITS) && __ARM_FEATURE_SVE_BITS >= 256)
#define SCALY_LANES 4
#elif defined(__SSE2__) || defined(__aarch64__)
#define SCALY_LANES 2
#else
#define SCALY_LANES 1
#endif
#endif
#if SCALY_LANES != 1 && SCALY_LANES != 2 && SCALY_LANES != 4 && SCALY_LANES != 8
#error "SCALY_LANES must be 1, 2, 4 or 8"
#endif
#if defined(__AVX512F__)
#define SCALY_REGISTER_SLOTS 256
#elif defined(__AVX__) || defined(__aarch64__)
#define SCALY_REGISTER_SLOTS 64
#elif defined(__SSE2__)
#define SCALY_REGISTER_SLOTS 32
#else
#define SCALY_REGISTER_SLOTS 16
#endif
#define SCALY_WIDTH_8_8_8_8 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 8 : SCALY_REGISTER_SLOTS == 64 ? 8 : SCALY_REGISTER_SLOTS == 32 ? 8 : 8) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 8 : SCALY_REGISTER_SLOTS == 64 ? 8 : SCALY_REGISTER_SLOTS == 32 ? 8 : 8))
typedef double lanes_8_8_8_8_vec __attribute__((vector_size(8 * SCALY_WIDTH_8_8_8_8)));
typedef lanes_8_8_8_8_vec lanes_8_8_8_8_vec_mem __attribute__((aligned(8), may_alias));
static inline __attribute__((always_inline, unused)) void lanes_8_8_8_8_vec_st(double* p, lanes_8_8_8_8_vec v, long long n) {
  if (n == SCALY_WIDTH_8_8_8_8) *(lanes_8_8_8_8_vec_mem*)p = v;
  else for (long long i = 0; i < n; ++i) p[i] = v[i];
}
static inline __attribute__((always_inline)) lanes_8_8_8_8_vec lanes_8_8_8_8_vec_sin(lanes_8_8_8_8_vec x) {
  lanes_8_8_8_8_vec result;
  for (int i = 0; i < SCALY_WIDTH_8_8_8_8; ++i) result[i] = sin(x[i]);
  return result;
}

#include <stdbool.h>
#ifndef SCALY_ALIGNAS
#ifdef __cplusplus
#define SCALY_ALIGNAS(n) alignas(n)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SCALY_ALIGNAS(n) _Alignas(n)
#elif defined(__GNUC__)
#define SCALY_ALIGNAS(n) __attribute__((aligned(n)))
#else
#define SCALY_ALIGNAS(n)
#endif
#endif
#ifndef SCALY_TYPES_workspace
#define SCALY_TYPES_workspace
typedef struct { SCALY_ALIGNAS(16) double data[16384]; } workspace_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } workspace_sum_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } workspace_sumsqr_t;
typedef struct {
  SCALY_ALIGNAS(64) double f64[16384];
} workspace_workspace_t;
#endif

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4
#define SCALY_ERR_ALLOC_WORK 5

#ifdef __cplusplus
extern "C" {
#endif

static inline __attribute__((always_inline)) void lanes_1(long long i, long long n, double* s0, const double* x) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&x[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = x[((i + k) < 16383 ? (i + k) : 16383)];
  lanes_8_8_8_8_vec_st(&s0[(i) + 0], lanes_8_8_8_8_vec_sin(l), n);
}
static inline __attribute__((always_inline)) void lanes_2(long long i, long long n, double* acc_sum, const double* s0) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&s0[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = s0[((i + k) < 16383 ? (i + k) : 16383)];
  for (long long k = 0; k < n; ++k) acc_sum[0] = (acc_sum[0] + l[k]);
}
static inline void SCALY_KERNEL_workspace(const double* x, double* sum, double* sumsqr, double* SCALY_W_f64) {
  double* s0 = SCALY_W_f64;
  for (long long i_t0_start = 0; i_t0_start <= 16384 - SCALY_WIDTH_8_8_8_8; i_t0_start += SCALY_WIDTH_8_8_8_8) {
    lanes_1(i_t0_start, SCALY_WIDTH_8_8_8_8, s0, x);
  }
#if (16384 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_1(16384 - 16384 % SCALY_WIDTH_8_8_8_8, 16384 % SCALY_WIDTH_8_8_8_8, s0, x);
#endif
  double acc_sum = 0.0;
  for (long long i_sum_start = 0; i_sum_start <= 16384 - SCALY_WIDTH_8_8_8_8; i_sum_start += SCALY_WIDTH_8_8_8_8) {
    lanes_2(i_sum_start, SCALY_WIDTH_8_8_8_8, &acc_sum, s0);
  }
#if (16384 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_2(16384 - 16384 % SCALY_WIDTH_8_8_8_8, 16384 % SCALY_WIDTH_8_8_8_8, &acc_sum, s0);
#endif
  sum[0] = acc_sum;
  double acc_sumsqr = 0.0;
  for (long long i_sumsqr = 0; i_sumsqr < 16384; ++i_sumsqr) {
    double v0 = s0[i_sumsqr];
    acc_sumsqr = (acc_sumsqr + (v0 * v0));
  }
  sumsqr[0] = acc_sumsqr;
}

int workspace_abi(const void* const* arg, void* const* res, workspace_workspace_t* w) {
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  if (!w) return SCALY_ERR_NULL_WORK;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  if (!res[1]) return SCALY_ERR_NULL_RESULT;
  SCALY_KERNEL_workspace((const double*)arg[0], (double*)res[0], (double*)res[1], (double*)w->f64);
  return SCALY_SUCCESS;
}

void workspace(const workspace_x_t* x, workspace_sum_t* sum, workspace_sumsqr_t* sumsqr, workspace_workspace_t* w) {
  SCALY_KERNEL_workspace((const double*)x->data, (double*)sum->data, (double*)sumsqr->data, (double*)w->f64);
}

#ifdef __cplusplus
}
#endif
