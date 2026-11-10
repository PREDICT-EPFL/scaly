/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c multi_output.c
 * clang -O3 -fno-math-errno -c multi_output.c
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

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline __attribute__((always_inline)) void lanes_2(long long i, long long n, double* s0, double* totals, const double* xs) {
  for (long long lane_i_vector = 0; lane_i_vector < 2; lane_i_vector += 1) {
    double ls[SCALY_WIDTH_8_8_8_8];
    for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) ls[k] = xs[((2 * ((i + k) < 4 ? (i + k) : 4)) + lane_i_vector)];
    lanes_8_8_8_8_vec l = *(const lanes_8_8_8_8_vec_mem*)ls;
    lanes_8_8_8_8_vec r = lanes_8_8_8_8_vec_sin(l);
    for (long long k = 0; k < n; ++k) s0[(((((i + k) < 4 ? (i + k) : 4) * 2) + lane_i_vector)) + 0] = r[k];
  }
  lanes_8_8_8_8_vec b;
  for (int k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) b[k] = 0.0;
  lanes_8_8_8_8_vec lane_acc_total = b;
  for (long long lane_i_total = 0; lane_i_total < 2; lane_i_total += 1) {
    double l_2s[SCALY_WIDTH_8_8_8_8];
    for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l_2s[k] = xs[((2 * ((i + k) < 4 ? (i + k) : 4)) + lane_i_total)];
    lanes_8_8_8_8_vec l_2 = *(const lanes_8_8_8_8_vec_mem*)l_2s;
    lane_acc_total = (lane_acc_total + l_2);
  }
  lanes_8_8_8_8_vec_st(&totals[((i + 0)) + 0], lane_acc_total, n);
}
static inline __attribute__((always_inline)) void lanes_3(long long i, long long n, const double* s0, double* vectors) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&s0[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = s0[((i + k) < 9 ? (i + k) : 9)];
  lanes_8_8_8_8_vec_st(&vectors[(i) + 0], l, n);
}
int multi_output(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  if (!res[1]) return SCALY_ERR_NULL_RESULT;
  double s0[10];
  for (long long it_t0_start = 0; it_t0_start <= 5 - SCALY_WIDTH_8_8_8_8; it_t0_start += SCALY_WIDTH_8_8_8_8) {
    lanes_2(it_t0_start, SCALY_WIDTH_8_8_8_8, s0, res[1], arg[0]);
  }
#if (5 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_2(5 - 5 % SCALY_WIDTH_8_8_8_8, 5 % SCALY_WIDTH_8_8_8_8, s0, res[1], arg[0]);
#endif
  for (long long c_vectors_start = 0; c_vectors_start <= 10 - SCALY_WIDTH_8_8_8_8; c_vectors_start += SCALY_WIDTH_8_8_8_8) {
    lanes_3(c_vectors_start, SCALY_WIDTH_8_8_8_8, s0, res[0]);
  }
#if (10 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_3(10 - 10 % SCALY_WIDTH_8_8_8_8, 10 % SCALY_WIDTH_8_8_8_8, s0, res[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
