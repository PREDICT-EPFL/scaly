/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c update_loop.c
 * clang -O3 -fno-math-errno -c update_loop.c
 * Link with: -lm
 */
#include "update_loop.h"

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

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline void update_four_inplace_raw(double* carry, double* w) {
  (void)w;
  static const int64_t k1[4] = {11, 29, 47, 83};
  for (long long scatter_carry = 0; scatter_carry < 4; ++scatter_carry) {
    int64_t v0 = k1[scatter_carry];
    carry[v0] = (carry[v0] + 1.0);
  }
}

static inline __attribute__((always_inline)) void lanes_1(long long i, long long n, const double* initial, double* s0) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&initial[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = initial[((i + k) < 99999 ? (i + k) : 99999)];
  lanes_8_8_8_8_vec_st(&s0[(i) + 0], l, n);
}
static inline __attribute__((always_inline)) void lanes_2(long long i, long long n, const double* s0, double* update_loop_2) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&s0[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = s0[((i + k) < 99999 ? (i + k) : 99999)];
  lanes_8_8_8_8_vec_st(&update_loop_2[(i) + 0], l, n);
}
int update_loop(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  if (!w) return SCALY_ERR_NULL_WORK;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  double* s0 = w + 0;
  for (long long c_t0_start = 0; c_t0_start <= 100000 - SCALY_WIDTH_8_8_8_8; c_t0_start += SCALY_WIDTH_8_8_8_8) {
    lanes_1(c_t0_start, SCALY_WIDTH_8_8_8_8, arg[0], s0);
  }
#if (100000 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_1(100000 - 100000 % SCALY_WIDTH_8_8_8_8, 100000 % SCALY_WIDTH_8_8_8_8, arg[0], s0);
#endif
  for (long long i_loop = 0; i_loop < 100; ++i_loop) {
    update_four_inplace_raw(s0, NULL);
  }
  for (long long c_update_loop_2_start = 0; c_update_loop_2_start <= 100000 - SCALY_WIDTH_8_8_8_8; c_update_loop_2_start += SCALY_WIDTH_8_8_8_8) {
    lanes_2(c_update_loop_2_start, SCALY_WIDTH_8_8_8_8, s0, res[0]);
  }
#if (100000 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_2(100000 - 100000 % SCALY_WIDTH_8_8_8_8, 100000 % SCALY_WIDTH_8_8_8_8, s0, res[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
