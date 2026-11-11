/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c seed_pair_map.c
 * clang -O3 -fno-math-errno -c seed_pair_map.c
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
#define SCALY_WIDTH_4_4_4_4 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4))
typedef double lanes_4_4_4_4_vec __attribute__((vector_size(8 * SCALY_WIDTH_4_4_4_4)));
typedef lanes_4_4_4_4_vec lanes_4_4_4_4_vec_mem __attribute__((aligned(8), may_alias));
static inline __attribute__((always_inline, unused)) void lanes_4_4_4_4_vec_st(double* p, lanes_4_4_4_4_vec v, long long n) {
  if (n == SCALY_WIDTH_4_4_4_4) *(lanes_4_4_4_4_vec_mem*)p = v;
  else for (long long i = 0; i < n; ++i) p[i] = v[i];
}
static inline __attribute__((always_inline)) lanes_4_4_4_4_vec lanes_4_4_4_4_vec_cos(lanes_4_4_4_4_vec x) {
  lanes_4_4_4_4_vec result;
  for (int i = 0; i < SCALY_WIDTH_4_4_4_4; ++i) result[i] = cos(x[i]);
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

static inline __attribute__((always_inline)) void lanes_1(long long i, long long n, double* s0, const double* x) {
  lanes_4_4_4_4_vec l;
  if (n == SCALY_WIDTH_4_4_4_4) l = *(const lanes_4_4_4_4_vec_mem*)(&x[i]);
  else for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l[k] = x[((i + k) < 3 ? (i + k) : 3)];
  lanes_4_4_4_4_vec_st(&s0[(i) + 0], lanes_4_4_4_4_vec_cos(l), n);
}
static inline __attribute__((always_inline)) void lanes_2(long long i, long long n, double* s1) {
  lanes_8_8_8_8_vec b;
  for (int k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) b[k] = 0.0;
  lanes_8_8_8_8_vec_st(&s1[(i) + 0], b, n);
}
static inline __attribute__((always_inline)) void lanes_3(long long i, long long n, double* y) {
  lanes_8_8_8_8_vec b;
  for (int k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) b[k] = 0.0;
  lanes_8_8_8_8_vec_st(&y[(i) + 0], b, n);
}
static inline __attribute__((always_inline)) void lanes_4(long long i, long long n, const double* s1, double* y) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&y[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = y[((i + k) < 11 ? (i + k) : 11)];
  double l_2s[SCALY_WIDTH_8_8_8_8];
  for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l_2s[k] = s1[(((((i + k) < 11 ? (i + k) : 11) / 4) * -4) + (((i + k) < 11 ? (i + k) : 11) * 5))];
  lanes_8_8_8_8_vec l_2 = *(const lanes_8_8_8_8_vec_mem*)l_2s;
  lanes_8_8_8_8_vec_st(&y[(i) + 0], (l + l_2), n);
}
static inline void seed_pair_stage_support_raw(const double* x, const double* d, double* y, double* w) {
  (void)w;
  double s0[4];
  double s1[48];
  for (long long i_t0_start = 0; i_t0_start <= 4 - SCALY_WIDTH_4_4_4_4; i_t0_start += SCALY_WIDTH_4_4_4_4) {
    lanes_1(i_t0_start, SCALY_WIDTH_4_4_4_4, s0, x);
  }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
  lanes_1(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, s0, x);
#endif
  for (long long zero_t1_start = 0; zero_t1_start <= 48 - SCALY_WIDTH_8_8_8_8; zero_t1_start += SCALY_WIDTH_8_8_8_8) {
    lanes_2(zero_t1_start, SCALY_WIDTH_8_8_8_8, s1);
  }
#if (48 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_2(48 - 48 % SCALY_WIDTH_8_8_8_8, 48 % SCALY_WIDTH_8_8_8_8, s1);
#endif
  for (long long seed_t1_0 = 0; seed_t1_0 < 4; ++seed_t1_0) {
    for (long long seed_t1 = 0; seed_t1 < 3; ++seed_t1) {
      int64_t v0 = ((seed_t1_0 * -43) + (((seed_t1_0 * 3) + seed_t1) * 16));
      s1[v0] = (d[v0] * s0[seed_t1_0]);
    }
  }
  for (long long zero_y_start = 0; zero_y_start <= 12 - SCALY_WIDTH_8_8_8_8; zero_y_start += SCALY_WIDTH_8_8_8_8) {
    lanes_3(zero_y_start, SCALY_WIDTH_8_8_8_8, y);
  }
#if (12 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_3(12 - 12 % SCALY_WIDTH_8_8_8_8, 12 % SCALY_WIDTH_8_8_8_8, y);
#endif
  for (long long support_y_start = 0; support_y_start <= 12 - SCALY_WIDTH_8_8_8_8; support_y_start += SCALY_WIDTH_8_8_8_8) {
    lanes_4(support_y_start, SCALY_WIDTH_8_8_8_8, s1, y);
  }
#if (12 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_4(12 - 12 % SCALY_WIDTH_8_8_8_8, 12 % SCALY_WIDTH_8_8_8_8, s1, y);
#endif
}

static inline __attribute__((always_inline)) void lanes_5(long long i, long long n, const double* d, const double* k0, double* s0) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&d[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = d[((i + k) < 143 ? (i + k) : 143)];
  double l_2s[SCALY_WIDTH_8_8_8_8];
  for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l_2s[k] = k0[((((((i + k) < 143 ? (i + k) : 143) / 4) % 12) * 4) + (((i + k) < 143 ? (i + k) : 143) % 4))];
  lanes_8_8_8_8_vec l_2 = *(const lanes_8_8_8_8_vec_mem*)l_2s;
  lanes_8_8_8_8_vec_st(&s0[(i) + 0], (l * l_2), n);
}
static inline __attribute__((always_inline)) void lanes_6(long long i, long long n, const double* s1, double* y) {
  lanes_8_8_8_8_vec l;
  if (n == SCALY_WIDTH_8_8_8_8) l = *(const lanes_8_8_8_8_vec_mem*)(&s1[i]);
  else for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) l[k] = s1[((i + k) < 35 ? (i + k) : 35)];
  lanes_8_8_8_8_vec_st(&y[(i) + 0], l, n);
}
int seed_pair_map(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  static const double k0[48] = {1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1};
  double s0[144];
  double s1[36];
  for (long long i_t1_start = 0; i_t1_start <= 144 - SCALY_WIDTH_8_8_8_8; i_t1_start += SCALY_WIDTH_8_8_8_8) {
    lanes_5(i_t1_start, SCALY_WIDTH_8_8_8_8, arg[1], k0, s0);
  }
#if (144 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_5(144 - 144 % SCALY_WIDTH_8_8_8_8, 144 % SCALY_WIDTH_8_8_8_8, arg[1], k0, s0);
#endif
  for (long long it_t2 = 0; it_t2 < 3; ++it_t2) {
    seed_pair_stage_support_raw((arg[0] + (4 * it_t2)), (s0 + (48 * it_t2)), (s1 + (it_t2 * 12)), NULL);
  }
  for (long long c_y_start = 0; c_y_start <= 36 - SCALY_WIDTH_8_8_8_8; c_y_start += SCALY_WIDTH_8_8_8_8) {
    lanes_6(c_y_start, SCALY_WIDTH_8_8_8_8, s1, res[0]);
  }
#if (36 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_6(36 - 36 % SCALY_WIDTH_8_8_8_8, 36 % SCALY_WIDTH_8_8_8_8, s1, res[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
