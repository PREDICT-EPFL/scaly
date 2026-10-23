/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c shooting.c
 * clang -O3 -fno-math-errno -c shooting.c
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
#define SCALY_WIDTH_4_4_4_4 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4))
typedef double lanes_4_4_4_4_vec __attribute__((vector_size(8 * SCALY_WIDTH_4_4_4_4)));
typedef lanes_4_4_4_4_vec lanes_4_4_4_4_vec_mem __attribute__((aligned(8), may_alias));
static inline __attribute__((always_inline, unused)) void lanes_4_4_4_4_vec_st(double* p, lanes_4_4_4_4_vec v, long long n) {
  if (n == SCALY_WIDTH_4_4_4_4) *(lanes_4_4_4_4_vec_mem*)p = v;
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

static inline __attribute__((always_inline)) void lanes_1(long long i, long long n, double* eq, const double* u, const double* z) {
  int64_t v1[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) v1[k] = (4 * ((i + k) < 2 ? (i + k) : 2));
  double ls[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) ls[k] = z[(v1[k] + 2)];
  lanes_4_4_4_4_vec l = *(const lanes_4_4_4_4_vec_mem*)ls;
  lanes_4_4_4_4_vec v2 = l;
  double l_2s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_2s[k] = z[(v1[k] + 3)];
  lanes_4_4_4_4_vec l_2 = *(const lanes_4_4_4_4_vec_mem*)l_2s;
  lanes_4_4_4_4_vec v3 = l_2;
  double l_3s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_3s[k] = z[v1[k]];
  lanes_4_4_4_4_vec l_3 = *(const lanes_4_4_4_4_vec_mem*)l_3s;
  lanes_4_4_4_4_vec b;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b[k] = 0.050000000000000003;
  double l_4s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_4s[k] = z[(4 + v1[k])];
  lanes_4_4_4_4_vec l_4 = *(const lanes_4_4_4_4_vec_mem*)l_4s;
  lanes_4_4_4_4_vec r = ((l_3 + (b * v2)) - l_4);
  for (long long k = 0; k < n; ++k) eq[((4 * ((i + k) < 2 ? (i + k) : 2))) + 0] = r[k];
  double l_5s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_5s[k] = z[(v1[k] + 1)];
  lanes_4_4_4_4_vec l_5 = *(const lanes_4_4_4_4_vec_mem*)l_5s;
  lanes_4_4_4_4_vec b_2;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_2[k] = 0.050000000000000003;
  double l_6s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_6s[k] = z[(5 + v1[k])];
  lanes_4_4_4_4_vec l_6 = *(const lanes_4_4_4_4_vec_mem*)l_6s;
  lanes_4_4_4_4_vec r_2 = ((l_5 + (b_2 * v3)) - l_6);
  for (long long k = 0; k < n; ++k) eq[((1 + (4 * ((i + k) < 2 ? (i + k) : 2)))) + 0] = r_2[k];
  int64_t v4[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) v4[k] = (2 * ((i + k) < 2 ? (i + k) : 2));
  lanes_4_4_4_4_vec b_3;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_3[k] = 0.10000000000000001;
  lanes_4_4_4_4_vec v5 = (b_3 * ((v2 * v2) + (v3 * v3)));
  lanes_4_4_4_4_vec b_4;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_4[k] = 0.050000000000000003;
  double l_7s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_7s[k] = u[v4[k]];
  lanes_4_4_4_4_vec l_7 = *(const lanes_4_4_4_4_vec_mem*)l_7s;
  double l_8s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_8s[k] = z[(6 + v1[k])];
  lanes_4_4_4_4_vec l_8 = *(const lanes_4_4_4_4_vec_mem*)l_8s;
  lanes_4_4_4_4_vec r_3 = ((v2 + (b_4 * (l_7 - (v5 * v2)))) - l_8);
  for (long long k = 0; k < n; ++k) eq[((2 + (4 * ((i + k) < 2 ? (i + k) : 2)))) + 0] = r_3[k];
  lanes_4_4_4_4_vec b_5;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_5[k] = 0.050000000000000003;
  double l_9s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_9s[k] = u[(v4[k] + 1)];
  lanes_4_4_4_4_vec l_9 = *(const lanes_4_4_4_4_vec_mem*)l_9s;
  double l_10s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_10s[k] = z[(7 + v1[k])];
  lanes_4_4_4_4_vec l_10 = *(const lanes_4_4_4_4_vec_mem*)l_10s;
  lanes_4_4_4_4_vec r_4 = ((v3 + (b_5 * (l_9 - (v5 * v3)))) - l_10);
  for (long long k = 0; k < n; ++k) eq[((3 + (4 * ((i + k) < 2 ? (i + k) : 2)))) + 0] = r_4[k];
}
int shooting(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  for (long long v0_start = 0; v0_start <= 3 - SCALY_WIDTH_4_4_4_4; v0_start += SCALY_WIDTH_4_4_4_4) {
    lanes_1(v0_start, SCALY_WIDTH_4_4_4_4, res[0], arg[1], arg[0]);
  }
#if (3 % SCALY_WIDTH_4_4_4_4) != 0
  lanes_1(3 - 3 % SCALY_WIDTH_4_4_4_4, 3 % SCALY_WIDTH_4_4_4_4, res[0], arg[1], arg[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
