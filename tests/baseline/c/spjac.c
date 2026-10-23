/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c shooting_spjac_eq_z.c
 * clang -O3 -fno-math-errno -c shooting_spjac_eq_z.c
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

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline __attribute__((always_inline)) void lanes_1(long long i, long long n, double* s0, const double* z) {
  double ls[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) ls[k] = z[((4 * ((i + k) < 2 ? (i + k) : 2)) + 2)];
  lanes_4_4_4_4_vec l = *(const lanes_4_4_4_4_vec_mem*)ls;
  lanes_4_4_4_4_vec lane_v0 = l;
  double l_2s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_2s[k] = z[((4 * ((i + k) < 2 ? (i + k) : 2)) + 3)];
  lanes_4_4_4_4_vec l_2 = *(const lanes_4_4_4_4_vec_mem*)l_2s;
  lanes_4_4_4_4_vec lane_v2 = l_2;
  lanes_4_4_4_4_vec b;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b[k] = 1.0;
  lanes_4_4_4_4_vec r = b;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 0)) + 0] = r[k];
  lanes_4_4_4_4_vec b_2;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_2[k] = 1.0;
  lanes_4_4_4_4_vec r_2 = b_2;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 1)) + 0] = r_2[k];
  lanes_4_4_4_4_vec b_3;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_3[k] = 0.0;
  lanes_4_4_4_4_vec r_3 = b_3;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 2)) + 0] = r_3[k];
  lanes_4_4_4_4_vec b_4;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_4[k] = 0.0;
  lanes_4_4_4_4_vec r_4 = b_4;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 3)) + 0] = r_4[k];
  lanes_4_4_4_4_vec b_5;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_5[k] = 0.050000000000000003;
  lanes_4_4_4_4_vec r_5 = b_5;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 4)) + 0] = r_5[k];
  lanes_4_4_4_4_vec b_6;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_6[k] = 0.0;
  lanes_4_4_4_4_vec r_6 = b_6;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 5)) + 0] = r_6[k];
  lanes_4_4_4_4_vec b_7;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_7[k] = 0.10000000000000001;
  lanes_4_4_4_4_vec b_8;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_8[k] = 2.0;
  lanes_4_4_4_4_vec lane_v1 = (b_7 * (b_8 * lane_v0));
  lanes_4_4_4_4_vec b_9;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_9[k] = 0.10000000000000001;
  lanes_4_4_4_4_vec lane_v3 = (b_9 * ((lane_v0 * lane_v0) + (lane_v2 * lane_v2)));
  lanes_4_4_4_4_vec b_10;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_10[k] = 1.0;
  lanes_4_4_4_4_vec b_11;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_11[k] = 0.050000000000000003;
  lanes_4_4_4_4_vec r_7 = (b_10 - (b_11 * ((lane_v1 * lane_v0) + lane_v3)));
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 6)) + 0] = r_7[k];
  lanes_4_4_4_4_vec b_12;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_12[k] = 0.050000000000000003;
  lanes_4_4_4_4_vec r_8 = (-(b_12 * (lane_v1 * lane_v2)));
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 7)) + 0] = r_8[k];
  lanes_4_4_4_4_vec b_13;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_13[k] = 0.0;
  lanes_4_4_4_4_vec r_9 = b_13;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 8)) + 0] = r_9[k];
  lanes_4_4_4_4_vec b_14;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_14[k] = 0.050000000000000003;
  lanes_4_4_4_4_vec r_10 = b_14;
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 9)) + 0] = r_10[k];
  lanes_4_4_4_4_vec b_15;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_15[k] = 0.10000000000000001;
  lanes_4_4_4_4_vec b_16;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_16[k] = 2.0;
  lanes_4_4_4_4_vec lane_v4 = (b_15 * (b_16 * lane_v2));
  lanes_4_4_4_4_vec b_17;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_17[k] = 0.050000000000000003;
  lanes_4_4_4_4_vec r_11 = (-(b_17 * (lane_v4 * lane_v0)));
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 10)) + 0] = r_11[k];
  lanes_4_4_4_4_vec b_18;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_18[k] = 1.0;
  lanes_4_4_4_4_vec b_19;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b_19[k] = 0.050000000000000003;
  lanes_4_4_4_4_vec r_12 = (b_18 - (b_19 * ((lane_v4 * lane_v2) + lane_v3)));
  for (long long k = 0; k < n; ++k) s0[(((((i + k) < 2 ? (i + k) : 2) * 12) + 11)) + 0] = r_12[k];
}
static inline __attribute__((always_inline)) void lanes_2(long long i, long long n, double* s1) {
  lanes_8_8_8_8_vec b;
  for (int k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) b[k] = 0.0;
  lanes_8_8_8_8_vec_st(&s1[(i) + 0], b, n);
}
static inline __attribute__((always_inline)) void lanes_3(long long i, long long n, const double* k5, double* s0, const double* s1, int64_t d0_t7) {
  int64_t v0[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) v0[k] = (d0_t7 + (((i + k) < 3 ? (i + k) : 3) * 12));
  double ls[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) ls[k] = s1[v0[k]];
  lanes_4_4_4_4_vec l = *(const lanes_4_4_4_4_vec_mem*)ls;
  double l_2s[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l_2s[k] = k5[v0[k]];
  lanes_4_4_4_4_vec l_2 = *(const lanes_4_4_4_4_vec_mem*)l_2s;
  lanes_4_4_4_4_vec_st(&s0[(((d0_t7 * 4) + i)) + 0], (l - l_2), n);
}
static inline __attribute__((always_inline)) void lanes_4(long long i, long long n, const int64_t* k8, const double* s0, double* spjac_eq_z) {
  double ls[SCALY_WIDTH_8_8_8_8];
  for (long long k = 0; k < SCALY_WIDTH_8_8_8_8; ++k) ls[k] = s0[k8[((i + k) < 35 ? (i + k) : 35)]];
  lanes_8_8_8_8_vec l = *(const lanes_8_8_8_8_vec_mem*)ls;
  lanes_8_8_8_8_vec_st(&spjac_eq_z[(i) + 0], l, n);
}
int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  double s0[48];
  static const int64_t k2[24] = {0, 1, 16, 18, 19, 25, 4, 6, 7, 13, 24, 33, 34, 35, 9, 10, 11, 12, 28, 30, 31, 21, 22, 23};
  double s1[48];
  static const int64_t k4[24] = {0, 1, 4, 6, 7, 9, 12, 14, 15, 17, 20, 21, 22, 23, 25, 26, 27, 28, 32, 34, 35, 41, 42, 43};
  static const double k5[48] = {0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0};
  static const int64_t k8[36] = {0, 1, 2, 4, 6, 5, 9, 10, 8, 13, 14, 15, 18, 16, 17, 21, 23, 20, 24, 27, 26, 28, 31, 29, 33, 34, 32, 36, 37, 38, 42, 41, 40, 46, 45, 44};
  for (long long it_t0_start = 0; it_t0_start <= 3 - SCALY_WIDTH_4_4_4_4; it_t0_start += SCALY_WIDTH_4_4_4_4) {
    lanes_1(it_t0_start, SCALY_WIDTH_4_4_4_4, s0, arg[0]);
  }
#if (3 % SCALY_WIDTH_4_4_4_4) != 0
  lanes_1(3 - 3 % SCALY_WIDTH_4_4_4_4, 3 % SCALY_WIDTH_4_4_4_4, s0, arg[0]);
#endif
  for (long long z_t3_start = 0; z_t3_start <= 48 - SCALY_WIDTH_8_8_8_8; z_t3_start += SCALY_WIDTH_8_8_8_8) {
    lanes_2(z_t3_start, SCALY_WIDTH_8_8_8_8, s1);
  }
#if (48 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_2(48 - 48 % SCALY_WIDTH_8_8_8_8, 48 % SCALY_WIDTH_8_8_8_8, s1);
#endif
  for (long long i_t3 = 0; i_t3 < 24; ++i_t3) {
    s1[k4[i_t3]] = s0[k2[i_t3]];
  }
  for (long long d0_t7 = 0; d0_t7 < 12; ++d0_t7) {
    for (long long d1_t7_start = 0; d1_t7_start <= 4 - SCALY_WIDTH_4_4_4_4; d1_t7_start += SCALY_WIDTH_4_4_4_4) {
      lanes_3(d1_t7_start, SCALY_WIDTH_4_4_4_4, k5, s0, s1, d0_t7);
    }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
    lanes_3(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, k5, s0, s1, d0_t7);
#endif
  }
  for (long long i_spjac_eq_z_start = 0; i_spjac_eq_z_start <= 36 - SCALY_WIDTH_8_8_8_8; i_spjac_eq_z_start += SCALY_WIDTH_8_8_8_8) {
    lanes_4(i_spjac_eq_z_start, SCALY_WIDTH_8_8_8_8, k8, s0, res[0]);
  }
#if (36 % SCALY_WIDTH_8_8_8_8) != 0
  lanes_4(36 - 36 % SCALY_WIDTH_8_8_8_8, 36 % SCALY_WIDTH_8_8_8_8, k8, s0, res[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
