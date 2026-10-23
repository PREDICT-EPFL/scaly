/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c dynamics_jac_znext_z.c
 * clang -O3 -fno-math-errno -c dynamics_jac_znext_z.c
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

static inline __attribute__((always_inline)) void lanes_1(long long i, long long n, double* s1) {
  lanes_4_4_4_4_vec b;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b[k] = 0.0;
  lanes_4_4_4_4_vec_st(&s1[(i) + 0], b, n);
}
static inline __attribute__((always_inline)) void lanes_2(long long i, long long n, double* s0, const double* s1) {
  lanes_4_4_4_4_vec b;
  for (int k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) b[k] = 0.10000000000000001;
  lanes_4_4_4_4_vec l;
  if (n == SCALY_WIDTH_4_4_4_4) l = *(const lanes_4_4_4_4_vec_mem*)(&s1[i]);
  else for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) l[k] = s1[((i + k) < 3 ? (i + k) : 3)];
  lanes_4_4_4_4_vec_st(&s0[(i) + 0], (b * l), n);
}
static inline __attribute__((always_inline)) void lanes_3(long long i, long long n, double* jac_znext_z, const double* s1, int64_t d0_jac_znext_z) {
  double ls[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) ls[k] = s1[(d0_jac_znext_z + (((i + k) < 3 ? (i + k) : 3) * 4))];
  lanes_4_4_4_4_vec l = *(const lanes_4_4_4_4_vec_mem*)ls;
  lanes_4_4_4_4_vec_st(&jac_znext_z[(((d0_jac_znext_z * 4) + i)) + 0], l, n);
}
int dynamics_jac_znext_z(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  static const double k0[8] = {1, 0, 0, 1, 0.050000000000000003, 0, 0, 0.050000000000000003};
  static const double k1[8] = {0, 0, 0, 0, 1, 0, 0, 1};
  const double* t5 = arg[0] + 2;
  double s0[4];
  double s1[16];
  double s2[1];
  for (long long i_t6 = 0; i_t6 < 2; ++i_t6) {
    s0[i_t6] = (2.0 * t5[i_t6]);
  }
  for (long long i_t9_start = 0; i_t9_start <= 4 - SCALY_WIDTH_4_4_4_4; i_t9_start += SCALY_WIDTH_4_4_4_4) {
    lanes_1(i_t9_start, SCALY_WIDTH_4_4_4_4, s1);
  }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
  lanes_1(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, s1);
#endif
  for (long long k_t9 = 0; k_t9 < 2; ++k_t9) {
    s1[0] = (s1[0] + (s0[k_t9] * k1[k_t9]));
    s1[1] = (s1[1] + (s0[k_t9] * k1[(2 + k_t9)]));
    s1[2] = (s1[2] + (s0[k_t9] * k1[(4 + k_t9)]));
    s1[3] = (s1[3] + (s0[k_t9] * k1[(6 + k_t9)]));
  }
  for (long long i_t10_start = 0; i_t10_start <= 4 - SCALY_WIDTH_4_4_4_4; i_t10_start += SCALY_WIDTH_4_4_4_4) {
    lanes_2(i_t10_start, SCALY_WIDTH_4_4_4_4, s0, s1);
  }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
  lanes_2(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, s0, s1);
#endif
  s1[0] = 0.0;
  for (long long i_t14 = 0; i_t14 < 2; ++i_t14) {
    double v0 = t5[i_t14];
    s1[0] = (s1[0] + (v0 * v0));
  }
  s2[0] = (0.10000000000000001 * s1[0]);
  for (long long j_t20_0_0 = 0; j_t20_0_0 < 4; ++j_t20_0_0) {
    for (long long j_t20_0_1 = 0; j_t20_0_1 < 2; ++j_t20_0_1) {
      s1[((j_t20_0_0 * 4) + j_t20_0_1)] = k0[((j_t20_0_0 * 2) + j_t20_0_1)];
    }
  }
  for (long long j_t20_1_0 = 0; j_t20_1_0 < 4; ++j_t20_1_0) {
    for (long long j_t20_1_1 = 0; j_t20_1_1 < 2; ++j_t20_1_1) {
      double v1 = k1[((j_t20_1_0 * 2) + j_t20_1_1)];
      s1[((j_t20_1_0 * 4) + (2 + j_t20_1_1))] = (v1 - (0.050000000000000003 * ((s0[j_t20_1_0] * t5[j_t20_1_1]) + (s2[0] * v1))));
    }
  }
  for (long long d0_jac_znext_z = 0; d0_jac_znext_z < 4; ++d0_jac_znext_z) {
    for (long long d1_jac_znext_z_start = 0; d1_jac_znext_z_start <= 4 - SCALY_WIDTH_4_4_4_4; d1_jac_znext_z_start += SCALY_WIDTH_4_4_4_4) {
      lanes_3(d1_jac_znext_z_start, SCALY_WIDTH_4_4_4_4, res[0], s1, d0_jac_znext_z);
    }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
    lanes_3(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, res[0], s1, d0_jac_znext_z);
#endif
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
