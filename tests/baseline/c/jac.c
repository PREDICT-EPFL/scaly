/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
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
#ifndef SCALY_TYPES_dynamics_jac_znext_z
#define SCALY_TYPES_dynamics_jac_znext_z
typedef struct { SCALY_ALIGNAS(16) double data[4]; } dynamics_jac_znext_z_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[2]; } dynamics_jac_znext_z_u_t;
typedef struct { SCALY_ALIGNAS(16) double data[16]; } dynamics_jac_znext_z_jac_znext_z_t;
typedef struct {
  unsigned char unused;
} dynamics_jac_znext_z_workspace_t;
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
static inline __attribute__((always_inline)) void lanes_3(long long i, long long n, double* jac_znext_z, const double* s1, int64_t i_jac_znext_z_0) {
  double ls[SCALY_WIDTH_4_4_4_4];
  for (long long k = 0; k < SCALY_WIDTH_4_4_4_4; ++k) ls[k] = s1[((((i + k) < 3 ? (i + k) : 3) * 4) + i_jac_znext_z_0)];
  lanes_4_4_4_4_vec l = *(const lanes_4_4_4_4_vec_mem*)ls;
  lanes_4_4_4_4_vec_st(&jac_znext_z[(((i_jac_znext_z_0 * 4) + i)) + 0], l, n);
}
static inline void SCALY_KERNEL_dynamics_jac_znext_z(const double* z, const double* u, double* jac_znext_z) {
  (void)u;
  double s0[4];
  double s1[16];
  static const double k2[8] = {0, 0, 0, 0, 1, 0, 0, 1};
  double s2[1];
  double s3[1];
  static const double k7[8] = {1, 0, 0, 1, 0.050000000000000003, 0, 0, 0.050000000000000003};
  for (long long i_t0 = 0; i_t0 < 2; ++i_t0) {
    s0[i_t0] = (2.0 * z[(2 + i_t0)]);
  }
  for (long long zero_t1_start = 0; zero_t1_start <= 4 - SCALY_WIDTH_4_4_4_4; zero_t1_start += SCALY_WIDTH_4_4_4_4) {
    lanes_1(zero_t1_start, SCALY_WIDTH_4_4_4_4, s1);
  }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
  lanes_1(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, s1);
#endif
  for (long long support_t1 = 0; support_t1 < 2; ++support_t1) {
    int64_t v0 = (2 + support_t1);
    int64_t v1 = (4 + (support_t1 * 3));
    s1[v0] = (s1[v0] + (s0[(v1 % 2)] * k2[v1]));
  }
  for (long long i_t3_start = 0; i_t3_start <= 4 - SCALY_WIDTH_4_4_4_4; i_t3_start += SCALY_WIDTH_4_4_4_4) {
    lanes_2(i_t3_start, SCALY_WIDTH_4_4_4_4, s0, s1);
  }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
  lanes_2(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, s0, s1);
#endif
  double acc_t4 = 0.0;
  for (long long i_t4 = 0; i_t4 < 2; ++i_t4) {
    double v2 = z[(2 + i_t4)];
    acc_t4 = (acc_t4 + (v2 * v2));
  }
  s2[0] = acc_t4;
  s3[0] = (0.10000000000000001 * s2[0]);
  for (long long j_t6_0_0 = 0; j_t6_0_0 < 4; ++j_t6_0_0) {
    for (long long j_t6_0 = 0; j_t6_0 < 2; ++j_t6_0) {
      s1[((j_t6_0_0 * 4) + j_t6_0)] = k7[((j_t6_0_0 * 2) + j_t6_0)];
    }
  }
  for (long long j_t6_1_0 = 0; j_t6_1_0 < 4; ++j_t6_1_0) {
    for (long long j_t6_1 = 0; j_t6_1 < 2; ++j_t6_1) {
      int64_t v3 = (2 + j_t6_1);
      double v4 = k2[((j_t6_1_0 * 2) + j_t6_1)];
      s1[((j_t6_1_0 * 4) + v3)] = (v4 - (0.050000000000000003 * ((s0[j_t6_1_0] * z[v3]) + (s3[0] * v4))));
    }
  }
  for (long long i_jac_znext_z_0 = 0; i_jac_znext_z_0 < 4; ++i_jac_znext_z_0) {
    for (long long i_jac_znext_z_start = 0; i_jac_znext_z_start <= 4 - SCALY_WIDTH_4_4_4_4; i_jac_znext_z_start += SCALY_WIDTH_4_4_4_4) {
      lanes_3(i_jac_znext_z_start, SCALY_WIDTH_4_4_4_4, jac_znext_z, s1, i_jac_znext_z_0);
    }
#if (4 % SCALY_WIDTH_4_4_4_4) != 0
    lanes_3(4 - 4 % SCALY_WIDTH_4_4_4_4, 4 % SCALY_WIDTH_4_4_4_4, jac_znext_z, s1, i_jac_znext_z_0);
#endif
  }
}

int dynamics_jac_znext_z_abi(const void* const* arg, void* const* res, dynamics_jac_znext_z_workspace_t* w) {
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  SCALY_KERNEL_dynamics_jac_znext_z((const double*)arg[0], (const double*)arg[1], (double*)res[0]);
  return SCALY_SUCCESS;
}

void dynamics_jac_znext_z(const dynamics_jac_znext_z_z_t* z, const dynamics_jac_znext_z_u_t* u, dynamics_jac_znext_z_jac_znext_z_t* jac_znext_z, dynamics_jac_znext_z_workspace_t* w) {
  (void)w;
  SCALY_KERNEL_dynamics_jac_znext_z((const double*)z->data, (const double*)u->data, (double*)jac_znext_z->data);
}

#ifdef __cplusplus
}
#endif
