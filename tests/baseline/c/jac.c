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
#define SCALY_WIDTH_dynamics_jac_znext_z_lanes_7 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4))
#define SCALY_WIDTH_dynamics_jac_znext_z_lanes_6 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4))
#define SCALY_WIDTH_dynamics_jac_znext_z_lanes_5 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4))
#define SCALY_WIDTH_dynamics_jac_znext_z_lanes_4 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2))
#define SCALY_WIDTH_dynamics_jac_znext_z_lanes_3 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2))
#define SCALY_WIDTH_dynamics_jac_znext_z_lanes_2 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2))
#define SCALY_WIDTH_dynamics_jac_znext_z_lanes_1 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 2 : SCALY_REGISTER_SLOTS == 64 ? 2 : SCALY_REGISTER_SLOTS == 32 ? 2 : 2))

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

typedef double dynamics_jac_znext_z_lanes_1_vec __attribute__((vector_size(8 * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1), aligned(8), may_alias));
static inline __attribute__((always_inline)) void dynamics_jac_znext_z_lanes_1(long long j_t7_0_chunk, long long dynamics_jac_znext_z_lanes_1_valid, const double* s0, double* s1) {
  dynamics_jac_znext_z_lanes_1_vec dynamics_jac_znext_z_lanes_1_load_1;
  if (dynamics_jac_znext_z_lanes_1_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) dynamics_jac_znext_z_lanes_1_load_1 = *(const dynamics_jac_znext_z_lanes_1_vec*)(&s0[(0 + (((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + 0) < 1 ? ((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + 0) : 1))]);
  else for (long long j_t7_0_lane = 0; j_t7_0_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_1; ++j_t7_0_lane) dynamics_jac_znext_z_lanes_1_load_1[j_t7_0_lane] = s0[(0 + (((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + j_t7_0_lane) < 1 ? ((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + j_t7_0_lane) : 1))];
  dynamics_jac_znext_z_lanes_1_vec dynamics_jac_znext_z_lanes_1_store_2 = dynamics_jac_znext_z_lanes_1_load_1;
  if (dynamics_jac_znext_z_lanes_1_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) *(dynamics_jac_znext_z_lanes_1_vec*)(&s1[((0 + (((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + 0) < 1 ? ((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + 0) : 1))) + 0]) = dynamics_jac_znext_z_lanes_1_store_2;
  else for (long long j_t7_0_lane = 0; j_t7_0_lane < dynamics_jac_znext_z_lanes_1_valid; ++j_t7_0_lane) s1[((0 + (((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + j_t7_0_lane) < 1 ? ((j_t7_0_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) + j_t7_0_lane) : 1))) + 0] = dynamics_jac_znext_z_lanes_1_store_2[j_t7_0_lane];
}
typedef double dynamics_jac_znext_z_lanes_2_vec __attribute__((vector_size(8 * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2), aligned(8), may_alias));
static inline __attribute__((always_inline)) void dynamics_jac_znext_z_lanes_2(long long j_t7_1_chunk, long long dynamics_jac_znext_z_lanes_2_valid, const double* s0, double* s1) {
  dynamics_jac_znext_z_lanes_2_vec dynamics_jac_znext_z_lanes_2_load_1;
  if (dynamics_jac_znext_z_lanes_2_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) dynamics_jac_znext_z_lanes_2_load_1 = *(const dynamics_jac_znext_z_lanes_2_vec*)(&s0[(0 + (((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + 0) < 1 ? ((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + 0) : 1))]);
  else for (long long j_t7_1_lane = 0; j_t7_1_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_2; ++j_t7_1_lane) dynamics_jac_znext_z_lanes_2_load_1[j_t7_1_lane] = s0[(0 + (((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + j_t7_1_lane) < 1 ? ((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + j_t7_1_lane) : 1))];
  dynamics_jac_znext_z_lanes_2_vec dynamics_jac_znext_z_lanes_2_store_2 = dynamics_jac_znext_z_lanes_2_load_1;
  if (dynamics_jac_znext_z_lanes_2_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) *(dynamics_jac_znext_z_lanes_2_vec*)(&s1[((2 + (0 + (((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + 0) < 1 ? ((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + 0) : 1)))) + 0]) = dynamics_jac_znext_z_lanes_2_store_2;
  else for (long long j_t7_1_lane = 0; j_t7_1_lane < dynamics_jac_znext_z_lanes_2_valid; ++j_t7_1_lane) s1[((2 + (0 + (((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + j_t7_1_lane) < 1 ? ((j_t7_1_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) + j_t7_1_lane) : 1)))) + 0] = dynamics_jac_znext_z_lanes_2_store_2[j_t7_1_lane];
}
typedef double dynamics_jac_znext_z_lanes_3_vec __attribute__((vector_size(8 * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3), aligned(8), may_alias));
static inline __attribute__((always_inline)) void dynamics_jac_znext_z_lanes_3(long long j_t7_2_chunk, long long dynamics_jac_znext_z_lanes_3_valid, const double* s0, double* s1) {
  dynamics_jac_znext_z_lanes_3_vec dynamics_jac_znext_z_lanes_3_load_1;
  if (dynamics_jac_znext_z_lanes_3_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) dynamics_jac_znext_z_lanes_3_load_1 = *(const dynamics_jac_znext_z_lanes_3_vec*)(&s0[(0 + (((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + 0) < 1 ? ((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + 0) : 1))]);
  else for (long long j_t7_2_lane = 0; j_t7_2_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_3; ++j_t7_2_lane) dynamics_jac_znext_z_lanes_3_load_1[j_t7_2_lane] = s0[(0 + (((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + j_t7_2_lane) < 1 ? ((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + j_t7_2_lane) : 1))];
  dynamics_jac_znext_z_lanes_3_vec dynamics_jac_znext_z_lanes_3_store_2 = dynamics_jac_znext_z_lanes_3_load_1;
  if (dynamics_jac_znext_z_lanes_3_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) *(dynamics_jac_znext_z_lanes_3_vec*)(&s1[((4 + (0 + (((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + 0) < 1 ? ((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + 0) : 1)))) + 0]) = dynamics_jac_znext_z_lanes_3_store_2;
  else for (long long j_t7_2_lane = 0; j_t7_2_lane < dynamics_jac_znext_z_lanes_3_valid; ++j_t7_2_lane) s1[((4 + (0 + (((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + j_t7_2_lane) < 1 ? ((j_t7_2_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) + j_t7_2_lane) : 1)))) + 0] = dynamics_jac_znext_z_lanes_3_store_2[j_t7_2_lane];
}
typedef double dynamics_jac_znext_z_lanes_4_vec __attribute__((vector_size(8 * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4), aligned(8), may_alias));
static inline __attribute__((always_inline)) void dynamics_jac_znext_z_lanes_4(long long j_t7_3_chunk, long long dynamics_jac_znext_z_lanes_4_valid, const double* s0, double* s1) {
  dynamics_jac_znext_z_lanes_4_vec dynamics_jac_znext_z_lanes_4_load_1;
  if (dynamics_jac_znext_z_lanes_4_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) dynamics_jac_znext_z_lanes_4_load_1 = *(const dynamics_jac_znext_z_lanes_4_vec*)(&s0[(0 + (((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + 0) < 1 ? ((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + 0) : 1))]);
  else for (long long j_t7_3_lane = 0; j_t7_3_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_4; ++j_t7_3_lane) dynamics_jac_znext_z_lanes_4_load_1[j_t7_3_lane] = s0[(0 + (((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + j_t7_3_lane) < 1 ? ((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + j_t7_3_lane) : 1))];
  dynamics_jac_znext_z_lanes_4_vec dynamics_jac_znext_z_lanes_4_store_2 = dynamics_jac_znext_z_lanes_4_load_1;
  if (dynamics_jac_znext_z_lanes_4_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) *(dynamics_jac_znext_z_lanes_4_vec*)(&s1[((6 + (0 + (((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + 0) < 1 ? ((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + 0) : 1)))) + 0]) = dynamics_jac_znext_z_lanes_4_store_2;
  else for (long long j_t7_3_lane = 0; j_t7_3_lane < dynamics_jac_znext_z_lanes_4_valid; ++j_t7_3_lane) s1[((6 + (0 + (((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + j_t7_3_lane) < 1 ? ((j_t7_3_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) + j_t7_3_lane) : 1)))) + 0] = dynamics_jac_znext_z_lanes_4_store_2[j_t7_3_lane];
}
typedef double dynamics_jac_znext_z_lanes_5_vec __attribute__((vector_size(8 * SCALY_WIDTH_dynamics_jac_znext_z_lanes_5), aligned(8), may_alias));
static inline __attribute__((always_inline)) void dynamics_jac_znext_z_lanes_5(long long i_t10_chunk, long long dynamics_jac_znext_z_lanes_5_valid, double* s0) {
  dynamics_jac_znext_z_lanes_5_vec dynamics_jac_znext_z_lanes_5_broadcast_1;
  for (int i_t10_lane = 0; i_t10_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_5; ++i_t10_lane) dynamics_jac_znext_z_lanes_5_broadcast_1[i_t10_lane] = 0.0;
  dynamics_jac_znext_z_lanes_5_vec dynamics_jac_znext_z_lanes_5_store_2 = dynamics_jac_znext_z_lanes_5_broadcast_1;
  if (dynamics_jac_znext_z_lanes_5_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_5) *(dynamics_jac_znext_z_lanes_5_vec*)(&s0[((0 + (((i_t10_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_5) + 0) < 3 ? ((i_t10_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_5) + 0) : 3))) + 0]) = dynamics_jac_znext_z_lanes_5_store_2;
  else for (long long i_t10_lane = 0; i_t10_lane < dynamics_jac_znext_z_lanes_5_valid; ++i_t10_lane) s0[((0 + (((i_t10_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_5) + i_t10_lane) < 3 ? ((i_t10_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_5) + i_t10_lane) : 3))) + 0] = dynamics_jac_znext_z_lanes_5_store_2[i_t10_lane];
}
typedef double dynamics_jac_znext_z_lanes_6_vec __attribute__((vector_size(8 * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6), aligned(8), may_alias));
static inline __attribute__((always_inline)) void dynamics_jac_znext_z_lanes_6(long long i_t11_chunk, long long dynamics_jac_znext_z_lanes_6_valid, const double* s0, double* s1) {
  dynamics_jac_znext_z_lanes_6_vec dynamics_jac_znext_z_lanes_6_broadcast_1;
  for (int i_t11_lane = 0; i_t11_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_6; ++i_t11_lane) dynamics_jac_znext_z_lanes_6_broadcast_1[i_t11_lane] = 0.10000000000000001;
  dynamics_jac_znext_z_lanes_6_vec dynamics_jac_znext_z_lanes_6_load_2;
  if (dynamics_jac_znext_z_lanes_6_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) dynamics_jac_znext_z_lanes_6_load_2 = *(const dynamics_jac_znext_z_lanes_6_vec*)(&s0[(0 + (((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + 0) < 3 ? ((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + 0) : 3))]);
  else for (long long i_t11_lane = 0; i_t11_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_6; ++i_t11_lane) dynamics_jac_znext_z_lanes_6_load_2[i_t11_lane] = s0[(0 + (((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + i_t11_lane) < 3 ? ((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + i_t11_lane) : 3))];
  dynamics_jac_znext_z_lanes_6_vec dynamics_jac_znext_z_lanes_6_store_3 = (dynamics_jac_znext_z_lanes_6_broadcast_1 * dynamics_jac_znext_z_lanes_6_load_2);
  if (dynamics_jac_znext_z_lanes_6_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) *(dynamics_jac_znext_z_lanes_6_vec*)(&s1[((0 + (((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + 0) < 3 ? ((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + 0) : 3))) + 0]) = dynamics_jac_znext_z_lanes_6_store_3;
  else for (long long i_t11_lane = 0; i_t11_lane < dynamics_jac_znext_z_lanes_6_valid; ++i_t11_lane) s1[((0 + (((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + i_t11_lane) < 3 ? ((i_t11_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) + i_t11_lane) : 3))) + 0] = dynamics_jac_znext_z_lanes_6_store_3[i_t11_lane];
}
typedef double dynamics_jac_znext_z_lanes_7_vec __attribute__((vector_size(8 * SCALY_WIDTH_dynamics_jac_znext_z_lanes_7), aligned(8), may_alias));
static inline __attribute__((always_inline)) void dynamics_jac_znext_z_lanes_7(long long d1_jac_znext_z_chunk, long long dynamics_jac_znext_z_lanes_7_valid, double* jac_znext_z, const double* s2, int64_t d0_jac_znext_z) {
  dynamics_jac_znext_z_lanes_7_vec dynamics_jac_znext_z_lanes_7_load_1;
  double dynamics_jac_znext_z_lanes_7_load_1_stage[8];
  for (long long d1_jac_znext_z_lane = 0; d1_jac_znext_z_lane < SCALY_WIDTH_dynamics_jac_znext_z_lanes_7; ++d1_jac_znext_z_lane) dynamics_jac_znext_z_lanes_7_load_1_stage[d1_jac_znext_z_lane] = s2[(d0_jac_znext_z + ((0 + (((d1_jac_znext_z_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) + d1_jac_znext_z_lane) < 3 ? ((d1_jac_znext_z_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) + d1_jac_znext_z_lane) : 3)) * 4))];
  dynamics_jac_znext_z_lanes_7_load_1 = *(const dynamics_jac_znext_z_lanes_7_vec*)dynamics_jac_znext_z_lanes_7_load_1_stage;
  dynamics_jac_znext_z_lanes_7_vec dynamics_jac_znext_z_lanes_7_store_2 = dynamics_jac_znext_z_lanes_7_load_1;
  if (dynamics_jac_znext_z_lanes_7_valid == SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) *(dynamics_jac_znext_z_lanes_7_vec*)(&jac_znext_z[(((d0_jac_znext_z * 4) + (0 + (((d1_jac_znext_z_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) + 0) < 3 ? ((d1_jac_znext_z_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) + 0) : 3)))) + 0]) = dynamics_jac_znext_z_lanes_7_store_2;
  else for (long long d1_jac_znext_z_lane = 0; d1_jac_znext_z_lane < dynamics_jac_znext_z_lanes_7_valid; ++d1_jac_znext_z_lane) jac_znext_z[(((d0_jac_znext_z * 4) + (0 + (((d1_jac_znext_z_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) + d1_jac_znext_z_lane) < 3 ? ((d1_jac_znext_z_chunk * SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) + d1_jac_znext_z_lane) : 3)))) + 0] = dynamics_jac_znext_z_lanes_7_store_2[d1_jac_znext_z_lane];
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
  double s0[8];
  double s1[8];
  double s2[16];
  double s3[1];
  for (long long i_t6 = 0; i_t6 < 2; ++i_t6) {
    s0[i_t6] = (2.0 * t5[i_t6]);
  }
  for (long long j_t7_0_chunk = 0; j_t7_0_chunk < 2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_1; ++j_t7_0_chunk) {
    dynamics_jac_znext_z_lanes_1(j_t7_0_chunk, SCALY_WIDTH_dynamics_jac_znext_z_lanes_1, s0, s1);
  }
#if (2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_1) != 0
  dynamics_jac_znext_z_lanes_1(2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_1, 2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_1, s0, s1);
#endif
  for (long long j_t7_1_chunk = 0; j_t7_1_chunk < 2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_2; ++j_t7_1_chunk) {
    dynamics_jac_znext_z_lanes_2(j_t7_1_chunk, SCALY_WIDTH_dynamics_jac_znext_z_lanes_2, s0, s1);
  }
#if (2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_2) != 0
  dynamics_jac_znext_z_lanes_2(2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_2, 2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_2, s0, s1);
#endif
  for (long long j_t7_2_chunk = 0; j_t7_2_chunk < 2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_3; ++j_t7_2_chunk) {
    dynamics_jac_znext_z_lanes_3(j_t7_2_chunk, SCALY_WIDTH_dynamics_jac_znext_z_lanes_3, s0, s1);
  }
#if (2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_3) != 0
  dynamics_jac_znext_z_lanes_3(2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_3, 2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_3, s0, s1);
#endif
  for (long long j_t7_3_chunk = 0; j_t7_3_chunk < 2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_4; ++j_t7_3_chunk) {
    dynamics_jac_znext_z_lanes_4(j_t7_3_chunk, SCALY_WIDTH_dynamics_jac_znext_z_lanes_4, s0, s1);
  }
#if (2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_4) != 0
  dynamics_jac_znext_z_lanes_4(2 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_4, 2 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_4, s0, s1);
#endif
  for (long long i_t10_chunk = 0; i_t10_chunk < 4 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_5; ++i_t10_chunk) {
    dynamics_jac_znext_z_lanes_5(i_t10_chunk, SCALY_WIDTH_dynamics_jac_znext_z_lanes_5, s0);
  }
#if (4 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_5) != 0
  dynamics_jac_znext_z_lanes_5(4 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_5, 4 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_5, s0);
#endif
  for (long long k_t10 = 0; k_t10 < 2; ++k_t10) {
    s0[0] = (s0[0] + (s1[k_t10] * k1[k_t10]));
    int64_t v0 = (2 + k_t10);
    s0[1] = (s0[1] + (s1[v0] * k1[v0]));
    int64_t v1 = (4 + k_t10);
    s0[2] = (s0[2] + (s1[v1] * k1[v1]));
    int64_t v2 = (6 + k_t10);
    s0[3] = (s0[3] + (s1[v2] * k1[v2]));
  }
  for (long long i_t11_chunk = 0; i_t11_chunk < 4 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_6; ++i_t11_chunk) {
    dynamics_jac_znext_z_lanes_6(i_t11_chunk, SCALY_WIDTH_dynamics_jac_znext_z_lanes_6, s0, s1);
  }
#if (4 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_6) != 0
  dynamics_jac_znext_z_lanes_6(4 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_6, 4 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_6, s0, s1);
#endif
  for (long long j_t12_0 = 0; j_t12_0 < 2; ++j_t12_0) {
    s0[j_t12_0] = t5[j_t12_0];
  }
  for (long long j_t12_1 = 0; j_t12_1 < 2; ++j_t12_1) {
    s0[(2 + j_t12_1)] = t5[j_t12_1];
  }
  for (long long j_t12_2 = 0; j_t12_2 < 2; ++j_t12_2) {
    s0[(4 + j_t12_2)] = t5[j_t12_2];
  }
  for (long long j_t12_3 = 0; j_t12_3 < 2; ++j_t12_3) {
    s0[(6 + j_t12_3)] = t5[j_t12_3];
  }
  s2[0] = 0.0;
  for (long long i_t15 = 0; i_t15 < 2; ++i_t15) {
    double v3 = t5[i_t15];
    s2[0] = (s2[0] + (v3 * v3));
  }
  s3[0] = (0.10000000000000001 * s2[0]);
  for (long long j_t21_0 = 0; j_t21_0 < 8; ++j_t21_0) {
    s2[(((j_t21_0 / 2) * 4) + (j_t21_0 % 2))] = k0[j_t21_0];
  }
  for (long long j_t21_1 = 0; j_t21_1 < 8; ++j_t21_1) {
    int64_t v4 = (j_t21_1 / 2);
    double v5 = k1[j_t21_1];
    s2[((v4 * 4) + (2 + (j_t21_1 % 2)))] = (v5 - (0.050000000000000003 * ((s1[v4] * s0[j_t21_1]) + (s3[0] * v5))));
  }
  for (long long d0_jac_znext_z = 0; d0_jac_znext_z < 4; ++d0_jac_znext_z) {
    for (long long d1_jac_znext_z_chunk = 0; d1_jac_znext_z_chunk < 4 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_7; ++d1_jac_znext_z_chunk) {
      dynamics_jac_znext_z_lanes_7(d1_jac_znext_z_chunk, SCALY_WIDTH_dynamics_jac_znext_z_lanes_7, res[0], s2, d0_jac_znext_z);
    }
#if (4 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_7) != 0
    dynamics_jac_znext_z_lanes_7(4 / SCALY_WIDTH_dynamics_jac_znext_z_lanes_7, 4 % SCALY_WIDTH_dynamics_jac_znext_z_lanes_7, res[0], s2, d0_jac_znext_z);
#endif
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
