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
#define SCALY_WIDTH_shooting_spjac_eq_z_lanes_1 (SCALY_LANES < (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4) ? SCALY_LANES : (SCALY_REGISTER_SLOTS == 256 ? 4 : SCALY_REGISTER_SLOTS == 64 ? 4 : SCALY_REGISTER_SLOTS == 32 ? 4 : 4))

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

typedef double shooting_spjac_eq_z_lanes_1_vec __attribute__((vector_size(8 * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1)));
typedef shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_vec_mem __attribute__((aligned(8), may_alias));
static inline __attribute__((always_inline)) void shooting_spjac_eq_z_lanes_1(long long v0_chunk, long long shooting_spjac_eq_z_lanes_1_valid, double* spjac_eq_z, const double* z) {
  int64_t v1[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) v1[v0_lane] = (4 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2)));
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_load_1;
  double shooting_spjac_eq_z_lanes_1_load_1_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_load_1_stage[v0_lane] = z[(v1[v0_lane] + 2)];
  shooting_spjac_eq_z_lanes_1_load_1 = *(const shooting_spjac_eq_z_lanes_1_vec_mem*)shooting_spjac_eq_z_lanes_1_load_1_stage;
  shooting_spjac_eq_z_lanes_1_vec v2 = shooting_spjac_eq_z_lanes_1_load_1;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_load_2;
  double shooting_spjac_eq_z_lanes_1_load_2_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_load_2_stage[v0_lane] = z[(v1[v0_lane] + 3)];
  shooting_spjac_eq_z_lanes_1_load_2 = *(const shooting_spjac_eq_z_lanes_1_vec_mem*)shooting_spjac_eq_z_lanes_1_load_2_stage;
  shooting_spjac_eq_z_lanes_1_vec v4 = shooting_spjac_eq_z_lanes_1_load_2;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_3;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_3[v0_lane] = 1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_4 = shooting_spjac_eq_z_lanes_1_broadcast_3;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2)))) + 0] = shooting_spjac_eq_z_lanes_1_store_4[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_5;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_5[v0_lane] = 1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_6 = shooting_spjac_eq_z_lanes_1_broadcast_5;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((3 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_6[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_7;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_7[v0_lane] = 0.050000000000000003;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_8 = shooting_spjac_eq_z_lanes_1_broadcast_7;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((1 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_8[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_9;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_9[v0_lane] = 0.050000000000000003;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_10 = shooting_spjac_eq_z_lanes_1_broadcast_9;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((4 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_10[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_11;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_11[v0_lane] = -1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_12 = shooting_spjac_eq_z_lanes_1_broadcast_11;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((2 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_12[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_13;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_13[v0_lane] = -1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_14 = shooting_spjac_eq_z_lanes_1_broadcast_13;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((5 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_14[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_15;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_15[v0_lane] = -1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_16 = shooting_spjac_eq_z_lanes_1_broadcast_15;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((8 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_16[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_17;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_17[v0_lane] = -1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_18 = shooting_spjac_eq_z_lanes_1_broadcast_17;
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((11 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_18[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_19;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_19[v0_lane] = 0.10000000000000001;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_20;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_20[v0_lane] = 2.0;
  shooting_spjac_eq_z_lanes_1_vec v3 = (shooting_spjac_eq_z_lanes_1_broadcast_19 * (shooting_spjac_eq_z_lanes_1_broadcast_20 * v2));
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_21;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_21[v0_lane] = 0.10000000000000001;
  shooting_spjac_eq_z_lanes_1_vec v5 = (shooting_spjac_eq_z_lanes_1_broadcast_21 * ((v2 * v2) + (v4 * v4)));
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_22;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_22[v0_lane] = 1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_23;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_23[v0_lane] = 0.050000000000000003;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_24 = (shooting_spjac_eq_z_lanes_1_broadcast_22 - (shooting_spjac_eq_z_lanes_1_broadcast_23 * ((v3 * v2) + v5)));
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((6 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_24[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_25;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_25[v0_lane] = 0.10000000000000001;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_26;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_26[v0_lane] = 2.0;
  shooting_spjac_eq_z_lanes_1_vec v6 = (shooting_spjac_eq_z_lanes_1_broadcast_25 * (shooting_spjac_eq_z_lanes_1_broadcast_26 * v4));
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_27;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_27[v0_lane] = 0.050000000000000003;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_28 = (-(shooting_spjac_eq_z_lanes_1_broadcast_27 * (v6 * v2)));
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((7 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_28[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_29;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_29[v0_lane] = 0.050000000000000003;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_30 = (-(shooting_spjac_eq_z_lanes_1_broadcast_29 * (v3 * v4)));
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((9 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_30[v0_lane];
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_31;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_31[v0_lane] = 1.0;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_broadcast_32;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_lane) shooting_spjac_eq_z_lanes_1_broadcast_32[v0_lane] = 0.050000000000000003;
  shooting_spjac_eq_z_lanes_1_vec shooting_spjac_eq_z_lanes_1_store_33 = (shooting_spjac_eq_z_lanes_1_broadcast_31 - (shooting_spjac_eq_z_lanes_1_broadcast_32 * ((v6 * v4) + v5)));
  for (long long v0_lane = 0; v0_lane < shooting_spjac_eq_z_lanes_1_valid; ++v0_lane) spjac_eq_z[((10 + (12 * (0 + (((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) + v0_lane) : 2))))) + 0] = shooting_spjac_eq_z_lanes_1_store_33[v0_lane];
}
int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  for (long long v0_chunk = 0; v0_chunk < 3 / SCALY_WIDTH_shooting_spjac_eq_z_lanes_1; ++v0_chunk) {
    shooting_spjac_eq_z_lanes_1(v0_chunk, SCALY_WIDTH_shooting_spjac_eq_z_lanes_1, res[0], arg[0]);
  }
#if (3 % SCALY_WIDTH_shooting_spjac_eq_z_lanes_1) != 0
  shooting_spjac_eq_z_lanes_1(3 / SCALY_WIDTH_shooting_spjac_eq_z_lanes_1, 3 % SCALY_WIDTH_shooting_spjac_eq_z_lanes_1, res[0], arg[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
