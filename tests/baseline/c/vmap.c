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

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline __attribute__((always_inline)) void shooting_lanes_1(long long v0_chunk, long long shooting_lanes_1_valid, double* eq, const double* u, const double* z) {
  int64_t v1[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) v1[v0_lane] = (4 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2)));
  lanes_4_4_4_4_vec shooting_lanes_1_load_1;
  double shooting_lanes_1_load_1_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_1_stage[v0_lane] = z[(v1[v0_lane] + 2)];
  shooting_lanes_1_load_1 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_1_stage;
  lanes_4_4_4_4_vec v2 = shooting_lanes_1_load_1;
  lanes_4_4_4_4_vec shooting_lanes_1_load_2;
  double shooting_lanes_1_load_2_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_2_stage[v0_lane] = z[(v1[v0_lane] + 3)];
  shooting_lanes_1_load_2 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_2_stage;
  lanes_4_4_4_4_vec v3 = shooting_lanes_1_load_2;
  lanes_4_4_4_4_vec shooting_lanes_1_load_3;
  double shooting_lanes_1_load_3_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_3_stage[v0_lane] = z[v1[v0_lane]];
  shooting_lanes_1_load_3 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_3_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_broadcast_4;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_broadcast_4[v0_lane] = 0.050000000000000003;
  lanes_4_4_4_4_vec shooting_lanes_1_load_5;
  double shooting_lanes_1_load_5_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_5_stage[v0_lane] = z[(4 + v1[v0_lane])];
  shooting_lanes_1_load_5 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_5_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_store_6 = ((shooting_lanes_1_load_3 + (shooting_lanes_1_broadcast_4 * v2)) - shooting_lanes_1_load_5);
  for (long long v0_lane = 0; v0_lane < shooting_lanes_1_valid; ++v0_lane) eq[((4 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2)))) + 0] = shooting_lanes_1_store_6[v0_lane];
  lanes_4_4_4_4_vec shooting_lanes_1_load_7;
  double shooting_lanes_1_load_7_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_7_stage[v0_lane] = z[(v1[v0_lane] + 1)];
  shooting_lanes_1_load_7 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_7_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_broadcast_8;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_broadcast_8[v0_lane] = 0.050000000000000003;
  lanes_4_4_4_4_vec shooting_lanes_1_load_9;
  double shooting_lanes_1_load_9_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_9_stage[v0_lane] = z[(5 + v1[v0_lane])];
  shooting_lanes_1_load_9 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_9_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_store_10 = ((shooting_lanes_1_load_7 + (shooting_lanes_1_broadcast_8 * v3)) - shooting_lanes_1_load_9);
  for (long long v0_lane = 0; v0_lane < shooting_lanes_1_valid; ++v0_lane) eq[((1 + (4 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = shooting_lanes_1_store_10[v0_lane];
  int64_t v4[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) v4[v0_lane] = (2 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2)));
  lanes_4_4_4_4_vec shooting_lanes_1_broadcast_11;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_broadcast_11[v0_lane] = 0.10000000000000001;
  lanes_4_4_4_4_vec v5 = (shooting_lanes_1_broadcast_11 * ((v2 * v2) + (v3 * v3)));
  lanes_4_4_4_4_vec shooting_lanes_1_broadcast_12;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_broadcast_12[v0_lane] = 0.050000000000000003;
  lanes_4_4_4_4_vec shooting_lanes_1_load_13;
  double shooting_lanes_1_load_13_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_13_stage[v0_lane] = u[v4[v0_lane]];
  shooting_lanes_1_load_13 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_13_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_load_14;
  double shooting_lanes_1_load_14_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_14_stage[v0_lane] = z[(6 + v1[v0_lane])];
  shooting_lanes_1_load_14 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_14_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_store_15 = ((v2 + (shooting_lanes_1_broadcast_12 * (shooting_lanes_1_load_13 - (v5 * v2)))) - shooting_lanes_1_load_14);
  for (long long v0_lane = 0; v0_lane < shooting_lanes_1_valid; ++v0_lane) eq[((2 + (4 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = shooting_lanes_1_store_15[v0_lane];
  lanes_4_4_4_4_vec shooting_lanes_1_broadcast_16;
  for (int v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_broadcast_16[v0_lane] = 0.050000000000000003;
  lanes_4_4_4_4_vec shooting_lanes_1_load_17;
  double shooting_lanes_1_load_17_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_17_stage[v0_lane] = u[(v4[v0_lane] + 1)];
  shooting_lanes_1_load_17 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_17_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_load_18;
  double shooting_lanes_1_load_18_stage[8];
  for (long long v0_lane = 0; v0_lane < SCALY_WIDTH_4_4_4_4; ++v0_lane) shooting_lanes_1_load_18_stage[v0_lane] = z[(7 + v1[v0_lane])];
  shooting_lanes_1_load_18 = *(const lanes_4_4_4_4_vec_mem*)shooting_lanes_1_load_18_stage;
  lanes_4_4_4_4_vec shooting_lanes_1_store_19 = ((v3 + (shooting_lanes_1_broadcast_16 * (shooting_lanes_1_load_17 - (v5 * v3)))) - shooting_lanes_1_load_18);
  for (long long v0_lane = 0; v0_lane < shooting_lanes_1_valid; ++v0_lane) eq[((3 + (4 * (0 + (((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) < 2 ? ((v0_chunk * SCALY_WIDTH_4_4_4_4) + v0_lane) : 2))))) + 0] = shooting_lanes_1_store_19[v0_lane];
}
int shooting(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  for (long long v0_chunk = 0; v0_chunk < 3 / SCALY_WIDTH_4_4_4_4; ++v0_chunk) {
    shooting_lanes_1(v0_chunk, SCALY_WIDTH_4_4_4_4, res[0], arg[1], arg[0]);
  }
#if (3 % SCALY_WIDTH_4_4_4_4) != 0
  shooting_lanes_1(3 / SCALY_WIDTH_4_4_4_4, 3 % SCALY_WIDTH_4_4_4_4, res[0], arg[1], arg[0]);
#endif
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
