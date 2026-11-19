/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c dynamics_float32.c
 * clang -O3 -fno-math-errno -c dynamics_float32.c
 * Link with: -lm
 */
#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

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
#ifndef SCALY_TYPES_dynamics_float32
#define SCALY_TYPES_dynamics_float32
typedef struct { SCALY_ALIGNAS(16) float data[4]; } dynamics_float32_z_t;
typedef struct { SCALY_ALIGNAS(16) float data[2]; } dynamics_float32_u_t;
typedef struct { SCALY_ALIGNAS(16) float data[4]; } dynamics_float32_znext_t;
typedef struct {
  unsigned char unused;
} dynamics_float32_workspace_t;
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

static inline void SCALY_KERNEL_dynamics_float32(const float* z, const float* u, float* znext) {
  float s0[1];
  float s1[1];
  float acc_t0 = 0.0f;
  for (long long i_t0 = 0; i_t0 < 2; ++i_t0) {
    float v0 = z[(2 + i_t0)];
    acc_t0 = (acc_t0 + (v0 * v0));
  }
  s0[0] = acc_t0;
  s1[0] = (0.100000001f * s0[0]);
  for (long long j_znext_0 = 0; j_znext_0 < 2; ++j_znext_0) {
    znext[j_znext_0] = (z[j_znext_0] + (0.0500000007f * z[(2 + j_znext_0)]));
  }
  for (long long j_znext_1 = 0; j_znext_1 < 2; ++j_znext_1) {
    int64_t v1 = (2 + j_znext_1);
    float v2 = z[v1];
    znext[v1] = (v2 + (0.0500000007f * (u[j_znext_1] - (s1[0] * v2))));
  }
}

int dynamics_float32_abi(const void* const* arg, void* const* res, dynamics_float32_workspace_t* w) {
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  SCALY_KERNEL_dynamics_float32((const float*)arg[0], (const float*)arg[1], (float*)res[0]);
  return SCALY_SUCCESS;
}

void dynamics_float32(const dynamics_float32_z_t* z, const dynamics_float32_u_t* u, dynamics_float32_znext_t* znext, dynamics_float32_workspace_t* w) {
  (void)w;
  SCALY_KERNEL_dynamics_float32((const float*)z->data, (const float*)u->data, (float*)znext->data);
}

#ifdef __cplusplus
}
#endif
