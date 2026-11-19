/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c dynamics.c
 * clang -O3 -fno-math-errno -c dynamics.c
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
#ifndef SCALY_TYPES_dynamics
#define SCALY_TYPES_dynamics
typedef struct { SCALY_ALIGNAS(16) double data[4]; } dynamics_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[2]; } dynamics_u_t;
typedef struct { SCALY_ALIGNAS(16) double data[4]; } dynamics_znext_t;
typedef struct {
  unsigned char unused;
} dynamics_workspace_t;
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

static inline void SCALY_KERNEL_dynamics(const double* z, const double* u, double* znext) {
  double s0[1];
  double s1[1];
  double acc_t0 = 0.0;
  for (long long i_t0 = 0; i_t0 < 2; ++i_t0) {
    double v0 = z[(2 + i_t0)];
    acc_t0 = (acc_t0 + (v0 * v0));
  }
  s0[0] = acc_t0;
  s1[0] = (0.10000000000000001 * s0[0]);
  for (long long j_znext_0 = 0; j_znext_0 < 2; ++j_znext_0) {
    znext[j_znext_0] = (z[j_znext_0] + (0.050000000000000003 * z[(2 + j_znext_0)]));
  }
  for (long long j_znext_1 = 0; j_znext_1 < 2; ++j_znext_1) {
    int64_t v1 = (2 + j_znext_1);
    double v2 = z[v1];
    znext[v1] = (v2 + (0.050000000000000003 * (u[j_znext_1] - (s1[0] * v2))));
  }
}

int dynamics_abi(const void* const* arg, void* const* res, dynamics_workspace_t* w) {
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  SCALY_KERNEL_dynamics((const double*)arg[0], (const double*)arg[1], (double*)res[0]);
  return SCALY_SUCCESS;
}

void dynamics(const dynamics_z_t* z, const dynamics_u_t* u, dynamics_znext_t* znext, dynamics_workspace_t* w) {
  (void)w;
  SCALY_KERNEL_dynamics((const double*)z->data, (const double*)u->data, (double*)znext->data);
}

#ifdef __cplusplus
}
#endif
