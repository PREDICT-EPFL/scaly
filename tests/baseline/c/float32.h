/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c dynamics_float32.c
 * clang -O3 -fno-math-errno -c dynamics_float32.c
 * Link with: -lm
 */
#pragma once

#include <stddef.h>

#ifndef SCALY_SUCCESS
#define SCALY_SUCCESS 0
#endif
#ifndef SCALY_ERR_NULL_ABI
#define SCALY_ERR_NULL_ABI 1
#endif
#ifndef SCALY_ERR_NULL_WORK
#define SCALY_ERR_NULL_WORK 2
#endif
#ifndef SCALY_ERR_NULL_RESULT
#define SCALY_ERR_NULL_RESULT 3
#endif
#ifndef SCALY_ERR_NULL_INPUT
#define SCALY_ERR_NULL_INPUT 4
#endif
#ifndef SCALY_ERR_ALLOC_WORK
#define SCALY_ERR_ALLOC_WORK 5
#endif

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

#define dynamics_float32_SZ_ARG 2
#define dynamics_float32_SZ_RES 1
#define dynamics_float32_SZ_IW 0
#define dynamics_float32_WORKSPACE_BYTES 0
#define dynamics_float32_FRAME_BYTES 4104

#ifdef __cplusplus
extern "C" {
#endif
void dynamics_float32(const dynamics_float32_z_t* z, const dynamics_float32_u_t* u, dynamics_float32_znext_t* znext, dynamics_float32_workspace_t* w);
int dynamics_float32_abi(const void* const* arg, void* const* res, dynamics_float32_workspace_t* w);
#ifdef __cplusplus
}
#endif

static inline int dynamics_float32_call(const dynamics_float32_z_t* z, const dynamics_float32_u_t* u, dynamics_float32_znext_t* znext, dynamics_float32_workspace_t* workspace) {
  const void* arg[dynamics_float32_SZ_ARG > 0 ? dynamics_float32_SZ_ARG : 1] = {z->data, u->data};
  void* res[dynamics_float32_SZ_RES > 0 ? dynamics_float32_SZ_RES : 1] = {znext->data};
  return dynamics_float32_abi(arg, res, workspace);
}
