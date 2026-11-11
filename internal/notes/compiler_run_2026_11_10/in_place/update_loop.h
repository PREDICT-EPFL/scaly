/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c update_loop.c
 * clang -O3 -fno-math-errno -c update_loop.c
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

#ifndef SCALY_ALIGNAS
#ifdef __cplusplus
#define SCALY_ALIGNAS(n) alignas(n)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SCALY_ALIGNAS(n) _Alignas(n)
#else
#define SCALY_ALIGNAS(n)
#endif
#endif

#define update_loop_SZ_ARG 1
#define update_loop_SZ_RES 1
#define update_loop_SZ_IW 0
#define update_loop_SZ_W 100000

// The pointer ABI for update_loop.
#ifdef __cplusplus
extern "C" {
#endif
int update_loop(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[100000]; } update_loop_initial_t;
typedef struct { SCALY_ALIGNAS(16) double data[100000]; } update_loop_update_loop__t;
typedef struct { SCALY_ALIGNAS(16) double data[update_loop_SZ_W > 0 ? update_loop_SZ_W : 1]; } update_loop_workspace_t;
static inline int update_loop_call(const update_loop_initial_t* initial, update_loop_update_loop__t* update_loop_, update_loop_workspace_t* workspace) {
  const double* arg[update_loop_SZ_ARG > 0 ? update_loop_SZ_ARG : 1] = {initial->data};
  double* res[update_loop_SZ_RES > 0 ? update_loop_SZ_RES : 1] = {update_loop_->data};
  return update_loop(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
