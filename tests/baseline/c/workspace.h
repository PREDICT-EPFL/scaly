/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c workspace.c
 * clang -O3 -fno-math-errno -c workspace.c
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

#ifndef SCALY_TYPES_workspace
#define SCALY_TYPES_workspace
typedef struct { SCALY_ALIGNAS(16) double data[16384]; } workspace_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } workspace_sum_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } workspace_sumsqr_t;
typedef struct {
  SCALY_ALIGNAS(64) double f64[16384];
} workspace_workspace_t;
#endif

#define workspace_SZ_ARG 1
#define workspace_SZ_RES 2
#define workspace_WORKSPACE_BYTES 131072
#define workspace_FRAME_BYTES 4096

#ifdef __cplusplus
extern "C" {
#endif
void workspace(const workspace_x_t* x, workspace_sum_t* sum, workspace_sumsqr_t* sumsqr, workspace_workspace_t* w);
int workspace_abi(const void* const* arg, void* const* res, workspace_workspace_t* w);
#ifdef __cplusplus
}
#endif

static inline int workspace_call(const workspace_x_t* x, workspace_sum_t* sum, workspace_sumsqr_t* sumsqr, workspace_workspace_t* workspace_2) {
  const void* arg[workspace_SZ_ARG > 0 ? workspace_SZ_ARG : 1] = {x->data};
  void* res[workspace_SZ_RES > 0 ? workspace_SZ_RES : 1] = {sum->data, sumsqr->data};
  return workspace_abi(arg, res, workspace_2);
}
