/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c wide.c
 * clang -O3 -fno-math-errno -c wide.c
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

#ifndef SCALY_TYPES_wide
#define SCALY_TYPES_wide
typedef struct { SCALY_ALIGNAS(16) double data[40]; } wide_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[40]; } wide_y_t;
typedef struct { SCALY_ALIGNAS(16) double data[40]; } wide_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[4]; } wide_tail_t;
typedef struct {
  unsigned char unused;
} wide_workspace_t;
#endif

#define wide_SZ_ARG 2
#define wide_SZ_RES 2
#define wide_WORKSPACE_BYTES 0
#define wide_FRAME_BYTES 4736

#ifdef __cplusplus
extern "C" {
#endif
void wide(const wide_x_t* x, const wide_y_t* y, wide_z_t* z, wide_tail_t* tail, wide_workspace_t* w);
int wide_abi(const void* const* arg, void* const* res, wide_workspace_t* w);
#ifdef __cplusplus
}
#endif

static inline int wide_call(const wide_x_t* x, const wide_y_t* y, wide_z_t* z, wide_tail_t* tail, wide_workspace_t* workspace) {
  const void* arg[wide_SZ_ARG > 0 ? wide_SZ_ARG : 1] = {x->data, y->data};
  void* res[wide_SZ_RES > 0 ? wide_SZ_RES : 1] = {z->data, tail->data};
  return wide_abi(arg, res, workspace);
}
