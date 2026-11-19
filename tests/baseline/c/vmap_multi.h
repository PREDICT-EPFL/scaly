/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c multi_output.c
 * clang -O3 -fno-math-errno -c multi_output.c
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

#ifndef SCALY_TYPES_multi_output
#define SCALY_TYPES_multi_output
typedef struct { SCALY_ALIGNAS(16) double data[10]; } multi_output_xs_t;
typedef struct { SCALY_ALIGNAS(16) double data[10]; } multi_output_vectors_t;
typedef struct { SCALY_ALIGNAS(16) double data[5]; } multi_output_totals_t;
typedef struct {
  unsigned char unused;
} multi_output_workspace_t;
#endif

#define multi_output_SZ_ARG 1
#define multi_output_SZ_RES 2
#define multi_output_SZ_IW 0
#define multi_output_WORKSPACE_BYTES 0
#define multi_output_FRAME_BYTES 4176

#ifdef __cplusplus
extern "C" {
#endif
void multi_output(const multi_output_xs_t* xs, multi_output_vectors_t* vectors, multi_output_totals_t* totals, multi_output_workspace_t* w);
int multi_output_abi(const void* const* arg, void* const* res, multi_output_workspace_t* w);
#ifdef __cplusplus
}
#endif

static inline int multi_output_call(const multi_output_xs_t* xs, multi_output_vectors_t* vectors, multi_output_totals_t* totals, multi_output_workspace_t* workspace) {
  const void* arg[multi_output_SZ_ARG > 0 ? multi_output_SZ_ARG : 1] = {xs->data};
  void* res[multi_output_SZ_RES > 0 ? multi_output_SZ_RES : 1] = {vectors->data, totals->data};
  return multi_output_abi(arg, res, workspace);
}
