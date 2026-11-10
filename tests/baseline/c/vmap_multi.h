/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
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

#ifndef SCALY_ALIGNAS
#ifdef __cplusplus
#define SCALY_ALIGNAS(n) alignas(n)
#elif defined(__STDC_VERSION__) && __STDC_VERSION__ >= 201112L
#define SCALY_ALIGNAS(n) _Alignas(n)
#else
#define SCALY_ALIGNAS(n)
#endif
#endif

#define multi_output_SZ_ARG 1
#define multi_output_SZ_RES 2
#define multi_output_SZ_IW 0
#define multi_output_SZ_W 0

// The pointer ABI for multi_output.
#ifdef __cplusplus
extern "C" {
#endif
int multi_output(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[10]; } multi_output_xs_t;
typedef struct { SCALY_ALIGNAS(16) double data[10]; } multi_output_vectors_t;
typedef struct { SCALY_ALIGNAS(16) double data[5]; } multi_output_totals_t;
typedef struct { SCALY_ALIGNAS(16) double data[multi_output_SZ_W > 0 ? multi_output_SZ_W : 1]; } multi_output_workspace_t;
static inline int multi_output_call(const multi_output_xs_t* xs, multi_output_vectors_t* vectors, multi_output_totals_t* totals, multi_output_workspace_t* workspace) {
  const double* arg[multi_output_SZ_ARG > 0 ? multi_output_SZ_ARG : 1] = {xs->data};
  double* res[multi_output_SZ_RES > 0 ? multi_output_SZ_RES : 1] = {vectors->data, totals->data};
  return multi_output(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
