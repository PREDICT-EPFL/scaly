/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c seed_pair_map.c
 * clang -O3 -fno-math-errno -c seed_pair_map.c
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

#define seed_pair_map_SZ_ARG 2
#define seed_pair_map_SZ_RES 1
#define seed_pair_map_SZ_IW 0
#define seed_pair_map_SZ_W 0
#define seed_pair_map_FRAME_BYTES 5952

// The pointer ABI for seed_pair_map.
#ifdef __cplusplus
extern "C" {
#endif
int seed_pair_map(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[12]; } seed_pair_map_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[144]; } seed_pair_map_d_t;
typedef struct { SCALY_ALIGNAS(16) double data[36]; } seed_pair_map_y_t;
typedef struct { SCALY_ALIGNAS(16) double data[seed_pair_map_SZ_W > 0 ? seed_pair_map_SZ_W : 1]; } seed_pair_map_workspace_t;
static inline int seed_pair_map_call(const seed_pair_map_x_t* x, const seed_pair_map_d_t* d, seed_pair_map_y_t* y, seed_pair_map_workspace_t* workspace) {
  const double* arg[seed_pair_map_SZ_ARG > 0 ? seed_pair_map_SZ_ARG : 1] = {x->data, d->data};
  double* res[seed_pair_map_SZ_RES > 0 ? seed_pair_map_SZ_RES : 1] = {y->data};
  return seed_pair_map(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
