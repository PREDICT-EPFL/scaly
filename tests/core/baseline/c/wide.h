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
#else
#define SCALY_ALIGNAS(n) _Alignas(n)
#endif
#endif

#define wide_SZ_ARG 2
#define wide_SZ_RES 2
#define wide_SZ_IW 0
#define wide_SZ_W 0

// The pointer ABI for wide.
#ifdef __cplusplus
extern "C" {
#endif
int wide(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[40]; } wide_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[40]; } wide_y_t;
typedef struct { SCALY_ALIGNAS(16) double data[40]; } wide_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[4]; } wide_tail_t;
typedef struct { SCALY_ALIGNAS(16) double data[wide_SZ_W > 0 ? wide_SZ_W : 1]; } wide_workspace_t;
static inline int wide_call(const wide_x_t* x, const wide_y_t* y, wide_z_t* z, wide_tail_t* tail, wide_workspace_t* workspace) {
  const double* arg[wide_SZ_ARG > 0 ? wide_SZ_ARG : 1] = {x->data, y->data};
  double* res[wide_SZ_RES > 0 ? wide_SZ_RES : 1] = {z->data, tail->data};
  return wide(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
