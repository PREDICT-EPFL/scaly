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

#define workspace_SZ_ARG 1
#define workspace_SZ_RES 2
#define workspace_SZ_IW 0
#define workspace_SZ_W 2048

// The pointer ABI for workspace.
#ifdef __cplusplus
extern "C" {
#endif
int workspace(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[2048]; } workspace_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } workspace_sum_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } workspace_sumsqr_t;
typedef struct { SCALY_ALIGNAS(16) double data[workspace_SZ_W > 0 ? workspace_SZ_W : 1]; } workspace_workspace_t;
static inline int workspace_call(const workspace_x_t* x, workspace_sum_t* sum, workspace_sumsqr_t* sumsqr, workspace_workspace_t* workspace) {
  const double* arg[workspace_SZ_ARG > 0 ? workspace_SZ_ARG : 1] = {x->data};
  double* res[workspace_SZ_RES > 0 ? workspace_SZ_RES : 1] = {sum->data, sumsqr->data};
  return workspace(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
