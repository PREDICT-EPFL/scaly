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

#define table_SZ_ARG 1
#define table_SZ_RES 1
#define table_SZ_IW 0
#define table_SZ_W 0

// The pointer ABI for table.
#ifdef __cplusplus
extern "C" {
#endif
int table(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[1]; } table_x_t;
typedef struct { SCALY_ALIGNAS(16) double data[1]; } table_y_t;
typedef struct { SCALY_ALIGNAS(16) double data[table_SZ_W > 0 ? table_SZ_W : 1]; } table_workspace_t;
static inline int table_call(const table_x_t* x, table_y_t* y, table_workspace_t* workspace) {
  const double* arg[table_SZ_ARG > 0 ? table_SZ_ARG : 1] = {x->data};
  double* res[table_SZ_RES > 0 ? table_SZ_RES : 1] = {y->data};
  return table(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
