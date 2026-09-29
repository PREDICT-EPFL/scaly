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

#define control_SZ_ARG 2
#define control_SZ_RES 2
#define control_SZ_IW 0
#define control_SZ_W 0

// The pointer ABI for control.
#ifdef __cplusplus
extern "C" {
#endif
int control(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[3]; } control_c0_t;
typedef struct { SCALY_ALIGNAS(16) double data[6]; } control_us_t;
typedef struct { SCALY_ALIGNAS(16) double data[3]; } control_final_t;
typedef struct { SCALY_ALIGNAS(16) double data[3]; } control_peaks_t;
typedef struct { SCALY_ALIGNAS(16) double data[control_SZ_W > 0 ? control_SZ_W : 1]; } control_workspace_t;
static inline int control_call(const control_c0_t* c0, const control_us_t* us, control_final_t* final, control_peaks_t* peaks, control_workspace_t* workspace) {
  const double* arg[control_SZ_ARG > 0 ? control_SZ_ARG : 1] = {c0->data, us->data};
  double* res[control_SZ_RES > 0 ? control_SZ_RES : 1] = {final->data, peaks->data};
  return control(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
