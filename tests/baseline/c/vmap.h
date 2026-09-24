/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False
 * Math library: scalar libm
 * gcc -O3 -fno-math-errno -c shooting.c
 * clang -O3 -fno-math-errno -c shooting.c
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

#define shooting_SZ_ARG 2
#define shooting_SZ_RES 1
#define shooting_SZ_IW 0
#define shooting_SZ_W 0

// The pointer ABI for shooting.
#ifdef __cplusplus
extern "C" {
#endif
int shooting(const double** arg, double** res, int* iw, double* w, int mem);
#ifdef __cplusplus
}
#endif

// Typed buffers: one struct per input and output, and the caller-owned workspace.
typedef struct { SCALY_ALIGNAS(16) double data[16]; } shooting_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[6]; } shooting_u_t;
typedef struct { SCALY_ALIGNAS(16) double data[12]; } shooting_eq_t;
typedef struct { SCALY_ALIGNAS(16) double data[shooting_SZ_W > 0 ? shooting_SZ_W : 1]; } shooting_workspace_t;
static inline int shooting_call(const shooting_z_t* z, const shooting_u_t* u, shooting_eq_t* eq, shooting_workspace_t* workspace) {
  const double* arg[shooting_SZ_ARG > 0 ? shooting_SZ_ARG : 1] = {z->data, u->data};
  double* res[shooting_SZ_RES > 0 ? shooting_SZ_RES : 1] = {eq->data};
  return shooting(arg, res, NULL, workspace ? workspace->data : NULL, 0);
}
