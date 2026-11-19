/* Scaly build recipe
 * CPU baseline: generic
 * lanes=auto, dialect=gnu, vector_libm=none, reciprocal=False, frame_budget=65536
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

#ifndef SCALY_TYPES_shooting
#define SCALY_TYPES_shooting
typedef struct { SCALY_ALIGNAS(16) double data[16]; } shooting_z_t;
typedef struct { SCALY_ALIGNAS(16) double data[6]; } shooting_u_t;
typedef struct { SCALY_ALIGNAS(16) double data[12]; } shooting_eq_t;
typedef struct {
  unsigned char unused;
} shooting_workspace_t;
#endif

#define shooting_SZ_ARG 2
#define shooting_SZ_RES 1
#define shooting_WORKSPACE_BYTES 0
#define shooting_FRAME_BYTES 4096

#ifdef __cplusplus
extern "C" {
#endif
void shooting(const shooting_z_t* z, const shooting_u_t* u, shooting_eq_t* eq, shooting_workspace_t* w);
int shooting_abi(const void* const* arg, void* const* res, shooting_workspace_t* w);
#ifdef __cplusplus
}
#endif

static inline int shooting_call(const shooting_z_t* z, const shooting_u_t* u, shooting_eq_t* eq, shooting_workspace_t* workspace) {
  const void* arg[shooting_SZ_ARG > 0 ? shooting_SZ_ARG : 1] = {z->data, u->data};
  void* res[shooting_SZ_RES > 0 ? shooting_SZ_RES : 1] = {eq->data};
  return shooting_abi(arg, res, workspace);
}
