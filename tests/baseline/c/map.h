#pragma once

#ifndef ALLOY_SUCCESS
#define ALLOY_SUCCESS 0
#endif
#ifndef ALLOY_ERR_NULL_ABI
#define ALLOY_ERR_NULL_ABI 1
#endif
#ifndef ALLOY_ERR_NULL_WORK
#define ALLOY_ERR_NULL_WORK 2
#endif
#ifndef ALLOY_ERR_NULL_RESULT
#define ALLOY_ERR_NULL_RESULT 3
#endif
#ifndef ALLOY_ERR_NULL_INPUT
#define ALLOY_ERR_NULL_INPUT 4
#endif

#define shooting_SZ_ARG 2
#define shooting_SZ_RES 1
#define shooting_SZ_IW 0
#define shooting_SZ_W 0

// Universal CasADi-style ABI for shooting.
#ifdef __cplusplus
extern "C" {
#endif
int shooting(const double** arg, double** res, int* iw, double* w, void* mem);
int shooting_sz_arg(void);
int shooting_sz_res(void);
int shooting_sz_iw(void);
int shooting_sz_w(void);
void* shooting_alloc_mem(void);
int shooting_init_mem(void* mem);
void shooting_free_mem(void* mem);
#ifdef __cplusplus
}
#endif

// Optional typed buffer wrappers for statically known shapes.
typedef struct { double data[16]; } shooting_z_in;
typedef struct { double data[6]; } shooting_u_in;
typedef struct { double data[12]; } shooting_eq_out;
#ifdef __cplusplus
static_assert(sizeof(shooting_z_in) == sizeof(double) * 16, "shooting_z_in size mismatch");
static_assert(sizeof(shooting_u_in) == sizeof(double) * 6, "shooting_u_in size mismatch");
static_assert(sizeof(shooting_eq_out) == sizeof(double) * 12, "shooting_eq_out size mismatch");
static inline int shooting_call(const shooting_z_in& in_z, const shooting_u_in& in_u, shooting_eq_out& out_eq) {
  double w[shooting_SZ_W > 0 ? shooting_SZ_W : 1];
  const double* arg[shooting_SZ_ARG > 0 ? shooting_SZ_ARG : 1] = {in_z.data, in_u.data};
  double* res[shooting_SZ_RES > 0 ? shooting_SZ_RES : 1] = {out_eq.data};
  return shooting(arg, res, nullptr, shooting_SZ_W ? w : nullptr, nullptr);
}
#endif
