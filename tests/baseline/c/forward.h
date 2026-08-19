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

#define dynamics_SZ_ARG 2
#define dynamics_SZ_RES 1
#define dynamics_SZ_IW 0
#define dynamics_SZ_W 0

// Universal CasADi-style ABI for dynamics.
#ifdef __cplusplus
extern "C" {
#endif
int dynamics(const double** arg, double** res, int* iw, double* w, void* mem);
int dynamics_sz_arg(void);
int dynamics_sz_res(void);
int dynamics_sz_iw(void);
int dynamics_sz_w(void);
void* dynamics_alloc_mem(void);
int dynamics_init_mem(void* mem);
void dynamics_free_mem(void* mem);
#ifdef __cplusplus
}
#endif

// Optional typed buffer wrappers for statically known shapes.
typedef struct { double data[4]; } dynamics_z_in;
typedef struct { double data[2]; } dynamics_u_in;
typedef struct { double data[4]; } dynamics_znext_out;
#ifdef __cplusplus
static_assert(sizeof(dynamics_z_in) == sizeof(double) * 4, "dynamics_z_in size mismatch");
static_assert(sizeof(dynamics_u_in) == sizeof(double) * 2, "dynamics_u_in size mismatch");
static_assert(sizeof(dynamics_znext_out) == sizeof(double) * 4, "dynamics_znext_out size mismatch");
static inline int dynamics_call(const dynamics_z_in& in_z, const dynamics_u_in& in_u, dynamics_znext_out& out_znext) {
  double w[dynamics_SZ_W > 0 ? dynamics_SZ_W : 1];
  const double* arg[dynamics_SZ_ARG > 0 ? dynamics_SZ_ARG : 1] = {in_z.data, in_u.data};
  double* res[dynamics_SZ_RES > 0 ? dynamics_SZ_RES : 1] = {out_znext.data};
  return dynamics(arg, res, nullptr, dynamics_SZ_W ? w : nullptr, nullptr);
}
#endif
