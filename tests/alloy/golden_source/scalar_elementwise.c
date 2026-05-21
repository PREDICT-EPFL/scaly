#include <math.h>
#include <stddef.h>

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

#ifdef __cplusplus
extern "C" {
#endif

static void scalar_elem_raw(const double* in0, double* out0, double* w) {
  (void)w;
  double s0[3];
  double s1[1];
  const double* v0 = in0;
  s0[0] = sin(v0[0]) + (v0[0] * v0[0]);
  s0[1] = sin(v0[1]) + (v0[1] * v0[1]);
  s0[2] = sin(v0[2]) + (v0[2] * v0[2]);
  s1[0] = 0.0;
  s1[0] += s0[0];
  s1[0] += s0[1];
  s1[0] += s0[2];
  out0[0] = s1[0];
}

int scalar_elem_sz_arg(void) { return 1; }
int scalar_elem_sz_res(void) { return 1; }
int scalar_elem_sz_iw(void) { return 0; }
int scalar_elem_sz_w(void) { return 0; }
void* scalar_elem_alloc_mem(void) { return NULL; }
int scalar_elem_init_mem(void* mem) { (void)mem; return ALLOY_SUCCESS; }
void scalar_elem_free_mem(void* mem) { (void)mem; }

int scalar_elem(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return ALLOY_ERR_NULL_ABI;
  if (!arg[0]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  scalar_elem_raw(arg[0], res[0], w);
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
