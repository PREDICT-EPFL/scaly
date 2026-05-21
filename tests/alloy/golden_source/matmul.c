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

static void mm_raw(const double* in0, const double* in1, double* out0, double* w) {
  (void)w;
  double s0[6];
  const double* v0 = in0;
  const double* v1 = in1;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 2; ++j) {
      s0[i * 2 + j] = 0.0;
      for (int k = 0; k < 4; ++k) s0[i * 2 + j] += v0[i * 4 + k] * v1[k * 2 + j];
    }
  }
  out0[0] = s0[0];
  out0[1] = s0[1];
  out0[2] = s0[2];
  out0[3] = s0[3];
  out0[4] = s0[4];
  out0[5] = s0[5];
}

int mm_sz_arg(void) { return 2; }
int mm_sz_res(void) { return 1; }
int mm_sz_iw(void) { return 0; }
int mm_sz_w(void) { return 0; }
void* mm_alloc_mem(void) { return NULL; }
int mm_init_mem(void* mem) { (void)mem; return ALLOY_SUCCESS; }
void mm_free_mem(void* mem) { (void)mem; }

int mm(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return ALLOY_ERR_NULL_ABI;
  if (!arg[0]) return ALLOY_ERR_NULL_INPUT;
  if (!arg[1]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  mm_raw(arg[0], arg[1], res[0], w);
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
