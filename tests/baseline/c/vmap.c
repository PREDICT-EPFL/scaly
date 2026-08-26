#include <math.h>
#include <stddef.h>
#include <stdint.h>

#define ALLOY_SUCCESS 0
#define ALLOY_ERR_NULL_ABI 1
#define ALLOY_ERR_NULL_WORK 2
#define ALLOY_ERR_NULL_RESULT 3
#define ALLOY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline void dynamics_raw(const double* z, const double* u, double* znext, double* w) {
  (void)w;
  const double* t0 = z;
  static const double k1[1] = {0.050000000000000003};
  const double* t2 = z + 2;
  static const double k5[1] = {0.10000000000000001};
  double s0[1];
  double s1[1];
  s0[0] = 0;
  for (long long i_t7 = 0; i_t7 < 2; ++i_t7) {
    s0[0] = (s0[0] + (t2[i_t7] * t2[i_t7]));
  }
  s1[0] = (k5[0] * s0[0]);
  for (long long j_znext_0 = 0; j_znext_0 < 2; ++j_znext_0) {
    znext[j_znext_0] = (t0[j_znext_0] + (k1[0] * t2[j_znext_0]));
  }
  for (long long j_znext_1 = 0; j_znext_1 < 2; ++j_znext_1) {
    znext[(2 + j_znext_1)] = (t2[j_znext_1] + (k1[0] * (u[j_znext_1] - (s1[0] * t2[j_znext_1]))));
  }
}

int shooting_sz_arg(void) { return 2; }
int shooting_sz_res(void) { return 1; }
int shooting_sz_iw(void) { return 0; }
int shooting_sz_w(void) { return 0; }
void* shooting_alloc_mem(void) { return NULL; }
int shooting_init_mem(void* mem) { (void)mem; return ALLOY_SUCCESS; }
void shooting_free_mem(void* mem) { (void)mem; }

int shooting(const double** arg, double** res, int* iw, double* w, void* mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return ALLOY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return ALLOY_ERR_NULL_INPUT;
  if (!arg[1]) return ALLOY_ERR_NULL_INPUT;
  if (!res[0]) return ALLOY_ERR_NULL_RESULT;
  double s0[12];
  const double* t1 = arg[0] + 4;
  for (long long it_t0 = 0; it_t0 < 3; ++it_t0) {
    dynamics_raw((arg[0] + (0 + (4 * it_t0))), (arg[1] + (0 + (2 * it_t0))), (s0 + (it_t0 * 4)), NULL);
  }
  for (long long i_eq = 0; i_eq < 12; ++i_eq) {
    res[0][i_eq] = (s0[i_eq] - t1[i_eq]);
  }
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
