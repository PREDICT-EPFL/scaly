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
  double v0 = z[2];
  double v1 = z[3];
  double v2 = (0.10000000000000001 * ((v0 * v0) + (v1 * v1)));
  znext[0] = (z[0] + (0.050000000000000003 * v0));
  znext[1] = (z[1] + (0.050000000000000003 * v1));
  znext[2] = (v0 + (0.050000000000000003 * (u[0] - (v2 * v0))));
  znext[3] = (v1 + (0.050000000000000003 * (u[1] - (v2 * v1))));
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
    dynamics_raw((arg[0] + (4 * it_t0)), (arg[1] + (2 * it_t0)), (s0 + (it_t0 * 4)), NULL);
  }
  for (long long i_eq = 0; i_eq < 12; ++i_eq) {
    res[0][i_eq] = (s0[i_eq] - t1[i_eq]);
  }
  return ALLOY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
