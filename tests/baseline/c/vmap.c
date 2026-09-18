#include <math.h>
#include <stddef.h>
#include <stdint.h>
typedef double double2 __attribute__((vector_size(16), aligned(8), may_alias));

#define SCALY_SUCCESS 0
#define SCALY_ERR_NULL_ABI 1
#define SCALY_ERR_NULL_WORK 2
#define SCALY_ERR_NULL_RESULT 3
#define SCALY_ERR_NULL_INPUT 4

#ifdef __cplusplus
extern "C" {
#endif

static inline void dynamics_raw(const double* z, const double* u, double* znext, double* w) {
  (void)w;
  double v0 = z[2];
  double v1 = z[3];
  double v2 = (0.10000000000000001 * ((v0 * v0) + (v1 * v1)));
  *(double2*)(znext) = (double2){(z[0] + (0.050000000000000003 * v0)), (z[1] + (0.050000000000000003 * v1))};
  *(double2*)(znext + 2) = (double2){(v0 + (0.050000000000000003 * (u[0] - (v2 * v0)))), (v1 + (0.050000000000000003 * (u[1] - (v2 * v1))))};
}

int shooting(const double** arg, double** res, int* iw, double* w, int mem) {
  (void)iw;
  (void)mem;
  if (!arg || !res) return SCALY_ERR_NULL_ABI;
  (void)w;
  if (!arg[0]) return SCALY_ERR_NULL_INPUT;
  if (!arg[1]) return SCALY_ERR_NULL_INPUT;
  if (!res[0]) return SCALY_ERR_NULL_RESULT;
  double s0[12];
  const double* t1 = arg[0] + 4;
  for (long long it_t0 = 0; it_t0 < 3; ++it_t0) {
    dynamics_raw((arg[0] + (4 * it_t0)), (arg[1] + (2 * it_t0)), (s0 + (it_t0 * 4)), NULL);
  }
  for (long long i_eq = 0; i_eq < 12; ++i_eq) {
    res[0][i_eq] = (s0[i_eq] - t1[i_eq]);
  }
  return SCALY_SUCCESS;
}

#ifdef __cplusplus
}
#endif
