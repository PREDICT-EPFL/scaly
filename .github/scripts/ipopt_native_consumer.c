#include <IpStdCInterface.h>
#include <stdint.h>
#ifdef SCALY_STANDALONE
#include <stdio.h>
#endif

#ifdef __cplusplus
static_assert(sizeof(ipnumber) == 8 && sizeof(ipindex) == 4, "Ipopt interface changed");
extern "C" {
#else
_Static_assert(sizeof(ipnumber) == 8 && sizeof(ipindex) == 4, "Ipopt interface changed");
#endif

#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif

#define N 512

static bool objective(ipindex n, ipnumber *x, bool fresh, ipnumber *f, void *data) {
  *f = 0;
  for (ipindex i = 0; i < n; ++i) *f += (x[i] - 5) * (x[i] - 5);
  for (ipindex i = 1; i < n; ++i) *f += 0.1 * (x[i] - x[i-1]) * (x[i] - x[i-1]);
  return true;
}

static bool gradient(ipindex n, ipnumber *x, bool fresh, ipnumber *g, void *data) {
  for (ipindex i = 0; i < n; ++i) {
    g[i] = 2 * (x[i] - 5);
    if (i > 0) g[i] += 0.2 * (x[i] - x[i-1]);
    if (i+1 < n) g[i] += 0.2 * (x[i] - x[i+1]);
  }
  return true;
}

static bool constraints(ipindex n, ipnumber *x, bool fresh, ipindex m, ipnumber *g, void *data) {
  for (ipindex i = 0; i < m; ++i) g[i] = x[2*i] + x[2*i+1];
  return true;
}

static bool jacobian(ipindex n, ipnumber *x, bool fresh, ipindex m, ipindex count,
                     ipindex *rows, ipindex *cols, ipnumber *values, void *data) {
  for (ipindex i = 0; i < n; ++i) {
    if (values) values[i] = 1;
    else { rows[i] = i/2; cols[i] = i; }
  }
  return true;
}

static bool hessian(ipindex n, ipnumber *x, bool fresh, ipnumber factor, ipindex m,
                    ipnumber *lambda, bool fresh_lambda, ipindex count,
                    ipindex *rows, ipindex *cols, ipnumber *values, void *data) {
  for (ipindex i = 0; i < n; ++i) {
    if (values) values[i] = factor * (2 + (i > 0 ? 0.2 : 0) + (i+1 < n ? 0.2 : 0));
    else rows[i] = cols[i] = i;
  }
  for (ipindex i = 1; i < n; ++i) {
    if (values) values[n+i-1] = -0.2 * factor;
    else { rows[n+i-1] = i; cols[n+i-1] = i-1; }
  }
  return true;
}

static bool intermediate(ipindex mode, ipindex iter, ipnumber f, ipnumber inf_pr,
                         ipnumber inf_du, ipnumber mu, ipnumber norm, ipnumber regularization,
                         ipnumber alpha_du, ipnumber alpha_pr, ipindex trials, void *data) {
  ((double *)data)[N+1] = iter;
  return true;
}

EXPORT void *probe_ipopt_address(void) { return (void *)(uintptr_t)&CreateIpoptProblem; }

EXPORT int probe(double *out) {
  ipnumber lower[N], upper[N], target[N/2], x[N];
  for (int i = 0; i < N; ++i) { lower[i] = -100; upper[i] = 100; x[i] = 0; }
  for (int i = 0; i < N/2; ++i) target[i] = 10;
  IpoptProblem p = CreateIpoptProblem(N, lower, upper, N/2, target, target, N, 2*N-1, 0,
                                     objective, constraints, gradient, jacobian, hessian);
  if (!p) return -999;
  if (!AddIpoptStrOption(p, "linear_solver", "mumps") || !AddIpoptStrOption(p, "sb", "yes") ||
      !AddIpoptIntOption(p, "print_level", 0) || !AddIpoptNumOption(p, "tol", 1e-10) ||
      !SetIntermediateCallback(p, intermediate)) { FreeIpoptProblem(p); return -998; }
  int status = IpoptSolve(p, x, 0, &out[N], 0, 0, 0, out);
  for (int i = 0; i < N; ++i) out[i] = x[i];
  FreeIpoptProblem(p);
  return status;
}

#ifdef SCALY_STANDALONE
int main(void) {
  double out[N+2] = {0};
  int status = probe(out);
  if (status != 0 || out[N+1] <= 0 || out[N] > 1e-10) return 1;
  for (int i = 0; i < N; ++i) if (out[i] < 5-1e-7 || out[i] > 5+1e-7) return 2;
  printf("{\"status\":%d,\"iter\":%.0f,\"x0\":%.17g,\"obj\":%.17g}\n", status, out[N+1], out[0], out[N]);
  return 0;
}
#endif

#ifdef __cplusplus
}
#endif
