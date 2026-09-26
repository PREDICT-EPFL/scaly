/* Benchmark-only baseline: the up-looking sparse LDL^T numeric factorization of Davis's LDL
   (the algorithm QDLDL uses), written from the published description. Not shipped. The input is
   the upper triangle of the permuted matrix in CSC; Lp and parent come from the same symbolic
   analysis the generated code uses. */
#include <time.h>

static int factor(int n, const int* Ap, const int* Ai, const double* Ax, const int* Lp, const int* parent, int* Li, double* Lx,
                  double* D, double* Y, int* pattern, int* flag, int* lnz) {
  for (int k = 0; k < n; ++k) {
    Y[k] = 0.0;
    int top = n;
    flag[k] = k;
    lnz[k] = 0;
    for (int p = Ap[k]; p < Ap[k + 1]; ++p) {
      int i = Ai[p];
      Y[i] += Ax[p];
      int len = 0;
      for (; flag[i] != k; i = parent[i]) {
        pattern[len++] = i;
        flag[i] = k;
      }
      while (len > 0) pattern[--top] = pattern[--len];
    }
    D[k] = Y[k];
    Y[k] = 0.0;
    for (; top < n; ++top) {
      int i = pattern[top];
      double yi = Y[i];
      Y[i] = 0.0;
      int p2 = Lp[i] + lnz[i];
      int p;
      for (p = Lp[i]; p < p2; ++p) Y[Li[p]] -= Lx[p] * yi;
      double lki = yi / D[i];
      D[k] -= lki * yi;
      Li[p] = k;
      Lx[p] = lki;
      lnz[i]++;
    }
    if (D[k] == 0.0) return k;
  }
  return n;
}

static void solve(int n, const int* Lp, const int* Li, const double* Lx, const double* D, double* x) {
  for (int j = 0; j < n; ++j)
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) x[Li[p]] -= Lx[p] * x[j];
  for (int j = 0; j < n; ++j) x[j] /= D[j];
  for (int j = n - 1; j >= 0; --j)
    for (int p = Lp[j]; p < Lp[j + 1]; ++p) x[j] -= Lx[p] * x[Li[p]];
}

static double now(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec + 1e-9 * t.tv_nsec;
}

double bench_factor(int n, const int* Ap, const int* Ai, const double* Ax, const int* Lp, const int* parent, int* Li, double* Lx, double* D,
                    double* Y, int* pattern, int* flag, int* lnz, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    double t0 = now();
    factor(n, Ap, Ai, Ax, Lp, parent, Li, Lx, D, Y, pattern, flag, lnz);
    double t = now() - t0;
    if (t < best) best = t;
  }
  return best;
}

double bench_solve(int n, const int* Lp, const int* Li, const double* Lx, const double* D, const double* b, double* x, int reps) {
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    for (int i = 0; i < n; ++i) x[i] = b[i];
    double t0 = now();
    solve(n, Lp, Li, Lx, D, x);
    double t = now() - t0;
    if (t < best) best = t;
  }
  return best;
}
