/* Replays an episode with the solver Scaly generated for it (tinympc_solve.c).
 *
 *   scaly_driver EPISODE RESULTS REPS
 *
 * The generated function takes the warm-start state and returns the next one; the driver keeps two
 * state buffers and swaps them after each solve, so nothing is copied between calls. Each run of the
 * episode starts from a zero state; only the call itself is timed.
 */
#define _POSIX_C_SOURCE 200809L /* clock_gettime under -std=c11 */
#include "tinympc_solve.h"

#include "episode.h"

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s EPISODE RESULTS REPS\n", argv[0]);
    return 2;
  }
  episode_t e;
  episode_read(argv[1], &e);
  int reps = atoi(argv[3]);
  const int nx = e.nx, nu = e.nu, N = e.N;
  /* x, u, v, vnew, z, znew, g, y, then the cone duals gc, yc when there are cones */
  size_t nstate = (size_t)4 * N * nx + (size_t)4 * (N - 1) * nu;
  if (e.n_state_cones) nstate += (size_t)N * nx;
  if (e.n_input_cones) nstate += (size_t)(N - 1) * nu;
  const size_t u_offset = (size_t)N * nx;

  double *state[2] = {(double *)calloc(nstate, sizeof(double)), (double *)calloc(nstate, sizeof(double))};
  double *w = (double *)malloc(sizeof(double) * (tinympc_solve_SZ_W > 0 ? tinympc_solve_SZ_W : 1));
  double iterations, solved, u0[64];
  const double *arg[tinympc_solve_SZ_ARG];
  double *res[4] = {NULL, &iterations, &solved, u0};
  if (nu > 64) return 3;

  results_t out;
  results_init(&out, e.steps, nu, reps);
  for (int rep = 0; rep < reps; ++rep) {
    memset(state[0], 0, nstate * sizeof(double));
    int cur = 0;
    double total = 0.0;
    for (int k = 0; k < e.steps; ++k) {
      int a = 0;
      arg[a++] = state[cur];
      arg[a++] = e.x0 + (size_t)k * nx;
      arg[a++] = e.xref + (size_t)k * N * nx;
      arg[a++] = e.uref + (size_t)k * (N - 1) * nu;
      if (e.en_state_bound && !e.fixed_bounds) arg[a++] = e.x_min, arg[a++] = e.x_max;
      if (e.en_input_bound && !e.fixed_bounds) arg[a++] = e.u_min, arg[a++] = e.u_max;
      res[0] = state[1 - cur];
      double t0 = now_ns();
      int status = tinympc_solve(arg, res, NULL, w, 0);
      double dt = now_ns() - t0;
      if (status != 0) return 4;
      cur = 1 - cur;
      total += dt;
      if (dt < out.best_ns[k]) out.best_ns[k] = dt;
      if (rep == 0) {
        out.iterations[k] = iterations;
        out.solved[k] = solved;
        /* u0 is also an output; take it from the state to check the packing */
        for (int i = 0; i < nu; ++i) out.u0[(size_t)k * nu + i] = state[cur][u_offset + i];
      }
    }
    out.total_ns[rep] = total;
  }
  results_write(argv[2], &out);
  return 0;
}
