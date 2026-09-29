// Replays an episode with the TinyMPC library (https://github.com/TinyMPC/TinyMPC).
//
//   tinympc_driver EPISODE RESULTS REPS [CACHE]
//
// Sets the solver up once from the episode's data (tiny_setup computes the Riccati cache), then
// runs the episode REPS times from a zero workspace: each step sets x0 and the references and
// times tiny_solve alone. With CACHE it also writes the cache it computed (Kinf, Pinf, Quu_inv,
// AmBKt row-major, then APf, BPf) so that the generated solver can be built on the same numbers.
#include <tinympc/tiny_api.hpp>

#include "episode.h"

using RowMajorMatrix = Matrix<tinytype, Dynamic, Dynamic, RowMajor>;

static tinyMatrix stage_major(const double *p, int rows, int cols) {
  // Stage-major storage (one stage's vector after the other) is column-major rows x cols.
  return Map<const Matrix<tinytype, Dynamic, Dynamic, ColMajor>>(p, rows, cols);
}

static void write_matrix(FILE *fp, const tinyMatrix &m) {
  RowMajorMatrix r = m;
  fwrite(r.data(), sizeof(double), r.size(), fp);
}

static void zero_workspace(TinyWorkspace *w) {
  for (tinyMatrix *m : {&w->x, &w->u, &w->q, &w->r, &w->p, &w->d, &w->v, &w->vnew, &w->z, &w->znew, &w->g, &w->y, &w->vc, &w->vcnew,
                        &w->zc, &w->zcnew, &w->gc, &w->yc, &w->vl, &w->vlnew, &w->zl, &w->zlnew, &w->gl, &w->yl}) {
    m->setZero();
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s EPISODE RESULTS REPS [CACHE]\n", argv[0]);
    return 2;
  }
  episode_t e;
  episode_read(argv[1], &e);
  int reps = atoi(argv[3]);
  const int nx = e.nx, nu = e.nu, N = e.N;

  tinyMatrix A = Map<RowMajorMatrix>(e.A, nx, nx);
  tinyMatrix B = Map<RowMajorMatrix>(e.B, nx, nu);
  tinyMatrix f = Map<tinyMatrix>(e.f, nx, 1);
  tinyVector Q = Map<tinyVector>(e.Q, nx), R = Map<tinyVector>(e.R, nu);
  TinySolver *solver;
  if (tiny_setup(&solver, A, B, f, Q.asDiagonal(), R.asDiagonal(), e.rho, nx, nu, N, 0)) return 1;
  tiny_set_bound_constraints(solver, stage_major(e.x_min, nx, N), stage_major(e.x_max, nx, N), stage_major(e.u_min, nu, N - 1),
                             stage_major(e.u_max, nu, N - 1));
  VectorXi acx(e.n_state_cones), qcx(e.n_state_cones), acu(e.n_input_cones), qcu(e.n_input_cones);
  tinyVector cx(e.n_state_cones), cu(e.n_input_cones);
  for (int i = 0; i < e.n_state_cones; ++i) acx(i) = (int)e.state_cones[3 * i], qcx(i) = (int)e.state_cones[3 * i + 1], cx(i) = e.state_cones[3 * i + 2];
  for (int i = 0; i < e.n_input_cones; ++i) acu(i) = (int)e.input_cones[3 * i], qcu(i) = (int)e.input_cones[3 * i + 1], cu(i) = e.input_cones[3 * i + 2];
  // The definition in tiny_api.cpp takes the state cones first (the header's names are swapped).
  tiny_set_cone_constraints(solver, acx, qcx, cx, acu, qcu, cu);
  TinySettings *s = solver->settings;
  s->abs_pri_tol = e.abs_pri_tol;
  s->abs_dua_tol = e.abs_dua_tol;
  s->max_iter = e.max_iter;
  s->check_termination = 1;
  s->en_state_bound = e.en_state_bound;
  s->en_input_bound = e.en_input_bound;
  s->en_state_soc = e.n_state_cones > 0;
  s->en_input_soc = e.n_input_cones > 0;
  s->adaptive_rho = 0;

  if (argc > 4) {
    FILE *fp = fopen(argv[4], "wb");
    TinyCache *c = solver->cache;
    for (const tinyMatrix *m : {&c->Kinf, &c->Pinf, &c->Quu_inv, &c->AmBKt}) write_matrix(fp, *m);
    fwrite(c->APf.data(), sizeof(double), nx, fp);
    fwrite(c->BPf.data(), sizeof(double), nu, fp);
    fclose(fp);
  }

  results_t res;
  results_init(&res, e.steps, nu, reps);
  for (int rep = 0; rep < reps; ++rep) {
    zero_workspace(solver->work);
    double total = 0.0;
    for (int k = 0; k < e.steps; ++k) {
      tiny_set_x0(solver, Map<tinyVector>(e.x0 + (size_t)k * nx, nx));
      tiny_set_x_ref(solver, stage_major(e.xref + (size_t)k * N * nx, nx, N));
      tiny_set_u_ref(solver, stage_major(e.uref + (size_t)k * (N - 1) * nu, nu, N - 1));
      double t0 = now_ns();
      tiny_solve(solver);
      double dt = now_ns() - t0;
      total += dt;
      if (dt < res.best_ns[k]) res.best_ns[k] = dt;
      if (rep == 0) {
        res.iterations[k] = solver->work->iter;
        res.solved[k] = solver->work->status == 1;
        for (int i = 0; i < nu; ++i) res.u0[(size_t)k * nu + i] = solver->work->u(i, 0);
      }
    }
    res.total_ns[rep] = total;
  }
  results_write(argv[2], &res);
  return 0;
}
