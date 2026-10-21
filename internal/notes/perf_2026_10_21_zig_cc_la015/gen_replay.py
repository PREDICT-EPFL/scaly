"""Generate a C harness that replays the captured solver calls and times each oracle."""

import json, sys
from pathlib import Path
cap = Path("/tmp/zigprof/capture"); calls = sorted(cap.iterdir())
meta = json.loads((calls[0] / "meta.json").read_text())
opts = ",\n".join(f'  {{ "{n}", {k}, {i}, {x!r}, NULL }}' for n, k, i, x, t in meta["options"])
src = f'''#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
typedef struct {{ const char* name; int kind; int64_t integer; double number; const char* text; }} scaly_solver_option;
typedef struct {{ int32_t version, status, native_status, iter; double obj, t_total, t_fe, t_solver, t_qp, t_globalization, t_glue;
  int32_t n_eval_f, n_eval_grad_f, n_eval_g, n_eval_jac_g, n_eval_h, _pad0; double primal_viol, step_inf, alpha, merit_penalty; int32_t backtracks, qp_iter; }} stats_t;
int {meta["name"]}_with_options(const double**, double**, int*, double*, int, const scaly_solver_option* const*);
int {meta["name"]}_stats(stats_t*);
static const scaly_solver_option opts[] = {{
{opts},
  {{ NULL, 0, 0, 0.0, NULL }}
}};
#define NC {len(calls)}
#define NA {meta["n_args"]}
static const int asz[NA] = {{ {", ".join(map(str, meta["arg_sizes"]))} }};
static const int rsz[] = {{ {", ".join(map(str, meta["res_sizes"]))} }};
static double* load(const char* path, int n) {{ double* p = malloc(sizeof(double) * (n ? n : 1)); FILE* f = fopen(path, "rb"); if (n && fread(p, sizeof(double), n, f) != (size_t)n) {{ fprintf(stderr, "short %s\\n", path); exit(1); }} fclose(f); return p; }}
int main(int argc, char** argv) {{
  int reps = argc > 1 ? atoi(argv[1]) : 20;
  const char* dirs[NC] = {{ {", ".join(f'"{c}"' for c in calls)} }};
  const double* args[NC][NA]; char path[512];
  for (int c = 0; c < NC; ++c) for (int a = 0; a < NA; ++a) {{ snprintf(path, sizeof path, "%s/arg%d.bin", dirs[c], a); args[c][a] = load(path, asz[a]); }}
  double* res[{len(meta["res_sizes"])}]; for (int r = 0; r < {len(meta["res_sizes"])}; ++r) res[r] = malloc(sizeof(double) * (rsz[r] ? rsz[r] : 1));
  double* w = malloc(sizeof(double) * {meta["sz_w"]});
  const scaly_solver_option* o[] = {{ opts }};
  double fe = 0, total = 0; long iters = 0, nh = 0, nf = 0;
  for (int rep = 0; rep < reps; ++rep) for (int c = 0; c < NC; ++c) {{
    if ({meta["name"]}_with_options(args[c], res, NULL, w, 0, o)) return 2;
    stats_t s; {meta["name"]}_stats(&s); if (rep > 0) {{ fe += s.t_fe; total += s.t_total; iters += s.iter; nh += s.n_eval_h; nf += s.n_eval_f; }}
  }}
  long n = (long)(reps - 1) * NC;
  printf("per solve: fe %.4f ms  total %.4f ms  iter %.3f  n_h %.3f  n_f %.3f\\n", 1e3 * fe / n, 1e3 * total / n, (double)iters / n, (double)nh / n, (double)nf / n);
  return 0;
}}
'''
Path(sys.argv[1]).write_text(src)
