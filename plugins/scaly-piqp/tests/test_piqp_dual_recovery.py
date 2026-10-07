"""The vendored PIQP reads its index of lower-bounded rows only up to its length.

PIQP 0.6.2's dual recovery (``KKTSystem::solve``) read the entry just past the rows with a finite
lower bound. That entry is uninitialized heap. When it happened to hold the index of the next row,
a row without a lower bound was taken as bounded, the first iterate went NaN and the solve ran to
its iteration limit: ``examples/ocp/linear_mpc.ipynb`` failed so in about one process in ten. PIQP
0.6.4 reads only up to the length. Here a preloaded ``malloc`` makes that garbage certain, filling
every block the size of the index with the index of the first unbounded row, and a C program linked
against the vendored library solves one QP through the dense and the sparse interface.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import scaly_piqp
from scaly.codegen.toolchain import find_c_compiler

ROWS, BOUNDED = 40, 25  # the first BOUNDED rows have a lower bound as well as an upper one

FILL = r"""
#include <stdint.h>
#include <stdlib.h>
static size_t fill_size;
static int64_t fill_value;
__attribute__((constructor)) static void read_env(void) {  /* getenv inside malloc is unsafe at startup */
  const char* size = getenv("FILL_SIZE");
  const char* value = getenv("FILL_VALUE");
  fill_size = size ? strtoull(size, NULL, 10) : 0;
  fill_value = value ? strtoll(value, NULL, 10) : 0;
}
static void* fill(void* p, size_t n) {
  if (p && n == fill_size) for (size_t k = 0; k < n / 8; ++k) ((int64_t*)p)[k] = fill_value;
  return p;
}
#ifdef __APPLE__
static void* filled_malloc(size_t n) { return fill(malloc(n), n); }
__attribute__((used, section("__DATA,__interpose"))) static const void* interpose[2] = {(const void*)filled_malloc, (const void*)malloc};
#else
extern void* __libc_malloc(size_t);
void* malloc(size_t n) { return fill(__libc_malloc(n), n); }
#endif
"""

# min 0.5 |x|^2 - (3, 1) x over -1 <= g_i x <= 1 (i < BOUNDED) and g_i x <= 1, g_i at 9 degree
# steps: the lower bounds repeat the opposite rows' upper ones, so the set is a regular 40-gon with
# inradius 1 and the solution the projection of (3, 1) onto its side with the normal at 18 degrees.
PROGRAM = r"""
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include "piqp/piqp.h"
enum { N = 2, M = %(rows)d, BOUNDED = %(bounded)d };
static void report(const char* form, piqp_workspace* ws) {
  piqp_solve(ws);
  printf("%%s %%d %%.17g %%.17g\n", form, (int)ws->result->info.status, ws->result->x[0], ws->result->x[1]);
  piqp_cleanup(ws);
}
int main(void) {
  volatile int64_t* probe = malloc(M * sizeof(int64_t));  /* the preloaded malloc is in place (volatile: */
                                                           /* an unwritten block is undefined to fold)  */
  printf("fill %%lld\n", (long long)probe[BOUNDED]);
  double P[N * N] = {1, 0, 0, 1}, c[N] = {-3, -1}, G[M * N], Gcol[M * N], lo[M], hi[M];
  for (int i = 0; i < M; ++i) {
    double angle = 2 * M_PI * i / M;
    G[N * i] = Gcol[i] = cos(angle);
    G[N * i + 1] = Gcol[M + i] = sin(angle);
    lo[i] = i < BOUNDED ? -1 : -PIQP_INF;
    hi[i] = 1;
  }
  piqp_settings settings;
  piqp_set_default_settings_dense(&settings);
  piqp_data_dense dense = {N, 0, M, P, c, NULL, NULL, G, lo, hi, NULL, NULL};
  piqp_workspace* ws = NULL;
  piqp_setup_dense(&ws, &dense, &settings);
  report("dense", ws);
  piqp_int P_p[N + 1] = {0, 1, 2}, P_i[N] = {0, 1}, G_p[N + 1] = {0, M, 2 * M}, G_i[N * M];
  double P_x[N] = {1, 1};
  for (int i = 0; i < M; ++i) G_i[i] = G_i[M + i] = i;
  piqp_csc P_csc = {N, N, N, P_p, P_i, P_x}, G_csc = {M, N, N * M, G_p, G_i, Gcol};
  piqp_set_default_settings_sparse(&settings);
  piqp_data_sparse sparse = {N, 0, M, &P_csc, c, NULL, NULL, &G_csc, lo, hi, NULL, NULL};
  piqp_setup_sparse(&ws, &sparse, &settings);
  report("sparse", ws);
  return 0;
}
"""


@pytest.mark.solver("piqp")
@pytest.mark.skipif(sys.platform not in ("darwin", "linux"), reason="preloads malloc through dyld or the glibc loader")
def test_rows_without_a_lower_bound_ignore_the_heap_past_the_index(tmp_path: Path) -> None:
  compiler = find_c_compiler()
  if compiler is None:
    pytest.skip("no C compiler")
  lib_dir, include_dir = scaly_piqp.lib_dir(), scaly_piqp.include_dir()
  fill = tmp_path / ("libfill.dylib" if sys.platform == "darwin" else "libfill.so")
  (tmp_path / "fill.c").write_text(FILL)
  shared = ["-dynamiclib"] if sys.platform == "darwin" else ["-shared", "-fPIC"]
  subprocess.run([compiler.cc, "-O2", *shared, str(tmp_path / "fill.c"), "-o", str(fill)], check=True)
  (tmp_path / "program.c").write_text(PROGRAM % {"rows": ROWS, "bounded": BOUNDED})
  program = tmp_path / "program"
  cmd = [
    compiler.cc,
    "-O2",
    f"-I{include_dir}",
    str(tmp_path / "program.c"),
    f"-L{lib_dir}",
    "-lpiqpc",
    f"-Wl,-rpath,{lib_dir}",
    "-lm",
    "-o",
    str(program),
  ]
  subprocess.run(cmd, check=True)

  preload = "DYLD_INSERT_LIBRARIES" if sys.platform == "darwin" else "LD_PRELOAD"
  env = {preload: str(fill), "FILL_SIZE": str(ROWS * 8), "FILL_VALUE": str(BOUNDED)}
  out = subprocess.run([str(program)], env=env, capture_output=True, text=True, timeout=60, check=True).stdout.split("\n")
  assert out[0] == f"fill {BOUNDED}", "the preloaded malloc did not take effect, so this test proves nothing"

  normal = np.array([np.cos(np.pi / 10), np.sin(np.pi / 10)])
  expected = np.array([3.0, 1.0]) - (normal @ [3.0, 1.0] - 1.0) * normal
  for line, form in zip(out[1:3], ("dense", "sparse"), strict=True):
    name, status, *x = line.split()
    assert name == form and int(status) == 1, f"{form}: PIQP status {status} (1 is solved), x = {x}"
    np.testing.assert_allclose(np.array(x, dtype=float), expected, atol=1e-7)
