// The class XLA's AOT compiler generates behind kernel.h. Its arguments must be 64-byte aligned, so
// setup copies them once; each evaluation runs on the calling thread and copies the result out.
#include <cstdlib>
#include <cstring>

#include "kernel.h"
#include "layout.h"
#include "npmpc_hess.h"

alignas(64) static double g_z[NZ], g_p[NP], g_lam_g[NLAM];
static NpmpcHess* g_fn;

void kernel_setup(const double* z, const double* p, const double* lam_g) {
  std::memcpy(g_z, z, sizeof g_z);
  std::memcpy(g_p, p, sizeof g_p);
  std::memcpy(g_lam_g, lam_g, sizeof g_lam_g);
  g_fn = new NpmpcHess(NpmpcHess::AllocMode::RESULTS_PROFILES_AND_TEMPS_ONLY);
  // The generated class orders its arguments its own way, so they are set by name.
  g_fn->set_arg_feed_z_data(g_z);
  g_fn->set_arg_feed_p_data(g_p);
  g_fn->set_arg_feed_lam_g_data(g_lam_g);
}

void kernel_evaluate(double* out) {
  if (!g_fn->Run()) std::abort();
  std::memcpy(out, g_fn->result0_data(), NNZ * sizeof(double));
}
