#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/string.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "piqp.h"

namespace nb = nanobind;
// Inputs are normalized by the Python compatibility wrapper before crossing the ABI.
using Array = nb::ndarray<nb::numpy, double, nb::c_contig, nb::device::cpu>;

class PIQPDenseSolver {
  piqp_int n_, p_, m_;
  piqp_workspace *ws_ = nullptr;
  piqp_settings settings_{};
  std::vector<double> P_, c_, A_, b_, G_, h_l_, h_u_, x_l_, x_u_;

  static size_t checked_size(piqp_int value) {
    if (value < 0) throw nb::value_error("PIQP dimensions must be nonnegative");
    return static_cast<size_t>(value);
  }

  static Array require_array(const nb::object &obj, size_t size, const char *name) {
    if (obj.is_none()) throw nb::value_error((std::string(name) + " must not be None").c_str());
    Array arr = nb::cast<Array>(obj);
    if (arr.size() != size) throw nb::value_error((std::string(name) + " has the wrong size").c_str());
    return arr;
  }

  static void copy_vector(std::vector<double> &dst, const nb::object &obj, const char *name) {
    Array arr = require_array(obj, dst.size(), name);
    std::copy_n(arr.data(), dst.size(), dst.data());
  }

  static void copy_matrix(std::vector<double> &dst, const nb::object &obj, size_t rows, size_t cols, const char *name) {
    Array arr = require_array(obj, rows * cols, name);
    for (size_t i = 0; i < rows; i++) for (size_t j = 0; j < cols; j++) dst[j * rows + i] = arr.data()[i * cols + j];
  }

  static double *ptr(std::vector<double> &v) { return v.empty() ? nullptr : v.data(); }

  void apply_settings(const nb::dict &settings) {
    for (auto [key_obj, value] : settings) {
      std::string key = nb::cast<std::string>(key_obj);
#define PIQP_FLOAT_SETTING(name) if (key == #name) { settings_.name = nb::cast<double>(value); continue; }
#define PIQP_INT_SETTING(name) if (key == #name) { settings_.name = nb::cast<piqp_int>(value); continue; }
      PIQP_FLOAT_SETTING(rho_init)
      PIQP_FLOAT_SETTING(delta_init)
      PIQP_FLOAT_SETTING(eps_abs)
      PIQP_FLOAT_SETTING(eps_rel)
      PIQP_INT_SETTING(check_duality_gap)
      PIQP_FLOAT_SETTING(eps_duality_gap_abs)
      PIQP_FLOAT_SETTING(eps_duality_gap_rel)
      PIQP_FLOAT_SETTING(infeasibility_threshold)
      PIQP_FLOAT_SETTING(reg_lower_limit)
      PIQP_FLOAT_SETTING(reg_finetune_lower_limit)
      PIQP_INT_SETTING(reg_finetune_primal_update_threshold)
      PIQP_INT_SETTING(reg_finetune_dual_update_threshold)
      PIQP_INT_SETTING(max_iter)
      PIQP_INT_SETTING(max_factor_retires)
      PIQP_INT_SETTING(preconditioner_scale_cost)
      PIQP_INT_SETTING(preconditioner_reuse_on_update)
      PIQP_INT_SETTING(preconditioner_iter)
      PIQP_FLOAT_SETTING(tau)
      if (key == "kkt_solver") { settings_.kkt_solver = static_cast<piqp_kkt_solver>(nb::cast<piqp_int>(value)); continue; }
      PIQP_INT_SETTING(iterative_refinement_always_enabled)
      PIQP_FLOAT_SETTING(iterative_refinement_eps_abs)
      PIQP_FLOAT_SETTING(iterative_refinement_eps_rel)
      PIQP_INT_SETTING(iterative_refinement_max_iter)
      PIQP_FLOAT_SETTING(iterative_refinement_min_improvement_rate)
      PIQP_FLOAT_SETTING(iterative_refinement_static_regularization_eps)
      PIQP_FLOAT_SETTING(iterative_refinement_static_regularization_rel)
      PIQP_INT_SETTING(verbose)
      PIQP_INT_SETTING(compute_timings)
#undef PIQP_FLOAT_SETTING
#undef PIQP_INT_SETTING
      throw nb::value_error(("unknown PIQP setting '" + key + "'").c_str());
    }
  }

  static Array copy_out(const double *src, size_t size) {
    double *data = new double[size];
    if (size) std::copy_n(src, size, data);
    nb::capsule owner(data, [](void *p) noexcept { delete[] static_cast<double *>(p); });
    return Array(data, {size}, owner);
  }

public:
  PIQPDenseSolver(piqp_int n, piqp_int p, piqp_int m, const nb::dict &settings)
      : n_(n), p_(p), m_(m), P_(checked_size(n) * checked_size(n)), c_(checked_size(n)), A_(checked_size(p) * checked_size(n)), b_(checked_size(p)),
        G_(checked_size(m) * checked_size(n)), h_l_(checked_size(m), -PIQP_INF), h_u_(checked_size(m), PIQP_INF), x_l_(checked_size(n), -PIQP_INF),
        x_u_(checked_size(n), PIQP_INF) {
    piqp_set_default_settings_dense(&settings_);
    apply_settings(settings);
  }

  ~PIQPDenseSolver() { cleanup(); }
  PIQPDenseSolver(const PIQPDenseSolver &) = delete;
  PIQPDenseSolver &operator=(const PIQPDenseSolver &) = delete;

  void update(const nb::object &P, const nb::object &c, const nb::object &A, const nb::object &b, const nb::object &G,
              const nb::object &h_l, const nb::object &h_u, const nb::object &x_l, const nb::object &x_u) {
    copy_matrix(P_, P, n_, n_, "P");
    copy_vector(c_, c, "c");
    if (p_) { copy_matrix(A_, A, p_, n_, "A_eq"); copy_vector(b_, b, "b_eq"); }
    if (m_) {
      copy_matrix(G_, G, m_, n_, "G_ineq"); copy_vector(h_l_, h_l, "l_ineq"); copy_vector(h_u_, h_u, "u_ineq");
      for (double &v : h_l_) if (std::isinf(v) && v < 0) v = -PIQP_INF;
      for (double &v : h_u_) if (std::isinf(v) && v > 0) v = PIQP_INF;
    }
    if (x_l.is_none()) std::fill(x_l_.begin(), x_l_.end(), -PIQP_INF); else copy_vector(x_l_, x_l, "x_lb");
    if (x_u.is_none()) std::fill(x_u_.begin(), x_u_.end(), PIQP_INF); else copy_vector(x_u_, x_u, "x_ub");
    for (double &v : x_l_) if (std::isinf(v) && v < 0) v = -PIQP_INF;
    for (double &v : x_u_) if (std::isinf(v) && v > 0) v = PIQP_INF;

    if (!ws_) {
      // PIQP must see real bounds at setup; placeholder infinities cause spurious free-constraint warnings.
      piqp_data_dense data{n_, p_, m_, ptr(P_), ptr(c_), ptr(A_), ptr(b_), ptr(G_), ptr(h_l_), ptr(h_u_), ptr(x_l_), ptr(x_u_)};
      nb::gil_scoped_release release;
      piqp_setup_dense(&ws_, &data, &settings_);
    } else {
      nb::gil_scoped_release release;
      piqp_update_dense(ws_, ptr(P_), ptr(c_), ptr(A_), ptr(b_), ptr(G_), ptr(h_l_), ptr(h_u_), ptr(x_l_), ptr(x_u_));
    }
  }

  nb::tuple solve() {
    if (!ws_) throw std::runtime_error("update() must be called before solve()");
    piqp_status status;
    { nb::gil_scoped_release release; status = piqp_solve(ws_); }
    const piqp_result &r = *ws_->result;
    return nb::make_tuple(static_cast<int>(status), r.info.iter, r.info.primal_obj, copy_out(r.x, n_), copy_out(r.y, p_), copy_out(r.z_l, m_),
                          copy_out(r.z_u, m_), copy_out(r.z_bl, n_), copy_out(r.z_bu, n_));
  }

  void cleanup() {
    if (ws_) { piqp_cleanup(ws_); ws_ = nullptr; }
  }
};

NB_MODULE(_piqp_ext, m) {
  nb::class_<PIQPDenseSolver>(m, "PIQPDenseSolver")
    .def(nb::init<piqp_int, piqp_int, piqp_int, const nb::dict &>(), nb::arg("n"), nb::arg("p"), nb::arg("m"), nb::arg("settings"))
    .def("update", &PIQPDenseSolver::update, nb::arg("P"), nb::arg("c"), nb::arg("A_eq") = nb::none(), nb::arg("b_eq") = nb::none(),
         nb::arg("G_ineq") = nb::none(), nb::arg("l_ineq") = nb::none(), nb::arg("u_ineq") = nb::none(), nb::arg("x_lb") = nb::none(),
         nb::arg("x_ub") = nb::none())
    .def("solve", &PIQPDenseSolver::solve)
    .def("cleanup", &PIQPDenseSolver::cleanup);
}
