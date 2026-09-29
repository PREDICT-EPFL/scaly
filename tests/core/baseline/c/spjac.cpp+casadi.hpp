#pragma once

#include <array>
#include <cassert>
#include <cstddef>

#ifndef SCALY_SUCCESS
#define SCALY_SUCCESS 0
#endif
#ifndef SCALY_ERR_NULL_ABI
#define SCALY_ERR_NULL_ABI 1
#endif
#ifndef SCALY_ERR_NULL_WORK
#define SCALY_ERR_NULL_WORK 2
#endif
#ifndef SCALY_ERR_NULL_RESULT
#define SCALY_ERR_NULL_RESULT 3
#endif
#ifndef SCALY_ERR_NULL_INPUT
#define SCALY_ERR_NULL_INPUT 4
#endif

#ifndef casadi_real
#define casadi_real double
#endif
#ifndef casadi_int
#define casadi_int long long int
#endif

#define shooting_spjac_eq_z_SZ_ARG 2
#define shooting_spjac_eq_z_SZ_RES 1
#define shooting_spjac_eq_z_SZ_IW 0
#define shooting_spjac_eq_z_SZ_W 36

#ifndef SCALY_BUFFER_HPP
#define SCALY_BUFFER_HPP
// A fixed-shape, row-major, 16-byte aligned buffer with inline storage: an aggregate to place on
// the stack, in a static, or behind make_unique. shape is the Expr shape; data is its flattening.
template <typename T, std::size_t... Ns>
struct alignas(16) Buffer {
  static constexpr std::size_t ndim = sizeof...(Ns);
  static constexpr std::array<std::size_t, ndim> shape = {Ns...};
  static constexpr std::size_t size = (Ns * ... * std::size_t{1});
  T data[size > 0 ? size : 1];
  T* ptr() { return data; }
  const T* ptr() const { return data; }
  template <typename... I> T& operator()(I... idx) { return data[offset(idx...)]; }
  template <typename... I> const T& operator()(I... idx) const { return data[offset(idx...)]; }
  template <typename... I> static constexpr std::size_t offset(I... idx) {
    static_assert(sizeof...(I) == ndim, "Buffer: one index per dimension");
    const std::size_t index[] = {static_cast<std::size_t>(idx)..., 0};
    std::size_t off = 0;
    for (std::size_t k = 0; k < ndim; ++k) {
      assert(index[k] < shape[k]);
      off = off * shape[k] + index[k];
    }
    return off;
  }
};
#endif

namespace shooting_spjac_eq_z {
// The pointer ABI for shooting_spjac_eq_z; the same C symbols the C header declares.
extern "C" int shooting_spjac_eq_z(const double** arg, double** res, int* iw, double* w, int mem);
extern "C" casadi_int shooting_spjac_eq_z_n_in(void);
extern "C" casadi_int shooting_spjac_eq_z_n_out(void);
extern "C" const char* shooting_spjac_eq_z_name_in(casadi_int i);
extern "C" const char* shooting_spjac_eq_z_name_out(casadi_int i);
extern "C" casadi_real shooting_spjac_eq_z_default_in(casadi_int i);
extern "C" const casadi_int* shooting_spjac_eq_z_sparsity_in(casadi_int i);
extern "C" const casadi_int* shooting_spjac_eq_z_sparsity_out(casadi_int i);
extern "C" int shooting_spjac_eq_z_work(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
extern "C" int shooting_spjac_eq_z_work_bytes(casadi_int* sz_arg, casadi_int* sz_res, casadi_int* sz_iw, casadi_int* sz_w);
extern "C" int shooting_spjac_eq_z_checkout(void);
extern "C" void shooting_spjac_eq_z_release(int mem);
extern "C" void shooting_spjac_eq_z_incref(void);
extern "C" void shooting_spjac_eq_z_decref(void);

using z_t = Buffer<double, 16>;
using u_t = Buffer<double, 6>;
using spjac_eq_z_t = Buffer<double, 36>;
using workspace_t = Buffer<double, shooting_spjac_eq_z_SZ_W>;
constexpr int sz_arg = shooting_spjac_eq_z_SZ_ARG;
constexpr int sz_res = shooting_spjac_eq_z_SZ_RES;
constexpr int sz_iw = shooting_spjac_eq_z_SZ_IW;
constexpr int sz_w = shooting_spjac_eq_z_SZ_W;

inline int call(const z_t& z, const u_t& u, spjac_eq_z_t& spjac_eq_z, workspace_t& workspace) {
  const double* arg[sz_arg > 0 ? sz_arg : 1] = {z.ptr(), u.ptr()};
  double* res[sz_res > 0 ? sz_res : 1] = {spjac_eq_z.ptr()};
  return shooting_spjac_eq_z(arg, res, nullptr, sz_w ? workspace.ptr() : nullptr, 0);
}

namespace spjac_eq_z {
constexpr int nrow = 12;
constexpr int ncol = 16;
constexpr int nnz = 36;
constexpr std::array<int, nnz> rows = {0, 1, 0, 2, 3, 1, 2, 3, 0, 4, 1, 5, 2, 4, 6, 7, 3, 5, 6, 7, 4, 8, 5, 9, 6, 8, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
constexpr std::array<int, nnz> cols = {0, 1, 2, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 9, 10, 10, 10, 10, 11, 11, 11, 11, 12, 13, 14, 15};
constexpr std::array<int, nrow + 1> csr_row_ptr = {0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36};
constexpr std::array<int, nnz> csr_col_ind = {0, 2, 4, 1, 3, 5, 2, 3, 6, 2, 3, 7, 4, 6, 8, 5, 7, 9, 6, 7, 10, 6, 7, 11, 8, 10, 12, 9, 11, 13, 10, 11, 14, 10, 11, 15};
constexpr std::array<int, nnz> csr_val_perm = {0, 2, 8, 1, 5, 10, 3, 6, 12, 4, 7, 16, 9, 13, 20, 11, 17, 22, 14, 18, 24, 15, 19, 28, 21, 25, 32, 23, 29, 33, 26, 30, 34, 27, 31, 35};
constexpr std::array<int, ncol + 1> csc_col_ptr = {0, 1, 2, 5, 8, 10, 12, 16, 20, 22, 24, 28, 32, 33, 34, 35, 36};
constexpr std::array<int, nnz> csc_row_ind = {0, 1, 0, 2, 3, 1, 2, 3, 0, 4, 1, 5, 2, 4, 6, 7, 3, 5, 6, 7, 4, 8, 5, 9, 6, 8, 10, 11, 7, 9, 10, 11, 8, 9, 10, 11};
constexpr std::array<int, nnz> csc_val_perm = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35};
}
}  // namespace shooting_spjac_eq_z
