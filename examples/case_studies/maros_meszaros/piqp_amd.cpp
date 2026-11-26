// PIQP's AMDOrdering::init, alone: Eigen's AMDOrdering on the upper triangle of a symmetric pattern.
//
// Reads from stdin n, nnz, the n + 1 column pointers and the nnz row indices of the upper triangle
// in CSC (int32, binary), and writes the n entries of P (int32, binary), PIQP's P[new] = old, to
// stdout. benchmark.py compiles it against Eigen 3.4 to compare PIQP's ordering with scaly's.
#include <Eigen/OrderingMethods>
#include <Eigen/SparseCore>
#include <cstdint>
#include <cstdio>
#include <vector>

int main() {
  int32_t n, nnz;
  if (std::fread(&n, 4, 1, stdin) != 1 || std::fread(&nnz, 4, 1, stdin) != 1) return 1;
  std::vector<int32_t> colptr(n + 1), rows(nnz);
  if (std::fread(colptr.data(), 4, n + 1, stdin) != size_t(n + 1)) return 1;
  if (std::fread(rows.data(), 4, nnz, stdin) != size_t(nnz)) return 1;
  std::vector<double> ones(nnz, 1.0);
  Eigen::Map<const Eigen::SparseMatrix<double, Eigen::ColMajor, int>> A(n, n, nnz, colptr.data(), rows.data(), ones.data());
  Eigen::SparseMatrix<double, Eigen::ColMajor, int> upper = A;
  // piqp/sparse/ordering.hpp, AMDOrdering::init
  Eigen::PermutationMatrix<Eigen::Dynamic, Eigen::Dynamic, int> P_eigen;
  Eigen::AMDOrdering<int> amd_ordering;
  amd_ordering(upper.selfadjointView<Eigen::Upper>(), P_eigen);
  std::fwrite(P_eigen.indices().data(), 4, n, stdout);
  return 0;
}
