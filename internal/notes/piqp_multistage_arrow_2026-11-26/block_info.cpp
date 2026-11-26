// Prints PIQP 0.7.0's MultistageKKT::block_info for one QP pattern, the reference for
// examples/piqp_multistage/multistage_kkt.py. Build and run with oracle.sh.
//
// Reads from stdin "n p m", then P's upper triangle, A^T and G^T, each as "rows cols nnz" followed
// by nnz lines "row col value". Writes one line "start diag_size off_diag_size" per block.

#include <iostream>
#include <vector>

#include "piqp/piqp.hpp"

using T = double;
using I = int;

struct Probe : piqp::sparse::MultistageKKT<T, I>
{
    using piqp::sparse::MultistageKKT<T, I>::MultistageKKT;
    const std::vector<piqp::sparse::BlockInfo<I>>& blocks() const { return this->block_info; }
};

static piqp::SparseMat<T, I> read_matrix()
{
    Eigen::Index rows, cols, nnz;
    std::cin >> rows >> cols >> nnz;
    std::vector<Eigen::Triplet<T, I>> triplets;
    for (Eigen::Index k = 0; k < nnz; k++) {
        I i, j;
        T v;
        std::cin >> i >> j >> v;
        triplets.emplace_back(i, j, v);
    }
    piqp::SparseMat<T, I> M(rows, cols);
    M.setFromTriplets(triplets.begin(), triplets.end());
    M.makeCompressed();
    return M;
}

int main()
{
    piqp::sparse::Data<T, I> data;
    std::cin >> data.n >> data.p >> data.m;
    data.P_utri = read_matrix();
    data.AT = read_matrix();
    data.GT = read_matrix();
    Probe kkt(data, 1);
    for (const auto& block : kkt.blocks()) {
        std::cout << block.start << " " << block.diag_size << " " << block.off_diag_size << "\n";
    }
    return 0;
}
