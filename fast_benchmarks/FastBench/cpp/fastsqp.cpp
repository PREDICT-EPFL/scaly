// FastSQP - a real-time-iteration / full sparse SQP for FastBench.
//
// Gauss-Newton-style SQP over a multiple-shooting OCP.  Problem derivatives
// (dynamics + Jacobians, cost gradients + Hessians) come from CasADi-generated
// C compiled into the same shared library (see casadi_abi.h).  Each SQP
// iteration assembles the sparse OCP-QP
//
//     min 0.5 dz' H dz + g' dz
//      s.t.  dx_0 = x_meas - x_nom_0
//            dx_{k+1} - A_k dx_k - B_k du_k = fd(x_nom_k,u_nom_k) - x_nom_{k+1}
//            lbx - x_nom_k <= dx_k <= ubx - x_nom_k   (and similarly for u)
//
// and solves it with a pluggable QP backend.  The nominal trajectory is updated
// by the (full) step; with sqp_iters==1 this is the real-time iteration.
//
// Backends are selected at compile time: -DHAVE_PROXQP / -DHAVE_OSQP /
// -DHAVE_PIQP / -DHAVE_QPOASES.  proxqp is the reference sparse backend.
//
// Exposed through a plain C ABI (ctypes-friendly): no Python headers needed.

#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <chrono>
#include <cstring>
#include <vector>

#include "casadi_abi.h"

#ifdef HAVE_PROXQP
#include <proxsuite/proxqp/sparse/sparse.hpp>
#endif
#ifdef HAVE_PIQP
#include <piqp/piqp.hpp>
#endif
#ifdef HAVE_OSQP
#include <osqp/osqp.h>
#endif
#ifdef HAVE_QPOASES
#include <qpOASES.hpp>
#endif

using Eigen::MatrixXd;
using Eigen::VectorXd;
using SpMat = Eigen::SparseMatrix<double>;
using Trip = Eigen::Triplet<double>;

// ---------------------------------------------------------------- casadi call
namespace {
struct Caller {
    casadi_int sz_arg = 0, sz_res = 0, sz_iw = 0, sz_w = 0;
    std::vector<casadi_real> w;
    std::vector<casadi_int> iw;
    void init(int (*wf)(casadi_int*, casadi_int*, casadi_int*, casadi_int*)) {
        wf(&sz_arg, &sz_res, &sz_iw, &sz_w);
        w.resize(std::max<casadi_int>(sz_w, 1));
        iw.resize(std::max<casadi_int>(sz_iw, 1));
    }
};

struct QPResult { VectorXd z; int iters; int status; double obj; double prim_res; };
}  // namespace

// ----------------------------------------------------------------- QP backend
class QPBackend {
public:
    virtual ~QPBackend() = default;
    virtual const char* name() const = 0;
    // box-constrained equality QP: 0.5 z'Hz+g'z s.t. Az=b, lb<=z<=ub
    virtual QPResult solve(const SpMat& H, const VectorXd& g, const SpMat& A,
                           const VectorXd& b, const VectorXd& lb,
                           const VectorXd& ub) = 0;
};

#ifdef HAVE_PROXQP
class ProxqpBackend : public QPBackend {
public:
    const char* name() const override { return "proxqp"; }
    QPResult solve(const SpMat& H, const VectorXd& g, const SpMat& A,
                   const VectorXd& b, const VectorXd& lb,
                   const VectorXd& ub) override {
        using namespace proxsuite::proxqp;
        const isize n = H.rows(), neq = A.rows(), nin = n;
        SpMat C(nin, n);
        C.setIdentity();
        sparse::QP<double, long long> qp(n, neq, nin);
        qp.settings.eps_abs = 1e-8;
        qp.settings.verbose = false;
        qp.settings.max_iter = 200;          // guarantee termination
        qp.settings.max_iter_in = 50;
        qp.init(H, g, A, b, C, lb, ub);
        qp.solve();
        QPResult r;
        r.z = qp.results.x;
        r.iters = (int)qp.results.info.iter;
        r.obj = qp.results.info.objValue;
        r.prim_res = qp.results.info.pri_res;
        r.status = (qp.results.info.status ==
                    proxsuite::proxqp::QPSolverOutput::PROXQP_SOLVED) ? 0 : 1;
        return r;
    }
};
#endif

#ifdef HAVE_PIQP
class PiqpBackend : public QPBackend {
public:
    const char* name() const override { return "piqp"; }
    QPResult solve(const SpMat& H, const VectorXd& g, const SpMat& A,
                   const VectorXd& b, const VectorXd& lb,
                   const VectorXd& ub) override {
        piqp::SparseSolver<double> s;
        s.settings().verbose = false;
        s.settings().eps_abs = 1e-8;
        SpMat Hc = H, Ac = A;
        SpMat G(0, H.rows());
        VectorXd h(0);
        s.setup(Hc, g, Ac, b, G, h, lb, ub);
        s.solve();
        QPResult r;
        r.z = s.result().x;
        r.iters = s.result().info.iter;
        r.obj = s.result().info.primal_obj;
        r.prim_res = s.result().info.primal_res;
        r.status = (s.result().info.status == piqp::Status::PIQP_SOLVED) ? 0 : 1;
        return r;
    }
};
#endif

static QPBackend* make_backend(int id) {
    // 0=proxqp 1=piqp 2=osqp 3=qpoases ; falls back to first available
#ifdef HAVE_PROXQP
    if (id == 0) return new ProxqpBackend();
#endif
#ifdef HAVE_PIQP
    if (id == 1) return new PiqpBackend();
#endif
    // default: first compiled backend
#ifdef HAVE_PROXQP
    return new ProxqpBackend();
#elif defined(HAVE_PIQP)
    return new PiqpBackend();
#else
    return nullptr;
#endif
}

// ------------------------------------------------------------------- the SQP
class FastSqp {
public:
    int nx, nu, N, nz, neq;
    VectorXd lbx, ubx, lbu, ubu;
    VectorXd Xnom, Unom;     // nominal trajectory ((N+1)*nx, N*nu)
    double reg = 1e-8;
    QPBackend* qp;
    Caller cdyn, ccost, ccostN;

    FastSqp(int nx_, int nu_, int N_, const double* lbx_, const double* ubx_,
            const double* lbu_, const double* ubu_, int backend)
        : nx(nx_), nu(nu_), N(N_) {
        nz = nx * (N + 1) + nu * N;
        neq = nx * (N + 1);
        lbx = Eigen::Map<const VectorXd>(lbx_, nx);
        ubx = Eigen::Map<const VectorXd>(ubx_, nx);
        lbu = Eigen::Map<const VectorXd>(lbu_, nu);
        ubu = Eigen::Map<const VectorXd>(ubu_, nu);
        Xnom = VectorXd::Zero((N + 1) * nx);
        Unom = VectorXd::Zero(N * nu);
        cdyn.init(dyn_work);
        ccost.init(costs_work);
        ccostN.init(costN_work);
        qp = make_backend(backend);
    }
    ~FastSqp() { delete qp; }

    int ix(int k) const { return k * nx; }
    int iu(int k) const { return nx * (N + 1) + k * nu; }

    void eval_dyn(const double* x, const double* u, double* xnext,
                  double* A, double* B) {
        const double* arg[2] = {x, u};
        double* res[3] = {xnext, A, B};
        dyn(arg, res, cdyn.iw.data(), cdyn.w.data(), 0);
    }
    void eval_cost(const double* x, const double* u, const double* xr,
                   const double* ur, double* gx, double* gu, double* Hxx,
                   double* Huu, double* Hxu) {
        const double* arg[4] = {x, u, xr, ur};
        double* res[5] = {gx, gu, Hxx, Huu, Hxu};
        costs(arg, res, ccost.iw.data(), ccost.w.data(), 0);
    }
    void eval_costN(const double* x, const double* xr, double* gx, double* Hxx) {
        const double* arg[2] = {x, xr};
        double* res[2] = {gx, Hxx};
        costN(arg, res, ccostN.iw.data(), ccostN.w.data(), 0);
    }

    // roll out a dynamically-consistent nominal from x0 with zero controls
    void reset(const double* x0) {
        Unom.setZero();
        Eigen::Map<VectorXd>(Xnom.data(), nx) = Eigen::Map<const VectorXd>(x0, nx);
        std::vector<double> An(nx * nx), Bn(nx * nu);
        for (int k = 0; k < N; ++k) {
            eval_dyn(Xnom.data() + ix(k), Unom.data() + k * nu,
                     Xnom.data() + ix(k + 1), An.data(), Bn.data());
        }
    }

    // one closed-loop step: returns u0, fills stats [iters,status,prim_res,obj]
    int solve(const double* x_meas, const double* xref, const double* uref,
              int sqp_iters, double* u0, double* stats) {
        std::vector<double> An(nx * nx), Bn(nx * nu), fd(nx);
        std::vector<double> gx(nx), gu(nu), Hxx(nx * nx), Huu(nu * nu),
            Hxu(nx * nu), gxN(nx), HxxN(nx * nx);
        int last_status = 0, last_iters = 0;
        double last_pr = 0, last_obj = 0;
        auto t0 = std::chrono::high_resolution_clock::now();

        for (int it = 0; it < sqp_iters; ++it) {
            std::vector<Trip> Ht, At;
            Ht.reserve(nz * 4);
            At.reserve(neq * (2 * nx + nu));
            VectorXd g = VectorXd::Zero(nz), b = VectorXd::Zero(neq);
            VectorXd lb(nz), ub(nz);

            // initial-value embedding: dx_0 = x_meas - x_nom_0
            for (int i = 0; i < nx; ++i) {
                At.emplace_back(i, ix(0) + i, 1.0);
                b(i) = x_meas[i] - Xnom(ix(0) + i);
            }
            // stages
            for (int k = 0; k < N; ++k) {
                eval_dyn(Xnom.data() + ix(k), Unom.data() + k * nu, fd.data(),
                         An.data(), Bn.data());
                eval_cost(Xnom.data() + ix(k), Unom.data() + k * nu,
                          xref + k * nx, uref + k * nu, gx.data(), gu.data(),
                          Hxx.data(), Huu.data(), Hxu.data());
                Eigen::Map<const MatrixXd> A_k(An.data(), nx, nx);
                Eigen::Map<const MatrixXd> B_k(Bn.data(), nx, nu);
                // dynamics equality
                int row = (k + 1) * nx;
                for (int i = 0; i < nx; ++i) {
                    At.emplace_back(row + i, ix(k + 1) + i, 1.0);
                    for (int j = 0; j < nx; ++j)
                        if (A_k(i, j) != 0.0)
                            At.emplace_back(row + i, ix(k) + j, -A_k(i, j));
                    for (int j = 0; j < nu; ++j)
                        if (B_k(i, j) != 0.0)
                            At.emplace_back(row + i, iu(k) + j, -B_k(i, j));
                    b(row + i) = fd[i] - Xnom(ix(k + 1) + i);
                }
                // Hessian + gradient blocks
                for (int i = 0; i < nx; ++i) {
                    g(ix(k) + i) += gx[i];
                    for (int j = 0; j < nx; ++j)
                        Ht.emplace_back(ix(k) + i, ix(k) + j, Hxx[i + j * nx]);
                    for (int j = 0; j < nu; ++j) {
                        Ht.emplace_back(ix(k) + i, iu(k) + j, Hxu[i + j * nx]);
                        Ht.emplace_back(iu(k) + j, ix(k) + i, Hxu[i + j * nx]);
                    }
                }
                for (int i = 0; i < nu; ++i) {
                    g(iu(k) + i) += gu[i];
                    for (int j = 0; j < nu; ++j)
                        Ht.emplace_back(iu(k) + i, iu(k) + j, Huu[i + j * nu]);
                }
            }
            // terminal cost
            eval_costN(Xnom.data() + ix(N), xref + N * nx, gxN.data(), HxxN.data());
            for (int i = 0; i < nx; ++i) {
                g(ix(N) + i) += gxN[i];
                for (int j = 0; j < nx; ++j)
                    Ht.emplace_back(ix(N) + i, ix(N) + j, HxxN[i + j * nx]);
            }
            // regularization
            for (int i = 0; i < nz; ++i) Ht.emplace_back(i, i, reg);
            // box bounds around nominal
            for (int k = 0; k <= N; ++k)
                for (int i = 0; i < nx; ++i) {
                    lb(ix(k) + i) = lbx(i) - Xnom(ix(k) + i);
                    ub(ix(k) + i) = ubx(i) - Xnom(ix(k) + i);
                }
            for (int k = 0; k < N; ++k)
                for (int i = 0; i < nu; ++i) {
                    lb(iu(k) + i) = lbu(i) - Unom(k * nu + i);
                    ub(iu(k) + i) = ubu(i) - Unom(k * nu + i);
                }

            SpMat H(nz, nz), A(neq, nz);
            H.setFromTriplets(Ht.begin(), Ht.end());
            A.setFromTriplets(At.begin(), At.end());
            QPResult r = qp->solve(H, g, A, b, lb, ub);
            last_status = r.status; last_iters = r.iters;
            last_pr = r.prim_res; last_obj = r.obj;
            if (r.status != 0 || !r.z.allFinite())
                break;          // do not step on an unsolved QP (hold previous)
            // full step update of the nominal trajectory
            Xnom += r.z.head((N + 1) * nx);
            Unom += r.z.tail(N * nu);
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < nu; ++i) u0[i] = Unom(i);
        stats[0] = last_iters;
        stats[1] = last_status;
        stats[2] = last_pr;
        stats[3] = std::chrono::duration<double>(t1 - t0).count();
        stats[4] = last_obj;
        return last_status;
    }
};

// ------------------------------------------------------------------- C ABI
extern "C" {
void* fastsqp_create(int nx, int nu, int N, const double* lbx, const double* ubx,
                     const double* lbu, const double* ubu, int backend) {
    return new FastSqp(nx, nu, N, lbx, ubx, lbu, ubu, backend);
}
void fastsqp_reset(void* h, const double* x0) {
    static_cast<FastSqp*>(h)->reset(x0);
}
int fastsqp_solve(void* h, const double* x_meas, const double* xref,
                  const double* uref, int sqp_iters, double* u0, double* stats) {
    return static_cast<FastSqp*>(h)->solve(x_meas, xref, uref, sqp_iters, u0, stats);
}
const char* fastsqp_backend_name(void* h) {
    auto* s = static_cast<FastSqp*>(h);
    return s->qp ? s->qp->name() : "none";
}
void fastsqp_destroy(void* h) { delete static_cast<FastSqp*>(h); }
}
