"""Build a sparse linear-MPC QP from an LTI problem.

Decision vector ``z = [x_0, ..., x_N, u_0, ..., u_{N-1}]``.

Objective   0.5 z' P z + q' z
Dynamics    x_0 = x_init ,  x_{k+1} = A x_k + B u_k + c
Bounds      lbx <= x_k <= ubx ,  lbu <= u_k <= ubu

The static structure (P, equality Jacobian) is built once; only the linear
term ``q`` and the equality right-hand side ``b`` change per step (initial
state + time-varying reference), so adapters update rather than rebuild.
"""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp


class MpcQp:
    def __init__(self, problem):
        m = problem.meta
        mats = problem.lti_matrices()
        self.nx, self.nu, self.N = m.nx, m.nu, m.N
        self.A = np.asarray(mats["A"], float)
        self.B = np.asarray(mats["B"], float)
        self.Q = np.asarray(mats["Q"], float)
        self.R = np.asarray(mats["R"], float)
        self.Qf = np.asarray(mats.get("Qf", mats["Q"]), float)
        self.c = np.asarray(mats.get("c", np.zeros(self.nx)), float)
        self.lbx, self.ubx, self.lbu, self.ubu = problem.bounds()
        self.problem = problem

        nx, nu, N = self.nx, self.nu, self.N
        self.nz = nx * (N + 1) + nu * N
        self.ix = lambda k: slice(k * nx, (k + 1) * nx)              # state k
        self.iu = lambda k: slice(nx * (N + 1) + k * nu,
                                  nx * (N + 1) + (k + 1) * nu)       # input k

        self._build_P()
        self._build_eq_structure()
        self._build_bounds()

    # -- objective ----------------------------------------------------------
    def _build_P(self):
        blocks = [self.Q] * self.N + [self.Qf] + [self.R] * self.N
        self.P = sp.block_diag(blocks, format="csc")

    def q(self, xrefs, urefs):
        nx, nu, N = self.nx, self.nu, self.N
        q = np.zeros(self.nz)
        for k in range(N):
            q[self.ix(k)] = -self.Q @ xrefs[k]
            q[self.iu(k)] = -self.R @ urefs[k]
        q[self.ix(N)] = -self.Qf @ xrefs[N]
        return q

    # -- equality (dynamics) ------------------------------------------------
    def _build_eq_structure(self):
        nx, nu, N = self.nx, self.nu, self.N
        neq = nx * (N + 1)
        rows, cols, data = [], [], []

        def add(blockrow, colslice, M):
            r0 = blockrow * nx
            cs = range(colslice.start, colslice.stop)
            for i in range(nx):
                for j, c in enumerate(cs):
                    rows.append(r0 + i); cols.append(c); data.append(M[i, j])

        # x_0 = x_init  -> I on x_0
        add(0, self.ix(0), np.eye(nx))
        # x_{k+1} - A x_k - B u_k = c
        for k in range(N):
            add(k + 1, self.ix(k + 1), np.eye(nx))
            add(k + 1, self.ix(k), -self.A)
            add(k + 1, self.iu(k), -self.B)
        self.Aeq = sp.csc_matrix((data, (rows, cols)), shape=(neq, self.nz))
        self.neq = neq

    def beq(self, x0):
        nx, N = self.nx, self.N
        b = np.zeros(self.neq)
        b[0:nx] = x0
        for k in range(N):
            b[(k + 1) * nx:(k + 2) * nx] = self.c
        return b

    # -- box bounds on z ----------------------------------------------------
    def _build_bounds(self):
        nx, nu, N = self.nx, self.nu, self.N
        lb = np.empty(self.nz); ub = np.empty(self.nz)
        for k in range(N + 1):
            lb[self.ix(k)] = self.lbx; ub[self.ix(k)] = self.ubx
        for k in range(N):
            lb[self.iu(k)] = self.lbu; ub[self.iu(k)] = self.ubu
        self.lb_z, self.ub_z = lb, ub

    # -- helper to pull horizon references at absolute time k ---------------
    def horizon_refs(self, k_abs):
        xr = [self.problem.reference_traj(k_abs + j)[0] for j in range(self.N + 1)]
        ur = [self.problem.reference_traj(k_abs + j)[1] for j in range(self.N)]
        return xr, ur
