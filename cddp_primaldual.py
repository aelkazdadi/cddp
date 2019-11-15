import itertools
import warnings

import numpy as np
from numpy.linalg import cond
from scipy.linalg import ldl, norm, solve, solve_triangular

from helper import CDDP, Function, Trajectory, trim_dtype


def ldlt_solve(ldl_factor, b):
    l, d, p = ldl_factor
    x = np.empty_like(b)
    x[p] = b
    x[:] = solve_triangular(l[p], x, lower=True)
    x[:] *= d
    x[:] = solve_triangular((l[p]).T, x, lower=False)
    x[:] = x[p]
    return x


def solve_eq_qp(H, g, A, b, p, mu, max_iter=0, debug_cond=None):
    n = H.shape[0]
    m = A.shape[0]

    M = np.empty([n + m, n + m], dtype=H.dtype)
    M[:n, :n] = H
    M[:n, n:] = A.T
    M[n:, :n] = A
    M[n:, n:] = -np.identity(m) / mu

    if debug_cond is not None:
        debug_cond[:] = cond(trim_dtype(M))

    l, d, perm = ldl(M, lower=True)

    # Check if d is diagonal (superdiagonal and subdiagonal are zero)
    if (d.flat[1 :: n + m + 1] == 0.0).all() and (
        d.flat[n + m :: n + m + 1] == 0.0
    ).all():
        d = 1 / d.diagonal()[:, np.newaxis]
        arg = (l, d, perm)
        sym_solve = ldlt_solve
    else:
        warnings.warn("Non diagonal matrix in augmented lagrangian LDLT")
        arg = M
        sym_solve = lambda mat, vec: solve(mat, vec, assume_a="sym")

    err = lambda g_, b_, x_, p_: [
        (H @ x_ + g_ + A.T @ p_),
        (A @ x_ - b_),
    ]
    optimality = lambda err_: norm(np.concatenate(err_))

    rhs = np.empty([n + m, g.shape[1]], dtype=H.dtype)
    rhs[n:] = b

    p_new = p.copy()
    x_new = np.zeros_like(g)
    opt = np.nan

    loop = itertools.count() if max_iter == 0 else range(max_iter)
    for i in loop:
        rhs[:n] = -g - A.T @ p_new
        x_p = sym_solve(arg, rhs)
        p_new += x_p[n:]
        x_new[:] = x_p[:n]

        if i % 5 == 0:
            new_opt = optimality(err(g, b, x_new, p_new))
        if opt < new_opt:
            break
        opt = new_opt
    x_new = x_p[:n]

    def deb_comp(f, y):
        return f[:, 0] + f[:, 1:] @ y

    def deb_opt(y):
        g_ = deb_comp(g, y)
        b_ = deb_comp(b, y)
        x_ = deb_comp(x_new, y)
        p_ = deb_comp(p_new, y)
        return list(map(norm, err(g_, b_, x_, p_)))

    if CDDP_Primal_Dual.pdb:
        y = np.random.randn(g.shape[1] - 1)
        # __import__("ipdb").set_trace()

    return x_new, p_new


class CDDP_Primal_Dual(CDDP):
    pdb = False

    def __init__(
        self,
        traj: Trajectory,
        f: Function,
        l: Function,
        l_f: Function,
        c: Function,
    ):
        super().__init__(traj, f, l, l_f, c)

        total_dims = self._c_idx[-1]
        self._p = np.zeros([total_dims, traj.x_dim + 1], dtype=self.dtype)
        self._p_new = np.zeros([total_dims, traj.x_dim + 1], dtype=self.dtype)

        self.qp_maxiter = 1

    def p_new(self, t):
        idx = self._c_idx
        return self._p_new[idx[t] : idx[t + 1]]

    def forward_pass(self, _ff, _fb):
        raise NotImplementedError

    def backward_pass(self):
        ff = np.empty([self.T, self.traj.u_dim], dtype=self.dtype)
        fb = np.empty(
            [self.T, self.traj.u_dim, self.traj.x_dim], dtype=self.dtype
        )

        x_T = self.traj.x(self.T)
        Vxx = self.l_f.xx(x_T)
        Vx = self.l_f.x(x_T)
        for t in reversed(range(self.T)):
            Q, C = self.get_model(t, Vx, Vxx)
            ff[t], fb[t] = self.get_feedback(t, Q, C)

            k, K = ff[t], fb[t]
            Vx = Q.x + K.T @ (Q.uu @ k + Q.u) + Q.ux.T @ k
            Vxx = Q.xx + K.T @ (Q.uu @ K + Q.ux) + Q.ux.T @ K
        return ff, fb

    def update_multipliers(self, old_traj: Trajectory):
        for t in range(self.T):
            p = self.p(t)
            p_new = self.p_new(t)
            p[:, 1:] = p_new[:, 1:]

            p[:, 0] = p_new[:, 0] - p[:, 1:] @ (self.traj.x(t) - old_traj.x(t))
            # p[:, 0] += self.mu[t] * self.c_violation(t) + p_new[:, 1:] @ (
            #     self.traj.x(t) - old_traj.x(t)
            # )

    def get_model(self, t, Vx, Vxx):
        Q, d = self.Q_and_derivatives(t, Vx, Vxx)
        C = Function()

        C.val = d.c
        C.x = d.cx
        C.u = d.cu
        C.xx = d.cxx
        C.uu = d.cuu
        C.ux = d.cux
        return Q, C

    def get_feedback(self, t, Q, C):
        q_grad = np.c_[Q.u, Q.ux]
        c_rhs = -np.c_[C.val, C.x]

        ff_fb, self.p_new(t)[:] = solve_eq_qp(
            Q.uu,
            q_grad,
            C.u,
            c_rhs,
            self.p(t),
            self.mu[t],
            max_iter=self.qp_maxiter,
            debug_cond=None
            if self.debug["cond"] is None
            else self.debug["cond"][t : t + 1],
        )
        ff, fb = ff_fb[:, 0], ff_fb[:, 1:]
        return ff, fb
