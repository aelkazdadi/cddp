import itertools
import warnings

import numpy as np
from numpy.linalg import cond
from scipy.linalg import ldl, norm, solve, solve_triangular

from helper import CDDP, Derivatives, Function, Trajectory, dot, trim_dtype


class CDDP_Primal(CDDP):
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
        self._p = np.zeros(total_dims, dtype=self.dtype)

    def forward_pass(self, ff, fb):
        T = self.T
        old_traj = self.traj
        new_traj = old_traj.copy()
        costs = np.empty([T + 1, 2], dtype=self.dtype)
        cost_initialized = False

        for alpha in map(lambda i: 2 ** (-i), range(8)):
            for t in range(T):
                old_u = old_traj.u(t)
                old_x = old_traj.x(t)
                new_u = new_traj.u(t)
                new_x = new_traj.x(t)

                new_traj.update(
                    t, old_u + alpha * ff[t] + fb[t] @ (new_x - old_x)
                )

                p = self.p(t)
                mu = self.mu[t]

                if not cost_initialized:
                    c = self.c(old_x, old_u, t)
                    costs[t, 0] = (
                        self.l(old_x, old_u, t) + (p + mu / 2 * c).T @ c
                    )

                c = self.c(new_x, new_u, t)
                costs[t, 1] = self.l(new_x, new_u, t) + (p + mu / 2 * c).T @ c

            if not cost_initialized:
                costs[T, :0] = self.l_f(new_traj.x(T))
                cost_initialized = True
            costs[T, :1] = self.l_f(new_traj.x(T))

            self.traj = new_traj
            if (costs[:, 1] - costs[:, 0]).sum() < 0:
                self.traj = new_traj
                return alpha
        raise Exception("Descent step not found")

    def backward_pass(self):
        T = self.T

        ff = np.empty([T, self.traj.u_dim], dtype=self.dtype)
        fb = np.empty([T, self.traj.u_dim, self.traj.x_dim], dtype=self.dtype)

        Vxx = self.l_f.xx(self.traj.x(T))
        Vx = self.l_f.x(self.traj.x(T))
        for t in reversed(range(T)):
            Q = self.get_model(t, Vx, Vxx)
            ff[t], fb[t] = self.get_feedback(t, Q)

            Vx = Q.x + Q.ux.T @ ff[t]
            Vxx = Q.xx + Q.ux.T @ fb[t]
        return ff, fb

    def get_model(self, t, Vx, Vxx):
        Q, d = self.Q_and_derivatives(t, Vx, Vxx)

        tmp = self.p(t) + self.mu[t] * d.c

        Q.x += tmp @ d.cx
        Q.u += tmp @ d.cu
        Q.xx += self.mu[t] * d.cx.T @ d.cx + dot(tmp, d.cxx)
        Q.uu += self.mu[t] * d.cu.T @ d.cu + dot(tmp, d.cuu)
        Q.ux += self.mu[t] * d.cu.T @ d.cx + dot(tmp, d.cux)
        return Q

    def get_feedback(self, t, Q):
        rhs = np.empty((self.traj.u_dim, self.traj.x_dim + 1), dtype=self.dtype)
        rhs[:, 0] = Q.u
        rhs[:, 1:] = Q.ux
        ff_fb = -solve(Q.uu, rhs, sym_pos=True)
        if self.debug["cond"] is not None:
            self.debug["cond"][t] = cond(trim_dtype(Q.uu))
        ff, fb = ff_fb[:, 0], ff_fb[:, 1:]
        return ff, fb

    def update_multipliers(self, _):
        for t in range(self.T):
            x = self.traj.x(t)
            u = self.traj.u(t)
            self.p(t)[:] += self.mu[t] * self.c(x, u, t)
