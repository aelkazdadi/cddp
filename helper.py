import warnings

import numpy as np


class Function:
    def __init__(
        self,
        f=None,
        fx=None,
        fxx=None,
        fu=None,
        fuu=None,
        fux=None,
        out_dim=None,
    ):
        self.val = f
        self.x = fx
        self.u = fu
        self.xx = fxx
        self.ux = fux
        self.uu = fuu

        self.out_dim = out_dim

    def __call__(self, *args, **kwargs):
        return self.val(*args, **kwargs)


class Derivatives:
    def __init__(self, x, u, t, prob):
        f = prob.f
        l = prob.l
        c = prob.c

        self.fx = f.x(x, u, t)
        self.fxx = f.xx(x, u, t)
        self.fu = f.u(x, u, t)
        self.fuu = f.uu(x, u, t)
        self.fux = f.ux(x, u, t)

        self.lx = l.x(x, u, t)
        self.lxx = l.xx(x, u, t)
        self.lu = l.u(x, u, t)
        self.luu = l.uu(x, u, t)
        self.lux = l.ux(x, u, t)

        self.c = c(x, u, t)
        self.cx = c.x(x, u, t)
        self.cxx = c.xx(x, u, t)
        self.cu = c.u(x, u, t)
        self.cuu = c.uu(x, u, t)
        self.cux = c.ux(x, u, t)


class Trajectory:
    def __init__(self, x_dim, u_dim, T, x_init, f, init=True):
        self.dtype = x_init.dtype
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.T = T
        self.data = np.empty([T + 1, x_dim + u_dim], dtype=self.dtype)
        self.data[0, :x_dim] = x_init
        self.f = f

        if init:
            self.data[:, x_dim:] = 0.0
            self.update_all()

    def copy(self):
        clone = Trajectory(
            self.x_dim, self.u_dim, self.T, self.x(0), self.f, init=False
        )
        clone.data = self.data.copy()
        return clone

    def x(self, i):
        return self.data[i, : self.x_dim]

    def u(self, i):
        return self.data[i, self.x_dim :]

    def update(self, t, u):
        x_t = self.data[t, : self.x_dim]
        u_t = self.data[t, self.x_dim :]
        x_next = self.data[t + 1, : self.x_dim]

        u_t[:] = u
        x_next[:] = self.f(x_t, u_t, t)

    def update_all(self):
        for t in range(self.T):
            u = self.data[t, self.x_dim :]
            self.update(t, u)


def dot(vec, tensor):
    return vec @ tensor.transpose(1, 0, 2)


def trim_dtype(M):
    return M.astype(dtype=np.float64) if M.dtype == np.dtype("float128") else M


class CDDP:
    def __init__(
        self,
        traj: Trajectory,
        f: Function,
        l: Function,
        l_f: Function,
        c: Function,
    ):
        self.traj = traj
        self.dtype = traj.dtype
        self.f = f
        self.l = l
        self.l_f = l_f
        self.c = c

        self.T = self.traj.T

        self.mu = 1e2 * np.ones(self.T)
        self._p = np.array([0])

        c_dims = np.array([c.out_dim(t) for t in range(self.T)], dtype=int)
        self._c_idx = np.concatenate([[0], np.cumsum(c_dims)])

        self.debug = dict()
        self.debug["linesearch"] = True
        self.debug["cond"] = None

    def p(self, t):
        idx = self._c_idx
        return self._p[idx[t] : idx[t + 1]]

    def log_condition_number(self, flag):
        self.debug["cond"] = np.zeros(self.T) if flag else None

    def one_iter(self):
        if self.debug["cond"] is not None:
            warnings.warn("Logging condition numbers")

        ff, fb = self.backward_pass()
        old_traj, _ = self.forward_pass_unchecked(ff, fb)
        self.update_multipliers(old_traj)

    def forward_pass(self, _ff, _fb):
        raise NotImplementedError

    def backward_pass(self):
        raise NotImplementedError

    def update_multipliers(self, old_traj):
        raise NotImplementedError

    def forward_pass_unchecked(self, ff, fb):
        warnings.warn("Skipping line search")
        old_traj = self.traj
        new_traj = old_traj.copy()
        for t in range(self.T):
            old_u = old_traj.u(t)
            old_x = old_traj.x(t)
            new_x = new_traj.x(t)
            new_traj.update(t, old_u + ff[t] + fb[t] @ (new_x - old_x))
        self.traj = new_traj
        return old_traj, 1.0

    def costs(self):
        T = self.T
        out = np.empty(T + 1, dtype=self.dtype)
        for t in range(T):
            out[t] = self.l(self.traj.x(t), self.traj.u(t), t)
        out[T] = self.l_f(self.traj.x(T))
        return out

    def c_violation(self, t=None):
        if t is None:
            return np.concatenate(
                [
                    self.c(self.traj.x(t), self.traj.u(t), t)
                    for t in range(self.T)
                ]
            )
        return self.c(self.traj.x(t), self.traj.u(t), t)

    def reset(self):
        self.traj.data[:-1, self.traj.x_dim :] = 0
        self.traj.update_all()
        self._p[:] = 0.0

    def Q_and_derivatives(self, t, Vx, Vxx):
        x = self.traj.x(t)
        u = self.traj.u(t)

        d = Derivatives(x, u, t, self)
        Q = Function()

        Q.x = d.lx + d.fx.T @ Vx
        Q.u = d.lu + d.fu.T @ Vx
        Q.xx = d.lxx + d.fx.T @ Vxx @ d.fx + dot(Vx, d.fxx)
        Q.uu = d.luu + d.fu.T @ Vxx @ d.fu + dot(Vx, d.fuu)
        Q.ux = d.lux + d.fu.T @ Vxx @ d.fx + dot(Vx, d.fux)

        return Q, d
