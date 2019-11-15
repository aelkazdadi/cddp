import numpy as np

from cddp_primal import CDDP_Primal
from cddp_primaldual import CDDP_Primal_Dual
from helper import Function, Trajectory


"""
Problem:
J(x, u) = sum_{t=0}^{t=T-1} l(x_t, u_t, t) + l_f(x_T)
min J(x, u)
s.t.    x_{t+1} = f(x_t, u_t, t)
and     c(x_t, u_t, t) = 0  for all t in [0, T)


LQR case:
f(x, u, t) = A_t x + B_t u + C_t
l(x, u, t) =                   [ R_t S_t.T ] [ x_t ]               [ x_t ]
             0.5 * [ x_t u_t ] [ S_t   Q_t ] [ u_t ] + [ r_t q_t ] [ u_t ]
c(x, u, t) =           [ x_t ]
             [ M1 M2 ] [ u_t ] - M0
"""


def toy_problem(CDDP_Class, dtype=np.float64):
    dt = 1e-1

    A = np.array([[1.0, dt], [-dt, 1.0]], dtype=dtype)
    B = dt * np.identity(2, dtype=dtype)
    C = np.array([dt, -dt], dtype=dtype)

    R = np.identity(2, dtype=dtype)
    Q = np.identity(2, dtype=dtype)
    S = np.zeros((2, 2), dtype=dtype)
    r = np.zeros(2, dtype=dtype)
    q = np.zeros(2, dtype=dtype)

    R_f = np.identity(2, dtype=dtype)
    r_f = np.zeros(2, dtype=dtype)

    M0 = np.array([1.0], dtype=dtype)
    M1 = np.array([[1.0, 1.0]], dtype=dtype)
    M2 = np.array([[1.0, 1.0]], dtype=dtype)

    x_init = np.array([1.0, 0.0], dtype=dtype)
    horizon = 3
    return from_parameters(
        CDDP_Class,
        A,
        B,
        C,
        R,
        Q,
        S,
        r,
        q,
        R_f,
        r_f,
        M0,
        M1,
        M2,
        x_init,
        horizon,
    )


def rand(shape, dtype=np.float64):
    out = np.zeros(shape, dtype=dtype)
    out[:] = np.random.randn(*shape)
    return out


def rand_sym(dim, dtype=np.float64):
    out = np.zeros([dim, dim], dtype=dtype)
    out[:] = np.random.randn(dim, dim)
    out = out.T @ out
    return out


def random_problem(x_dim, u_dim, c_dim, horizon, dt=1e-2, dtype=np.float64):
    A = np.identity(x_dim) + dt * rand([x_dim, x_dim], dtype)
    B = dt * rand([x_dim, u_dim], dtype)
    C = dt * rand([x_dim], dtype=dtype)

    R = dt * rand_sym(x_dim, dtype=dtype)
    Q = dt * rand_sym(u_dim, dtype=dtype)
    S = dt * rand([u_dim, x_dim], dtype=dtype)
    S[:] = 0
    r = dt * rand([x_dim], dtype=dtype)
    q = dt * rand([u_dim], dtype=dtype)

    R_f = rand_sym(x_dim, dtype=dtype)
    r_f = rand([x_dim], dtype=dtype)

    M0 = rand([c_dim], dtype=dtype)
    M1 = rand([c_dim, x_dim], dtype=dtype)
    M2 = rand([c_dim, u_dim], dtype=dtype)

    x_init = rand([x_dim], dtype=dtype)
    return [
        from_parameters(
            CDDP, A, B, C, R, Q, S, r, q, R_f, r_f, M0, M1, M2, x_init, horizon,
        )
        for CDDP in [CDDP_Primal, CDDP_Primal_Dual]
    ]


def from_parameters(
    CDDP_Class, A, B, C, R, Q, S, r, q, R_f, r_f, M0, M1, M2, x_init, horizon
):
    n = A.shape[0]
    m = B.shape[1]
    f = Function(
        f=lambda x, u, t: A @ x + B @ u + C,
        fx=lambda x, u, t: A,
        fxx=lambda x, u, t: np.zeros([n, n, n]),
        fu=lambda x, u, t: B,
        fuu=lambda x, u, t: np.zeros([n, m, m]),
        fux=lambda x, u, t: np.zeros([n, m, n]),
    )

    l = Function(
        f=lambda x, u, t: 0.5 * x.T @ R @ x
        + 0.5 * u.T @ Q @ u
        + u @ S @ x
        + r @ x
        + q @ u,
        fx=lambda x, u, t: R @ x + u @ S + r,
        fxx=lambda x, u, t: R,
        fu=lambda x, u, t: Q @ u + S @ x + q,
        fuu=lambda x, u, t: Q,
        fux=lambda x, u, t: S,
    )

    l_f = Function(
        f=lambda x: 0.5 * x.T @ R_f @ x + r_f @ x,
        fx=lambda x: R_f @ x + r_f,
        fxx=lambda x: R_f,
    )

    c_dim = M0.size
    c = Function(
        f=lambda x, u, t: M1 @ x + M2 @ u - M0,
        fx=lambda x, u, t: M1,
        fxx=lambda x, u, t: np.zeros([c_dim, n, n]),
        fu=lambda x, u, t: M2,
        fuu=lambda x, u, t: np.zeros([c_dim, m, m]),
        fux=lambda x, u, t: np.zeros([c_dim, m, n]),
        out_dim=lambda t: c_dim,
    )

    traj = Trajectory(n, m, horizon, x_init, f)
    traj.data[:-1, n:] = np.random.randn(horizon, m)
    return CDDP_Class(traj, f, l, l_f, c)


def full_qp(problem):
    horizon = problem.traj.T
    n = problem.traj.x_dim
    m = problem.traj.u_dim
    n_var = horizon * (n + m) + n

    H = np.zeros((n_var, n_var), dtype=problem.dtype)
    g = np.zeros(n_var, dtype=problem.dtype)
    c = [np.zeros((n, n_var), dtype=problem.dtype)]
    c[0][:, :n] = np.identity(n)
    c_rhs = [problem.traj.x(0)]

    x = np.zeros(n, dtype=problem.dtype)
    u = np.zeros(m, dtype=problem.dtype)
    for t in range(horizon):

        x_begin = t * (n + m)
        u_begin = t * (n + m) + n
        end = t * (n + m) + n + m

        H[x_begin:u_begin, x_begin:u_begin] = problem.l.xx(x, u, t)
        H[u_begin:end, u_begin:end] = problem.l.uu(x, u, t)

        H[u_begin:end, x_begin:u_begin] = problem.l.ux(x, u, t)
        H[x_begin:u_begin, u_begin:end] = problem.l.ux(x, u, t).T

        g[x_begin:u_begin] = problem.l.x(x, u, t)
        g[u_begin:end] = problem.l.u(x, u, t)

        c_rhs.append(
            -problem.c(
                np.zeros(n, dtype=problem.dtype),
                np.zeros(m, dtype=problem.dtype),
                0,
            )
        )
        c.append(
            np.zeros(
                (problem.c.out_dim(t), horizon * (n + m) + n),
                dtype=problem.dtype,
            )
        )
        c[-1][:, x_begin:u_begin] = problem.c.x(x, u, t)
        c[-1][:, u_begin:end] = problem.c.u(x, u, t)

        c_rhs.append(np.zeros(n, dtype=problem.dtype))
        c.append(np.zeros((n, horizon * (n + m) + n), dtype=problem.dtype))
        c[-1][:, x_begin:u_begin] = problem.f.x(x, u, t)
        c[-1][:, u_begin:end] = problem.f.u(x, u, t)
        c[-1][:, end : end + n] = -np.identity(n)

    begin = horizon * (n + m)
    H[begin:, begin:] = problem.l_f.xx(x)
    g[begin:] = problem.l_f.x(x)

    c = np.vstack(c)
    c_rhs = np.hstack(c_rhs)
    _, _, vh = np.linalg.svd(c.astype(dtype=np.float64))
    K = vh.T[:, c.shape[0] :]
    return H, g, c, c_rhs, K
