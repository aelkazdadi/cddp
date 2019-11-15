import itertools
import os
import sys

import numpy as np
from numpy.linalg import norm

from gen_problem import full_qp, random_problem

np.random.seed(1)
problem_primal, problem_primal_dual = random_problem(
    50, 20, 10, 10, 1e-1, dtype=getattr(np, "float128", np.float64)
)

problem_primal.log_condition_number(True)
problem_primal_dual.log_condition_number(True)

H, g, c, c_rhs, K = full_qp(problem_primal)

col_titles = [
    "Iteration",
    "Constr error",
    "Cost",
    "Proj. grad",
    "Min cond.",
    "Index",
    "Max cond.",
    "Index",
    "Mean (×) cond.",
]
formats = ["", ".2e", ".2e", ".2e", ".2e", "", ".2e", "", ".2e"]
col_widths = [max(10, len(title) + 1) for title in col_titles]
ncols = len(col_widths)

fmt_str = "{:>{w[0]}{f[0]}}"
for i in map(str, range(1, ncols)):
    fmt_str = fmt_str + " | {:>{w[" + i + "]}{f[" + i + "]}}"


def print_titles():
    print(fmt_str.format(*col_titles, w=col_widths, f=ncols * [""]), sep="")


def print_optimization_info(iter_count, problem):
    print(
        fmt_str.format(
            iter_count,
            norm(problem.c_violation()),
            (problem.costs() - COST_MIN).sum(),
            norm(K.T @ (H @ traj_flat + g)),
            problem.debug["cond"].min(),
            problem.debug["cond"].argmin(),
            problem.debug["cond"].max(),
            problem.debug["cond"].argmax(),
            (problem.debug["cond"].prod()) ** (1 / problem.traj.T),
            w=col_widths,
            f=formats,
        )
    )


problem_primal_dual.mu[:] = np.inf
COST_MIN = problem_primal_dual.costs()
for _ in range(10):
    problem_primal_dual.one_iter()
    cost = problem_primal_dual.costs()
    if cost.sum() < COST_MIN.sum():
        COST_MIN = cost

counter = 0

start = int(sys.argv[1]) if len(sys.argv) > 1 else 2
end = int(sys.argv[2]) if len(sys.argv) > 2 else 10
for mu in map(lambda x: 10 ** x, range(start, end)):
    for prob in [problem_primal_dual]:
        prob.mu[:] = mu
        print("-" * os.get_terminal_size(0)[0])
        print(f"μ = {mu:.0e}, Method: {type(prob).__name__}")
        prob.reset()
        min_constraint_err = np.inf
        print_titles()
        for i in itertools.count():
            traj_flat = prob.traj.data.flatten()[: g.size]
            print_optimization_info(i, prob)
            prob.one_iter()

            constraint_err = norm(prob.c_violation())
            min_constraint_err = min(min_constraint_err, constraint_err)
            counter += 1
            if constraint_err == min_constraint_err:
                counter = 0

prob = problem_primal_dual
mu = float(sys.argv[3])
try:
    prob.mu[:] = np.inf
    print("-" * os.get_terminal_size(0)[0])
    print(f"μ = {mu:.0e}, Method: {type(prob).__name__.replace('_', ' ')}")
    prob.reset()
    print_titles()
    # for i in range(20):
    for i in itertools.count():
        if i == 10:
            prob.mu[:] = mu
        if i >= 30:
            type(prob).pdb = True
            # __import__('ipdb').set_trace()
        traj_flat = prob.traj.data.flatten()[: g.size]
        print_optimization_info(i, prob)
        prob.one_iter()
except KeyboardInterrupt:
    pass
