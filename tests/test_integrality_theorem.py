import cvxpy
import numpy
# import pytest

from src import service_rate_utils
from src.debug_utils import *


def test_integrality_theorem():
    # k, n = 2, 2
    M = numpy.zeros((4, 4))
    b = numpy.zeros((4, 1))

    # obj-0, 1
    M[0, :] = [1, 1, 0, 0]
    b[0] = 1
    M[1, :] = [0, 0, 1, 1]
    b[1] = 1
    # node-0, 1
    M[2, :] = [1, 0, 1, 0]
    b[2] = 1
    M[3, :] = [0, 1, 0, 1]
    b[3] = 1

    x = cvxpy.Variable(shape=(4, 1), name="x", integer=True)

    obj = cvxpy.Minimize(cvxpy.sum(x))
    # obj = cvxpy.Minimize(cvxpy.norm(x, 1))
    # obj = cvxpy.Minimize(cvxpy.min(x) - cvxpy.max(x))
    constraints = [M @ x == b, x >= 0]

    prob = cvxpy.Problem(obj, constraints)
    # opt_value = service_rate_utils.solve_prob(prob)
    # prob.solve(solver="GLPK")
    prob.solve()
    check(prob.status == cvxpy.OPTIMAL, "")

    log(DEBUG, "", prob_value=prob.value, x_value=x.value)
