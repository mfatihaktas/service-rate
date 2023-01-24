import cvxpy
import numpy
# import pytest

from src import service_rate_utils
from src.debug_utils import *


def test_w_integer_programming():
    k, n = 3, 3
    # Obj choice constraints
    obj_req_left_list, obj_req_right_list = [], []
    obj_req_left_list.append([1, 1, 1, 0, 0, 0, 0, 0, 0])
    obj_req_right_list.append([2])
    obj_req_left_list.append([0, 0, 0, 1, 1, 1, 0, 0, 0])
    obj_req_right_list.append([2])
    obj_req_left_list.append([0, 0, 0, 0, 0, 0, 1, 1, 1])
    obj_req_right_list.append([1])

    obj_req_left_matrix = numpy.array(obj_req_left_list)
    obj_req_right_matrix = numpy.array(obj_req_right_list)

    # Node constraints
    # node_req_left_list, node_req_right_list = [], []
    # node_req_left_list.append([1, 0, 0, 1, 0, 0, 1, 0, 0])
    # node_req_right_list.append([1])

    x = cvxpy.Variable(shape=(k*n, 1), name="x", integer=True)

    obj = cvxpy.Minimize(cvxpy.sum(x))
    # obj = cvxpy.Minimize(cvxpy.norm(x, 1))
    # obj = cvxpy.Minimize(cvxpy.min(x) - cvxpy.max(x))
    constraints = [obj_req_left_matrix @ x >= obj_req_right_matrix, x >= 0, x <= 1]

    prob = cvxpy.Problem(obj, constraints)
    # opt_value = service_rate_utils.solve_prob(prob)
    prob.solve(solver="GLPK")
    prob.solve()
    # check(prob.status == cvxpy.OPTIMAL, "")

    log(DEBUG, "", prob_value=prob.value, x_value=x.value)
