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

    x_0 = cvxpy.vstack([x[i] for i in [0, 1, 2]])
    x_1 = cvxpy.vstack([x[i] for i in [3, 4, 5]])
    x_2 = cvxpy.vstack([x[i] for i in [6, 7, 8]])

    obj = cvxpy.Minimize(cvxpy.sum(x))
    # obj = cvxpy.Minimize(cvxpy.norm(x, 1))
    # obj = cvxpy.Minimize(cvxpy.min(x) - cvxpy.max(x))
    constraints = [
        obj_req_left_matrix @ x >= obj_req_right_matrix,
        x_0.T @ x_1 >= 2,  # cvxpy.error.DCPError: Problem does not follow DCP rules.
        x >= 0, x <= 1,
    ]

    prob = cvxpy.Problem(obj, constraints)
    # opt_value = service_rate_utils.solve_prob(prob)
    # prob.solve(solver="GLPK")
    prob.solve()
    # check(prob.status == cvxpy.OPTIMAL, "")

    log(DEBUG, "", prob_value=prob.value, x_value=x.value)


def test_w_l1_norm():
    k, n = 3, 7
    demand_list = [2, 2, 1]

    x = cvxpy.Variable(shape=(k, n), name="x")
    constraint_list = []

    # Demand constraints
    for i in range(k):
        constraint_list.append(cvxpy.sum(x[i, :]) == demand_list[i])

    # Node constraints
    # for i in range(n):
    #     constraint_list.append(cvxpy.sum(x[:i]) <= 1)

    # Range constraints
    constraint_list.extend([x >= 0, x <= 1])

    # obj = cvxpy.Minimize(cvxpy.sum(x))
    obj = cvxpy.Minimize(cvxpy.norm(x, 1))

    prob = cvxpy.Problem(obj, constraint_list)
    prob.solve()

    log(DEBUG, "",
        prob_status=prob.status,
        prob_value=prob.value,
        x=x.value,
        x_row_0=x.value[0],
        sum_x_row_0=numpy.sum(x.value[0, :]),
        sum_x_row_1=numpy.sum(x.value[1, :]),
        sum_x_row_2=numpy.sum(x.value[2, :]),
    )
