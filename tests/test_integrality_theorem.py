import cvxpy
import numpy
import pytest

from src.utils.debug import *


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

    x = cvxpy.Variable(shape=(k * n, 1), name="x", integer=True)

    x_0 = cvxpy.vstack([x[i] for i in [0, 1, 2]])
    x_1 = cvxpy.vstack([x[i] for i in [3, 4, 5]])
    # x_2 = cvxpy.vstack([x[i] for i in [6, 7, 8]])

    obj = cvxpy.Minimize(cvxpy.sum(x))
    # obj = cvxpy.Minimize(cvxpy.norm(x, 1))
    # obj = cvxpy.Minimize(cvxpy.min(x) - cvxpy.max(x))
    constraints = [
        obj_req_left_matrix @ x >= obj_req_right_matrix,
        x_0.T @ x_1 >= 2,  # cvxpy.error.DCPError: Problem does not follow DCP rules.
        x >= 0,
        x <= 1,
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

    log(
        DEBUG,
        "",
        prob_status=prob.status,
        prob_value=prob.value,
        x=x.value,
        x_row_0=x.value[0],
        sum_x_row_0=numpy.sum(x.value[0, :]),
        sum_x_row_1=numpy.sum(x.value[1, :]),
        sum_x_row_2=numpy.sum(x.value[2, :]),
    )


@pytest.fixture(
    scope="session",
    params=[
        # dict(
        #     # Goal:
        #     # [a, b]
        #     # [a, b]
        #     # [a, c]
        #     k=3,
        #     n=7,
        #     obj_ids_to_min_span_size_map={
        #         0: 3,
        #         1: 2,
        #         2: 1,
        #         (0, 1): 3,
        #         (0, 2): 3,
        #         (1, 2): 3,
        #         (0, 1, 2): 3,
        #     }
        # ),

        # dict(
        #     # Goal:
        #     # [a]
        #     # [a, b]
        #     # [a, c]
        #     # [b, c]
        #     k=3,
        #     n=7,
        #     obj_ids_to_min_span_size_map={
        #         0: 2,
        #         1: 2,
        #         2: 2,
        #         (0, 1): 3,
        #         (0, 2): 4,
        #         (1, 2): 3,
        #         (0, 1, 2): 4,
        #     }
        # ),

        # dict(
        #     # Goal:
        #     # [a, b]
        #     # [a, b]
        #     # [a, c]
        #     # [a, c]
        #     # [d]
        #     k=4,
        #     n=7,
        #     obj_ids_to_min_span_size_map={
        #         0: 4,
        #         1: 2,
        #         2: 2,
        #         3: 1,
        #         (0, 1): 4,
        #         (0, 2): 4,
        #         (0, 3): 5,
        #         (1, 2): 4,
        #         (1, 3): 3,
        #         (2, 3): 3,
        #         (0, 1, 2): 4,
        #         (0, 1, 3): 5,
        #         (0, 2, 3): 5,
        #         (1, 2, 3): 5,
        #     },
        # ),

        dict(
            k=4,
            n=16,
            obj_ids_to_min_span_size_map={
                0: 1,
                1: 3,
                2: 4,
                3: 2,
                (0, 1): 4,
                (0, 2): 5,
                (0, 3): 3,
                (1, 2): 7,
                (1, 3): 5,
                (2, 3): 6,
                (0, 1, 2): 8,
                (0, 1, 3): 6,
                (0, 2, 3): 7,
                (1, 2, 3): 9,
                (0, 1, 2, 3): 10,
            }
        ),
    ],
)
def storage_info_w_span_sizes(request) -> dict:
    return request.param


def test_w_integer_programming_refined(storage_info_w_span_sizes: dict):
    k = storage_info_w_span_sizes["k"]
    n = storage_info_w_span_sizes["n"]
    obj_ids_to_min_span_size_map = storage_info_w_span_sizes["obj_ids_to_min_span_size_map"]

    x = cvxpy.Variable(shape=(k, n), name="x", boolean=True)
    constraint_list = []

    # Span constraints
    z_list = []
    for counter, (obj_ids, min_span_size) in enumerate(obj_ids_to_min_span_size_map.items()):
        log(DEBUG, f">> obj_ids= {obj_ids}", min_span_size=min_span_size)

        if isinstance(obj_ids, int):
            constraint_list.append(cvxpy.sum(x[obj_ids, :]) >= min_span_size)
            continue

        z = cvxpy.Variable(shape=(n, 1), name=f"z_{counter}", boolean=True)
        z_list.append(z)

        for i in obj_ids:
            constraint_list.append(1 - cvxpy.reshape(x[i, :], shape=(n, 1)) >= z)

        num_objs = len(obj_ids)
        # x_i_in_columns = cvxpy.vstack([x[i, :] for i in obj_ids]).T
        # sum_x_i = x_i_in_columns @ numpy.ones((num_objs, 1))
        # log(DEBUG, "", x_i_in_columns=x_i_in_columns, sum_x_i=sum_x_i)
        # constraint_list.append(sum_x_i - len(obj_ids) + 1 <= z)

        # obj_choice_union_size = sum(obj_ids_to_min_span_size_map[i] for i in obj_ids)
        # constraint_list.append(cvxpy.sum(z) <= obj_choice_union_size - min_span_size)
        # log(DEBUG, "", obj_choice_union_size=obj_choice_union_size, min_span_size=min_span_size)

        one_minus_x_i_in_columns = cvxpy.vstack([1 - x[i, :] for i in obj_ids]).T
        sum_one_minus_x_i = one_minus_x_i_in_columns @ numpy.ones((num_objs, 1))
        log(DEBUG, "", one_minus_x_i_in_columns=one_minus_x_i_in_columns, sum_one_minus_x_i=sum_one_minus_x_i)
        constraint_list.append(sum_one_minus_x_i - len(obj_ids) + 1 <= z)

        constraint_list.append(cvxpy.sum(z) <= n - min_span_size)
        log(DEBUG, "", min_span_size=min_span_size)

    C = numpy.array([[i + 1] for i in range(n)])
    log(DEBUG, "", C=C, constraint_list=constraint_list)
    obj = cvxpy.Minimize(cvxpy.sum(x @ C))

    prob = cvxpy.Problem(obj, constraint_list)
    prob.solve(solver="SCIP")

    log(DEBUG, "",
        prob_status=prob.status,
        prob_value=prob.value,
        x=x.value,
        z_list=[z.value for z in z_list],
    )


def test_integer_programming_w_dot_product_constraint():
    k = 10
    x = cvxpy.Variable(shape=k, name="x", boolean=True)
    y = cvxpy.Variable(shape=k, name="y", boolean=True)
    constraint_list = []

    # Span of x and y
    constraint_list.append(cvxpy.sum(x) >= 4)
    constraint_list.append(cvxpy.sum(y) >= 4)

    # <x, y> <= 2
    z = cvxpy.Variable(shape=k, name="z", boolean=True)

    constraint_list.append(x >= z)
    constraint_list.append(y >= z)

    constraint_list.append(x + y - 1 <= z)

    constraint_list.append(cvxpy.sum(z) <= 2)

    # Objective
    C = numpy.hstack([numpy.array([[i + 1] for i in range(k)]) for _ in range(2)])
    # log(DEBUG, "", C=C, constraint_list=constraint_list)
    obj = cvxpy.Minimize(cvxpy.sum(x @ C) + cvxpy.sum(y @ C))

    prob = cvxpy.Problem(obj, constraint_list)
    prob.solve(solver="SCIP")

    log(DEBUG, "",
        prob_status=prob.status,
        prob_value=prob.value,
        x=x.value,
        y=y.value,
        z=z.value,
    )


def test_integer_programming_w_or_constraint():
    k = 10
    x = cvxpy.Variable(shape=k, name="x", boolean=True)
    y = cvxpy.Variable(shape=k, name="y", boolean=True)
    constraint_list = []

    # Span of x and y
    constraint_list.append(cvxpy.sum(x) >= 3)
    constraint_list.append(cvxpy.sum(y) >= 2)

    # |x or y| >= min_span_size
    min_span_size = 3
    z = cvxpy.Variable(shape=k, name="z", boolean=True)

    constraint_list.append(1 - x >= z)
    constraint_list.append(1 - y >= z)

    constraint_list.append(1 - x + 1 - y - 1 <= z)

    constraint_list.append(cvxpy.sum(z) <= k - min_span_size)

    # Objective
    C = numpy.hstack([numpy.array([[i + 1] for i in range(k)]) for _ in range(2)])
    # log(DEBUG, "", C=C, constraint_list=constraint_list)
    obj = cvxpy.Minimize(cvxpy.sum(x @ C) + cvxpy.sum(y @ C))

    prob = cvxpy.Problem(obj, constraint_list)
    prob.solve(solver="SCIP")

    log(DEBUG, "",
        prob_status=prob.status,
        prob_value=prob.value,
        x=x.value,
        y=y.value,
        z=z.value,
    )


def test_w_integer_programming_2(storage_info_w_span_sizes: dict):
    k = storage_info_w_span_sizes["k"]
    n = storage_info_w_span_sizes["n"]
    obj_ids_to_min_span_size_map = storage_info_w_span_sizes[
        "obj_ids_to_min_span_size_map"
    ]

    x = cvxpy.Variable(shape=(k, n), name="x", integer=True)
    # x = cvxpy.Variable(shape=(k, n), name="x")
    constraint_list = []

    # Span constraints
    # for obj_ids, min_span_size in obj_ids_to_min_span_size_map.items():
    #     v_list = [x[i, :] for i in obj_ids]
    #     constraint_list.append(cvxpy.sum(cvxpy.vstack(v_list)) >= min_span_size)

    # constraint_list.append(x[0, :].T @ x[1, :] <= 2)
    # z = cvxpy.Variable(shape=(n,), name="z")  # , integer=True
    # constraint_list.append(x[0, :] >= z)
    # constraint_list.append(x[1, :] >= z)
    # constraint_list.append(x[0, :] + x[1, :] - 1 <= z)
    # constraint_list.append(cvxpy.sum(z) >= 4)

    # Span constraints
    for obj_ids, min_span_size in obj_ids_to_min_span_size_map.items():
        log(DEBUG, f">> obj_ids= {obj_ids}", min_span_size=min_span_size)

        if isinstance(obj_ids, int):
            constraint_list.append(cvxpy.sum(x[obj_ids, :]) >= min_span_size)
            continue

        z = cvxpy.Variable(shape=(n, 1), name="z")

        num_objs = len(obj_ids)
        x_i_in_columns = cvxpy.vstack([x[i, :] for i in obj_ids]).T
        sum_x_i = x_i_in_columns @ numpy.ones((num_objs, 1))
        log(DEBUG, "", x_i_in_columns=x_i_in_columns, sum_x_i=sum_x_i)
        constraint_list.append(sum_x_i - len(obj_ids) + 1 <= z)

        obj_choice_union_size = sum(obj_ids_to_min_span_size_map[i] for i in obj_ids)
        constraint_list.append(cvxpy.sum(z) <= obj_choice_union_size - min_span_size)
        log(DEBUG, "", obj_choice_union_size=obj_choice_union_size, min_span_size=min_span_size)

    # constraint_list.append(cvxpy.quad_form(x[0, :], numpy.ones((n, n))) <= 2)
    # constraint_list.append(cvxpy.pnorm(x, 2) <= 2)

    # Node constraints
    # for i in range(n):
    #     constraint_list.append(cvxpy.sum(x[:, i]) <= 1)

    # Range constraints
    constraint_list.extend([x >= 0, x <= 1])

    # C = numpy.array([[i + 1 for i in range(n)] for _ in range(k)])
    # obj = cvxpy.Minimize(cvxpy.sum(cvxpy.multiply(C, x)))
    C = numpy.hstack([numpy.array([[i + 1] for i in range(n)]) for _ in range(k)])
    log(DEBUG, "", C=C, constraint_list=constraint_list)
    obj = cvxpy.Minimize(cvxpy.sum(x @ C))

    # obj = cvxpy.Minimize(cvxpy.norm(x, 1))
    # obj = cvxpy.Minimize(cvxpy.sum_squares(x))
    # obj = cvxpy.Minimize(cvxpy.tv(x))
    # obj = cvxpy.Minimize(cvxpy.tv(cvxpy.cumsum(x, axis=1)))

    prob = cvxpy.Problem(obj, constraint_list)
    # prob.solve(solver="GLPK_MI")
    # prob.solve(solver="ECOS_BB")
    prob.solve(solver="SCIP")
    # prob.solve()

    log(
        DEBUG,
        "",
        prob_status=prob.status,
        prob_value=prob.value,
        x=x.value,
    )


def test_mixed_integer_quadratic_program():
    # Generate a random problem
    numpy.random.seed(0)
    m, n = 40, 25

    A = numpy.random.rand(m, n)
    b = numpy.random.randn(m)

    x = cvxpy.Variable(n, integer=True)
    objective = cvxpy.Minimize(cvxpy.sum_squares(A @ x - b))
    prob = cvxpy.Problem(objective)
    # prob.solve()
    # prob.solve(solver="ECOS_BB")
    prob.solve(solver="SCIP")

    log(
        DEBUG,
        "",
        prob_status=prob.status,
        prob_value=prob.value,
        x=x.value,
    )


def test_StorageOptimizerReplicationAndMDS_wSingleObjPerNode_preliminary(demand_vector_list: list[float]):
    k = 3
    n_a = cvxpy.Variable(name="n_a", integer=True)
    n_b = cvxpy.Variable(name="n_b", integer=True)
    n_c = cvxpy.Variable(name="n_c", integer=True)

    n_mds = cvxpy.Variable(name="n_mds", integer=True)
    constraint_list = []

    # Constraints for `a`
    m_a = cvxpy.Variable(name="m_a", integer=True)
    constraint_list.append(cvxpy.maximum(m_a - n_a, 0) + cvxpy.maximum(m_a - n_b, 0) + cvxpy.maximum(m_a - n_c, 0) + m_a <= n_mds)

    constraint_list.append(n_a + m_a + (n_mds - m_a) / k >= 2)

    # Constraints for `b`
    m_b = cvxpy.Variable(name="m_b", integer=True)
    constraint_list.append(cvxpy.maximum(m_b - n_a, 0) + cvxpy.maximum(m_b - n_b, 0) + cvxpy.maximum(m_b - n_c, 0) + m_b <= n_mds)

    constraint_list.append(n_b + m_b + (n_mds - m_b) / k >= 2)

    # Constraints for `c`
    m_c = cvxpy.Variable(name="m_c", integer=True)
    constraint_list.append(cvxpy.maximum(m_c - n_a, 0) + cvxpy.maximum(m_c - n_b, 0) + cvxpy.maximum(m_c - n_c, 0) + m_c <= n_mds)

    constraint_list.append(n_c + m_c + (n_mds - m_c) / k >= 3)

    # Constraints for `a, b`
    m_ab = cvxpy.Variable(name="m_ab", integer=True)
    constraint_list.append(cvxpy.maximum(m_ab - n_c, 0) + 2 * m_ab <= n_mds)

    constraint_list.append(n_a + n_b + m_ab + (n_mds - m_ab) / k >= 4)

    # Range constraints
    # constraint_list.extend([n_a >= 0, n_b >= 0, n_c >= 0, n_mds >= 0])

    obj = cvxpy.Minimize(n_a + n_b + n_c + n_mds)

    prob = cvxpy.Problem(obj, constraint_list)
    prob.solve(solver="SCIP")

    log(DEBUG, "",
        prob_value=prob.value, n_a=n_a.value, n_b=n_b.value, n_c=n_c.value, n_mds=n_mds.value,
        m_a=m_a.value, m_b=m_b.value, m_c=m_c.value, m_ab=m_ab.value
    )
