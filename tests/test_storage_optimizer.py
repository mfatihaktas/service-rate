import pytest

from src.opt_storage import storage_optimizer as storage_optimizer_module
from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # [[1, 2]],
        # [[1, 3, 4, 2]],
        # [[3.5, 1, 0.2, 0.1]],

        # [
        #     [4, 1, 0.3, 0.2],
        #     [0.2, 0.3, 1, 4],
        # ],

        [
            [5, 0.3, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.3, 5],
        ],
    ],
)
def demand_vector_list(request) -> list[list[float]]:
    return request.param


def test_StorageOptimizerReplication(demand_vector_list: list[float]):
    storage_optimizer = storage_optimizer_module.StorageOptimizerReplication(demand_vector_list=demand_vector_list)
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(DEBUG, "", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)


def test_StorageOptimizerReplicationAndMDS(demand_vector_list: list[float]):
    storage_optimizer = storage_optimizer_module.StorageOptimizerReplicationAndMDS(demand_vector_list=demand_vector_list)
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(DEBUG, "", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)


def test_StorageOptimizerReplicationAndMDS_wSingleObjPerNode(demand_vector_list: list[float]):
    import cvxpy

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
    constraint_list.extend([n_a >= 1, n_b >= 1, n_c >= 1, n_mds >= 0])

    obj = cvxpy.Minimize(n_a + n_b + n_c + n_mds)

    prob = cvxpy.Problem(obj, constraint_list)
    prob.solve(solver="SCIP")

    log(DEBUG, "",
        prob_value=prob.value, n_a=n_a.value, n_b=n_b.value, n_c=n_c.value, n_mds=n_mds.value,
        m_a=m_a.value, m_b=m_b.value, m_c=m_c.value, m_ab=m_ab.value
    )
