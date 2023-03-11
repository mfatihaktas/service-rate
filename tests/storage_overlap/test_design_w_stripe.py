import pytest

from typing import Tuple

from src.storage_overlap import design_w_stripe

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        (4, 6, 3),
        # (4, 4, 2),
        # (12, 12, 3),
    ],
)
def k_n_d(request) -> Tuple[int, int, int]:
    return request.param


def test_ClusteringDesignWithStripe(
    k_n_d: Tuple[int, int, int],
):
    k, n, d = k_n_d
    s = 2
    clustering_design = design_w_stripe.ClusteringDesignWithStripe(k=k, n=n, d=(s * d), s=s, use_cvxpy=False)

    log(INFO, "",
        clustering_design=clustering_design,
        obj_id_to_node_id_set_map=clustering_design.obj_id_to_node_id_set_map,
    )

    assert clustering_design.is_demand_vector_covered(demand_vector=[d] + (k - 1) * [0])


def test_CyclicDesignWithStripe(
    k_n_d: Tuple[int, int, int],
):
    k, n, d = k_n_d
    s = 2
    cyclic_design = design_w_stripe.CyclicDesignWithStripe(k=k, n=n, d=(s * d), s=s, use_cvxpy=False)

    log(INFO, "",
        cyclic_design=cyclic_design,
        obj_id_to_node_id_set_map=cyclic_design.obj_id_to_node_id_set_map,
    )

    assert cyclic_design.is_demand_vector_covered(demand_vector=[d] + (k - 1) * [0])


def test_RandomBlockDesignWithStripe(
    k_n_d: Tuple[int, int, int],
):
    k, n, d = k_n_d
    s = d
    random_block_design = design_w_stripe.RandomBlockDesignWithStripe(k=k, n=n, d=d, s=s, use_cvxpy=False)

    log(INFO, "",
        random_block_design=random_block_design,
        obj_id_to_node_id_set_map=random_block_design.obj_id_to_node_id_set_map,
    )

    assert random_block_design.is_demand_vector_covered(demand_vector=[d] + (k - 1) * [0])
