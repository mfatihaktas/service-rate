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
    clustering_design = design_w_stripe.ClusteringDesignWithStripe(k=k, n=n, d=s*d, s=s, use_cvxpy=False)

    log(INFO, "",
        clustering_design=clustering_design,
        obj_id_to_node_id_set_map=clustering_design.obj_id_to_node_id_set_map,
    )

    assert clustering_design.is_demand_vector_covered(demand_vector=[d] + (k - 1) * [0])
