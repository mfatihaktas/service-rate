import pytest

from typing import Tuple

from src.storage_overlap import design

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # (4, 4, 2),
        (12, 12, 3),
    ],
)
def k_n_d(request) -> Tuple[int, int, int]:
    return request.param


def test_ClusteringDesign(k_n_d: Tuple[int, int, int]):
    k, n, d = k_n_d
    clustering_design = design.ClusteringDesign(k=k, n=n, d=d, use_cvxpy=False)

    log(INFO, "",
        clustering_design=clustering_design,
        obj_id_to_node_id_set_map=clustering_design.obj_id_to_node_id_set_map,
    )


def test_CyclicDesign(k_n_d: Tuple[int, int, int]):
    k, n, d = k_n_d

    cyclic_design = design.CyclicDesign(k=k, n=n, d=d, shift_size=1, use_cvxpy=False)

    log(INFO, "",
        cyclic_design=cyclic_design,
        obj_id_to_node_id_set_map=cyclic_design.obj_id_to_node_id_set_map,
    )


def test_TwoXORDesign():
    # k, n = 7, 7
    # k = 106
    k = 121
    # k = 124
    n = k
    xor_design = design.TwoXORDesign(k=k, n=n, d=3)

    log(INFO, "",
        xor_design=xor_design,
    )
