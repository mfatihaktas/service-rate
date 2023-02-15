import pytest

from typing import Tuple

from src.storage_overlap import storage_design

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


def test_get_obj_id_to_node_id_set_map_w_replicas_clustering_design(k_n_d: Tuple[int, int, int]):
    k, n, d = k_n_d
    log(DEBUG, "", k=k, n=n, d=d)

    obj_id_to_node_id_set_map = storage_design.get_obj_id_to_node_id_set_map_w_replicas_clustering_design(
        k=k, n=n, d=d,
    )

    log(INFO, "",
        obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
    )


def test_get_obj_id_to_node_id_set_map_w_replicas_cyclic_design(k_n_d: Tuple[int, int, int]):
    k, n, d = k_n_d
    log(DEBUG, "", k=k, n=n, d=d)

    obj_id_to_node_id_set_map = storage_design.get_obj_id_to_node_id_set_map_w_replicas_cyclic_design(
        k=k, n=n, d=d,
    )

    log(INFO, "",
        obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
    )
