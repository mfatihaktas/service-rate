import pytest

from src.storage_opt import single_obj_per_node as single_obj_per_node_module
from src.model import demand as demand_module

from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        {
            "num_objects": 3,
            "demand_ordered_for_most_popular_objs": [9, 0],
        }
    ],
)
def demand_vector_list(request) -> list[list[float]]:
    input_map = request.param

    return demand_module.get_demand_vectors(
        num_objs=input_map["num_objects"],
        demand_ordered_for_most_popular_objs=input_map["demand_ordered_for_most_popular_objs"],
    )


def test_StorageOptimizerReplicationAndXOR_wSingleObjPerNode(demand_vector_list: list[float]):
    log(DEBUG, "", demand_vector_list=demand_vector_list)

    storage_optimizer = single_obj_per_node_module.StorageOptimizerReplicationAndXOR_wSingleObjPerNode(
        demand_vector_list=demand_vector_list,
    )

    obj_to_num_copies_map = storage_optimizer.access_graph.get_obj_to_num_copies_map()
    log(INFO, "",
        obj_to_num_copies_map=obj_to_num_copies_map,
    )
