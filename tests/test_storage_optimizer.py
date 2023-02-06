import pytest

from src.opt_storage import storage_optimizer as storage_optimizer_module
from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        # [[1, 2]],
        # [[1, 3, 4, 2]],
        [[3.5, 1, 0.2, 0.1]],
        # [
        #     [4, 1, 1, 1],
        #     # [1, 1, 3, 4],
        # ],
    ],
)
def demand_vector_list(request) -> list[float]:
    return request.param


def test_StorageOptimizerForDemandVectors(demand_vector_list: list[float]):
    storage_optimizer = storage_optimizer_module.StorageOptimizerForDemandVectors(demand_vector_list=demand_vector_list)
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(DEBUG, "", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)
