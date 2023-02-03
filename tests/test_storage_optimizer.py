import pytest

from src.opt_storage import storage_optimizer as storage_optimizer_module
from src.utils.debug import *


@pytest.fixture(
    scope="session",
    params=[
        [1, 2],
        # [1, 3, 4, 2],
    ],
)
def demand_list(request) -> list[float]:
    return request.param


def test_StorageOptimizerForDemandVector(demand_list: list[float]):
    storage_optimizer = storage_optimizer_module.StorageOptimizerForDemandVector(demand_list=demand_list)
    obj_id_to_node_id_set_map = storage_optimizer.get_obj_id_to_node_id_set_map()
    log(DEBUG, "", obj_id_to_node_id_set_map=obj_id_to_node_id_set_map)
