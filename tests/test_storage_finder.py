from src import (
    storage_finder as storage_finder_module,
)
from src.utils import demand as demand_utils
from src.utils.debug import *


def test_StorageFinder():
    k = 4
    cum_demand = 10
    demand_list = demand_utils.sample_demand_on_simplex(k=k, cum_demand=cum_demand)
    log(DEBUG, "", demand_list=demand_list)

    storage_finder_module.StorageFinder(demand_list=demand_list)
