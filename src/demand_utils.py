import random

from src.debug_utils import *


def sample_demand_on_simplex(k: int, cum_demand: float) -> list[float]:
    uniform_sample_list = [0] + sorted([random.uniform(0, 1) for _ in range(k - 1)]) + [1]
    demand_fraction_list = [uniform_sample_list[i] - uniform_sample_list[i - 1] for i in range(1, k + 1)]
    # log(DEBUG, "", uniform_sample_list=uniform_sample_list, demand_fraction_list=demand_fraction_list)

    demand_list = [demand_fraction * cum_demand for demand_fraction in demand_fraction_list]
    return sorted(demand_list, reverse=True)
