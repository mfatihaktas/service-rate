from src.model import demand

from src.utils.debug import *


def test_gen_demand_vector_w_zipf_law():
    for demand_vector in demand.sample_demand_vectors_w_zipf_law(
        num_obj=3,
        num_popular_obj=2,
        cum_demand=2,
        zipf_tail_index=0,
        num_samples=5,
    ):
        log(DEBUG, "", demand_vector=demand_vector)
