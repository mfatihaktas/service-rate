import collections
import functools
import numpy
import random

from typing import Tuple

from src.allocation_w_complexes import sim as allocation_w_complexes_sim
from src.model import demand
from src.storage_overlap import design

from src.utils.debug import *


def sim_object_span(
    storage_design: design.StorageDesign,
    m: int,
) -> float:
    # node_id_list = list(range(storage_design.n))
    # obj_id_to_node_id_set_map = {
    #     obj_id: set(random.sample(node_id_list, storage_design.d))
    #     for obj_id in range(storage_design.k)
    # }

    node_id_set = set()
    obj_id_list = random.sample(list(range(storage_design.k)), m)
    for obj_id in obj_id_list:
        node_id_set_ = storage_design.obj_id_to_node_id_set_map[obj_id]

        # node_id_set_ = obj_id_to_node_id_set_map[obj_id]
        # log(DEBUG, "", obj_id=obj_id, node_id_set_=node_id_set_)

        # node_id_set_ = random.sample(list(range(storage_design.n)), storage_design.d)
        for node_id in node_id_set_:
            node_id_set.add(node_id)

    return len(node_id_set)


def sim_object_span_to_prob_map(
    storage_design: design.StorageDesign,
    m: int,
    num_sample: int,
) -> dict[int, float]:
    log(DEBUG, "Started",
        storage_design=storage_design,
        m=m,
        num_sample=num_sample,
    )

    object_span_to_counter_map = collections.defaultdict(int)
    for _ in range(num_sample):
        storage_design.reset()
        object_span = sim_object_span(storage_design=storage_design, m=m)
        object_span_to_counter_map[object_span] += 1

    object_span_to_prob_map = {
        object_span: counter / num_sample
        for object_span, counter in object_span_to_counter_map.items()
    }
    log(DEBUG, "Done",
        storage_design=storage_design,
        m=m,
        object_span_to_prob_map=object_span_to_prob_map,
    )

    return object_span_to_prob_map


def sim_frac_of_demand_vectors_covered(
    storage_design: design.StorageDesign,
    num_popular_obj: int,
    cum_demand: float,
    zipf_tail_index: float,
    num_sample: int,
    num_sim_run: int = 1,
    combination_size_for_is_demand_vector_covered: int = None,
) -> list[float]:
    log(DEBUG, "Started",
        num_popular_obj=num_popular_obj,
        cum_demand=cum_demand,
        zipf_tail_index=zipf_tail_index,
        num_sample=num_sample,
        num_sim_run=num_sim_run,
        combination_size_for_is_demand_vector_covered=combination_size_for_is_demand_vector_covered,
    )

    frac_of_demand_vectors_covered_list = []

    for sim_id in range(num_sim_run):
        log(DEBUG, f"> sim_id= {sim_id}")

        num_covered = 0
        for demand_vector in demand.sample_demand_vectors_w_zipf_law(
            num_obj=storage_design.k,
            num_popular_obj=num_popular_obj,
            cum_demand=cum_demand,
            zipf_tail_index=zipf_tail_index,
            num_sample=num_sample,
        ):
            storage_design.reset()

            if combination_size_for_is_demand_vector_covered is not None:
                if storage_design.is_demand_vector_covered_for_given_combination_size(
                        demand_vector=demand_vector,
                        combination_size=combination_size_for_is_demand_vector_covered,
                ):
                    num_covered += 1

            elif storage_design.is_demand_vector_covered(demand_vector=demand_vector):
                num_covered += 1

        frac_of_demand_vectors_covered = num_covered / num_sample
        frac_of_demand_vectors_covered_list.append(frac_of_demand_vectors_covered)

    log(DEBUG, "Done")
    return frac_of_demand_vectors_covered_list


def sim_frac_of_demand_vectors_covered_lower_and_upper_bound(
    storage_design: design.StorageDesign,
    num_popular_obj: int,
    cum_demand: float,
    zipf_tail_index: float,
    num_sample: int,
    num_sim_run: int = 1,
) -> Tuple[float, float]:
    log(DEBUG, "Started",
        num_popular_obj=num_popular_obj,
        cum_demand=cum_demand,
        zipf_tail_index=zipf_tail_index,
        num_sample=num_sample,
        num_sim_run=num_sim_run,
    )

    combination_size_to_frac_demand_vectors_covered_list_map = collections.defaultdict(list)
    for sim_id in range(num_sim_run):
        log(DEBUG, f"> sim_id= {sim_id}")

        combination_size_to_num_covered_map = collections.defaultdict(int)
        for demand_vector in demand.sample_demand_vectors_w_zipf_law(
                num_obj=storage_design.k,
                num_popular_obj=num_popular_obj,
                cum_demand=cum_demand,
                zipf_tail_index=zipf_tail_index,
                num_sample=num_sample,
        ):
            for combination_size in range(1, num_popular_obj + 1):
                if storage_design.is_demand_vector_covered_for_given_combination_size(
                    demand_vector=demand_vector,
                    combination_size=combination_size,
                ):
                    combination_size_to_num_covered_map[combination_size] += 1

        for combination_size, num_covered in combination_size_to_num_covered_map.items():
            frac_of_demand_vectors_covered = num_covered / num_sample
            combination_size_to_frac_demand_vectors_covered_list_map[combination_size].append(frac_of_demand_vectors_covered)

    log(DEBUG, "Done",
        combination_size_to_frac_demand_vectors_covered_list_map=combination_size_to_frac_demand_vectors_covered_list_map,
    )

    E_frac_demand_vectors_covered_list = [
        numpy.mean(frac_demand_vectors_covered_list)
        for frac_demand_vectors_covered_list in combination_size_to_frac_demand_vectors_covered_list_map.values()
    ]
    lower_bound = functools.reduce(lambda x, y: x * y, E_frac_demand_vectors_covered_list)
    # lower_bound = max(0, 1 - sum(1 - p for p in E_frac_demand_vectors_covered_list))

    upper_bound = min(E_frac_demand_vectors_covered_list)

    return lower_bound, upper_bound
