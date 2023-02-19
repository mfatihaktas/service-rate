import functools
import itertools
import math
import random

from typing import Tuple

from src.utils.debug import *


@functools.cache
def get_demand_vectors(
    num_objs: int,
    demand_ordered_for_most_popular_objs: Tuple[float],
) -> list[list[float]]:
    num_popular_objs = len(demand_ordered_for_most_popular_objs)
    base_demand_vector = (
        demand_ordered_for_most_popular_objs
        + (num_objs - num_popular_objs) * (0,)
    )

    return [list(p) for p in itertools.permutations(base_demand_vector)]


def gen_demand_vector(
    num_objs: int,
    demand_ordered_for_most_popular_objs: Tuple[float],
) -> list[float]:
    num_popular_objs = len(demand_ordered_for_most_popular_objs)
    base_demand_vector = (
        demand_ordered_for_most_popular_objs
        + (num_objs - num_popular_objs) * (0,)
    )

    for p in itertools.permutations(base_demand_vector):
        yield list(p)


def get_ordered_demand_vector(
    num_popular_objs: int,
    cum_demand: float,
    zipf_tail_index: float,
) -> list[float]:
    k, a = num_popular_objs, zipf_tail_index

    # kth generalized Harmonic number
    H_k_a = sum(math.pow(1 / i, a) for i in range(1, k + 1))

    ordered_prob_list = [math.pow(1 / i, a) / H_k_a for i in range(1, k + 1)]
    ordered_demand_vector = tuple([p * cum_demand for p in ordered_prob_list])

    return ordered_demand_vector


@functools.cache
def get_demand_vectors_w_zipf_law(
    num_objs: int,
    num_popular_objs: int,
    cum_demand: float,
    zipf_tail_index: float,
) -> list[list[float]]:
    ordered_demand_vector = get_ordered_demand_vector(
        num_popular_objs=num_popular_objs,
        cum_demand=cum_demand,
        zipf_tail_index=zipf_tail_index,
    )

    # log(DEBUG, "",
    #     ordered_demand_vector=ordered_demand_vector,
    #     cum_demand=cum_demand,
    #     zipf_tail_index=zipf_tail_index,
    # )

    return get_demand_vectors(
        num_objs=num_objs,
        demand_ordered_for_most_popular_objs=ordered_demand_vector,
    )


def gen_demand_vector_w_zipf_law(
    num_objs: int,
    num_popular_objs: int,
    cum_demand: float,
    zipf_tail_index: float,
) -> list[float]:
    ordered_demand_vector = get_ordered_demand_vector(
        num_popular_objs=num_popular_objs,
        cum_demand=cum_demand,
        zipf_tail_index=zipf_tail_index,
    )

    # log(DEBUG, "",
    #     ordered_demand_vector=ordered_demand_vector,
    #     cum_demand=cum_demand,
    #     zipf_tail_index=zipf_tail_index,
    # )

    for demand_vector in gen_demand_vector(
        num_objs=num_objs,
        demand_ordered_for_most_popular_objs=ordered_demand_vector,
    ):
        yield demand_vector


def sample_demand_vectors_w_zipf_law(
    num_objs: int,
    num_popular_objs: int,
    cum_demand: float,
    zipf_tail_index: float,
    num_samples: int,
) -> list[float]:
    ordered_demand_vector = get_ordered_demand_vector(
        num_popular_objs=num_popular_objs,
        cum_demand=cum_demand,
        zipf_tail_index=zipf_tail_index,
    )

    # log(DEBUG, "",
    #     ordered_demand_vector=ordered_demand_vector,
    #     cum_demand=cum_demand,
    #     zipf_tail_index=zipf_tail_index,
    # )

    for sample_id in range(num_samples):
        log(DEBUG, f"> sample_id= {sample_id}")

        demand_vector = num_objs * [0]
        index_list = random.sample(list(range(num_objs)), num_popular_objs)
        for i, index in enumerate(index_list):
            demand_vector[index] = ordered_demand_vector[i]

        yield demand_vector
