import itertools
import math

from typing import Tuple


def is_demand_vector_covered(
    demand_vector: list[float],
    obj_id_to_node_id_set_map: dict[int, set[int]],
) -> bool:
    k = len(demand_vector)
    for index_combination in itertools.combinations(list(range(k))):
        cum_demand = 0
        node_id_set = set()
        for i in index_combination:
            node_id_set |= obj_id_to_node_id_set_map[i]
            cum_demand += demand_vector[i]

        if math.ceil(cum_demand) > len(node_id_set):
            return False

    return True


def get_demand_vector_to_covered_or_not_map(
    demand_vector_list: list[list[float]],
    obj_id_to_node_id_set_map: dict[int, set[int]],
) -> dict[Tuple[float], bool]:
    return {
        tuple(demand_vector): is_demand_vector_covered(
            demand_vector=demand_vector,
            obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
        )
        for demand_vector in demand_vector_list
    }


def frac_of_demand_vectors_covered(
    demand_vector_list: list[list[float]],
    obj_id_to_node_id_set_map: dict[int, set[int]],
) -> float:
    num_covered = 0

    for demand_vector in demand_vector_list:
        if is_demand_vector_covered(
            demand_vector=demand_vector,
            obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
        ):
            num_covered += 1

    return num_covered / len(demand_vector_list)
