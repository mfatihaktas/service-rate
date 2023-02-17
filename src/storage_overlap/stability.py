import itertools
import math

from typing import Tuple

from src.storage_overlap import design


# def get_demand_vector_to_covered_or_not_map(
#     demand_vector_list: list[list[float]],
#     obj_id_to_node_id_set_map: dict[int, set[int]],
# ) -> dict[Tuple[float], bool]:
#     return {
#         tuple(demand_vector): is_demand_vector_covered(
#             demand_vector=demand_vector,
#             obj_id_to_node_id_set_map=obj_id_to_node_id_set_map,
#         )
#         for demand_vector in demand_vector_list
#     }
