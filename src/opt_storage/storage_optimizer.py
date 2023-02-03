import itertools
import math

from typing import Tuple

from src.debug_utils import *


class StorageOptimizer:
    def __init__(self, demand_list: list[float]):
        self.demand_list = demand_list

        self.n = len(self.demand_list)
        self.obj_id_subset_to_min_span_size_map = (
            self.get_obj_id_subset_to_min_span_size_map()
        )

        self.obj_id_to_node_id_set_map = {}

    def get_obj_id_subset_to_min_span_size_map(self) -> dict[Tuple[int], int]:
        obj_id_subset_to_min_span_size_map = {}

        obj_id_list = list(range(self.n))
        for subset_size in range(1, self.n):
            for obj_id_subset in itertools.combinations(obj_id_list, subset_size):
                min_span_size = math.ceil(
                    sum(self.demand_list[obj_id] for obj_id in obj_id_subset)
                )
                obj_id_subset_to_min_span_size_map[
                    frozenset(obj_id_subset)
                ] = min_span_size

        log(
            DEBUG,
            "Done",
            obj_id_subset_to_min_span_size_map=obj_id_subset_to_min_span_size_map,
        )
        return obj_id_subset_to_min_span_size_map

    # def assign_objects_to_nodes(self):
    #     """Assumes nodes are ranked from 0 to n - 1 w.r.t their popularity.
    #     """

    #     def is_proper_set():
    #     obj_id_to_node_id_set_map = {}

    #     def get_possible_node_ids(obj_id: int, ) -> list[int]:
    #         possible_node_id_set = set()
    #         num_possible_node_ids = 0

    #         # obj_min_span_size = self.obj_id_subset_to_min_span_size_map[frozenset([obj_id])]

    #         obj_id_list = list(range(obj_id))
    #         for subset_size in range(1, obj_id + 1):
    #             for obj_id_subset in itertools.combinations(obj_id_list, subset_size):
    #                 _span = set()
    #                 for _obj_id in obj_id_subset:
    #                     _span |= obj_id_to_node_id_set_map[_obj_id]

    #                 obj_id_subset_ = obj_id_subset + (obj_id,)
    #                 min_span_size = self.obj_id_subset_to_min_span_size_map[frozenset(obj_id_subset_)]
    #                 max_intersection_size + _span = min_span_size - len()

    #     def helper(obj_id: int):
    #         if obj_id == self.n:
    #             return obj_id_to_node_id_set_map

    #         min_span_size = self.obj_id_subset_to_min_span_size_map[frozenset([obj_id])]
    #         if obj_id == 0:
    #             node_id_set = set(range(min_span_size))
    #             obj_id_to_node_id_set_map[obj_id] = node_id_set
    #             helper(obj_id=obj_id + 1)

    #         else:

    #             min_span_size
    #             helper(obj_id=obj_id + 1)
