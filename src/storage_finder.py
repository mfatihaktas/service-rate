import collections
import itertools
import math

from typing import Tuple

from src.debug_utils import *


class StorageFinder:
    def __init__(self, demand_list: list[float]):
        self.demand_list = demand_list

        self.n = len(self.demand_list)
        self.obj_id_subset_to_min_span_size_map = self.get_obj_id_subset_to_min_span_size_map()

        self.obj_id_to_node_id_set_map = {}

    def get_obj_id_subset_to_min_span_size_map(self) -> dict[Tuple[int], int]:
        obj_id_subset_to_min_span_size_map = {}

        obj_id_list = list(range(self.n))
        for subset_size in range(1, self.n):
            for obj_id_subset in itertools.combinations(obj_id_list, subset_size):
                min_span_size = math.ceil(sum(self.demand_list[obj_id] for obj_id in obj_id_subset))
                obj_id_subset_to_min_span_size_map[frozenset(obj_id_subset)] = min_span_size

        log(DEBUG, "Done", obj_id_subset_to_min_span_size_map=obj_id_subset_to_min_span_size_map)
        return obj_id_subset_to_min_span_size_map

    # def find_max_intersection_sizes(self):
    #     # Construct `obj_id_subset_to_min_span_size_map`
    #     obj_id_subset_to_min_span_size_map = {}
    #     obj_id_list = list(range(self.n))
    #     for subset_size in range(1, self.n):
    #         for obj_id_subset in itertools.combinations(obj_id_list, subset_size):
    #             min_span_size = math.ceil(sum(self.demand_list[obj_id] for obj_id in obj_id_subset))
    #             obj_id_subset_to_min_span_size_map[frozenset(obj_id_subset)] = min_span_size
    #     log(DEBUG, "Done", obj_id_subset_to_min_span_size_map=self.obj_id_subset_to_min_span_size_map)

    #     # Construct `obj_id_subset_to_max_intersect_size_map`
    #     obj_id_subset_to_max_intersect_size_map = {}
    #     for obj_id_subset, min_span_size in obj_id_subset_to_min_span_size_map.items():
    #         # TODO
    #         obj_id_subset_to_max_intersect_size_map[frozenset(obj_id_subset)] = min_span_size
    #     log(DEBUG, "Done", obj_id_subset_to_min_span_size_map=self.obj_id_subset_to_min_span_size_map)


    def assign_objects_to_nodes(self):
        """Assumes nodes are ranked from 0 to n - 1 w.r.t their popularity.
        """
        node_id_to_objs_map = collections.defaultdict(list)
        min_num_objs_on_a_node = 0

        def can_put_in_node(obj_id: int, node_id: int) -> bool:
            obj_list = node_id_to_objs_map[node_id]
            obj_id_subset = frozenset([obj_id, *obj_list])
            min_span_size = self.obj_id_subset_to_min_span_size_map[obj_id_subset]


        for obj_id in range(self.n):
            min_span_size = self.obj_id_subset_to_min_span_size_map[frozenset([obj_id])]



            if obj_id == 0:
                node_id_set = set(range(min_span_size))
                self.obj_id_to_node_id_set_map[obj_id] = node_id_set

                for node_id in node_id_set:
                    node_id_to_objs_map[node_id].add(obj_id)
                min_num_objs_on_a_node = 1

                continue

            for node_id, obj_list in node_id_to_objs_map.items():
                if len(obj_list) > min_num_objs_on_a_node:
                    continue



            min_num_objs_on_a_node

            for _obj_id in range(obj_id):

            node_id_to_objs_map
