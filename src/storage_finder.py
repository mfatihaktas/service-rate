import math
import itertools


class StorageFinder:
    def __init__(self, demand_list: list[float]):
        self.demand_list = sorted(demand_list)

        self.n = len(self.demand_list)
        self.obj_id_subset_to_min_span_size_map = {}
        self.find_min_span_sizes()

    def find_min_span_sizes(self):
        obj_id_list = list(range(self.n))
        for subset_size in range(1, self.n):
            for obj_id_subset in itertools.combinations(obj_id_list, subset_size):
                min_span_size = math.ceil(sum(self.demand_list[obj_id] for obj_id in obj_id_subset))
                self.obj_id_subset_to_min_span_size_map[frozenset(obj_id_subset)] = min_span_size

        log(DEBUG, "Done", obj_id_subset_to_min_span_size_map=self.obj_id_subset_to_min_span_size_map)
