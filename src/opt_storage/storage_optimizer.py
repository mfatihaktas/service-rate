import itertools
import math

from typing import Tuple

from src.debug_utils import *


class StorageOptimizer:
    def __init__(self, demand_list: list[float]):
        self.demand_list = demand_list

        self.k = len(self.demand_list)
        self.obj_id_set_to_min_span_size_map = self.get_obj_id_subset_to_min_span_size_map()
        self.max_num_nodes = max(min_span_size for _, min_span_size in self.obj_id_set_to_min_span_size_map.items())

        self.obj_id_to_node_id_set_map = {}

    def get_obj_id_subset_to_min_span_size_map(self) -> dict[Tuple[int], int]:
        obj_id_set_to_min_span_size_map = {}

        for subset_size in range(1, self.k):
            for obj_id_subset in itertools.combinations(list(range(self.k)), subset_size):
                min_span_size = math.ceil(sum(self.demand_list[obj_id] for obj_id in obj_id_subset))
                obj_id_set_to_min_span_size_map[frozenset(obj_id_subset)] = min_span_size

        log(DEBUG, "Done",
            obj_id_set_to_min_span_size_map=obj_id_set_to_min_span_size_map,
        )

        return obj_id_set_to_min_span_size_map

    def get_obj_id_to_node_id_set_map(self) -> dict[int, set[int]]:
        k, n = self.k, self.max_num_nodes
        log(DEBUG, "Started", k=k, n=n)

        x = cvxpy.Variable(shape=(k, n), name="x", boolean=True)
        constraint_list = []

        # Span constraints
        for counter, (obj_id_set, min_span_size) in enumerate(self.obj_id_set_to_min_span_size_map.items()):
            log(DEBUG, f">> counter= {counter}", obj_id_set=obj_id_set, min_span_size=min_span_size)

            if len(obj_id_set) == 1:
                obj_id = next(iter(obj_id_set))
                constraint_list.append(cvxpy.sum(x[obj_id, :]) >= min_span_size)
                continue

            # Let
            # - i, j, k be the object indices.
            # - C_i be the set of nodes/choices for obj_i.
            # |C_i ^ C_j ^ C_l| = |C_i| + |C_j| + |C_l| -
            #
            z = cvxpy.Variable(shape=(n, 1), name=f"z_{counter}", boolean=True)

            for i in obj_id_set:
                constraint_list.append(cvxpy.reshape(x[i, :], shape=(n, 1)) >= z)

            num_objs = len(obj_id_set)
            x_i_in_columns = cvxpy.vstack([x[i, :] for i in obj_id_set]).T
            sum_x_i = x_i_in_columns @ numpy.ones((num_objs, 1))
            log(DEBUG, "", x_i_in_columns=x_i_in_columns, sum_x_i=sum_x_i)
            constraint_list.append(sum_x_i - len(obj_id_set) + 1 <= z)

            obj_choice_union_size = sum(self.obj_id_set_to_min_span_size_map[i] for i in obj_id_set)
            constraint_list.append(cvxpy.sum(z) <= obj_choice_union_size - min_span_size)
            log(DEBUG, "", obj_choice_union_size=obj_choice_union_size, min_span_size=min_span_size)

        C = numpy.array([[i + 1] for i in range(n)])
        log(DEBUG, "", C=C, constraint_list=constraint_list)
        obj = cvxpy.Minimize(cvxpy.sum(x @ C))

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")
