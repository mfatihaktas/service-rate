import cvxpy
import itertools
import math
import numpy

from typing import Tuple

from src.utils.debug import *


class StorageOptimizer:
    def __init__(self, demand_vector_list: list[list[float]], max_num_nodes: int = None):
        self.demand_vector_list = demand_vector_list
        self.max_num_nodes = max_num_nodes

        self.k = len(self.demand_vector_list[0])
        self.obj_id_set_to_min_span_size_map = self.get_obj_id_subset_to_min_span_size_map()
        log(DEBUG, "", obj_id_set_to_min_span_size_map=self.obj_id_set_to_min_span_size_map)

        max_min_span_size = max(min_span_size for _, min_span_size in self.obj_id_set_to_min_span_size_map.items())
        if self.max_num_nodes is None or self.max_num_nodes < max_min_span_size:
            log(INFO, f"Setting max_num_nodes to max_min_span_size= {max_min_span_size}")
            self.max_num_nodes = max_min_span_size

    def get_obj_id_subset_to_min_span_size_map(self) -> dict[Tuple[int], int]:
        obj_id_set_to_min_span_size_map = {}

        for subset_size in range(1, self.k + 1):
            for obj_id_subset in itertools.combinations(list(range(self.k)), subset_size):
                min_span_size = float("-Inf")

                for demand_vector in self.demand_vector_list:
                    min_span_size_ = math.ceil(sum(demand_vector[obj_id] for obj_id in obj_id_subset))
                    min_span_size = max(min_span_size, min_span_size_)

                obj_id_set_to_min_span_size_map[frozenset(obj_id_subset)] = min_span_size

        log(DEBUG, "Done",
            obj_id_set_to_min_span_size_map=obj_id_set_to_min_span_size_map,
        )

        return obj_id_set_to_min_span_size_map


class StorageOptimizerReplication(StorageOptimizer):
    def __init__(self, demand_vector_list: list[list[float]], max_num_nodes: int = None):
        super().__init__(demand_vector_list=demand_vector_list, max_num_nodes=max_num_nodes)

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

            # x v y v z = [x' ^ y' ^ z']' = [(1 - x) ^ (1 - y) ^ (1 - z)]'
            # => |x v y v z| >= s is equivalent to |(1 - x) ^ (1 - y) ^ (1 - z)| <= k - s
            z = cvxpy.Variable(shape=(n, 1), name=f"z_{counter}", boolean=True)

            for i in obj_id_set:
                constraint_list.append(1 - cvxpy.reshape(x[i, :], shape=(n, 1)) >= z)

            num_objs = len(obj_id_set)
            x_i_complement_in_columns = cvxpy.vstack([1 - x[i, :] for i in obj_id_set]).T
            sum_x_i_complement = x_i_complement_in_columns @ numpy.ones((num_objs, 1))
            # log(DEBUG, "", x_i_complement_in_columns=x_i_complement_in_columns, sum_x_i_complement=sum_x_i_complement)
            constraint_list.append(sum_x_i_complement - len(obj_id_set) + 1 <= z)

            constraint_list.append(cvxpy.sum(z) <= n - min_span_size)

        C = numpy.array([[i + 1] for i in range(n)])
        # log(DEBUG, "", C=C, constraint_list=constraint_list)
        obj = cvxpy.Minimize(cvxpy.sum(x @ C))

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        log(DEBUG, "", prob_value=prob.value, x_value=x.value)

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!", x=x.value)

        obj_id_to_node_id_set_map = {
            obj_id : set(
                node_id
                for node_id in range(n)
                if x.value[obj_id, node_id] == 1
            )
            for obj_id in range(k)
        }
        return obj_id_to_node_id_set_map
    def __init__(self, demand_vector_list: list[list[float]], max_num_nodes: int = None):
        self.demand_vector_list = demand_vector_list
        self.max_num_nodes = max_num_nodes

        self.k = len(self.demand_vector_list[0])
        self.obj_id_set_to_min_span_size_map = self.get_obj_id_subset_to_min_span_size_map()
        log(DEBUG, "", obj_id_set_to_min_span_size_map=self.obj_id_set_to_min_span_size_map)

        max_min_span_size = max(min_span_size for _, min_span_size in self.obj_id_set_to_min_span_size_map.items())
        if self.max_num_nodes is None or self.max_num_nodes < max_min_span_size:
            log(INFO, f"Setting max_num_nodes to max_min_span_size= {max_min_span_size}")
            self.max_num_nodes = max_min_span_size

        self.obj_id_to_node_id_set_map = {}

    def get_obj_id_subset_to_min_span_size_map(self) -> dict[Tuple[int], int]:
        obj_id_set_to_min_span_size_map = {}

        for subset_size in range(1, self.k + 1):
            for obj_id_subset in itertools.combinations(list(range(self.k)), subset_size):
                min_span_size = float("-Inf")

                for demand_vector in self.demand_vector_list:
                    min_span_size_ = math.ceil(sum(demand_vector[obj_id] for obj_id in obj_id_subset))
                    min_span_size = max(min_span_size, min_span_size_)

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

            # x v y v z = [x' ^ y' ^ z']' = [(1 - x) ^ (1 - y) ^ (1 - z)]'
            # => |x v y v z| >= s is equivalent to |(1 - x) ^ (1 - y) ^ (1 - z)| <= k - s
            z = cvxpy.Variable(shape=(n, 1), name=f"z_{counter}", boolean=True)

            for i in obj_id_set:
                constraint_list.append(1 - cvxpy.reshape(x[i, :], shape=(n, 1)) >= z)

            num_objs = len(obj_id_set)
            x_i_complement_in_columns = cvxpy.vstack([1 - x[i, :] for i in obj_id_set]).T
            sum_x_i_complement = x_i_complement_in_columns @ numpy.ones((num_objs, 1))
            # log(DEBUG, "", x_i_complement_in_columns=x_i_complement_in_columns, sum_x_i_complement=sum_x_i_complement)
            constraint_list.append(sum_x_i_complement - len(obj_id_set) + 1 <= z)

            constraint_list.append(cvxpy.sum(z) <= n - min_span_size)

        C = numpy.array([[i + 1] for i in range(n)])
        # log(DEBUG, "", C=C, constraint_list=constraint_list)
        obj = cvxpy.Minimize(cvxpy.sum(x @ C))

        prob = cvxpy.Problem(obj, constraint_list)
        prob.solve(solver="SCIP")

        log(DEBUG, "", prob_value=prob.value, x_value=x.value)

        check(prob.status == cvxpy.OPTIMAL, "Solution to optimization problem is NOT optimal!", x=x.value)

        obj_id_to_node_id_set_map = {
            obj_id : set(
                node_id
                for node_id in range(n)
                if x.value[obj_id, node_id] == 1
            )
            for obj_id in range(k)
        }
        return obj_id_to_node_id_set_map
